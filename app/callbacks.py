"""
app/callbacks.py — MYTIGATE_futures Dash callbacks.

State machine:
  IDLE      — app opened, or scenario changed → Steps 2/3/4 reset
  COMPUTED  — farm computed for current data → Steps 3/4 unlocked

Map toggle (Baseline/Scenario): visible in IDLE for env browsing,
  hidden after compute (results fixed to computed scenario).
Crop toggle (Farm/Full grid): visible after compute only.

Polygon persists across scenario changes — user can recompute same farm
for different scenarios without redrawing.

Farm ID increments each time Compute succeeds. Used in export filenames.
Export filenames: mytigate_farm{ID}_{scenario_slug}_{type}.csv
"""
from __future__ import annotations
import io
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import dash
from dash import Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate

from config import (
    BASELINE_GPKG, GRID_GPKG, PRIMARY_HARVEST, HARVEST_MONTHS,
)
from app.map_utils import (build_map, build_env_map, empty_map,
                           build_exclusion_outline_map)
from app.layout import render_farm_summary, render_cell_report
from model.site_selector import cell_report
from model.growth import run_module2, run_module3, run_module4, N_CPU
from model.module12_farm import discretise_farm
from utils.raster_sampler import run_module1
from utils.climate_projections import load_scenario_env

HARVEST_MONTHS_STR = [f"{h:02d}" for h in HARVEST_MONTHS]
N_SC_APP = 100
RDEP_MAP  = {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0}
ILOOP_MAP = {0: 0.7, 1: 1.0, 2: 1.5, 3: 2.0}


# ── Server-side storage ───────────────────────────────────────────────────────

def load_all_grids(app: dash.Dash) -> None:
    app.server._baseline       = _load_grid(BASELINE_GPKG)
    app.server._base_env_grid  = _load_grid(GRID_GPKG)
    app.server._scenario       = None
    app.server._farm_cells     = None   # farm_cells after M12+M2+M3+M4
    app.server._m12_result     = None
    app.server._scenario_label = ""
    app.server._drawn_shapes   = []
    app.server._computed_label = "baseline"   # scenario label at compute time
    print("[App] Startup complete")


def _load_grid(path) -> gpd.GeoDataFrame | None:
    if not Path(path).exists():
        print(f"[App] Not found: {path}")
        return None
    gdf = gpd.read_file(path)
    if "lat" not in gdf.columns:
        pts = gdf.geometry.centroid.to_crs("EPSG:4326")
        gdf["lon"], gdf["lat"] = pts.x, pts.y
    print(f"[App] Loaded {Path(path).name}: {len(gdf):,} cells")
    return gdf


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _parse_path_to_lonlat(path_str, map_bounds=None, map_center=None, map_zoom=None):
    if not path_str:
        return None
    try:
        segments = re.split(r'[MLZ]', path_str)
        paper_coords = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            parts = seg.split(",")
            if len(parts) == 2:
                paper_coords.append((float(parts[0]), float(parts[1])))
    except Exception as e:
        print(f"[Shape] Parse error: {e}")
        return None
    if len(paper_coords) < 3:
        return None
    if map_bounds:
        lon_min = map_bounds.get("lon_min", 9.0)
        lon_max = map_bounds.get("lon_max", 14.0)
        lat_min = map_bounds.get("lat_min", 54.0)
        lat_max = map_bounds.get("lat_max", 58.0)
        lonlat = [(lon_min + px*(lon_max-lon_min),
                   lat_min + py*(lat_max-lat_min))
                  for px, py in paper_coords]
    elif map_center and map_zoom is not None:
        center_lon = map_center.get("lon", 11.5)
        center_lat = map_center.get("lat", 56.3)
        world_px = 256.0 * (2 ** map_zoom)
        def _lon2x(lo): return world_px * (lo + 180) / 360
        def _lat2y(la):
            r = math.radians(la)
            return world_px * (1 - math.log(math.tan(r)+1/math.cos(r))/math.pi)/2
        cx, cy = _lon2x(center_lon), _lat2y(center_lat)
        lonlat = []
        for px, py in paper_coords:
            tx = cx + (px - 0.5) * 1260
            ty = cy + (0.5 - py) * 1034
            lo = tx / world_px * 360 - 180
            n  = math.pi - 2*math.pi*ty/world_px
            la = math.degrees(math.atan(math.sinh(n)))
            lonlat.append((lo, la))
    else:
        return None
    print(f"[Shape] {len(lonlat)} vertices — "
          f"lon {min(l[0] for l in lonlat):.3f}–{max(l[0] for l in lonlat):.3f}  "
          f"lat {min(l[1] for l in lonlat):.3f}–{max(l[1] for l in lonlat):.3f}")
    return lonlat


def _slug(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_").replace("(","").replace(")","").replace("~","").replace("+","").strip("_")


def _env_cols(gdf):
    cols = ["cell_id", "lon", "lat", "bathymetry_m"]
    for prefix in ["temp_mean_", "sal_mean_", "lnchla_mean_", "vh_"]:
        cols += [c for c in gdf.columns if c.startswith(prefix)]
    return [c for c in cols if c in gdf.columns]


def _result_cols(gdf):
    cols = ["cell_id", "lon", "lat", "excluded", "conflict_score", "rdep"]
    for prefix in ["harvest_WW_", "harvest_DW_", "shell_DW_",
                   "N_red_", "N_red_ha_", "P_red_",
                   "mbio_p50_", "vh_ratio_", "food_limitation_"]:
        cols += [f"{prefix}{mk}" for mk in HARVEST_MONTHS_STR
                 if f"{prefix}{mk}" in gdf.columns]
    return [c for c in cols if c in gdf.columns]


def _to_csv(gdf, cols, fmt="%.4f"):
    buf = io.StringIO()
    pd.DataFrame(gdf[cols]).to_csv(buf, index=False, float_format=fmt)
    return buf.getvalue()


# ── Register callbacks ────────────────────────────────────────────────────────

def register_callbacks(app: dash.Dash) -> None:

    # ── Show/hide CMIP6 / delta panels ───────────────────────────────────────
    @app.callback(
        Output("div-cmip6-panel", "style"),
        Output("div-delta-panel", "style"),
        Input("radio-mode", "value"),
    )
    def toggle_mode_panels(mode):
        show, hide = {"display": "block"}, {"display": "none"}
        return (show if mode == "cmip6" else hide,
                show if mode == "delta" else hide)

    # ── Scenario / mode change → reset Steps 2/3/4 ───────────────────────────
    @app.callback(
        Output("store-compute-done",   "data", allow_duplicate=True),
        Output("store-scenario-ready", "data", allow_duplicate=True),
        Output("store-scenario-label", "data", allow_duplicate=True),
        Output("store-active-view",    "data", allow_duplicate=True),
        Output("div-scenario-status",  "children", allow_duplicate=True),
        Output("div-delta-status",     "children", allow_duplicate=True),
        Output("topbar-status",        "children", allow_duplicate=True),
        Input("radio-mode", "value"),
        prevent_initial_call=True,
    )
    def reset_on_mode_change(mode):
        # Clear scenario, reset compute state, keep drawn polygon
        app.server._scenario       = None
        app.server._farm_cells     = None
        app.server._computed_label = "baseline"
        label = "Baseline" if mode == "baseline" else "Select scenario and load"
        return False, False, "", "baseline", "", "", f"Mode: {label} — draw farm to compute"

    # ── Load CMIP6 scenario ───────────────────────────────────────────────────
    @app.callback(
        Output("store-scenario-ready", "data",  allow_duplicate=True),
        Output("store-scenario-label", "data",  allow_duplicate=True),
        Output("div-scenario-status",  "children"),
        Output("store-compute-done",   "data",  allow_duplicate=True),
        Output("topbar-status",        "children", allow_duplicate=True),
        Input("btn-load-scenario", "n_clicks"),
        State("dd-ssp",    "value"),
        State("dd-period", "value"),
        prevent_initial_call=True,
    )
    def load_cmip6(n, ssp, period):
        if not n:
            raise PreventUpdate
        base_env = app.server._base_env_grid
        if base_env is None:
            return False, "", "❌ Base grid not found", False, ""
        try:
            grid_scen = load_scenario_env(base_env, ssp=ssp, period=period)
        except FileNotFoundError as e:
            return False, "", f"❌ {str(e)[:80]}", False, ""
        app.server._scenario       = grid_scen
        app.server._farm_cells     = None   # reset farm results — must recompute
        app.server._computed_label = ""
        from config_scenario import SSP_SCENARIOS
        label  = f"{SSP_SCENARIOS.get(ssp, ssp)} / {period}"
        status = f"Scenario: {label} — draw farm and compute"
        return True, label, f"✅ {label} loaded", False, status

    # ── Load custom delta scenario ────────────────────────────────────────────
    @app.callback(
        Output("store-scenario-ready", "data",  allow_duplicate=True),
        Output("store-scenario-label", "data",  allow_duplicate=True),
        Output("div-delta-status",     "children"),
        Output("store-compute-done",   "data",  allow_duplicate=True),
        Output("topbar-status",        "children", allow_duplicate=True),
        Input("btn-run-delta", "n_clicks"),
        State("slider-delta-temp",  "value"),
        State("slider-delta-sal",   "value"),
        State("slider-chla-mult",   "value"),
        prevent_initial_call=True,
    )
    def run_delta(n, dtemp, dsal, chla_mult):
        if not n:
            raise PreventUpdate
        base_env = app.server._base_env_grid
        if base_env is None:
            return False, "", "❌ Base grid not found", False, ""
        dtemp     = dtemp or 0.0
        dsal      = dsal  or 0.0
        chla_mult = chla_mult or 1.0
        grid_scen = run_module1(base_env, delta_temp=dtemp, delta_sal=dsal,
                                chla_mult=chla_mult, force_resample=False)
        app.server._scenario       = grid_scen
        app.server._farm_cells     = None
        app.server._computed_label = ""
        label  = f"ΔT={dtemp:+.1f}°C ΔS={dsal:+.1f}psu ChlA×{chla_mult:.2f}"
        status = f"Scenario: {label} — draw farm and compute"
        return True, label, f"✅ {label} ready", False, status

    # ── Env browse map toggle: show when scenario loaded, hide after compute ──
    @app.callback(
        Output("div-map-toggle", "style"),
        Input("store-scenario-ready", "data"),
        Input("store-compute-done",   "data"),
    )
    def show_env_toggle(scen_ready, compute_done):
        if compute_done:
            return {"display": "none"}   # hide after compute
        return {"display": "flex"} if scen_ready else {"display": "none"}

    # ── Auto-switch view to scenario when scenario loads ──────────────────────
    @app.callback(
        Output("btn-view-baseline", "className", allow_duplicate=True),
        Output("btn-view-scenario", "className", allow_duplicate=True),
        Output("store-active-view", "data",      allow_duplicate=True),
        Input("store-scenario-label", "data"),
        prevent_initial_call=True,
    )
    def auto_switch_to_scenario(label):
        if not label:
            raise PreventUpdate
        a, i = "map-toggle-btn active", "map-toggle-btn inactive"
        return i, a, "scenario"

    # ── Map toggle button styling ─────────────────────────────────────────────
    @app.callback(
        Output("btn-view-baseline", "className"),
        Output("btn-view-scenario", "className"),
        Output("store-active-view", "data"),
        Input("btn-view-baseline",  "n_clicks"),
        Input("btn-view-scenario",  "n_clicks"),
        State("store-active-view",  "data"),
        prevent_initial_call=True,
    )
    def toggle_view(nb, ns, current):
        view = ("baseline" if ctx.triggered_id == "btn-view-baseline"
                else "scenario" if ctx.triggered_id == "btn-view-scenario"
                else current or "baseline")
        a, i = "map-toggle-btn active", "map-toggle-btn inactive"
        return (a if view=="baseline" else i,
                a if view=="scenario" else i, view)

    # ── Crop toggle ───────────────────────────────────────────────────────────
    @app.callback(
        Output("div-crop-toggle", "style"),
        Input("store-compute-done", "data"),
    )
    def show_crop_toggle(done):
        return {"display": "flex"} if done else {"display": "none"}

    @app.callback(
        Output("btn-crop-farm",   "className"),
        Output("btn-crop-full",   "className"),
        Output("store-crop-mode", "data"),
        Input("btn-crop-farm",    "n_clicks"),
        Input("btn-crop-full",    "n_clicks"),
        State("store-crop-mode",  "data"),
        prevent_initial_call=True,
    )
    def toggle_crop(nf, ng, current):
        mode = ("farm" if ctx.triggered_id == "btn-crop-farm"
                else "full" if ctx.triggered_id == "btn-crop-full"
                else current or "farm")
        a, i = "map-toggle-btn active", "map-toggle-btn inactive"
        return (a if mode=="farm" else i, a if mode=="full" else i, mode)

    # ── Live zoom warning ─────────────────────────────────────────────────────
    @app.callback(
        Output("div-zoom-warning", "children"),
        Output("div-zoom-warning", "style"),
        Input("store-map-viewport", "data"),
    )
    def zoom_warning(viewport):
        zoom = (viewport or {}).get("zoom", 0)
        s = {"marginTop": "6px", "fontSize": ".70rem", "fontWeight": "600"}
        if zoom == 0:
            return "", {**s, "display": "none"}
        if zoom < 9:
            return (f"⚠  Zoom in more  (current: {zoom:.1f}, need ≥ 10)",
                    {**s, "color": "#f06a6a"})
        if zoom < 10:
            return (f"⚠  Almost there  (current: {zoom:.1f})",
                    {**s, "color": "#f0a040"})
        return (f"✓  Ready to draw  (zoom {zoom:.1f})",
                {**s, "color": "#4caf79"})

    # ── Capture drawn shape ───────────────────────────────────────────────────
    @app.callback(
        Output("store-drawn-shape",           "data"),
        Output("div-restriction-warning",     "children"),
        Output("div-restriction-warning",     "style"),
        Output("store-new-polygon-drawn",     "data"),
        Input("map-graph", "relayoutData"),
        State("store-map-viewport",  "data"),
        State("store-drawn-shape",   "data"),
        prevent_initial_call=True,
    )
    def capture_shape(relayout_data, viewport, current_shape):
        bs = {"fontSize": ".70rem", "fontWeight": "600",
              "marginBottom": "6px", "minHeight": "14px"}
        if not relayout_data:
            raise PreventUpdate
        shapes = relayout_data.get("shapes", [])
        if not shapes:
            raise PreventUpdate
        path = shapes[-1].get("path", "")
        if not path:
            raise PreventUpdate
        # Only update if path is genuinely new
        current_path = (current_shape or {}).get("shape_path", "")
        if path == current_path:
            raise PreventUpdate

        print(f"[Shape] New polygon drawn ({len(path)} chars)")
        app.server._drawn_shapes = shapes

        warn_text  = ""
        warn_style = {**bs, "color": "#4caf79"}
        vp = viewport or {}
        coords = _parse_path_to_lonlat(path, map_bounds=vp.get("bounds"),
                                       map_center=vp.get("center"),
                                       map_zoom=vp.get("zoom"))
        if coords and app.server._base_env_grid is not None:
            try:
                from model.module12_farm import lonlat_to_utm
                polygon_utm = lonlat_to_utm(coords)
                grid     = app.server._base_env_grid
                grid_utm = grid.to_crs("EPSG:32632") \
                           if grid.crs.to_epsg() != 32632 else grid
                cands  = list(grid_utm.sindex.query(polygon_utm, predicate="intersects"))
                cells  = grid_utm.iloc[cands]
                n_total = len(cells)
                if n_total == 0:
                    warn_text  = "⚠  No grid cells found in polygon"
                    warn_style = {**bs, "color": "#f0a040"}
                elif "excluded" in cells.columns:
                    n_excl = int(cells["excluded"].sum())
                    pct    = 100 * n_excl / n_total
                    if n_excl == n_total:
                        warn_text  = f"🚫  Fully restricted — all {n_total} cells excluded"
                        warn_style = {**bs, "color": "#f06a6a"}
                    elif n_excl > 0:
                        warn_text  = f"⚠  Partial restriction — {n_excl}/{n_total} excluded ({pct:.0f}%)"
                        warn_style = {**bs, "color": "#f0a040"}
                    else:
                        warn_text  = f"✓  No restrictions — {n_total} cells available"
                        warn_style = {**bs, "color": "#4caf79"}
            except Exception as e:
                print(f"[Shape] Exclusion check error: {e}")

        return {"shape_path": path}, warn_text, warn_style, True

    # ── Track viewport ────────────────────────────────────────────────────────
    @app.callback(
        Output("store-map-viewport", "data"),
        Input("map-graph", "relayoutData"),
        State("store-map-viewport", "data"),
        prevent_initial_call=True,
    )
    def track_viewport(relayout_data, current):
        if not relayout_data:
            raise PreventUpdate
        vp = current or {"center": {"lon": 11.5, "lat": 56.3}, "zoom": 7.0}
        updated = False
        if "map.center" in relayout_data:
            vp["center"] = relayout_data["map.center"]; updated = True
        if "map.zoom" in relayout_data:
            vp["zoom"]   = relayout_data["map.zoom"];   updated = True
        if "map._derived" in relayout_data:
            derived = relayout_data["map._derived"]
            coords  = derived.get("coordinates", {})
            if coords:
                try:
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    vp["bounds"] = {"lon_min": min(lons), "lon_max": max(lons),
                                    "lat_min": min(lats), "lat_max": max(lats)}
                    updated = True
                    print(f"[Viewport] bounds: "
                          f"lon {vp['bounds']['lon_min']:.3f}–{vp['bounds']['lon_max']:.3f}  "
                          f"lat {vp['bounds']['lat_min']:.3f}–{vp['bounds']['lat_max']:.3f}")
                except Exception:
                    pass
        return vp if updated else no_update

    # ── Step 2: Compute farm ──────────────────────────────────────────────────
    @app.callback(
        Output("store-compute-done",      "data"),
        Output("div-compute-status",      "children"),
        Output("div-farm-summary",        "children"),
        Output("dd-result-layer",         "disabled"),
        Output("dd-harvest-month",        "disabled"),
        Output("btn-dl-farm-summary",     "disabled"),
        Output("btn-dl-cell-summary",     "disabled"),
        Output("btn-dl-env-baseline",     "disabled"),
        Output("btn-dl-results-baseline", "disabled"),
        Output("export-hint",             "children"),
        Output("store-farm-cells-ids",    "data"),
        Output("store-computed-label",    "data"),
        Output("store-farm-lonlat",           "data"),
        Output("store-new-polygon-drawn",     "data", allow_duplicate=True),
        Output("topbar-status",               "children", allow_duplicate=True),
        Input("btn-compute",                  "n_clicks"),
        State("store-drawn-shape",            "data"),
        State("slider-max-rdep",              "value"),
        State("slider-iloop",                 "value"),
        State("store-map-viewport",           "data"),
        State("radio-mode",                   "value"),
        State("store-farm-lonlat",            "data"),
        State("store-new-polygon-drawn",      "data"),
        prevent_initial_call=True,
    )
    def compute_farm(n, shape_store, max_rdep, iloop, viewport, mode,
                     stored_lonlat, new_polygon_drawn):
        # All-disabled return helper — 16 outputs now
        def _fail(msg):
            return (False, msg, no_update,
                    True, True, True, True, True, True,
                    no_update, no_update, no_update,
                    no_update, no_update, no_update)

        if not n:
            raise PreventUpdate

        vp   = viewport or {}
        zoom = vp.get("zoom", 0)

        # Coordinate resolution:
        # - Use stored lon/lat if user has NOT drawn a new polygon since last compute
        # - Use fresh parsed coords ONLY when new_polygon_drawn flag is True
        if new_polygon_drawn:
            shape_path   = (shape_store or {}).get("shape_path", "")
            fresh_coords = _parse_path_to_lonlat(shape_path,
                                                  map_bounds=vp.get("bounds"),
                                                  map_center=vp.get("center"),
                                                  map_zoom=vp.get("zoom"))
            if fresh_coords:
                if zoom < 10:
                    return _fail(f"⚠  Zoom in before computing — current zoom {zoom:.1f}, need ≥ 10")
                coords = fresh_coords
                print(f"[Compute] Using NEW polygon: {len(coords)} vertices")
            elif stored_lonlat:
                coords = [tuple(c) for c in stored_lonlat]
                print(f"[Compute] Parse failed — reusing stored polygon: {len(coords)} vertices")
            else:
                return _fail("⚠  Draw a polygon on the map first, then press Compute.")
        elif stored_lonlat:
            # Recompute same farm — use stored geographic coords
            coords = [tuple(c) for c in stored_lonlat]
            print(f"[Compute] Reusing stored polygon: {len(coords)} vertices")
        else:
            return _fail("⚠  Draw a polygon on the map first, then press Compute.")

        # Use whatever data is currently loaded
        base_env = app.server._base_env_grid
        if base_env is None:
            return _fail("❌ Base grid not found — run pipeline.py first")

        # Determine which grid to use for env conditions
        if mode in ("cmip6", "delta") and app.server._scenario is not None:
            env_grid = app.server._scenario
            computed_label = app.server._scenario_label or "scenario"
        else:
            env_grid = app.server._baseline
            computed_label = "baseline"

        max_rdep = RDEP_MAP.get(max_rdep, 2.0)
        iloop    = ILOOP_MAP.get(iloop,    0.7)
        month    = PRIMARY_HARVEST   # M12 orientation uses primary harvest month
        m_str    = f"{month:02d}"

        # M12
        try:
            m12 = discretise_farm(polygon_lonlat=coords, grid=env_grid,
                                  max_rdep=max_rdep, iloop=iloop,
                                  harvest_month=month)
        except Exception as e:
            return _fail(f"❌ M12 error: {e}")

        farm_cells = m12.get("farm_cells")
        if farm_cells is None or farm_cells.empty:
            return _fail("⚠  No viable cells in polygon — try a different location.")

        # M2 → M3 → M4 on farm_cells
        try:
            farm_cells = run_module2(farm_cells, n_scenarios=N_SC_APP, n_workers=N_CPU)
            farm_cells = run_module3(farm_cells)
            farm_cells = run_module4(farm_cells,
                                     D_short      = m12.get("D_short"),
                                     farm_area_m2 = m12.get("farm_area_m2"))
            farm_cells = farm_cells.copy()
        except Exception as e:
            return _fail(f"❌ Compute error: {e}")

        # Farm ID set when polygon was drawn — just read it here
        app.server._farm_cells     = farm_cells
        app.server._m12_result     = m12
        app.server._drawn_shapes   = []
        app.server._computed_label = computed_label

        n_viable = int(farm_cells["viable"].sum()) \
                   if "viable" in farm_cells.columns else len(farm_cells)
        nr = float(farm_cells[f"N_red_{m_str}"].sum()) \
             if f"N_red_{m_str}" in farm_cells.columns else 0.0

        summary     = render_farm_summary(m12, farm_cells, month)
        farm_ids    = farm_cells["cell_id"].tolist()
        status      = (f"{computed_label} · {n_viable} viable cells · N-red {nr:.1f} tN (month {m_str})")
        export_hint = f"{computed_label} — exports ready"

        return (True, f"✅ {status}", summary,
                False, False, False, False, False, False,
                export_hint, farm_ids, computed_label,
                coords,  # store lon/lat for stable recompute
                False,   # reset new-polygon-drawn flag
                status)

    # ── Update right panel when harvest month changes ─────────────────────────
    @app.callback(
        Output("div-farm-summary", "children", allow_duplicate=True),
        Input("dd-harvest-month",  "value"),
        State("store-compute-done", "data"),
        prevent_initial_call=True,
    )
    def update_farm_summary_month(month, compute_done):
        if not compute_done:
            raise PreventUpdate
        m12        = app.server._m12_result
        farm_cells = app.server._farm_cells
        if m12 is None or farm_cells is None:
            raise PreventUpdate
        return render_farm_summary(m12, farm_cells, month or PRIMARY_HARVEST)

    # ── Main map render ───────────────────────────────────────────────────────
    @app.callback(
        Output("map-graph",     "figure"),
        Output("topbar-status", "children"),
        Input("dd-env-layer",         "value"),
        Input("dd-result-layer",      "value"),
        Input("dd-env-month",         "value"),
        Input("dd-harvest-month",     "value"),
        Input("store-active-view",    "data"),
        Input("store-compute-done",   "data"),
        Input("store-scenario-ready", "data"),
        Input("store-scenario-label", "data"),
        Input("store-crop-mode",      "data"),
        Input("slider-marker-size",   "value"),
    )
    def render_map(env_layer, result_layer, env_month, harvest_month, view,
                   compute_done, scen_ready, scenario_label, crop_mode, marker_size):
        scen_label = scenario_label or "baseline"
        msize      = int(marker_size or 6)
        shapes     = getattr(app.server, "_drawn_shapes", [])
        e_month    = env_month    or 7
        h_month    = harvest_month or PRIMARY_HARVEST

        # ── Pre-compute: env browsing ─────────────────────────────────────────
        if not compute_done:
            if view == "scenario" and app.server._scenario is not None:
                gdf   = app.server._scenario
                glabel = scen_label
            else:
                gdf    = app.server._baseline
                glabel = "Baseline"
            if gdf is None:
                return empty_map("baseline.gpkg not found — run pipeline.py first"), ""

            if env_layer is None or env_layer == "none":
                status = f"{glabel} · {len(gdf):,} cells — select a layer in Step 1"
                return empty_map(""), status

            rev = f"{env_layer}|{e_month}|{scen_label}|{view}"
            if env_layer == "exclusions":
                fig    = build_exclusion_outline_map(gdf, shapes=shapes, marker_size=msize)
                fig.update_layout(uirevision=rev)
                return fig, f"{glabel} · exclusion zones"

            fig = build_env_map(gdf, env_layer, e_month, shapes=shapes,
                                marker_size=msize, uirevision=rev)
            return fig, f"{glabel} · {env_layer} · month {e_month:02d}"

        # ── Post-compute: result visualisation ────────────────────────────────
        # Map shows full grid (baseline always — it has full M2/M3/M4 results)
        # Farm boundary shown as geo-anchored cyan outline
        full_gdf   = app.server._baseline
        farm_cells = app.server._farm_cells
        comp_label = app.server._computed_label or "baseline"

        if full_gdf is None:
            return empty_map("baseline.gpkg not found"), ""

        farm_ids = farm_cells["cell_id"].tolist() if farm_cells is not None else []

        if crop_mode == "farm" and farm_cells is not None:
            gdf   = full_gdf[full_gdf["cell_id"].isin(farm_ids)]
            title = f"{comp_label} — farm area"
        else:
            gdf   = full_gdf
            title = f"{comp_label} — full grid"

        rev = f"{result_layer}|{h_month}|{comp_label}|{crop_mode}"
        fig = build_map(gdf, result_layer or "N_red", h_month,
                        title=title, shapes=shapes, farm_gdf=farm_cells,
                        marker_size=msize, uirevision=rev)
        status = (f"{comp_label} · {result_layer or 'N_red'} · month {h_month:02d}")
        return fig, status

    # ── Cell click report ─────────────────────────────────────────────────────
    @app.callback(
        Output("div-cell-report", "children"),
        Input("map-graph",        "clickData"),
        State("dd-harvest-month", "value"),
        State("store-compute-done", "data"),
        prevent_initial_call=True,
    )
    def on_click(click_data, month, compute_done):
        if not click_data:
            raise PreventUpdate
        try:
            cell_id = int(click_data["points"][0]["customdata"])
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate
        month = month or PRIMARY_HARVEST
        gdf   = app.server._farm_cells if compute_done else app.server._baseline
        if gdf is None:
            raise PreventUpdate
        rep = cell_report(gdf, cell_id, month)
        return render_cell_report(rep)

    # ── Export: farm summary ──────────────────────────────────────────────────
    @app.callback(
        Output("dl-farm-summary", "data"),
        Input("btn-dl-farm-summary", "n_clicks"),
        State("dd-harvest-month", "value"),
        prevent_initial_call=True,
    )
    def dl_farm_summary(n, month):
        if not n:
            raise PreventUpdate
        m12  = app.server._m12_result
        farm = app.server._farm_cells
        if m12 is None or farm is None:
            raise PreventUpdate
        month = month or PRIMARY_HARVEST
        m_str = f"{month:02d}"
        clabel = _slug(app.server._computed_label or "baseline")
        rows = []
        geom_fields = [
            ("Farm area [ha]",    m12.get("farm_area_m2", 0)/1e4),
            ("Sections",          m12.get("n_sections")),
            ("D_short [m]",       m12.get("D_short")),
            ("D_long [m]",        m12.get("D_long")),
            ("l_col mean [m]",    m12.get("l_col")),
            ("Orientation [deg]", m12.get("orientation_deg")),
            ("Flow mean [m/s]",   m12.get("vh_mean")),
            ("Viable cells",      int(farm["viable"].sum())
                                  if "viable" in farm.columns else len(farm)),
        ]
        for label, val in geom_fields:
            rows.append({"section": "Farm geometry", "metric": label,
                         "value": round(float(val), 4) if val is not None else ""})
        result_fields = [
            ("N-reduction [tN]",  f"N_red_{m_str}"),
            ("P-reduction [tP]",  f"P_red_{m_str}"),
            ("Harvest WW [t]",    f"harvest_WW_{m_str}"),
            ("Shell DW [t]",      f"shell_DW_{m_str}"),
            ("mbio p05 [gDW]",    f"mbio_p05_{m_str}"),
            ("mbio p50 [gDW]",    f"mbio_p50_{m_str}"),
            ("mbio p95 [gDW]",    f"mbio_p95_{m_str}"),
            ("N_red p05 [tN]",    f"N_red_p05_{m_str}"),
            ("N_red p95 [tN]",    f"N_red_p95_{m_str}"),
            ("vh_ratio",          f"vh_ratio_{m_str}"),
            ("food_limitation",   f"food_limitation_{m_str}"),
        ]
        for label, col in result_fields:
            if col in farm.columns:
                val = float(farm[col].fillna(0).sum()) \
                      if any(col.startswith(p) for p in ("N_red","P_red","harvest","shell")) \
                      else float(farm[col].fillna(0).mean())
                rows.append({"section": f"Results — month {m_str}",
                             "metric": label, "value": round(val, 4)})
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return dict(content=buf.getvalue(),
                    filename=f"mytigate_{clabel}_summary.csv")

    # ── Export: farm cell summary ─────────────────────────────────────────────
    @app.callback(
        Output("dl-cell-summary", "data"),
        Input("btn-dl-cell-summary", "n_clicks"),
        State("dd-harvest-month", "value"),
        prevent_initial_call=True,
    )
    def dl_cell_summary(n, month):
        if not n:
            raise PreventUpdate
        farm = app.server._farm_cells
        if farm is None:
            raise PreventUpdate
        month = month or PRIMARY_HARVEST
        m_str = f"{month:02d}"
        clabel = _slug(app.server._computed_label or "baseline")
        keep = ["cell_id", "lat", "lon", "bathymetry_m", "rdep",
                "excluded", "conflict_score", "viable"]
        for prefix in ["mbio_p05_", "mbio_p50_", "mbio_p95_",
                       "N_red_", "P_red_", "harvest_WW_", "harvest_DW_",
                       "shell_DW_", "N_red_ha_", "vh_ratio_", "food_limitation_"]:
            col = f"{prefix}{m_str}"
            if col in farm.columns:
                keep.append(col)
        keep = [c for c in keep if c in farm.columns]
        buf  = io.StringIO()
        pd.DataFrame(farm[keep]).to_csv(buf, index=False, float_format="%.4f")
        return dict(content=buf.getvalue(),
                    filename=f"mytigate_{clabel}_cells.csv")

    # ── Export: env grid ──────────────────────────────────────────────────────
    @app.callback(
        Output("dl-env-baseline", "data"),
        Input("btn-dl-env-baseline", "n_clicks"),
        prevent_initial_call=True,
    )
    def dl_env(n):
        if not n:
            raise PreventUpdate
        clabel = _slug(app.server._computed_label or "baseline")
        gdf = (app.server._scenario
               if app.server._scenario is not None
               else app.server._baseline)
        if gdf is None:
            raise PreventUpdate
        return dict(content=_to_csv(gdf, _env_cols(gdf), "%.6f"),
                    filename=f"mytigate_{clabel}_env.csv")

    # ── Export: results grid ──────────────────────────────────────────────────
    @app.callback(
        Output("dl-results-baseline", "data"),
        Input("btn-dl-results-baseline", "n_clicks"),
        prevent_initial_call=True,
    )
    def dl_results(n):
        if not n:
            raise PreventUpdate
        farm = app.server._farm_cells
        if farm is None:
            raise PreventUpdate
        clabel = _slug(app.server._computed_label or "baseline")
        return dict(content=_to_csv(farm, _result_cols(farm)),
                    filename=f"mytigate_{clabel}_results.csv")