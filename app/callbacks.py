"""
app/callbacks.py — MYTIGATE Dash callbacks.

Flow:
  Step 1: map shows env layer (no computation needed)
  Step 2: user draws polygon → presses Compute → M12 → M2/M3/M4 on farm cells
  Step 3: result layer dropdown unlocks, scenario mode available
  Step 4: export buttons unlock

Map: single Scattermap, toggled between baseline/scenario view via store.
Polygon: captured from relayoutData["shapes"], parsed to lon/lat coords.
"""
from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import dash
from dash import Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

from config import (
    BASELINE_GPKG, GRID_GPKG, PRIMARY_HARVEST, HARVEST_MONTHS,
    GRID_ENV_TEMP_GPKG, GRID_ENV_SAL_GPKG,
    GRID_ENV_CHLA_GPKG, GRID_ENV_FLOW_GPKG,
)
from app.map_utils import (build_map, build_env_map, empty_map,
                           build_exclusion_outline_map)
from app.layout import (render_farm_summary, render_cell_report,
                        render_compare_table)
from model.site_selector import cell_report
from model.growth import run_module2, run_module3, run_module4, N_CPU
from model.module12_farm import discretise_farm
from utils.raster_sampler import run_module1
from utils.climate_projections import load_scenario_env

HARVEST_MONTHS_STR = [f"{h:02d}" for h in HARVEST_MONTHS]
ALL_MONTHS_STR     = [f"{i:02d}" for i in range(1, 13)]
N_SC_APP = 100

# Snapping slider index → actual value
RDEP_MAP  = {2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0}   # slider value = actual value
ILOOP_MAP = {0: 0.7, 1: 1.0, 2: 1.5, 3: 2.0}    # slider index → m


# ── Server-side storage ───────────────────────────────────────────────────────

def load_all_grids(app: dash.Dash) -> None:
    app.server._baseline      = _load_grid(BASELINE_GPKG)
    app.server._base_env_grid = _load_grid(GRID_GPKG)
    app.server._scenario      = None
    app.server._farm_baseline = None   # M2/M3/M4 result on farm cells (baseline)
    app.server._farm_scenario = None   # M2/M3/M4 result on farm cells (scenario)
    app.server._m12_result    = None   # discretise_farm() dict
    app.server._scenario_label = ""
    app.server._drawn_shapes   = []
    print("[App] Startup complete")


def _load_grid(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"[App] Not found: {path}")
        return None
    gdf = gpd.read_file(path)
    if "lat" not in gdf.columns:
        pts = gdf.geometry.centroid.to_crs("EPSG:4326")
        gdf["lon"], gdf["lat"] = pts.x, pts.y
    print(f"[App] Loaded {path.name}: {len(gdf):,} cells")
    return gdf


# ── Helpers ───────────────────────────────────────────────────────────────────




def _paper_to_lonlat_derived(
        px: float, py: float,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
) -> tuple[float, float]:
    """
    Convert Plotly paper coords (0–1) to lon/lat using the actual map bounds
    from map._derived (exposed in relayoutData when the map moves).
    Simple linear interpolation — accurate because _derived gives true tile bounds.
    Paper (0,0) = bottom-left, (1,1) = top-right.
    """
    lon = lon_min + px * (lon_max - lon_min)
    lat = lat_min + py * (lat_max - lat_min)
    return lon, lat


def _parse_path_to_lonlat(
        path_str: str,
        map_bounds: dict | None = None,   # from map._derived
        map_center: dict | None = None,   # fallback
        map_zoom: float | None = None,    # fallback
) -> list[tuple] | None:
    """
    Convert SVG path string (Plotly paper coords 0–1) to lon/lat.

    Preferred: use map_bounds from map._derived (accurate).
    Fallback: Web Mercator math from center+zoom (approximate).
    """
    if not path_str:
        return None

    import re, math
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

    # Use map._derived bounds if available — most accurate
    if map_bounds:
        lon_min = map_bounds.get("lon_min", 9.0)
        lon_max = map_bounds.get("lon_max", 14.0)
        lat_min = map_bounds.get("lat_min", 54.0)
        lat_max = map_bounds.get("lat_max", 58.0)
        lonlat = [
            _paper_to_lonlat_derived(px, py, lon_min, lon_max, lat_min, lat_max)
            for px, py in paper_coords
        ]
    elif map_center and map_zoom is not None:
        # Fallback: Web Mercator
        center_lon = map_center.get("lon", 11.5)
        center_lat = map_center.get("lat", 56.3)
        world_px   = 256.0 * (2 ** map_zoom)
        def _lon2x(lo): return world_px * (lo + 180) / 360
        def _lat2y(la):
            r = math.radians(la)
            return world_px * (1 - math.log(math.tan(r) + 1/math.cos(r)) / math.pi) / 2
        cx, cy = _lon2x(center_lon), _lat2y(center_lat)
        W, H   = 1260, 1034
        lonlat = []
        for px, py in paper_coords:
            tx = cx + (px - 0.5) * W
            ty = cy + (0.5 - py) * H
            lo = tx / world_px * 360 - 180
            n  = math.pi - 2 * math.pi * ty / world_px
            la = math.degrees(math.atan(math.sinh(n)))
            lonlat.append((lo, la))
    else:
        print("[Shape] No viewport info available")
        return None

    print(f"[Shape] {len(lonlat)} vertices — "
          f"lon {min(l[0] for l in lonlat):.3f}–{max(l[0] for l in lonlat):.3f}  "
          f"lat {min(l[1] for l in lonlat):.3f}–{max(l[1] for l in lonlat):.3f}")
    return lonlat


def _env_cols(gdf: gpd.GeoDataFrame) -> list[str]:
    cols = ["cell_id", "lon", "lat", "bathymetry_m"]
    for prefix in ["temp_mean_", "sal_mean_", "lnchla_mean_", "vh_"]:
        cols += [c for c in gdf.columns if c.startswith(prefix)]
    return [c for c in cols if c in gdf.columns]


def _result_cols(gdf: gpd.GeoDataFrame) -> list[str]:
    cols = ["cell_id", "lon", "lat", "excluded", "conflict_score", "rdep"]
    for prefix in ["harvest_WW_", "harvest_DW_", "shell_DW_",
                   "N_red_", "N_red_ha_", "P_red_",
                   "mbio_p50_", "vh_ratio_", "food_limitation_"]:
        cols += [f"{prefix}{mk}" for mk in HARVEST_MONTHS_STR
                 if f"{prefix}{mk}" in gdf.columns]
    return [c for c in cols if c in gdf.columns]


def _to_csv(gdf: gpd.GeoDataFrame, cols: list[str], fmt: str = "%.4f") -> str:
    buf = io.StringIO()
    pd.DataFrame(gdf[cols]).to_csv(buf, index=False, float_format=fmt)
    return buf.getvalue()


# ── Register callbacks ────────────────────────────────────────────────────────

def register_callbacks(app: dash.Dash) -> None:

    # ── Capture drawn shape path as soon as it's drawn ───────────────────────
    # relayoutData["shapes"] fires immediately when user closes polygon.
    # Store the path string so Compute can read it reliably via State.
    @app.callback(
        Output("store-drawn-shape",        "data"),
        Output("div-restriction-warning",  "children"),
        Output("div-restriction-warning",  "style"),
        Input("map-graph", "relayoutData"),
        State("store-map-viewport", "data"),
        prevent_initial_call=True,
    )
    def capture_shape(relayout_data, viewport):
        base_style = {"fontSize": ".70rem", "fontWeight": "600",
                      "marginBottom": "6px", "minHeight": "14px"}
        if not relayout_data:
            raise PreventUpdate
        shapes = relayout_data.get("shapes", [])
        if not shapes:
            raise PreventUpdate
        path = shapes[-1].get("path", "")
        if not path:
            raise PreventUpdate
        print(f"[Shape] Captured path ({len(path)} chars)")

        # Quick exclusion check using current viewport
        warning_text  = ""
        warning_style = {**base_style, "color": "#4caf79"}

        vp     = viewport or {}
        coords = _parse_path_to_lonlat(
            path,
            map_bounds = vp.get("bounds"),
            map_center = vp.get("center"),
            map_zoom   = vp.get("zoom"),
        )

        if coords and app.server._base_env_grid is not None:
            try:
                from model.module12_farm import lonlat_to_utm

                polygon_utm = lonlat_to_utm(coords)
                grid        = app.server._base_env_grid
                grid_utm    = grid.to_crs("EPSG:32632") \
                              if grid.crs.to_epsg() != 32632 else grid
                sindex      = grid_utm.sindex
                cands       = list(sindex.query(polygon_utm, predicate="intersects"))
                cells       = grid_utm.iloc[cands]
                n_total     = len(cells)

                if n_total == 0:
                    warning_text  = "⚠  No grid cells found in polygon"
                    warning_style = {**base_style, "color": "#f0a040"}
                elif "excluded" in cells.columns:
                    n_excl = int(cells["excluded"].sum())
                    pct    = 100 * n_excl / n_total
                    if n_excl == n_total:
                        warning_text  = (f"🚫  Fully restricted area — "
                                         f"all {n_total} cells excluded")
                        warning_style = {**base_style, "color": "#f06a6a"}
                    elif n_excl > 0:
                        warning_text  = (f"⚠  Partial restriction — "
                                         f"{n_excl}/{n_total} cells excluded ({pct:.0f}%)")
                        warning_style = {**base_style, "color": "#f0a040"}
                    else:
                        warning_text  = f"✓  No restrictions — {n_total} cells available"
                        warning_style = {**base_style, "color": "#4caf79"}
            except Exception as e:
                print(f"[Shape] Exclusion check error: {e}")

        # Store the full shape dict (not just path) so we can re-inject it
        app.server._drawn_shapes = shapes
        return {"shape_path": path}, warning_text, warning_style

    # ── Track map viewport (center + zoom) ───────────────────────────────────
    @app.callback(
        Output("store-map-viewport", "data"),
        Input("map-graph", "relayoutData"),
        State("store-map-viewport", "data"),
        prevent_initial_call=True,
    )
    def track_viewport(relayout_data, current):
        if not relayout_data:
            raise PreventUpdate
        viewport = current or {"center": {"lon": 11.5, "lat": 56.3}, "zoom": 7.0}
        updated  = False

        if "map.center" in relayout_data:
            viewport["center"] = relayout_data["map.center"]
            updated = True
        if "map.zoom" in relayout_data:
            viewport["zoom"] = relayout_data["map.zoom"]
            updated = True
        # map._derived contains the actual geographic bounds of the visible tiles
        if "map._derived" in relayout_data:
            derived = relayout_data["map._derived"]
            coords  = derived.get("coordinates", {})
            if coords:
                # coordinates = [[NW_lon,NW_lat],[NE_lon,NE_lat],
                #                 [SE_lon,SE_lat],[SW_lon,SW_lat]]
                try:
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    viewport["bounds"] = {
                        "lon_min": min(lons), "lon_max": max(lons),
                        "lat_min": min(lats), "lat_max": max(lats),
                    }
                    updated = True
                    print(f"[Viewport] bounds: "
                          f"lon {viewport['bounds']['lon_min']:.3f}–"
                          f"{viewport['bounds']['lon_max']:.3f}  "
                          f"lat {viewport['bounds']['lat_min']:.3f}–"
                          f"{viewport['bounds']['lat_max']:.3f}")
                except Exception:
                    pass

        return viewport if updated else no_update

    # ── Show/hide scenario panels ─────────────────────────────────────────────
    @app.callback(
        Output("div-cmip6-panel",     "style"),
        Output("div-delta-panel",     "style"),
        Output("div-export-scenario", "style"),
        Output("div-compare-panel",   "style"),
        Input("radio-mode", "value"),
    )
    def toggle_mode_panels(mode):
        show = {"display": "block"}
        hide = {"display": "none"}
        scen = mode in ("cmip6", "delta")
        return (
            show if mode == "cmip6" else hide,
            show if mode == "delta" else hide,
            show if scen else hide,
            show if scen else hide,
        )

    # ── Live zoom warning ─────────────────────────────────────────────────────
    @app.callback(
        Output("div-zoom-warning", "children"),
        Output("div-zoom-warning", "style"),
        Input("store-map-viewport", "data"),
    )
    def zoom_warning(viewport):
        zoom = (viewport or {}).get("zoom", 0)
        base_style = {"marginTop": "6px", "fontSize": ".70rem", "fontWeight": "600"}
        if zoom == 0:
            return "", {**base_style, "display": "none"}
        if zoom < 9:
            return (f"⚠  Zoom in more before drawing  (current: {zoom:.1f}, need ≥ 10)",
                    {**base_style, "color": "#f06a6a"})
        if zoom < 10:
            return (f"⚠  Almost there — zoom in a bit more  (current: {zoom:.1f})",
                    {**base_style, "color": "#f0a040"})
        return (f"✓  Ready to draw  (zoom {zoom:.1f})",
                {**base_style, "color": "#4caf79"})

    # ── Show crop toggle after compute ───────────────────────────────────────
    @app.callback(
        Output("div-crop-toggle", "style"),
        Input("store-compute-done", "data"),
    )
    def show_crop_toggle(done):
        base = {"top": "10px", "left": "calc(50% + 160px)", "transform": "none"}
        return {**base, "display": "flex"} if done else {**base, "display": "none"}

    # ── Crop toggle state ─────────────────────────────────────────────────────
    @app.callback(
        Output("btn-crop-farm",  "className"),
        Output("btn-crop-full",  "className"),
        Output("store-crop-mode","data"),
        Input("btn-crop-farm",   "n_clicks"),
        Input("btn-crop-full",   "n_clicks"),
        State("store-crop-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_crop(nf, ng, current):
        mode = ("farm" if ctx.triggered_id == "btn-crop-farm"
                else "full" if ctx.triggered_id == "btn-crop-full"
                else current or "farm")
        a, i = "map-toggle-btn active", "map-toggle-btn inactive"
        return (a if mode == "farm" else i,
                a if mode == "full" else i,
                mode)

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
        scen_label   = scenario_label or "baseline"
        msize        = int(marker_size or 6)
        shapes       = getattr(app.server, "_drawn_shapes", [])
        e_month      = env_month     or 7
        h_month      = harvest_month or PRIMARY_HARVEST

        # Before compute: show env layer — use scenario grid if loaded and active
        if not compute_done:
            if view == "scenario" and app.server._scenario is not None:
                gdf = app.server._scenario
            else:
                gdf = app.server._baseline
            if gdf is None:
                return empty_map("baseline.gpkg not found — run pipeline.py first"), ""

            if env_layer is None or env_layer == "none":
                fig = empty_map("")
                return fig, f"Baseline · {len(gdf):,} cells — select a layer in Step 1"

            rev = f"{env_layer}|{e_month}|{scen_label}|{view}"
            if env_layer == "exclusions":
                fig    = build_exclusion_outline_map(gdf, shapes=shapes,
                                                     marker_size=msize)
                fig.update_layout(uirevision=rev)
                status = f"{scen_label} · {len(gdf):,} cells · exclusion zones"
                return fig, status

            fig    = build_env_map(gdf, env_layer, e_month,
                                   shapes=shapes, marker_size=msize,
                                   uirevision=rev)
            status = f"{scen_label} · {len(gdf):,} cells · {env_layer} month {e_month:02d}"
            return fig, status

        # After compute: use harvest_month for result layers
        farm_gdf = app.server._farm_baseline

        if view == "scenario" and app.server._scenario is not None:
            full_gdf = app.server._scenario
            label    = app.server._scenario_label or "Scenario"
        else:
            full_gdf = app.server._baseline
            label    = "Baseline"

        if full_gdf is None:
            return empty_map("baseline.gpkg not found — run pipeline.py first"), ""

        farm_ids = (farm_gdf["cell_id"].tolist()
                    if farm_gdf is not None else [])

        if crop_mode == "farm" and farm_gdf is not None:
            gdf   = full_gdf[full_gdf["cell_id"].isin(farm_ids)]
            title = f"{label} — farm area"
        else:
            gdf   = full_gdf
            title = f"{label} — full grid"

        rev = f"{result_layer}|{h_month}|{scen_label}|{view}|{crop_mode}"
        fig    = build_map(gdf, result_layer or "N_red", h_month,
                           title=title, shapes=shapes, farm_gdf=farm_gdf,
                           marker_size=msize, uirevision=rev)
        status = (f"{label} · {len(full_gdf):,} cells"
                  + (f" · {len(farm_ids)} farm cells" if farm_ids else ""))
        return fig, status

    # ── Show map toggle as soon as scenario is loaded ─────────────────────────
    @app.callback(
        Output("div-map-toggle", "style"),
        Input("store-scenario-ready", "data"),
    )
    def show_toggle(scen_ready):
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

    # ── Step 2: Compute farm ──────────────────────────────────────────────────
    @app.callback(
        Output("store-compute-done",        "data"),
        Output("div-compute-status",        "children"),
        Output("div-farm-summary",          "children"),
        Output("dd-result-layer",           "disabled"),
        Output("dd-harvest-month",          "disabled"),
        Output("btn-dl-env-baseline",       "disabled"),
        Output("btn-dl-results-baseline",   "disabled"),
        Output("btn-dl-cell-summary",       "disabled"),
        Output("btn-dl-farm-summary",       "disabled"),
        Output("export-hint",               "children"),
        Output("store-farm-cells-ids",      "data"),
        Input("btn-compute",                "n_clicks"),
        State("store-drawn-shape",           "data"),   # holds {"shape_path": "..."}
        State("slider-max-rdep",            "value"),
        State("slider-iloop",               "value"),
        State("dd-env-month",               "value"),
        State("store-map-viewport",         "data"),
        prevent_initial_call=True,
    )
    def compute_farm(n, shape_store, max_rdep, iloop, month, viewport):
        if not n:
            raise PreventUpdate

        # Zoom guard — polygon must be drawn at zoom ≥ 11 for farm-scale accuracy
        vp   = viewport or {}
        zoom = vp.get("zoom", 0)
        if zoom < 10:
            msg = (f"⚠  Zoom in before computing — current zoom {zoom:.1f}, "
                   f"need ≥ 10. Scroll the map wheel to zoom into your site, "
                   f"then redraw the polygon.")
            return (no_update, msg, no_update,
                    True, True, True, True, True, True, no_update, no_update)

        shape_path = (shape_store or {}).get("shape_path", "")
        coords = _parse_path_to_lonlat(
            shape_path,
            map_bounds = vp.get("bounds"),      # accurate if user panned/zoomed
            map_center = vp.get("center"),
            map_zoom   = vp.get("zoom"),
        )
        if not coords:
            return (no_update,
                    "⚠  Draw a polygon on the map first, then press Compute.",
                    no_update, True, True, True, True, True, True, no_update, no_update)

        base_env = app.server._base_env_grid
        if base_env is None:
            return (False, "❌ Base grid not found — run pipeline.py first",
                    no_update, True, True, True, True, True, True, no_update, no_update)

        max_rdep = RDEP_MAP.get(max_rdep, 2.0)
        iloop    = ILOOP_MAP.get(iloop, 0.7)
        month    = month or PRIMARY_HARVEST
        m_str    = f"{month:02d}"

        # M12: discretise farm polygon
        try:
            m12 = discretise_farm(
                polygon_lonlat        = coords,
                grid                  = base_env,
                max_rdep              = max_rdep,
                iloop                 = iloop,
                harvest_month         = month,
            )
        except Exception as e:
            return (False, f"❌ M12 error: {e}",
                    no_update, True, True, True, True, True, True, no_update, no_update)

        farm_cells = m12.get("farm_cells")
        if farm_cells is None or farm_cells.empty:
            return (False,
                    "⚠  No viable cells in polygon — all cells are excluded "
                    "(MPA, military, cables, shallow water etc). "
                    "Try a different location.",
                    no_update, True, True, True, True, True, True, no_update, no_update)

        # ── Exactly as in pipeline_debug.ipynb ───────────────────────────────
        # farm_cells from M12 already has env columns (from baseline.gpkg).
        # Run M2 → M3 → M4 on farm_cells only — for the RIGHT PANEL summary.
        # The MAP always shows the full baseline grid (already has M2/M3/M4).
        # This matches notebook: farm_cells = run_module2(farm_cells, ...)
        #                                      run_module3(farm_cells)
        #                                      run_module4(farm_cells,
        #                                          D_short=..., farm_area_m2=...)
        try:
            farm_cells = run_module2(farm_cells, n_scenarios=N_SC_APP,
                                     n_workers=N_CPU)
            farm_cells = run_module3(farm_cells)
            farm_cells = run_module4(
                farm_cells,
                D_short      = m12.get("D_short"),
                farm_area_m2 = m12.get("farm_area_m2"),
            )
            farm_cells = farm_cells.copy()
        except Exception as e:
            return (False, f"❌ Compute error: {e}",
                    no_update, True, True, True, True, True, True, no_update, no_update)

        # Store M12 result + computed farm cells for right panel
        app.server._farm_baseline = farm_cells
        app.server._m12_result    = m12
        # Clear paper-anchored drawn shape — farm boundary trace replaces it
        app.server._drawn_shapes  = []

        n_viable = int(farm_cells["viable"].sum()) \
                   if "viable" in farm_cells.columns else len(farm_cells)
        nr = float(farm_cells[f"N_red_{m_str}"].sum()) \
             if f"N_red_{m_str}" in farm_cells.columns else 0.0

        status  = (f"✅ Done — {n_viable} viable cells · "
                   f"N-red: {nr:.1f} tN (month {m_str})")
        summary = render_farm_summary(
            {**m12, "farm_cells": farm_cells}, m_str)
        farm_ids_json = farm_cells["cell_id"].tolist()

        return (True, status, summary,
                False,   # unlock result layer dropdown
                False,   # unlock harvest month dropdown
                False,   # unlock env export
                False,   # unlock results export
                False,   # unlock cell summary export
                False,   # unlock farm summary export
                "Step 2 complete — exports ready.",
                farm_ids_json)

    # ── Step 3: CMIP6 scenario load ───────────────────────────────────────────
    @app.callback(
        Output("store-scenario-ready",       "data",  allow_duplicate=True),
        Output("div-scenario-status",        "children"),
        Output("btn-dl-env-scenario",        "disabled", allow_duplicate=True),
        Output("btn-dl-results-scenario",    "disabled", allow_duplicate=True),
        Output("store-scenario-label",       "data",  allow_duplicate=True),
        Input("btn-load-scenario",           "n_clicks"),
        State("dd-ssp",                      "value"),
        State("dd-period",                   "value"),
        State("slider-max-rdep",             "value"),
        State("slider-iloop",                "value"),
        State("dd-harvest-month",            "value"),
        State("store-compute-done",          "data"),
        prevent_initial_call=True,
    )
    def load_cmip6(n, ssp, period, max_rdep, iloop, month, compute_done):
        max_rdep = RDEP_MAP.get(max_rdep, 2.0)
        iloop    = ILOOP_MAP.get(iloop, 0.7)
        if not n:
            raise PreventUpdate

        base_env = app.server._base_env_grid
        if base_env is None:
            return False, "❌ Base grid not found", True, True, ""

        try:
            grid_scen = load_scenario_env(base_env, ssp=ssp, period=period)
        except FileNotFoundError as e:
            return False, f"❌ {str(e)[:100]}", True, True, ""

        # Always store full scenario grid — needed for env layer browsing
        app.server._scenario = grid_scen

        # Only run M2/M3/M4 on farm_cells if farm has been computed
        m12 = app.server._m12_result
        if compute_done and m12 is not None:
            farm_ids  = m12["farm_cells"]["cell_id"].values
            farm_scen = grid_scen[grid_scen["cell_id"].isin(farm_ids)].copy()
            if not farm_scen.empty:
                farm_scen = run_module2(farm_scen, n_scenarios=N_SC_APP,
                                        n_workers=N_CPU)
                farm_scen = run_module3(farm_scen)
                farm_scen = run_module4(
                    farm_scen,
                    D_short      = m12.get("D_short"),
                    farm_area_m2 = m12.get("farm_area_m2"),
                )
                app.server._farm_scenario = farm_scen

        from config_scenario import SSP_SCENARIOS
        label = f"{SSP_SCENARIOS.get(ssp, ssp)} / {period}"
        app.server._scenario_label = label
        return True, f"✅ {label}", False, False, label

    # ── Step 3: Custom delta scenario ─────────────────────────────────────────
    @app.callback(
        Output("store-scenario-ready",       "data",  allow_duplicate=True),
        Output("div-delta-status",           "children"),
        Output("btn-dl-env-scenario",        "disabled", allow_duplicate=True),
        Output("btn-dl-results-scenario",    "disabled", allow_duplicate=True),
        Output("store-scenario-label",       "data",  allow_duplicate=True),
        Input("btn-run-delta",               "n_clicks"),
        State("slider-delta-temp",           "value"),
        State("slider-delta-sal",            "value"),
        State("slider-chla-mult",            "value"),
        State("slider-max-rdep",             "value"),
        State("slider-iloop",                "value"),
        State("store-compute-done",          "data"),
        prevent_initial_call=True,
    )
    def run_delta(n, dtemp, dsal, chla_mult, max_rdep, iloop, compute_done):
        max_rdep = RDEP_MAP.get(max_rdep, 2.0)
        iloop    = ILOOP_MAP.get(iloop, 0.7)
        if not n:
            raise PreventUpdate

        base_env  = app.server._base_env_grid
        if base_env is None:
            return False, "❌ Base grid not found", True, True, ""

        dtemp     = dtemp     or 0.0
        dsal      = dsal      or 0.0
        chla_mult = chla_mult or 1.0

        grid_scen = run_module1(base_env, delta_temp=dtemp, delta_sal=dsal,
                                chla_mult=chla_mult, force_resample=False)

        # Always store full scenario grid — needed for env layer browsing
        app.server._scenario = grid_scen

        # Only run M2/M3/M4 on farm_cells if farm has been computed
        m12 = app.server._m12_result
        if compute_done and m12 is not None:
            farm_ids  = m12["farm_cells"]["cell_id"].values
            farm_scen = grid_scen[grid_scen["cell_id"].isin(farm_ids)].copy()
            if not farm_scen.empty:
                farm_scen = run_module2(farm_scen, n_scenarios=N_SC_APP,
                                        n_workers=N_CPU)
                farm_scen = run_module3(farm_scen)
                farm_scen = run_module4(
                    farm_scen,
                    D_short      = m12.get("D_short"),
                    farm_area_m2 = m12.get("farm_area_m2"),
                )
                app.server._farm_scenario = farm_scen

        label = f"ΔT={dtemp:+.1f}°C ΔS={dsal:+.1f}psu ChlA×{chla_mult:.2f}"
        app.server._scenario_label = label
        return True, f"✅ {label}", False, False, label

    # ── Cell click ────────────────────────────────────────────────────────────
    @app.callback(
        Output("div-cell-report",   "children"),
        Output("div-compare-table", "children"),
        Input("map-graph",          "clickData"),
        State("dd-harvest-month",   "value"),
        State("radio-mode",         "value"),
        State("store-active-view",  "data"),
        State("store-compute-done", "data"),
        prevent_initial_call=True,
    )
    def on_click(click_data, month, mode, active_view, compute_done):
        if not click_data:
            raise PreventUpdate
        try:
            cell_id = int(click_data["points"][0]["customdata"])
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate

        month = month or PRIMARY_HARVEST

        # Pre-compute: report from baseline full grid
        if not compute_done:
            gdf = app.server._baseline
            if gdf is None:
                raise PreventUpdate
            rep  = cell_report(gdf, cell_id, month)
            return render_cell_report(rep), no_update

        # Post-compute: report from farm cells
        gdf = app.server._farm_baseline
        if gdf is None:
            raise PreventUpdate

        rep_base = cell_report(gdf, cell_id, month)
        cell_div = render_cell_report(rep_base)
        comp_div = no_update

        if mode in ("cmip6", "delta") and app.server._farm_scenario is not None:
            rep_scen = cell_report(app.server._farm_scenario, cell_id, month)
            label    = app.server._scenario_label or "Scenario"
            comp_div = render_compare_table(rep_base, rep_scen, label)

        return cell_div, comp_div

    # ── Export: farm geometry + results summary CSV ───────────────────────────
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
        farm = app.server._farm_baseline
        if m12 is None or farm is None:
            raise PreventUpdate
        month = month or PRIMARY_HARVEST
        m_str = f"{month:02d}"

        # Build summary rows matching the right-panel display
        rows = []

        # Farm geometry
        geom_fields = [
            ("Farm area [ha]",    m12.get("farm_area_m2", 0) / 1e4),
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

        # Results
        result_fields = [
            ("N-reduction [tN]",   f"N_red_{m_str}"),
            ("P-reduction [tP]",   f"P_red_{m_str}"),
            ("Harvest WW [t]",     f"harvest_WW_{m_str}"),
            ("Shell DW [t]",       f"shell_DW_{m_str}"),
            ("mbio p50 [gDW]",     f"mbio_p50_{m_str}"),
            ("mbio p05 [gDW]",     f"mbio_p05_{m_str}"),
            ("mbio p95 [gDW]",     f"mbio_p95_{m_str}"),
            ("N_red p05 [tN]",     f"N_red_p05_{m_str}"),
            ("N_red p95 [tN]",     f"N_red_p95_{m_str}"),
            ("vh_ratio",           f"vh_ratio_{m_str}"),
            ("food_limitation",    f"food_limitation_{m_str}"),
        ]
        for label, col in result_fields:
            if col in farm.columns:
                val = float(farm[col].fillna(0).sum()) \
                      if col.startswith(("N_red","P_red","harvest","shell")) \
                      else float(farm[col].fillna(0).mean())
                rows.append({"section": f"Results — month {m_str}",
                             "metric": label, "value": round(val, 4)})

        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return dict(content=buf.getvalue(),
                    filename=f"mytigate_farm_summary_{m_str}.csv")

    # ── Export: farm cell summary CSV ────────────────────────────────────────
    @app.callback(
        Output("dl-cell-summary", "data"),
        Input("btn-dl-cell-summary", "n_clicks"),
        State("dd-harvest-month", "value"),
        prevent_initial_call=True,
    )
    def dl_cell_summary(n, month):
        if not n:
            raise PreventUpdate
        farm = app.server._farm_baseline
        if farm is None:
            raise PreventUpdate
        month  = month or PRIMARY_HARVEST
        m_str  = f"{month:02d}"
        # Select key columns for summary
        keep = ["cell_id", "lat", "lon", "bathymetry_m", "rdep",
                "excluded", "conflict_score", "viable"]
        for prefix in ["mbio_p05_", "mbio_p50_", "mbio_p95_",
                       "N_red_", "P_red_", "harvest_WW_", "harvest_DW_",
                       "shell_DW_", "N_red_ha_", "vh_ratio_",
                       "food_limitation_"]:
            col = f"{prefix}{m_str}"
            if col in farm.columns:
                keep.append(col)
        keep = [c for c in keep if c in farm.columns]
        buf  = io.StringIO()
        pd.DataFrame(farm[keep]).to_csv(buf, index=False, float_format="%.4f")
        return dict(content=buf.getvalue(),
                    filename=f"mytigate_farm_cells_{m_str}.csv")

    # ── Export: baseline env ──────────────────────────────────────────────────
    @app.callback(
        Output("dl-env-baseline", "data"),
        Input("btn-dl-env-baseline", "n_clicks"),
        prevent_initial_call=True,
    )
    def dl_env_base(n):
        if not n:
            raise PreventUpdate
        gdf = app.server._baseline
        if gdf is None:
            raise PreventUpdate
        return dict(content=_to_csv(gdf, _env_cols(gdf), "%.6f"),
                    filename="mytigate_env_baseline.csv")

    # ── Export: baseline results ──────────────────────────────────────────────
    @app.callback(
        Output("dl-results-baseline", "data"),
        Input("btn-dl-results-baseline", "n_clicks"),
        prevent_initial_call=True,
    )
    def dl_results_base(n):
        if not n:
            raise PreventUpdate
        gdf = app.server._farm_baseline
        if gdf is None:
            raise PreventUpdate
        return dict(content=_to_csv(gdf, _result_cols(gdf)),
                    filename="mytigate_results_baseline.csv")

    # ── Export: scenario env ──────────────────────────────────────────────────
    @app.callback(
        Output("dl-env-scenario", "data"),
        Input("btn-dl-env-scenario", "n_clicks"),
        State("store-scenario-label", "data"),
        prevent_initial_call=True,
    )
    def dl_env_scen(n, label):
        if not n:
            raise PreventUpdate
        gdf = app.server._baseline  # full grid env (baseline env is the base)
        if gdf is None:
            raise PreventUpdate
        slug = (label or "scenario").replace(" ", "_").replace("/", "_")
        return dict(content=_to_csv(gdf, _env_cols(gdf), "%.6f"),
                    filename=f"mytigate_env_{slug}.csv")

    # ── Export: scenario results ──────────────────────────────────────────────
    @app.callback(
        Output("dl-results-scenario", "data"),
        Input("btn-dl-results-scenario", "n_clicks"),
        State("store-scenario-label", "data"),
        prevent_initial_call=True,
    )
    def dl_results_scen(n, label):
        if not n:
            raise PreventUpdate
        gdf = app.server._farm_scenario
        if gdf is None:
            raise PreventUpdate
        slug = (label or "scenario").replace(" ", "_").replace("/", "_")
        return dict(content=_to_csv(gdf, _result_cols(gdf)),
                    filename=f"mytigate_results_{slug}.csv")