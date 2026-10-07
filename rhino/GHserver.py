"""
GHserver.py
-----------
MYTIGATE Flask server for Rhino / Grasshopper integration.

Compute strategy (mirrors callbacks.py exactly):
  - Spatial maps  : _baseline (full 61k-cell grid from baseline.gpkg),
                    already has M2/M3/M4 from pipeline.py — returned as-is
  - Farm summary  : M12 → M2 → M3 → M4 on farm_cells only (2-50 cells)
  - Scenario      : same but on scenario env grid farm_cells

Usage:
    python rhino/GHserver.py

Endpoints:
    GET  /status   — health check, confirms grids loaded
    POST /compute  — run M12 → M2 → M3 → M4 on farm cells
    GET  /result   — retrieve computed arrays

Coordinate convention:
    Heron delivers WGS84 lon/lat. The server receives
    polygon_lonlat [[lon, lat], ...] and passes them straight into
    discretise_farm() which handles the UTM conversion internally.

Scenario modes (optional fields in POST body):
    "scenario_mode": "baseline"          — default, no scenario
    "scenario_mode": "cmip6"             — requires "ssp" and "period"
    "scenario_mode": "delta"             — requires "delta_temp", "delta_sal",
                                           "chla_mult"
"""

import sys
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import geopandas as gpd
from flask import Flask, request, jsonify

from config import (
    BASELINE_GPKG, GRID_GPKG, PRIMARY_HARVEST,
    HARVEST_MONTHS, N_SCENARIOS,
)
from model.module12_farm import discretise_farm
from model.growth import run_module2, run_module3, run_module4, N_CPU
from utils.raster_sampler import run_module1
from utils.climate_projections import load_scenario_env

app = Flask(__name__)

_baseline      = None
_base_env_grid = None
_latest_result = None

HARVEST_MONTHS_STR = [f"{h:02d}" for h in HARVEST_MONTHS]


# ── Startup ───────────────────────────────────────────────────────────────────

def _load_grid(path):
    if not path.exists():
        print(f"[Server] NOT FOUND: {path}")
        return None
    gdf = gpd.read_file(path)
    if "lat" not in gdf.columns:
        pts = gdf.geometry.centroid.to_crs("EPSG:4326")
        gdf["lon"], gdf["lat"] = pts.x, pts.y
    print(f"[Server] Loaded {path.name}: {len(gdf):,} cells, {len(gdf.columns)} columns")
    return gdf


def load_all_grids():
    global _baseline, _base_env_grid
    print("[Server] Loading grids...")
    _baseline      = _load_grid(BASELINE_GPKG)
    _base_env_grid = _load_grid(GRID_GPKG)
    if _baseline is None:
        print("[Server] WARNING: baseline.gpkg not found — run pipeline.py first")
    if _base_env_grid is None:
        print("[Server] WARNING: grid_base.gpkg not found — run pipeline.py first")
    print("[Server] Startup complete")


# ── JSON serialiser ───────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            cleaned = np.where(np.isnan(obj), None, obj) \
                      if np.issubdtype(obj.dtype, np.floating) else obj
            return cleaned.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)


def _jsonify_numpy(data):
    return json.dumps(data, cls=_NumpyEncoder)


# ── Farm summary helper ───────────────────────────────────────────────────────

def _farm_summary(fc):
    """Aggregated totals per harvest month over farm_cells."""
    summary = {}
    for ms in HARVEST_MONTHS_STR:
        row = {}
        for prefix in ["N_red_", "N_red_p05_", "N_red_p95_",
                        "P_red_", "harvest_WW_", "harvest_DW_", "shell_DW_"]:
            col = f"{prefix}{ms}"
            if col in fc.columns:
                row[col] = float(fc[col].fillna(0).sum())
        for prefix in ["mbio_p05_", "mbio_p50_", "mbio_p95_",
                        "vh_ratio_", "food_limitation_"]:
            col = f"{prefix}{ms}"
            if col in fc.columns:
                row[col] = float(fc[col].fillna(0).mean())
        summary[ms] = row

    return summary


# ── Grid arrays helper ────────────────────────────────────────────────────────

def _grid_arrays(g):
    """All result columns across the full grid."""
    out = {
        "cell_id":        g["cell_id"].values,
        "lat":            g["lat"].values,
        "lon":            g["lon"].values,
        "bathymetry_m":   g["bathymetry_m"].fillna(0).values
                          if "bathymetry_m" in g.columns else np.zeros(len(g)),
        "rdep":           g["rdep"].fillna(0).values
                          if "rdep" in g.columns else np.zeros(len(g)),
        "excluded":       g["excluded"].astype(bool).values
                          if "excluded" in g.columns else np.zeros(len(g), dtype=bool),
        "conflict_score": g["conflict_score"].fillna(0).values
                          if "conflict_score" in g.columns else np.zeros(len(g)),
    }
    all_months_str = [f"{m:02d}" for m in range(1, 13)]
    for ms in all_months_str:
        for prefix in ["uo_", "vo_", "vh_",
                       "temp_mean_", "temp_sd_",
                       "sal_mean_",  "sal_sd_",
                       "lnchla_mean_", "lnchla_sd_"]:
            col = f"{prefix}{ms}"
            if col in g.columns:
                out[col] = g[col].fillna(0).values
    for ms in HARVEST_MONTHS_STR:
        for prefix in [
            "mbio_p05_", "mbio_p50_", "mbio_p95_",
            "harvest_DW_", "harvest_WW_", "shell_DW_",
            "N_red_", "N_red_p05_", "N_red_p95_", "N_red_ha_",
            "P_red_",
            "vh_ratio_", "vh_crit_", "food_limitation_",
        ]:
            col = f"{prefix}{ms}"
            if col in g.columns:
                out[col] = g[col].fillna(0).values
    if "food_limitation_severity" in g.columns:
        out["food_limitation_severity"] = g["food_limitation_severity"].fillna(0).values
    if "food_ok" in g.columns:
        out["food_ok"] = g["food_ok"].astype(bool).values
    return out


# ── Result builder ────────────────────────────────────────────────────────────

def _build_result(grid, m12, farm_cells, scenario_label="baseline",
                  grid_scen=None, farm_cells_scen=None):

    farm_geometry = {
        "farm_area_ha":          float(m12.get("farm_area_m2", 0) / 1e4),
        "n_sections":            int(m12.get("n_sections", 0)),
        "D_short":               float(m12.get("D_short", 0)),
        "D_long":                float(m12.get("D_long", 0)),
        "l_col_mean_m":          float(m12.get("l_col", 0)),
        "orientation_deg":       float(m12.get("orientation_deg", 0)),
        "uo":                    float(m12.get("uo", 0)),
        "vo":                    float(m12.get("vo", 0)),
        "vh_mean":               float(m12.get("vh_mean", 0)),
        "max_rdep":              float(m12.get("max_rdep", 2.0)),
        "iloop":                 float(m12.get("iloop", 0.7)),
        "longline_length":       float(m12.get("longline_length", 200)),
        "longline_spacing":      float(m12.get("longline_spacing", 8)),
        "longlines_per_section": int(m12.get("longlines_per_section", 30)),
        "viable_cells":          int(farm_cells["viable"].sum())
                                 if "viable" in farm_cells.columns else len(farm_cells),
        "total_cells":           int(len(farm_cells)),
    }

    farm_cell_ids = farm_cells["cell_id"].tolist()

    result = {
        "status":         "ok",
        "scenario_label": scenario_label,
        "farm_geometry":  farm_geometry,
        "farm_cell_ids":  farm_cell_ids,
        "baseline": {
            "grid":         _grid_arrays(grid),
            "farm_summary": _farm_summary(farm_cells),
        },
    }

    # ── Scenario block ────────────────────────────────────────────────────────
    if grid_scen is not None and farm_cells_scen is not None:
        result["scenario"] = {
            "grid":         _grid_arrays(grid_scen),
            "farm_summary": _farm_summary(farm_cells_scen),
        }

    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    b_cells = f"{len(_baseline):,} cells" if _baseline is not None else "NOT LOADED"
    e_cells = f"{len(_base_env_grid):,} cells" if _base_env_grid is not None else "NOT LOADED"
    return (
        f"<h2>MYTIGATE server is running</h2>"
        f"<pre>"
        f"baseline.gpkg  : {b_cells}\n"
        f"grid_base.gpkg : {e_cells}\n"
        f"result ready   : {_latest_result is not None}\n"
        f"</pre>"
        f"<p>Endpoints: "
        f"<code>GET /status</code> &nbsp;|&nbsp; "
        f"<code>POST /compute</code> &nbsp;|&nbsp; "
        f"<code>GET /result</code></p>"
    )


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":          "running",
        "baseline_loaded": _baseline is not None,
        "base_env_loaded": _base_env_grid is not None,
        "baseline_cells":  len(_baseline) if _baseline is not None else 0,
        "result_ready":    _latest_result is not None,
    })


@app.route("/compute", methods=["POST"])
def compute():
    global _latest_result

    data = request.json
    if data is None:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    if _baseline is None or _base_env_grid is None:
        return jsonify({
            "status":  "error",
            "message": "Grids not loaded. Run pipeline.py first.",
        }), 500

    polygon_lonlat = data.get("polygon_lonlat")
    max_rdep       = float(data.get("max_rdep",      2.0))
    iloop          = float(data.get("iloop",         0.7))
    harvest_month  = int(data.get("harvest_month",   PRIMARY_HARVEST))
    n_scenarios    = int(data.get("n_scenarios",     N_SCENARIOS))
    scenario_mode  = data.get("scenario_mode",       "baseline")
    ssp            = data.get("ssp",                 "ssp2_4_5")
    period         = data.get("period",              "end")
    delta_temp     = float(data.get("delta_temp",    0.0))
    delta_sal      = float(data.get("delta_sal",     0.0))
    chla_mult      = float(data.get("chla_mult",     1.0))

    print(f"\n[Server] scenario_mode received: {scenario_mode!r}")

    if not polygon_lonlat or len(polygon_lonlat) < 3:
        return jsonify({
            "status":  "error",
            "message": "polygon_lonlat must have at least 3 points [[lon,lat],...]",
        }), 400

    try:
        # ── M12 baseline ──────────────────────────────────────────────────────
        print(f"[Server] Running M12 — {len(polygon_lonlat)} vertices, "
              f"max_rdep={max_rdep}, iloop={iloop}, month={harvest_month:02d}")
        print(f"[Server] First vertex: lon={polygon_lonlat[0][0]:.4f}, "
              f"lat={polygon_lonlat[0][1]:.4f}")

        m12 = discretise_farm(
            polygon_lonlat = polygon_lonlat,
            grid           = _base_env_grid,
            max_rdep       = max_rdep,
            iloop          = iloop,
            harvest_month  = harvest_month,
        )

        if m12["farm_cells"].empty:
            return jsonify({
                "status":  "error",
                "message": "No viable cells in polygon.",
            }), 400

        # ── M2/M3/M4 on baseline farm_cells ──────────────────────────────────
        print(f"[Server] Running M2/M3/M4 on farm_cells "
              f"({len(m12['farm_cells'])} cells, {n_scenarios} scenarios)...")
        farm_cells_baseline = m12["farm_cells"].copy()
        farm_cells_baseline = run_module2(farm_cells_baseline,
                                          n_scenarios=n_scenarios,
                                          n_workers=N_CPU)
        farm_cells_baseline = run_module3(farm_cells_baseline)
        farm_cells_baseline = run_module4(
            farm_cells_baseline,
            D_short      = m12.get("D_short"),
            farm_area_m2 = m12.get("farm_area_m2"),
        )

        scenario_label  = "baseline"
        grid_scen       = None
        farm_cells_scen = None

        # ── Scenario ──────────────────────────────────────────────────────────
        if scenario_mode in ("cmip6", "delta"):
            print(f"[Server] Scenario mode: {scenario_mode}")

            if scenario_mode == "cmip6":
                print(f"[Server] Loading CMIP6 scenario: {ssp} / {period}")
                grid_env_scen  = load_scenario_env(_base_env_grid, ssp=ssp, period=period)
                scenario_label = f"{ssp} / {period}"
            else:
                print(f"[Server] Applying delta: "
                      f"ΔT={delta_temp:+.1f} ΔS={delta_sal:+.1f} ChlA×{chla_mult:.2f}")
                grid_env_scen  = run_module1(
                    _base_env_grid,
                    delta_temp    = delta_temp,
                    delta_sal     = delta_sal,
                    chla_mult     = chla_mult,
                    force_resample= False,
                )
                scenario_label = (f"ΔT={delta_temp:+.1f}°C "
                                  f"ΔS={delta_sal:+.1f}psu "
                                  f"ChlA×{chla_mult:.2f}")

            # M12 on scenario env grid — get farm geometry for this scenario
            m12_scen = discretise_farm(
                polygon_lonlat = polygon_lonlat,
                grid           = grid_env_scen,
                max_rdep       = max_rdep,
                iloop          = iloop,
                harvest_month  = harvest_month,
            )

            # M2/M3/M4 on the FULL scenario grid (61k cells) — same as baseline
            print(f"[Server] Running M2/M3/M4 on full scenario grid "
                  f"({len(grid_env_scen):,} cells, {n_scenarios} scenarios)...")
            grid_scen = grid_env_scen.copy()
            grid_scen = run_module2(grid_scen,
                                    n_scenarios=n_scenarios,
                                    n_workers=N_CPU)
            grid_scen = run_module3(grid_scen)
            grid_scen = run_module4(
                grid_scen,
                D_short      = m12_scen.get("D_short"),
                farm_area_m2 = m12_scen.get("farm_area_m2"),
            )

            # Farm cells from scenario full grid for farm summary
            farm_cells_scen = grid_scen[
                grid_scen["cell_id"].isin(m12_scen["farm_cells"]["cell_id"].values)
            ].copy()
            print(f"[Server] Scenario full grid done: {grid_scen.shape}, "
                  f"farm cells: {len(farm_cells_scen)}")
        else:
            print(f"[Server] Baseline only — no scenario block will be written")

        # ── Build result ──────────────────────────────────────────────────────
        print("[Server] Building result payload...")
        _latest_result = _build_result(
            grid            = _baseline,
            m12             = m12,
            farm_cells      = farm_cells_baseline,
            scenario_label  = scenario_label,
            grid_scen       = grid_scen if scenario_mode in ("cmip6", "delta") else None,
            farm_cells_scen = farm_cells_scen,
        )



        m_str    = f"{harvest_month:02d}"
        n_viable = int(farm_cells_baseline["viable"].sum()) \
                   if "viable" in farm_cells_baseline.columns \
                   else len(farm_cells_baseline)
        n_red    = float(farm_cells_baseline[f"N_red_{m_str}"].sum()) \
                   if f"N_red_{m_str}" in farm_cells_baseline.columns else 0.0

        print(f"[Server] Done — {n_viable} viable farm cells, "
              f"N-red: {n_red:.1f} tN (month {m_str}), "
              f"scenario: {scenario_label}")

        return jsonify({
            "status":        "ok",
            "viable_cells":  n_viable,
            "n_red_tN":      round(n_red, 3),
            "harvest_month": m_str,
            "scenario":      scenario_label,
            "grid_cells":    len(_baseline),
        })

    except FileNotFoundError as e:
        msg = str(e)
        print(f"[Server] FileNotFoundError: {msg}")
        return jsonify({"status": "error", "message": msg}), 404

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Server] ERROR:\n{tb}")
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500


@app.route("/result", methods=["GET"])
def result():
    if _latest_result is None:
        return jsonify({
            "status":  "error",
            "message": "No result yet — POST to /compute first",
        }), 404

    print(f"[Server] /result requested — "
          f"{'with scenario' if 'scenario' in _latest_result else 'baseline only'}")

    response = app.response_class(
        response = _jsonify_numpy(_latest_result),
        mimetype = "application/json",
    )
    return response


if __name__ == "__main__":
    load_all_grids()
    print("\n[Server] Endpoints:")
    print("  GET  http://127.0.0.1:8000/status")
    print("  POST http://127.0.0.1:8000/compute")
    print("  GET  http://127.0.0.1:8000/result")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=False)