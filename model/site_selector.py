"""
model/site_selector.py
----------------------
Site ranking and selection algorithm.
Works identically for baseline and scenario grids.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd
from config import PRIMARY_HARVEST


def rank_and_select(
    grid: gpd.GeoDataFrame,
    harvest_month: int = PRIMARY_HARVEST,
    criterion: str = "N_red",
    target_mode: str = "n_farms",
    target_value: float = 10,
    conflict_weight: float = 1.0,
    food_filter: bool = True,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Rank all viable cells and select until target is reached.

    Returns (grid_with_selected_col, summary_dict)
    """
    grid = grid.copy()
    m    = f"{harvest_month:02d}"
    raw_col = f"{criterion}_{m}"

    if raw_col not in grid.columns:
        raise KeyError(f"Column '{raw_col}' not found. Run pipeline first.")

    scores = grid[raw_col].fillna(0).values.copy()

    # Zero out excluded / too-shallow cells
    if "excluded" in grid.columns:
        scores[grid["excluded"].values] = 0.0
    if "rdep" in grid.columns:
        scores[grid["rdep"].fillna(0).values <= 0] = 0.0

    # Conflict penalty
    if "conflict_score" in grid.columns and conflict_weight > 0:
        scores *= (1.0 - conflict_weight * grid["conflict_score"].fillna(0).values)

    # Food limitation filter
    if food_filter:
        food_col = f"food_ok_{m}"
        fc = grid.get(food_col, grid.get("food_ok", pd.Series(True, index=grid.index)))
        scores[~fc.fillna(True).values] = 0.0

    # Farm area derived from grid column written by M3/M12 — no hardcoded constant
    farm_ha = (grid["farm_area_m2"].fillna(0).mean() / 1e4
               if "farm_area_m2" in grid.columns else 0.0)

    grid["adj_score"] = np.maximum(scores, 0.0)
    order = np.argsort(-grid["adj_score"].values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    grid["rank"] = ranks
    grid["selected"] = False

    candidates = grid[grid["adj_score"] > 0].sort_values("rank")
    sel_idx    = []
    cum_N      = 0.0

    for idx, row in candidates.iterrows():
        if target_mode == "n_farms" and len(sel_idx) >= int(target_value):
            break
        if target_mode == "N_reduction" and cum_N >= float(target_value):
            break
        sel_idx.append(idx)
        cum_N += row.get(f"N_red_{m}", 0.0)

    grid.loc[sel_idx, "selected"] = True
    sel = grid[grid["selected"]]

    summary = {
        "n_farms":          int(len(sel)),
        "total_N_tN":       float(sel.get(f"N_red_{m}", pd.Series(dtype=float)).sum()),
        "total_P_tP":       float(sel.get(f"P_red_{m}", pd.Series(dtype=float)).sum()),
        "total_WW_t":       float(sel.get(f"harvest_WW_{m}", pd.Series(dtype=float)).sum()),
        "total_area_ha":    float(len(sel) * farm_ha),
        "mean_N_per_farm":  float(sel.get(f"N_red_{m}", pd.Series(dtype=float)).mean()),
        "mean_N_ha":        float((sel.get(f"N_red_{m}", pd.Series(dtype=float)) / farm_ha).mean()
                                  if farm_ha > 0 else 0.0),
        "harvest_month":    harvest_month,
        "target_mode":      target_mode,
        "target_value":     target_value,
        "selected_cell_ids": sel["cell_id"].tolist(),
    }
    return grid, summary


def cell_report(
    grid: gpd.GeoDataFrame,
    cell_id: int,
    harvest_month: int = PRIMARY_HARVEST,
) -> dict:
    """Full model output for one clicked cell — fed to the right panel."""
    m   = f"{harvest_month:02d}"
    row = grid[grid["cell_id"] == cell_id]
    if row.empty:
        return {"error": f"Cell {cell_id} not found"}
    r = row.iloc[0]

    def g(col, default=None):
        v = r.get(col, default)
        return float(v) if v is not None and pd.notna(v) else default

    return {
        "cell_id":        int(cell_id),
        "lat":            g("lat"),
        "lon":            g("lon"),
        "bathymetry_m":   g("bathymetry_m"),
        "rdep_m":         g("rdep"),
        "lcol_m":         g("lcol_m"),
        "excluded":       bool(r.get("excluded", False)),
        "conflict_score": g("conflict_score", 0.0),
        "food_ok":        bool(r.get(f"food_ok_{m}", r.get("food_ok", True))),
        "vh_ratio":       g(f"vh_ratio_{m}"),
        "mbio_p05_gDW":   g(f"mbio_p05_{m}"),
        "mbio_p50_gDW":   g(f"mbio_p50_{m}"),
        "mbio_p95_gDW":   g(f"mbio_p95_{m}"),
        "harvest_WW_t":   g(f"harvest_WW_{m}"),
        "N_red_tN":       g(f"N_red_{m}"),
        "N_red_p05_tN":   g(f"N_red_p05_{m}"),
        "N_red_p95_tN":   g(f"N_red_p95_{m}"),
        "P_red_tP":       g(f"P_red_{m}"),
        "N_red_ha":       g(f"N_red_ha_{m}"),
        "rank":           int(r["rank"]) if "rank" in r.index else None,
        "selected":       bool(r.get("selected", False)),
        "temp_mean_C":    g(f"temp_mean_{m}"),
        "sal_mean_psu":   g(f"sal_mean_{m}"),
        "chla_ugL":       float(np.exp(r[f"lnchla_mean_{m}"])) if f"lnchla_mean_{m}" in r.index and pd.notna(r[f"lnchla_mean_{m}"]) else None,
    }