"""
model/module12_farm.py
----------------------
Module 12 — Farm orientation, discretisation, geometry.

OPTIMISATIONS:
- STRtree spatial index for polygon cell intersection
- Vectorised rdep computation
- Transformer cached (not recreated per call)
"""

from __future__ import annotations
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from pyproj import Transformer

from config import (
    GRID_CRS,
    LONGLINE_LENGTH, LONGLINE_SPACING, LONGLINES_PER_SECTION,
    MIN_DEPTH_M, PRIMARY_HARVEST,
    MOCK_FARM_COORDS_LONLAT,
)

# Cache transformer (expensive to create)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)


def lonlat_to_utm(coords_lonlat: list[tuple[float, float]]) -> Polygon:
    utm_coords = [_TRANSFORMER.transform(lon, lat) for lon, lat in coords_lonlat]
    return Polygon(utm_coords)


def dominant_current_direction(
    cells: gpd.GeoDataFrame,
    month: int = PRIMARY_HARVEST,
) -> tuple[float, float]:
    if cells.empty:
        return 0.05, 0.0
    m = f"{month:02d}"
    for uo_col, vo_col in [(f"uo_{m}", f"vo_{m}")]:
        if uo_col in cells.columns and vo_col in cells.columns:
            uo = cells[uo_col].fillna(0).mean()
            vo = cells[vo_col].fillna(0).mean()
            if np.isfinite(uo) and np.isfinite(vo) and abs(uo) + abs(vo) > 1e-6:
                return float(uo), float(vo)
    vh_col = f"vh_{m}"
    if vh_col in cells.columns:
        vh = cells[vh_col].fillna(0.05).mean()
        vh = float(vh) if np.isfinite(vh) else 0.05
        print(f"  [M12] No direction data — assuming eastward {vh:.3f} m/s")
        return vh, 0.0
    return 0.05, 0.0


def orientation_from_current(uo: float, vo: float) -> float:
    return float(np.degrees(np.arctan2(uo, vo)) % 360)


def discretise_farm(
    polygon_lonlat: list[tuple[float, float]],
    grid: gpd.GeoDataFrame,
    longline_length: float       = LONGLINE_LENGTH,
    longline_spacing: float      = LONGLINE_SPACING,
    longlines_per_section: int   = LONGLINES_PER_SECTION,
    max_rdep: float              = 2.0,
    iloop: float                 = 0.7,
    harvest_month: int           = PRIMARY_HARVEST,
) -> dict:
    """
    Discretise user polygon into standard farm sections aligned with current.

    Optimisations:
    - STRtree for polygon cell intersection (vs brute force intersects)
    - Vectorised rdep and lcol computation
    - Cached CRS transformer
    """
    print("\n=== Module 12: Farm Discretisation ===")

    polygon_utm = lonlat_to_utm(polygon_lonlat)
    print(f"  [M12] Polygon area: {polygon_utm.area/1e4:.1f} ha")

    # STRtree intersection — fast for large grids
    grid_utm = grid.to_crs(GRID_CRS) if grid.crs.to_epsg() != 32632 else grid
    sindex   = grid_utm.sindex
    cands    = list(sindex.query(polygon_utm, predicate="intersects"))
    farm_cells = grid_utm.iloc[cands].copy()
    print(f"  [M12] Cells inside polygon: {len(farm_cells)}")

    if farm_cells.empty:
        print("  [M12] WARNING: No cells in polygon")
        return _empty_result(polygon_utm)

    if "excluded" in farm_cells.columns:
        n_before = len(farm_cells)
        farm_cells = farm_cells[~farm_cells["excluded"]].copy()
        print(f"  [M12] After exclusion: {len(farm_cells)} "
              f"({n_before-len(farm_cells)} excluded)")

    # Vectorised depth check
    bathy  = farm_cells["bathymetry_m"].fillna(0).values
    viable = bathy >= MIN_DEPTH_M
    farm_cells["viable"] = viable
    print(f"  [M12] Viable cells (depth ≥{MIN_DEPTH_M}m): {viable.sum()}")

    # Current direction
    uo, vo = dominant_current_direction(farm_cells, month=harvest_month)
    orientation_deg = orientation_from_current(uo, vo)
    vh_mean = float(np.sqrt(uo**2 + vo**2))
    print(f"  [M12] Current: uo={uo:.4f} vo={vo:.4f} m/s  "
          f"orientation={orientation_deg:.1f}°")

    # Farm geometry from exposed parameters
    D_short = longlines_per_section * longline_spacing     # e.g. 30×8 = 240m
    D_long  = longline_length                              # e.g. 200m
    section_area_m2  = D_short * D_long                   # 48,000 m²
    # Use actual drawn polygon area — not grid cell count × 1km²
    # Grid cells only determine env conditions; farm fills the polygon area
    viable_area_m2   = polygon_utm.area
    n_sections       = max(1, int(viable_area_m2 / section_area_m2))
    farm_area_m2     = n_sections * section_area_m2

    print(f"  [M12] D_short={D_short}m  D_long={D_long}m  "
          f"sections={n_sections}  area={farm_area_m2/1e4:.1f}ha")

    # Vectorised rdep and lcol
    rdep = np.where(bathy - 2 <= max_rdep, np.maximum(bathy - 2, 0), max_rdep)
    rdep = np.where(viable, rdep, 0.0)
    farm_cells["rdep"] = rdep

    total_lines  = longlines_per_section * n_sections
    lcol_per_cell = np.where(
        rdep > 0,
        (2 * rdep + iloop) * (longline_length * total_lines / iloop),
        0.0,
    )
    farm_cells["lcol_m"]       = lcol_per_cell
    farm_cells["farm_area_m2"] = farm_area_m2 / max(len(farm_cells), 1)

    l_col_mean = float(lcol_per_cell[lcol_per_cell > 0].mean()) \
                 if (lcol_per_cell > 0).any() else 0.0

    rdep_pos = rdep[rdep > 0]
    rdep_str = (f"{rdep_pos.min():.1f}–{rdep.max():.1f} m"
                if len(rdep_pos) else "no viable depth")
    print(f"  [M12] l_col mean: {l_col_mean:.0f} m  rdep range: {rdep_str}")

    return {
        "farm_cells"           : farm_cells,
        "n_sections"           : n_sections,
        "D_short"              : D_short,
        "D_long"               : D_long,
        "farm_area_m2"         : farm_area_m2,
        "l_col"                : l_col_mean,
        "orientation_deg"      : orientation_deg,
        "uo"                   : uo,
        "vo"                   : vo,
        "vh_mean"              : vh_mean,
        "r_dep"                : rdep,
        "lcol_per_cell"        : lcol_per_cell,
        "polygon_utm"          : polygon_utm,
        "max_rdep"             : max_rdep,
        "iloop"                : iloop,
        "longline_length"      : longline_length,
        "longline_spacing"     : longline_spacing,
        "longlines_per_section": longlines_per_section,
    }


def _empty_result(polygon_utm) -> dict:
    return {
        "farm_cells": gpd.GeoDataFrame(), "n_sections": 0,
        "D_short": 0.0, "D_long": 0.0, "farm_area_m2": 0.0,
        "l_col": 0.0, "orientation_deg": 0.0, "uo": 0.0, "vo": 0.0,
        "vh_mean": 0.0, "r_dep": np.array([]), "lcol_per_cell": np.array([]),
        "polygon_utm": polygon_utm,
    }