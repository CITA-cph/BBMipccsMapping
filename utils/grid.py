"""
utils/grid.py
-------------
1km UTM32N grid builder.

CHANGELOG:
- A: clip_to_sea() Python for-loop replaced with geopandas sindex.query_bulk()
     No per-cell Python loop — pure vectorised spatial index query
     Expected: 3 min → <30s
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm

from config import (X_MIN, X_MAX, Y_MIN, Y_MAX, CELL_SIZE,
                    GRID_CRS, COASTLINE_SHP, GRID_GPKG)


def build_grid() -> gpd.GeoDataFrame:
    xs = np.arange(X_MIN, X_MAX, CELL_SIZE)
    ys = np.arange(Y_MIN, Y_MAX, CELL_SIZE)
    print(f"[Grid] Building {len(xs)} × {len(ys)} = {len(xs)*len(ys):,} cells...")

    x_grid, y_grid = np.meshgrid(xs, ys, indexing='ij')
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    cells = [box(x, y, x + CELL_SIZE, y + CELL_SIZE)
             for x, y in zip(x_flat, y_flat)]

    gdf = gpd.GeoDataFrame(
        {"cell_id": np.arange(len(cells)),
         "x_center": x_flat + CELL_SIZE / 2,
         "y_center": y_flat + CELL_SIZE / 2},
        geometry=cells, crs=GRID_CRS,
    )
    pts        = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["lon"] = pts.x
    gdf["lat"] = pts.y
    print(f"[Grid] Built {len(gdf):,} cells")
    return gdf


def clip_to_sea(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove land cells — fully vectorised via geopandas sindex.

    CHANGE A: Replaced Python for-loop over STRtree candidates with
    geopandas sindex.query_bulk(centroids, predicate='within').
    This is a single vectorised C-level operation — no Python iteration.
    Simplify coastline first (500m) to keep dissolve fast.
    """
    if not COASTLINE_SHP.exists():
        print("[Grid] No coastline → all cells kept (land gets NaN bathymetry)")
        return gdf

    print("[Grid] Loading coastline...")
    coast = gpd.read_file(COASTLINE_SHP).to_crs(gdf.crs)
    print(f"[Grid] Simplifying ({len(coast)} polygons, 500m tolerance)...")
    coast["geometry"] = coast.geometry.simplify(500, preserve_topology=True)
    print("[Grid] Dissolving...")
    land = coast.union_all()
    print("[Grid] Dissolve done")

    # CHANGE A: vectorised bulk query — no Python loop
    land_gdf   = gpd.GeoDataFrame(geometry=[land], crs=gdf.crs)
    centroids  = gpd.GeoDataFrame(
        {"idx": np.arange(len(gdf))},
        geometry=gdf.geometry.centroid,
        crs=gdf.crs,
    )
    # query_bulk returns (centroid_indices, land_indices) for all within matches
    result     = land_gdf.sindex.query(centroids.geometry, predicate="within")
    # result[0] = centroid indices that are within land
    if len(result) > 0 and hasattr(result, '__len__') and len(result) == 2:
        land_cell_idx = np.unique(result[0])
    else:
        # fallback for different geopandas versions
        land_cell_idx = np.array([], dtype=int)

    print(f"[Grid] {len(land_cell_idx):,} land cells identified via sindex")
    sea = gdf[~gdf.index.isin(land_cell_idx)].reset_index(drop=True)
    print(f"[Grid] {len(land_cell_idx):,} land cells removed → {len(sea):,} sea cells")
    return sea


def add_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts        = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf        = gdf.copy()
    gdf["lon"] = pts.x
    gdf["lat"] = pts.y
    return gdf


def save_grid(gdf: gpd.GeoDataFrame, path: Path = GRID_GPKG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    print(f"[Grid] Saved → {path}  ({len(gdf):,} cells)")


def load_or_build_grid() -> gpd.GeoDataFrame:
    if GRID_GPKG.exists():
        print(f"[Grid] Loading cached grid from {GRID_GPKG}")
        gdf = gpd.read_file(GRID_GPKG)
        print(f"[Grid] Loaded {len(gdf):,} cells")
        return gdf
    GRID_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf = build_grid()
    gdf = clip_to_sea(gdf)
    gdf = add_latlon(gdf)
    save_grid(gdf)
    return gdf
