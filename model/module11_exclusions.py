"""
model/module11_exclusions.py
----------------------------
Module 11 — Exclusion and conflict layers.

CHANGELOG:
- D: Removed dead _intersects_strtree() slow function (Python per-cell loop)
- E: Replaced zonal_stats with rasterio point sampling at centroids
     No polygon overlap needed at 1km resolution — centroid is sufficient
     Expected: 15 min → ~30s for shipping layer
- F: All exclusion layers processed concurrently (ThreadPoolExecutor)
     Expected: 5 layers × 1 min → ~1 min total
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    MPA_SHP, MILITARY_SHP, WINDFARMS_SHP, PIPELINES_SHP,
    FISHING_SHP, HARBOURS_SHP, CABLES_DIRS, SHIPPING_TIF_DIR,
    MIN_DEPTH_M,
)

SIMPLIFY_M = 500   # metres — invisible at 1km resolution


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_shp(path: Path, crs: str, simplify: bool = True) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"  [M11] MISSING: {path.name} → skipped")
        return None
    gdf = gpd.read_file(path).to_crs(crs)
    if simplify and not gdf.empty:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    return gdf


def _merge_shapefiles(dirs: list[Path], crs: str) -> gpd.GeoDataFrame | None:
    frames = []
    for d in dirs:
        if not d.exists():
            continue
        for shp in sorted(d.glob("*.shp")):
            try:
                gdf = gpd.read_file(shp).to_crs(crs)
                gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
                frames.append(gdf[["geometry"]])
            except Exception as e:
                print(f"  [M11] Error: {shp.name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else None


def _sindex_intersects(grid: gpd.GeoDataFrame,
                       layer: gpd.GeoDataFrame) -> np.ndarray:
    """
    Fast bulk intersection via geopandas sindex.query_bulk.
    No union_all() needed — queries each layer feature separately.
    Returns bool array length len(grid).
    """
    result = layer.sindex.query(grid.geometry, predicate="intersects")
    # result[0] = grid indices that intersect any layer feature
    mask = np.zeros(len(grid), dtype=bool)
    if len(result) > 0:
        if hasattr(result, '__len__') and len(result) == 2:
            mask[np.unique(result[0])] = True
        else:
            mask[np.unique(result)] = True
    return mask


def _shipping_score_point_sample(
    grid: gpd.GeoDataFrame,
    tif_dir: Path,
) -> np.ndarray:
    """
    CHANGE E: Point-sample shipping rasters at cell centroids using rasterio.

    At 1km cell resolution, sampling at the centroid is equivalent to
    zonal statistics — no polygon overlap computation needed.
    Pre-averages all annual TIFs into one array first (in memory),
    then does one vectorised point read.

    Expected: 15 min (zonal_stats on 480k cells) → ~30s.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        print("  [M11] rasterio not available → shipping score = 0")
        return np.zeros(len(grid))

    tifs = sorted(tif_dir.glob("*.tif")) if tif_dir.exists() else []
    if not tifs:
        print(f"  [M11] No TIFs in {tif_dir.name} → shipping = 0")
        return np.zeros(len(grid))

    print(f"  [M11] Shipping: point-sampling {len(tifs)} rasters at centroids...")

    # Get centroid coordinates in WGS84
    centroids_wgs = grid.geometry.centroid.to_crs("EPSG:4326")
    xs = centroids_wgs.x.values
    ys = centroids_wgs.y.values

    # Read and average all TIFs in memory
    arrays, transform, nodata = [], None, None
    for tif in tifs:
        with rasterio.open(tif) as src:
            data = src.read(1).astype(float)
            nd   = src.nodata if src.nodata is not None else -9999
            data[data == nd] = np.nan
            arrays.append(data)
            if transform is None:
                transform = src.transform
                nodata    = nd

    avg = np.nanmean(arrays, axis=0)

    # Point sample: convert lon/lat → row/col indices (vectorised)
    with rasterio.open(tifs[0]) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)

    rows = np.clip(np.array(rows), 0, avg.shape[0] - 1)
    cols = np.clip(np.array(cols), 0, avg.shape[1] - 1)
    vals = avg[rows, cols]
    vals = np.where(np.isnan(vals), 0.0, vals)

    mx = vals.max()
    return (vals / mx).clip(0, 1) if mx > 0 else vals


def _check_excl_layer(args) -> tuple[str, np.ndarray]:
    """Worker for parallel exclusion layer check."""
    name, path, crs, grid_geom_wkt = args
    # Reconstruct grid geometry in worker
    from shapely import wkt as swkt
    import geopandas as gpd
    import numpy as np

    if not path.exists():
        return name, np.zeros(len(grid_geom_wkt), dtype=bool)

    layer = gpd.read_file(path).to_crs(crs)
    if layer.empty:
        return name, np.zeros(len(grid_geom_wkt), dtype=bool)

    layer["geometry"] = layer.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    grid_tmp = gpd.GeoDataFrame(
        geometry=[swkt.loads(w) for w in grid_geom_wkt], crs=crs
    )
    result = layer.sindex.query(grid_tmp.geometry, predicate="intersects")
    mask   = np.zeros(len(grid_geom_wkt), dtype=bool)
    if len(result) > 0 and hasattr(result, '__len__') and len(result) == 2:
        mask[np.unique(result[0])] = True
    return name, mask


# ── Module 11 ──────────────────────────────────────────────────────────────────

def run_module11(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Module 11: Exclusion masks and conflict gradients.

    CHANGE D: Removed slow _intersects_strtree() dead code.
    CHANGE E: Shipping raster uses point sampling (not zonal_stats).
    CHANGE F: All 5 exclusion layers processed concurrently.
    """
    print("\n=== Module 11: Exclusions and Conflict Layers ===")
    grid = grid.copy()
    crs  = grid.crs

    # ── Depth exclusion (instant) ──────────────────────────────────────────────
    grid["excl_shallow"] = grid["bathymetry_m"].fillna(0) < MIN_DEPTH_M
    print(f"  [M11] excl_shallow: {grid['excl_shallow'].sum():,} cells")

    # ── CHANGE F: polygon exclusion layers in parallel ─────────────────────────
    print("  [M11] Running polygon exclusion layers in parallel...")

    # Cables need merging first (done outside thread to avoid file I/O contention)
    cables = _merge_shapefiles(CABLES_DIRS, crs)

    single_layers = {
        "mpa":        MPA_SHP,
        "military":   MILITARY_SHP,
        "wind_farms": WINDFARMS_SHP,
        "pipelines":  PIPELINES_SHP,
    }

    # Serialize grid geometry as WKT for subprocess safety
    grid_wkt = [g.wkt for g in grid.geometry]

    def _check(name, path):
        if not path.exists():
            print(f"  [M11] MISSING: {name} → skipped")
            return name, np.zeros(len(grid), dtype=bool)
        layer = _load_shp(path, crs)
        if layer is None or layer.empty:
            return name, np.zeros(len(grid), dtype=bool)
        mask = _sindex_intersects(grid, layer)
        return name, mask

    with ThreadPoolExecutor(max_workers=len(single_layers)) as ex:
        futs = {ex.submit(_check, name, path): name
                for name, path in single_layers.items()}
        for fut in tqdm(as_completed(futs), total=len(single_layers),
                        desc="  Exclusion layers"):
            name, mask = fut.result()
            col = f"excl_{name}"
            grid[col] = mask
            print(f"  [M11] {col:20s}: {mask.sum():,} cells")

    # Cables
    if cables is not None:
        grid["excl_cables"] = _sindex_intersects(grid, cables)
        print(f"  [M11] excl_cables           : {grid['excl_cables'].sum():,} cells")
    else:
        grid["excl_cables"] = False

    excl_cols       = [c for c in grid.columns if c.startswith("excl_")]
    grid["excluded"] = grid[excl_cols].any(axis=1)
    pct = 100 * grid["excluded"].mean()
    print(f"\n  [M11] Total excluded: {grid['excluded'].sum():,} ({pct:.1f}%)")
    print(f"  [M11] Available:      {(~grid['excluded']).sum():,} cells")

    # ── Conflict gradients (display only) ──────────────────────────────────────
    print("\n  [M11] Computing conflict gradients...")

    # CHANGE E: shipping via point sampling
    grid["conflict_shipping"] = _shipping_score_point_sample(grid, SHIPPING_TIF_DIR)

    # Fishing
    fishing = _load_shp(FISHING_SHP, crs)
    if fishing is not None:
        attr = next((c for c in ["SAR","density","EFFORT"] if c in fishing.columns), None)
        if attr:
            joined = gpd.sjoin(
                grid[["cell_id","geometry"]], fishing[["geometry", attr]],
                how="left", predicate="intersects",
            ).groupby("cell_id")[attr].mean()
            vals = grid["cell_id"].map(joined).fillna(0)
            mx   = vals.max()
            grid["conflict_fishing"] = (vals / mx).clip(0, 1) if mx > 0 else 0.0
        else:
            grid["conflict_fishing"] = _sindex_intersects(grid, fishing).astype(float)
    else:
        grid["conflict_fishing"] = 0.0

    # Harbours — distance to nearest port per cell
    harbours = _load_shp(HARBOURS_SHP, crs, simplify=False)
    if harbours is not None:
        harbour_union = harbours.geometry.union_all()
        # Reset index to avoid misalignment, use numpy array directly
        centroids = grid.geometry.centroid.reset_index(drop=True)
        dists = np.array([pt.distance(harbour_union) for pt in centroids])
        mx = dists.max()
        grid["conflict_harbours"] = (dists / mx).clip(0, 1) if mx > 0 else 0.0
    else:
        grid["conflict_harbours"] = 0.0

    grid["conflict_score"] = (
        grid["conflict_shipping"] +
        grid["conflict_fishing"]  +
        grid["conflict_harbours"]
    ) / 3.0

    print(f"  [M11] Conflict score: "
          f"{grid['conflict_score'].min():.3f} – {grid['conflict_score'].max():.3f}")
    return grid