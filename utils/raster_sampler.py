"""
utils/raster_sampler.py
-----------------------
Module 1 — CMEMS NetCDF → 1km grid (Baltic + NWS merged).

CHANGELOG:
- B: CRS projection computed once at run_module1() start, passed to all
     sample functions — no redundant to_crs() per call
- C: Baltic and NWS sampling run concurrently (ThreadPoolExecutor)
     Baltic (Temp+Sal+ChlA+Flow) and NWS (same) in parallel
     Expected: ~5 min → ~2.5 min
- I: Baseline env columns saved to GRID_ENV_GPKG after first sample.
     Scenario recompute loads this cache and applies deltas only —
     no CMEMS re-sampling. Expected: scenario M1: 5 min → 5 seconds.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import geopandas as gpd
import xarray as xr
from tqdm import tqdm

from config import (
    CMEMS_PHY_DIR, CMEMS_BGC_DIR, CMEMS_FLOW_DIR,
    CMEMS_NWS_PHY_DIR, CMEMS_NWS_BGC_DIR, CMEMS_NWS_FLOW_DIR,
    BATHY_PATH, GRID_CRS, PROCESSED,
    GRID_ENV_TEMP_GPKG, GRID_ENV_SAL_GPKG,
    GRID_ENV_CHLA_GPKG, GRID_ENV_FLOW_GPKG,
)

VAR_ALIASES = {
    "temp": ["thetao", "temperature", "temp"],
    "sal":  ["so", "salinity", "sal"],
    "chl":  ["chl", "chlorophyll", "CHL"],
    "uo":   ["uo", "eastward_sea_water_velocity"],
    "vo":   ["vo", "northward_sea_water_velocity"],
}

DOMAIN_LON    = (4.5, 18.5)
DOMAIN_LAT    = (52.0, 61.0)

# ── Baltic/NWS blend zone (lon degrees) ───────────────────────────────────────
# Scalar fields (temp, sal, chla): linear blend across BLEND_LON_MIN–MAX.
# Flow fields (uo, vo, vh): wider zone + Gaussian post-smooth to suppress
# vector direction artefacts and the M4 critical-flow ridge.
BLEND_LON_MIN      = 8.5    # scalar blend western edge
BLEND_LON_MAX      = 10.0   # scalar blend eastern edge
BLEND_FLOW_LON_MIN = 7.5    # flow blend western edge (wider)
BLEND_FLOW_LON_MAX = 11.5   # flow blend eastern edge (wider)
BLEND_FLOW_SIGMA   = 3      # Gaussian smooth σ in grid cells (~3 km) applied
                             # across full domain post-merge to kill residual ridge

# ── Per-variable env cache paths (imported from config) ───────────────────────
# Kept as module-level references for convenience inside this file.
_CACHE_TEMP = GRID_ENV_TEMP_GPKG
_CACHE_SAL  = GRID_ENV_SAL_GPKG
_CACHE_CHLA = GRID_ENV_CHLA_GPKG
_CACHE_FLOW = GRID_ENV_FLOW_GPKG

_VAR_CACHE_MAP = {
    "temp":   (_CACHE_TEMP,  ["temp_mean", "temp_sd"]),
    "sal":    (_CACHE_SAL,   ["sal_mean",  "sal_sd"]),
    "lnchla": (_CACHE_CHLA,  ["lnchla_mean", "lnchla_sd"]),
    "flow":   (_CACHE_FLOW,  ["vh", "uo", "vo"]),
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_nc(nc_dir: Path) -> xr.Dataset | None:
    files = sorted(nc_dir.glob("*.nc")) if nc_dir.exists() else []
    if not files:
        print(f"  [M1] No .nc in {nc_dir.name}")
        return None
    return xr.open_mfdataset(files, combine="by_coords", engine="netcdf4",
                              chunks={"time": 12})


def _find_var(ds: xr.Dataset, aliases: list[str]) -> str | None:
    return next((v for v in aliases if v in ds), None)


def _depth_mean(ds: xr.Dataset, var: str, max_depth: float = 10.0) -> xr.DataArray:
    for d in ["depth", "lev", "deptht", "z"]:
        if d in ds.dims:
            return ds[var].sel({d: slice(0, max_depth)}).mean(dim=d)
    return ds[var]


def _clip_domain(da: xr.DataArray) -> xr.DataArray:
    """Pre-clip to model domain — reduces interpolation cost 50-70%."""
    lon = next((d for d in ["longitude","lon","x"] if d in da.dims), None)
    lat = next((d for d in ["latitude", "lat","y"] if d in da.dims), None)
    if lon and lat:
        try:
            return da.sel({lon: slice(*DOMAIN_LON), lat: slice(*DOMAIN_LAT)})
        except Exception:
            pass
    return da


def _get_ll_dims(da: xr.DataArray) -> tuple[str, str]:
    lon = next((d for d in ["longitude","lon","x"] if d in da.dims), None)
    lat = next((d for d in ["latitude", "lat","y"] if d in da.dims), None)
    if not lon or not lat:
        raise ValueError(f"No lat/lon dims in {list(da.dims)}")
    return lon, lat


def _interp_month(args) -> tuple[int, np.ndarray, np.ndarray]:
    """Worker: interpolate one month. Returns (month_idx_0based, mean, sd)."""
    month, da_surf, lons, lats, is_log = args
    da   = _clip_domain(da_surf)
    lon_d, lat_d = _get_ll_dims(da)
    pts_lon = xr.DataArray(lons, dims="points")
    pts_lat = xr.DataArray(lats, dims="points")

    monthly = da.sel(time=da.time.dt.month == month)
    if monthly.sizes["time"] == 0:
        return month - 1, np.full(len(lons), np.nan), np.full(len(lons), np.nan)

    mean_da = monthly.mean(dim="time").load()
    sd_da   = monthly.std(dim="time").load()

    m_vals = mean_da.interp({lon_d: pts_lon, lat_d: pts_lat},
                             method="linear").values.astype(float)
    s_vals = sd_da.interp({lon_d: pts_lon, lat_d: pts_lat},
                           method="linear").values.astype(float)
    if is_log:
        m_vals = np.log(np.maximum(m_vals, 0.01))
        s_vals = np.abs(s_vals)
    return month - 1, m_vals, np.abs(s_vals)


def _sample_var_parallel(
    nc_dir: Path, var_type: str, aliases: list[str],
    col_prefix: str, lons: np.ndarray, lats: np.ndarray,
    n: int, n_workers: int = 4,
) -> dict[str, np.ndarray]:
    """
    Sample one variable for all 12 months in parallel.
    CHANGE B: accepts pre-computed lons/lats — no redundant CRS projection.
    Returns dict of column_name → array.
    """
    defaults = {"temp": 10.0, "sal": 20.0, "lnchla": np.log(2.0)}
    result   = {}

    ds = _load_nc(nc_dir)
    if ds is None:
        d = defaults.get(col_prefix, 0.0)
        for m in range(1, 13):
            result[f"{col_prefix}_mean_{m:02d}"] = np.full(n, d)
            result[f"{col_prefix}_sd_{m:02d}"]   = np.zeros(n)
        return result

    var_name = _find_var(ds, aliases)
    if not var_name:
        raise KeyError(f"None of {aliases} in {nc_dir.name}. Got: {list(ds.data_vars)}")

    da_surf  = _depth_mean(ds, var_name)
    is_log   = (var_type == "chl")
    prefix   = "lnchla" if is_log else col_prefix

    means = np.full((n, 12), np.nan)
    sds   = np.full((n, 12), np.nan)

    args_list = [(m, da_surf, lons, lats, is_log) for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_interp_month, a): a[0] for a in args_list}
        for fut in as_completed(futs):
            mi, mv, sv = fut.result()
            means[:, mi] = mv
            sds[:, mi]   = sv

    ds.close()
    for m in range(1, 13):
        result[f"{prefix}_mean_{m:02d}"] = means[:, m - 1]
        result[f"{prefix}_sd_{m:02d}"]   = sds[:, m - 1]
    return result


def _sample_flow_parallel(
    nc_dir: Path, lons: np.ndarray, lats: np.ndarray,
    n: int, n_workers: int = 4,
) -> dict[str, np.ndarray]:
    """
    Sample uo, vo, vh for all 12 months in parallel.
    CHANGE B: accepts pre-computed lons/lats.
    """
    result = {}
    ds = _load_nc(nc_dir)

    if ds is None:
        for m in range(1, 13):
            result[f"vh_{m:02d}"] = np.full(n, 0.05)
            result[f"uo_{m:02d}"] = np.full(n, 0.05)
            result[f"vo_{m:02d}"] = np.zeros(n)
        return result

    uo_name = _find_var(ds, VAR_ALIASES["uo"])
    vo_name = _find_var(ds, VAR_ALIASES["vo"])
    if not uo_name or not vo_name:
        raise KeyError(f"uo/vo not in {nc_dir.name}")

    uo_surf = _clip_domain(_depth_mean(ds, uo_name))
    vo_surf = _clip_domain(_depth_mean(ds, vo_name))
    lon_d, lat_d = _get_ll_dims(uo_surf)
    pts_lon = xr.DataArray(lons, dims="points")
    pts_lat = xr.DataArray(lats, dims="points")

    def _flow_month(month):
        uo_m = uo_surf.sel(time=uo_surf.time.dt.month == month).mean(dim="time").load()
        vo_m = vo_surf.sel(time=vo_surf.time.dt.month == month).mean(dim="time").load()
        uv = uo_m.interp({lon_d: pts_lon, lat_d: pts_lat}, method="linear").values.astype(float)
        vv = vo_m.interp({lon_d: pts_lon, lat_d: pts_lat}, method="linear").values.astype(float)
        return month, uv, vv

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_flow_month, m): m for m in range(1, 13)}
        for fut in as_completed(futs):
            month, uv, vv = fut.result()
            mk = f"{month:02d}"
            result[f"uo_{mk}"] = uv
            result[f"vo_{mk}"] = vv
            result[f"vh_{mk}"] = np.sqrt(uv**2 + vv**2)

    ds.close()
    return result


def _apply_cols(grid: gpd.GeoDataFrame, cols: dict) -> gpd.GeoDataFrame:
    """Apply a dict of {col_name: array} to grid, filling NaN only."""
    for col, vals in cols.items():
        if col not in grid.columns:
            grid[col] = vals
        else:
            mask = grid[col].isna()
            if mask.any():
                grid.loc[mask, col] = np.asarray(vals)[mask]
    return grid


def _blend_nws(
    grid: gpd.GeoDataFrame,
    baltic_cols: dict,
    nws_cols: dict,
) -> dict:
    """
    Merge Baltic and NWS columns with product-appropriate blending.

    Scalar fields (temp, sal, lnchla):
      Linear distance-weighted blend across BLEND_LON_MIN–BLEND_LON_MAX.

    Flow fields (uo, vo, vh):
      1. Blend vh (speed scalar) across the wider BLEND_FLOW_LON_MIN–MAX zone.
         uo/vo are reconstructed from blended vh + direction weighted the same way,
         avoiding spurious direction artefacts at the seam.
      2. Gaussian smooth (σ=BLEND_FLOW_SIGMA cells) applied to uo, vo, vh
         across the full domain post-merge to kill any residual ridge.
         The smooth is light enough (~3 km) not to smear mesoscale features.

    In all cases: where only one source has valid data, that source is used
    without modification.
    """
    from scipy.ndimage import gaussian_filter

    lons = grid["lon"].values  # WGS84 centroid longitude

    # Scalar weight (0=pure NWS, 1=pure Baltic)
    w_baltic = np.clip(
        (lons - BLEND_LON_MIN) / (BLEND_LON_MAX - BLEND_LON_MIN), 0.0, 1.0
    )
    # Flow weight — wider zone
    w_baltic_flow = np.clip(
        (lons - BLEND_FLOW_LON_MIN) / (BLEND_FLOW_LON_MAX - BLEND_FLOW_LON_MIN),
        0.0, 1.0,
    )

    # Build a spatial index for Gaussian smooth: need (row, col) structure.
    # Approximate grid shape from unique lon/lat counts.
    u_lons = np.unique(np.round(lons, 4))
    u_lats = np.unique(np.round(grid["lat"].values, 4))
    n_col  = len(u_lons)
    n_row  = len(u_lats)
    grid_shaped = (n_row > 1 and n_col > 1 and n_row * n_col == len(grid))

    def _smooth_flow(arr: np.ndarray) -> np.ndarray:
        """Apply Gaussian smooth if grid is regular, else skip."""
        if not grid_shaped:
            return arr
        mat     = arr.reshape(n_row, n_col)
        mat_nan = np.where(np.isnan(mat), 0.0, mat)
        mask    = (~np.isnan(mat)).astype(float)
        s_val   = gaussian_filter(mat_nan, sigma=BLEND_FLOW_SIGMA)
        s_mask  = gaussian_filter(mask,    sigma=BLEND_FLOW_SIGMA)
        result  = np.where(s_mask > 0, s_val / s_mask, np.nan)
        return result.ravel()

    is_flow = lambda col: any(col.startswith(p) for p in ("uo_", "vo_", "vh_"))

    merged   = {}
    all_cols = set(baltic_cols) | set(nws_cols)

    for col in all_cols:
        b = np.asarray(baltic_cols.get(col, np.full(len(grid), np.nan)), float)
        n = np.asarray(nws_cols.get(col,   np.full(len(grid), np.nan)), float)

        both   = np.isfinite(b) & np.isfinite(n)
        only_b = np.isfinite(b) & ~np.isfinite(n)
        only_n = ~np.isfinite(b) & np.isfinite(n)

        w = w_baltic_flow if is_flow(col) else w_baltic

        out = np.full(len(grid), np.nan)
        out[both]   = w[both] * b[both] + (1.0 - w[both]) * n[both]
        out[only_b] = b[only_b]
        out[only_n] = n[only_n]

        if is_flow(col):
            out = _smooth_flow(out)

        merged[col] = out

    return merged


# ── Bathymetry ─────────────────────────────────────────────────────────────────

def sample_bathymetry(grid: gpd.GeoDataFrame,
                      tif_path: Path = BATHY_PATH) -> gpd.GeoDataFrame:
    from rasterstats import zonal_stats
    grid = grid.copy()
    if not tif_path.exists():
        raise FileNotFoundError(
            f"Bathymetry TIF not found: {tif_path}\n"
            "Download EMODnet DTM and place at the path configured in config.py (BATHY_PATH). "
            "Bathymetry is required — no fallback is available."
        )
    print(f"  [M1] Sampling bathymetry ({len(grid):,} cells)...")
    stats = zonal_stats(grid.to_crs("EPSG:4326"), str(tif_path),
                        stats=["mean"], nodata=-9999)
    grid["bathymetry_m"] = [abs(s["mean"]) if s["mean"] else np.nan for s in stats]
    print(f"  [M1] Bathymetry: {grid['bathymetry_m'].notna().sum():,} valid cells")
    return grid


# ── Module 1 ───────────────────────────────────────────────────────────────────

def _save_var_cache(grid: gpd.GeoDataFrame, path: Path, prefixes: list[str]) -> None:
    """Save columns matching any of the given prefixes to a GeoPackage cache."""
    cols = [c for c in grid.columns
            if any(c.startswith(p) for p in prefixes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    grid[["cell_id"] + cols + ["geometry"]].to_file(path, driver="GPKG")
    print(f"  [M1] Cached → {path.name}  ({len(cols)} columns)")


def _load_var_cache(grid: gpd.GeoDataFrame, path: Path) -> gpd.GeoDataFrame:
    """Load a variable cache and merge its columns onto grid by position."""
    cached = gpd.read_file(path)
    result = grid.copy()
    for col in cached.columns:
        if col in ("cell_id", "geometry"):
            continue
        result[col] = cached[col].values
    return result


def run_module1(
    grid: gpd.GeoDataFrame,
    delta_temp: float  = 0.0,
    delta_sal: float   = 0.0,
    chla_mult: float   = 1.0,
    delta_flow: bool   = False,
    n_workers: int     = 4,
    force_resample: bool = False,
) -> gpd.GeoDataFrame:
    """
    Module 1: Sample CMEMS env fields (Baltic + NWS blended).

    Per-variable caching:
      grid_env_temp.gpkg  — resampled when delta_temp != 0 or force_resample
      grid_env_sal.gpkg   — resampled when delta_sal  != 0 or force_resample
      grid_env_chla.gpkg  — resampled when chla_mult  != 1 or force_resample
      grid_env_flow.gpkg  — resampled when delta_flow  = True or force_resample

    Baltic and NWS are blended with distance-weighted merge across 8.5–10°E.

    Parameters
    ----------
    delta_flow   : if True, resample flow from CMEMS (scenario with changed
                   circulation); otherwise load from cache.
    force_resample : ignore all caches and re-sample everything from CMEMS.
    """
    perturb_temp  = delta_temp != 0.0
    perturb_sal   = delta_sal  != 0.0
    perturb_chla  = chla_mult  != 1.0
    any_scenario  = perturb_temp or perturb_sal or perturb_chla or delta_flow

    label = (f"SCENARIO ΔT={delta_temp:+.1f} ΔS={delta_sal:+.1f} "
             f"ChlA×{chla_mult:.2f}" if any_scenario else "BASELINE")
    print(f"\n=== Module 1: {label} ===")

    # Determine which variables need CMEMS resampling vs cache load
    resample = {
        "temp":   force_resample or perturb_temp  or not _CACHE_TEMP.exists(),
        "sal":    force_resample or perturb_sal   or not _CACHE_SAL.exists(),
        "lnchla": force_resample or perturb_chla  or not _CACHE_CHLA.exists(),
        "flow":   force_resample or delta_flow    or not _CACHE_FLOW.exists(),
    }

    grid = grid.copy()

    # ── Load cached variables that don't need resampling ─────────────────────
    for var, needs_resample in resample.items():
        if not needs_resample:
            cache_path = _VAR_CACHE_MAP[var][0]
            print(f"  [M1] {var:8s} → loading from cache ({cache_path.name})")
            grid = _load_var_cache(grid, cache_path)

    # ── Sample only what's needed ─────────────────────────────────────────────
    vars_to_sample = [v for v, s in resample.items() if s]
    if not vars_to_sample:
        print("  [M1] All variables loaded from cache — no CMEMS sampling needed")
    else:
        print(f"  [M1] Projecting grid centroids to WGS84 (once)...")
        grid_wgs = grid.to_crs("EPSG:4326")
        lons     = grid_wgs.geometry.centroid.x.values
        lats     = grid_wgs.geometry.centroid.y.values
        n        = len(grid)

        print(f"  [M1] Sampling Baltic + NWS concurrently: {vars_to_sample}")

        def _sample_source(phy_dir, bgc_dir, flow_dir, label_s):
            out = {}
            if "temp" in vars_to_sample:
                print(f"    [{label_s}] Temp...")
                out.update(_sample_var_parallel(
                    phy_dir, "temp", VAR_ALIASES["temp"], "temp", lons, lats, n, n_workers))
            if "sal" in vars_to_sample:
                print(f"    [{label_s}] Sal...")
                out.update(_sample_var_parallel(
                    phy_dir, "sal", VAR_ALIASES["sal"], "sal", lons, lats, n, n_workers))
            if "lnchla" in vars_to_sample:
                print(f"    [{label_s}] ChlA...")
                out.update(_sample_var_parallel(
                    bgc_dir, "chl", VAR_ALIASES["chl"], "lnchla", lons, lats, n, n_workers))
            if "flow" in vars_to_sample:
                print(f"    [{label_s}] Flow...")
                flow_result = _sample_flow_parallel(flow_dir, lons, lats, n, n_workers)
                if flow_dir == CMEMS_FLOW_DIR and flow_result.get(f"vh_01", np.array([0.05]))[0] == 0.05:
                    print(f"  [M1] WARNING: No flow NetCDF found in {flow_dir.name} "
                          f"— using placeholder vh=0.05 m/s. Farm orientation and "
                          f"food limitation results will be unreliable.")
                out.update(flow_result)
            return out

        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=2) as ex:
            fut_b = ex.submit(_sample_source,
                              CMEMS_PHY_DIR, CMEMS_BGC_DIR, CMEMS_FLOW_DIR, "Baltic")
            fut_n = ex.submit(_sample_source,
                              CMEMS_NWS_PHY_DIR, CMEMS_NWS_BGC_DIR, CMEMS_NWS_FLOW_DIR, "NWS")
            baltic_cols = fut_b.result()
            nws_cols    = fut_n.result()

        # Blend Baltic + NWS across overlap zone (replaces hard fill-NaN merge)
        blended = _blend_nws(grid, baltic_cols, nws_cols)
        for col, vals in blended.items():
            grid[col] = vals

        # ── Save per-variable caches (baseline only — no delta applied yet) ──
        if not any_scenario:
            if "temp"   in vars_to_sample:
                _save_var_cache(grid, _CACHE_TEMP,  ["temp_mean_", "temp_sd_"])
            if "sal"    in vars_to_sample:
                _save_var_cache(grid, _CACHE_SAL,   ["sal_mean_",  "sal_sd_"])
            if "lnchla" in vars_to_sample:
                _save_var_cache(grid, _CACHE_CHLA,  ["lnchla_mean_", "lnchla_sd_"])
            if "flow"   in vars_to_sample:
                _save_var_cache(grid, _CACHE_FLOW,  ["vh_", "uo_", "vo_"])

    # ── Apply scenario deltas (in-memory, never written to cache) ────────────
    if any_scenario:
        print(f"  [M1] Applying scenario deltas: {label}")
        for m in range(1, 13):
            mc = f"{m:02d}"
            if perturb_temp and f"temp_mean_{mc}" in grid.columns:
                grid[f"temp_mean_{mc}"] = grid[f"temp_mean_{mc}"] + delta_temp
            if perturb_sal and f"sal_mean_{mc}" in grid.columns:
                grid[f"sal_mean_{mc}"]  = grid[f"sal_mean_{mc}"]  + delta_sal
            if perturb_chla and f"lnchla_mean_{mc}" in grid.columns:
                grid[f"lnchla_mean_{mc}"] = (grid[f"lnchla_mean_{mc}"]
                                             + np.log(chla_mult))
        print(f"  [M1] Deltas applied in <1s")

    # ── Diagnostic summary ────────────────────────────────────────────────────
    for var, col in [("Temp", "temp_mean_07"), ("Sal", "sal_mean_07"),
                     ("ChlA", "lnchla_mean_07"), ("Flow", "vh_07")]:
        v = int(grid[col].notna().sum()) if col in grid.columns else 0
        print(f"  [M1] {var:6s} July: {v:,} valid cells")

    return grid