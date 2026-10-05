"""
utils/climate_projections.py
----------------------------
Assumes all raw data is already downloaded:
  - CDS (tos, sos): run utils/download_cds.py first
  - ESGF (uo, vo, chl): run utils/download_esgf.py first

Interpolates to 1km grid, computes delta vs historical baseline,
applies delta to baseline .gpkg files, saves scenario .gpkg files.

Output per SSP × period:
    data/processed/<ssp_key>/<period>/grid_env_temp_<ssp_key>_<period>.gpkg
    data/processed/<ssp_key>/<period>/grid_env_sal_<ssp_key>_<period>.gpkg
    data/processed/<ssp_key>/<period>/grid_env_chla_<ssp_key>_<period>.gpkg
    data/processed/<ssp_key>/<period>/grid_env_flow_<ssp_key>_<period>.gpkg

These are drop-in replacements for the baseline .gpkg files passed to Module 2.

Run standalone:
    python utils/climate_projections.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
xr.set_options(use_new_combine_kwarg_defaults=True)
from config import (
    PROCESSED, GRID_GPKG,
    GRID_ENV_TEMP_GPKG, GRID_ENV_SAL_GPKG,
    GRID_ENV_CHLA_GPKG, GRID_ENV_FLOW_GPKG,
)

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DIR = PROCESSED / "cmip6_raw"

SSP_SCENARIOS = {
    "ssp1_2_6": "SSP1-2.6",
    "ssp2_4_5": "SSP2-4.5",
    "ssp3_7_0": "SSP3-7.0",
    "ssp5_8_5": "SSP5-8.5",
}

SSP_ESGF_MAP = {
    "ssp1_2_6": "ssp126",
    "ssp2_4_5": "ssp245",
    "ssp3_7_0": "ssp370",
    "ssp5_8_5": "ssp585",
}

# CDS folders use the raw experiment name (ssp126, not ssp1_2_6)
SSP_CDS_DIR = {
    "ssp1_2_6": "ssp126_cds",
    "ssp2_4_5": "ssp245_cds",
    "ssp3_7_0": "ssp370_cds",
    "ssp5_8_5": "ssp585_cds",
}

PERIODS = {
    "mid": ("2041-01-01", "2060-12-31"),
    "end": ("2071-01-01", "2100-12-31"),
}

HIST_DATE = ("1985-01-01", "2014-12-31")
MONTHS    = [f"{m:02d}" for m in range(1, 13)]


# ── NetCDF helpers ─────────────────────────────────────────────────────────────

def _open_nc(pattern: str) -> xr.Dataset:
    files = sorted(glob.glob(pattern))
    if not files:
        # Fall back to all .nc in the same directory
        from pathlib import Path as _Path
        folder = str(_Path(pattern).parent)
        files  = sorted(glob.glob(folder + "/*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=time_coder)
    rename = {}
    for wrong, right in [("latitude", "lat"), ("longitude", "lon"),
                          ("nav_lat",  "lat"), ("nav_lon",  "lon")]:
        if wrong in ds.dims or wrong in ds.coords:
            rename[wrong] = right
    return ds.rename(rename) if rename else ds


def _time_slice(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    return ds.sel(time=slice(start, end))


# ── Interpolation ──────────────────────────────────────────────────────────────

LAT_NAMES = ["lat", "latitude", "nav_lat", "y", "rlat"]
LON_NAMES = ["lon", "longitude", "nav_lon", "x", "rlon"]


def _find_coord(da: xr.DataArray, names: list[str]) -> str | None:
    """Find the first matching coordinate name in a DataArray."""
    for n in names:
        if n in da.coords or n in da.dims:
            return n
    return None


def _surface(da: xr.DataArray) -> xr.DataArray:
    """Select surface level if depth dimension exists."""
    for d in ["depth", "lev", "deptht", "z", "olevel"]:
        if d in da.dims:
            return da.isel({d: 0})
    return da


def _interp_regular(da_mean: xr.DataArray, lat_name: str, lon_name: str,
                    lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    src_lats = da_mean[lat_name].values
    src_lons = da_mean[lon_name].values
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        da_mean  = da_mean.isel({lat_name: slice(None, None, -1)})
    if src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        da_mean  = da_mean.isel({lon_name: slice(None, None, -1)})
    interp = RegularGridInterpolator(
        (src_lats, src_lons), da_mean.values,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    return interp(np.column_stack([lats, lons]))


def _interp_curvilinear(da_mean: xr.DataArray, lat_name: str, lon_name: str,
                        lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    src_lat = da_mean[lat_name].values.ravel()
    src_lon = da_mean[lon_name].values.ravel()
    src_val = da_mean.values.ravel()
    valid   = np.isfinite(src_val)
    tree    = cKDTree(np.column_stack([src_lat[valid], src_lon[valid]]))
    dists, idx = tree.query(np.column_stack([lats, lons]), k=4)
    dists   = np.where(dists == 0, 1e-10, dists)
    weights = 1.0 / dists
    weights /= weights.sum(axis=1, keepdims=True)
    return (src_val[valid][idx] * weights).sum(axis=1)


def _interpolate(da: xr.DataArray,
                 lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    da_mean  = _surface(da.mean(dim="time").load())
    lat_name = _find_coord(da_mean, LAT_NAMES)
    lon_name = _find_coord(da_mean, LON_NAMES)
    if not lat_name or not lon_name:
        raise ValueError(f"Cannot find lat/lon coords in {list(da_mean.coords)}. Dims: {list(da_mean.dims)}")
    # Regular grid: lat/lon are 1D dims
    if lat_name in da_mean.dims and da_mean[lat_name].ndim == 1:
        return _interp_regular(da_mean, lat_name, lon_name, lons, lats)
    # Curvilinear: lat/lon are 2D coords
    return _interp_curvilinear(da_mean, lat_name, lon_name, lons, lats)


def _monthly_interp(
    da: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
    is_log: bool = False,
    unit_scale: float = 1.0,
    var_label: str = "",
) -> dict[str, np.ndarray]:
    """
    Interpolate a DataArray to grid centroids for all 12 months.

    Parameters
    ----------
    unit_scale : multiplicative factor applied BEFORE log transform.
                 Use 1e6 for CMIP6 chl (kg m-3 → µg L-1) so values are
                 on the same scale as the CMEMS baseline (already µg L-1).
                 Default 1.0 (no conversion).
    var_label  : label printed in diagnostics to confirm scale applied.
    """
    if unit_scale != 1.0:
        src_units = da.attrs.get("units", "unknown")
        print(f"    [unit_scale] {var_label}: ×{unit_scale:.0e}  "
              f"({src_units} → µg L⁻¹ equivalent for log transform)")

    results = {}
    for m in range(1, 13):
        mk      = f"{m:02d}"
        monthly = da.sel(time=da.time.dt.month == m)
        if monthly.sizes["time"] == 0:
            results[mk] = np.full(len(lons), np.nan)
            continue
        vals = _interpolate(monthly, lons, lats)
        if unit_scale != 1.0:
            vals = vals * unit_scale
        if is_log:
            # Floor matches CHLA_ZERO in config (0.01 µg L-1 after unit conversion)
            vals = np.log(np.maximum(vals, 0.01))
        results[mk] = vals
    return results


# ── Delta ──────────────────────────────────────────────────────────────────────

def _compute_delta(fut: dict, hist: dict) -> dict:
    return {mk: fut[mk] - hist[mk] for mk in MONTHS}


# ── Output .gpkg ───────────────────────────────────────────────────────────────

def _apply_delta(baseline_gpkg: Path, col_prefix: str,
                 deltas: dict, out_path: Path) -> None:
    """
    Apply monthly deltas to baseline gpkg columns and save.

    For lnchla columns: delta is ln(fut_µgL) - ln(hist_µgL), both converted
    from kg m-3 × 1e6 before log. Result added to CMEMS ln(ChlA µg L-1)
    baseline — units consistent throughout.
    """
    gdf = gpd.read_file(baseline_gpkg).copy()
    n_applied = 0
    for mk in MONTHS:
        col = f"{col_prefix}_mean_{mk}"
        if col in gdf.columns and mk in deltas:
            delta_vals = deltas[mk]
            non_nan = np.isfinite(delta_vals).sum()
            gdf[col] = gdf[col] + delta_vals
            n_applied += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    # Report mean delta for first valid month as a sanity indicator
    first_mk = next((mk for mk in MONTHS if np.isfinite(deltas[mk]).any()), None)
    delta_summary = (f"  mean Δ month 01 = {np.nanmean(deltas[first_mk]):+.4f}"
                     if first_mk else "")
    print(f"  [OUT] {out_path.name}  ({n_applied} months applied){delta_summary}")


def _build_flow_gpkg(baseline_gpkg: Path, uo_deltas: dict,
                     vo_deltas: dict, out_path: Path) -> None:
    gdf = gpd.read_file(baseline_gpkg).copy()
    for mk in MONTHS:
        if f"uo_{mk}" in gdf.columns:
            new_uo = gdf[f"uo_{mk}"] + uo_deltas.get(mk, 0.0)
            new_vo = gdf[f"vo_{mk}"] + vo_deltas.get(mk, 0.0)
            gdf[f"uo_{mk}"] = new_uo
            gdf[f"vo_{mk}"] = new_vo
            gdf[f"vh_{mk}"] = np.sqrt(new_uo**2 + new_vo**2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    print(f"  [OUT] {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_climate_projections(
    grid: gpd.GeoDataFrame,
    ssp: str | None = None,
    period: str = "end",
) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Interpolate downloaded CMIP6 data to 1km grid and produce scenario .gpkg files.

    Parameters
    ----------
    grid   : base grid GeoDataFrame
    ssp    : one of 'ssp1_2_6','ssp2_4_5','ssp3_7_0','ssp5_8_5' or None for all
    period : 'mid', 'end', or 'both'

    Returns
    -------
    paths[ssp_key][period_key] = {'temp', 'sal', 'chla', 'flow'} → Path
    """
    ssps    = {ssp: SSP_SCENARIOS[ssp]} if ssp else SSP_SCENARIOS
    periods = list(PERIODS.items()) if period == "both" else [(period, PERIODS[period])]

    # Grid centroids in WGS84
    centroids_utm = grid.geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_utm, crs=grid.crs).to_crs("EPSG:4326")
    lons = centroids_wgs.x.values
    lats = centroids_wgs.y.values

    # ── Historical baseline ───────────────────────────────────────────────────
    print("\n=== Interpolating historical baseline ===")
    hist_dir     = RAW_DIR / "historical_cds"
    # CDS downloads tos and sos as separate files — open all and merge
    all_hist_files = sorted(glob.glob(str(hist_dir / "*.nc")))
    print(f"  [CDS hist] files found: {[Path(f).name for f in all_hist_files]}")
    ds_hist_cds = xr.open_mfdataset(
        all_hist_files, combine="by_coords",
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    )
    ds_hist_cds = _time_slice(ds_hist_cds, *HIST_DATE)
    print(f"  [CDS hist] variables: {list(ds_hist_cds.data_vars)}")
    tos_var = next((v for v in ds_hist_cds.data_vars if v.lower() in ("tos", "sea_surface_temperature")), None)
    sos_var = next((v for v in ds_hist_cds.data_vars if v.lower() in ("sos", "sea_surface_salinity")),    None)
    if not tos_var:
        raise KeyError(f"tos not found in {list(ds_hist_cds.data_vars)} — check CDS historical download")
    if not sos_var:
        raise KeyError(f"sos not found in {list(ds_hist_cds.data_vars)} — sos may not have downloaded yet. Re-run download_cds.py")
    print(f"  [CDS hist] tos={tos_var}, sos={sos_var}")
    hist_temp = _monthly_interp(ds_hist_cds[tos_var], lons, lats)
    hist_sal  = _monthly_interp(ds_hist_cds[sos_var], lons, lats)
    hist_uo   = _monthly_interp(
        _time_slice(_open_nc(str(RAW_DIR / "historical_esgf_uo"  / "uo_*.nc")),  *HIST_DATE)["uo"],  lons, lats)
    hist_vo   = _monthly_interp(
        _time_slice(_open_nc(str(RAW_DIR / "historical_esgf_vo"  / "vo_*.nc")),  *HIST_DATE)["vo"],  lons, lats)
    hist_chl  = _monthly_interp(
        _time_slice(_open_nc(str(RAW_DIR / "historical_esgf_chl" / "chl_*.nc")), *HIST_DATE)["chl"],
        lons, lats, is_log=True, unit_scale=1e6, var_label="chl historical (kg m-3 → µg L-1)")

    output_paths: dict[str, dict[str, dict[str, Path]]] = {}

    for ssp_key, ssp_label in ssps.items():
        output_paths[ssp_key] = {}
        ssp_dir = RAW_DIR / SSP_CDS_DIR[ssp_key]

        # Open SSP CDS dataset (may be single file with both tos+sos)
        ds_cds_ssp = _open_nc(str(ssp_dir / "*.nc"))
        print(f"  [CDS {ssp_key}] variables: {list(ds_cds_ssp.data_vars)}")
        tos_var = next((v for v in ds_cds_ssp.data_vars if "tos" in v.lower() or "temperature" in v.lower()), None)
        sos_var = next((v for v in ds_cds_ssp.data_vars if "sos" in v.lower() or "salinity" in v.lower()), None)
        if not tos_var or not sos_var:
            raise KeyError(f"Could not find tos/sos in {list(ds_cds_ssp.data_vars)}")
        ds_tos = ds_cds_ssp
        ds_sos = ds_cds_ssp
        ds_uo  = _open_nc(str(RAW_DIR / f"{ssp_key}_esgf_uo"  / "uo_*.nc"))
        ds_vo  = _open_nc(str(RAW_DIR / f"{ssp_key}_esgf_vo"  / "vo_*.nc"))
        ds_chl = _open_nc(str(RAW_DIR / f"{ssp_key}_esgf_chl" / "chl_*.nc"))

        for period_key, (p_start, p_end) in periods:
            print(f"\n=== {ssp_label} / {period_key} ({p_start[:4]}–{p_end[:4]}) ===")

            # Output paths — computed first so we can skip if all exist
            out_dir  = PROCESSED / ssp_key / period_key
            suffix   = f"{ssp_key}_{period_key}"
            out_temp = out_dir / f"grid_env_temp_{suffix}.gpkg"
            out_sal  = out_dir / f"grid_env_sal_{suffix}.gpkg"
            out_chla = out_dir / f"grid_env_chla_{suffix}.gpkg"
            out_flow = out_dir / f"grid_env_flow_{suffix}.gpkg"

            all_exist = all(p.exists() for p in [out_temp, out_sal, out_chla, out_flow])
            if all_exist:
                print(f"  [SKIP] All 4 gpkg files already exist — skipping {ssp_label} {period_key}")
                output_paths[ssp_key][period_key] = {
                    "temp": out_temp, "sal": out_sal,
                    "chla": out_chla, "flow": out_flow,
                }
                continue

            # Interpolate future monthly means
            fut_temp = _monthly_interp(_time_slice(ds_tos, p_start, p_end)[tos_var], lons, lats)
            fut_sal  = _monthly_interp(_time_slice(ds_sos, p_start, p_end)[sos_var], lons, lats)
            fut_uo   = _monthly_interp(_time_slice(ds_uo,  p_start, p_end)["uo"],  lons, lats)
            fut_vo   = _monthly_interp(_time_slice(ds_vo,  p_start, p_end)["vo"],  lons, lats)
            fut_chl  = _monthly_interp(
                _time_slice(ds_chl, p_start, p_end)["chl"],
                lons, lats, is_log=True, unit_scale=1e6,
                var_label=f"chl {ssp_key} {period_key} (kg m-3 → µg L-1)")

            # Deltas
            delta_temp = _compute_delta(fut_temp, hist_temp)
            delta_sal  = _compute_delta(fut_sal,  hist_sal)
            delta_uo   = _compute_delta(fut_uo,   hist_uo)
            delta_vo   = _compute_delta(fut_vo,   hist_vo)
            delta_chl  = _compute_delta(fut_chl,  hist_chl)

            # Write only files that are missing (partial re-run support)
            if not out_temp.exists():
                _apply_delta(GRID_ENV_TEMP_GPKG, "temp",   delta_temp, out_temp)
            else:
                print(f"  [SKIP] {out_temp.name} exists")
            if not out_sal.exists():
                _apply_delta(GRID_ENV_SAL_GPKG,  "sal",    delta_sal,  out_sal)
            else:
                print(f"  [SKIP] {out_sal.name} exists")
            if not out_chla.exists():
                _apply_delta(GRID_ENV_CHLA_GPKG, "lnchla", delta_chl,  out_chla)
            else:
                print(f"  [SKIP] {out_chla.name} exists")
            if not out_flow.exists():
                _build_flow_gpkg(GRID_ENV_FLOW_GPKG, delta_uo, delta_vo, out_flow)
            else:
                print(f"  [SKIP] {out_flow.name} exists")

            output_paths[ssp_key][period_key] = {
                "temp": out_temp, "sal": out_sal,
                "chla": out_chla, "flow": out_flow,
            }
            print(f"  [DONE] {ssp_label} {period_key}")

    print("\n=== All projections complete ===")
    return output_paths


# ── Scenario loader (Path 2: CMIP6 precomputed) ───────────────────────────────

def scenario_gpkg_paths(ssp: str, period: str) -> dict[str, Path]:
    """
    Return the 4 expected .gpkg paths for a given SSP × period combination.
    Does not check existence — call load_scenario_env() for a validated load.
    """
    if ssp not in SSP_SCENARIOS:
        raise ValueError(f"Unknown SSP '{ssp}'. Valid: {list(SSP_SCENARIOS)}")
    if period not in PERIODS:
        raise ValueError(f"Unknown period '{period}'. Valid: {list(PERIODS)}")
    out_dir = PROCESSED / ssp / period
    suffix  = f"{ssp}_{period}"
    return {
        "temp": out_dir / f"grid_env_temp_{suffix}.gpkg",
        "sal":  out_dir / f"grid_env_sal_{suffix}.gpkg",
        "chla": out_dir / f"grid_env_chla_{suffix}.gpkg",
        "flow": out_dir / f"grid_env_flow_{suffix}.gpkg",
    }


def load_scenario_env(
    grid: gpd.GeoDataFrame,
    ssp: str,
    period: str,
) -> gpd.GeoDataFrame:
    """
    Load precomputed CMIP6 scenario env columns onto the base grid.

    This is Path 2 (CMIP6 scenario) — completely separate from Path 1
    (interactive delta sliders via run_module1). The returned grid has
    identical column structure to what run_module1 returns and is passed
    directly to run_module2.

    Parameters
    ----------
    grid   : base grid GeoDataFrame (must have cell_id + geometry + exclusions)
    ssp    : 'ssp1_2_6' | 'ssp2_4_5' | 'ssp3_7_0' | 'ssp5_8_5'
    period : 'mid' | 'end'

    Returns
    -------
    GeoDataFrame with all env columns replaced by scenario values,
    all other columns (bathymetry, exclusions, etc.) unchanged.

    Raises
    ------
    FileNotFoundError if scenario gpkg files are missing — run pipeline
    with --precompute-scenarios first.
    """
    paths = scenario_gpkg_paths(ssp, period)

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Scenario gpkg files not found for {ssp} / {period}:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nRun: python pipeline.py --precompute-scenarios"
        )

    label = f"{SSP_SCENARIOS[ssp]} / {period}"
    print(f"\n=== load_scenario_env: {label} ===")

    result = grid.copy()

    # Env column prefixes each file contributes
    var_prefixes = {
        "temp": ["temp_mean_", "temp_sd_"],
        "sal":  ["sal_mean_",  "sal_sd_"],
        "chla": ["lnchla_mean_", "lnchla_sd_"],
        "flow": ["vh_", "uo_", "vo_"],
    }

    for var, path in paths.items():
        cached = gpd.read_file(path)
        prefixes = var_prefixes[var]
        cols = [c for c in cached.columns
                if any(c.startswith(p) for p in prefixes)]
        for col in cols:
            result[col] = cached[col].values
        print(f"  [LOAD] {var:6s} → {path.name}  ({len(cols)} columns)")

    # Sanity check: flag env_source so downstream code can log it
    result.attrs["env_source"] = f"CMIP6:{ssp}:{period}"
    print(f"  [OK] env_source = CMIP6:{ssp}:{period}")
    print(f"  [OK] Grid ready for Module 2 ({len(result):,} cells)")
    return result


if __name__ == "__main__":
    print("Loading base grid...")
    grid = gpd.read_file(GRID_GPKG)
    print(f"Grid loaded: {len(grid):,} cells")
    run_climate_projections(grid, period="both")