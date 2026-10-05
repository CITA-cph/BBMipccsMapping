"""
pipeline.py
-----------
Full MYTIGATE preprocessing pipeline.

Mirrors exactly the validated notebook workflow:
  Step 1: Grid
  Step 2: Bathymetry
  Step 3: Module 1 (CMEMS env)
  Step 4: Module 11 (exclusions + conflicts)
  Step 5: Module 2 (mussel growth)
  Step 6: Module 3 (farm upscaling, rdep from bathymetry)
  Step 7: Module 4 (food limitation)
  Step 8: Save baseline

Optional (--precompute-scenarios only):
  Step 9:  Download CDS data (tos, sos) — skips if already present
  Step 10: Download ESGF data (uo, vo, chl) — skips if already present
  Step 11: Build CMIP6 scenario gpkg files (all SSP × period) — skips per-file

Usage:
    python pipeline.py                            # full baseline run
    python pipeline.py --n-scenarios 50           # fast debug (~5 min)
    python pipeline.py --skip-env                 # skip CMEMS if already done
    python pipeline.py --max-rdep 4 --iloop 1.0
    python pipeline.py --scenario --delta-temp 1.8 --delta-sal -0.5 --chla-mult 1.1
    python pipeline.py --n-workers 4              # parallel workers
    python pipeline.py --precompute-scenarios     # run Steps 9-11 after baseline
    python pipeline.py --precompute-scenarios --skip-baseline  # Steps 9-11 only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import geopandas as gpd

from config import (
    PROCESSED, GRID_GPKG, BASELINE_GPKG,
    BATHY_PATH, N_SCENARIOS, PRIMARY_HARVEST,
)
from utils.grid import load_or_build_grid
from utils.raster_sampler import run_module1, sample_bathymetry
from model.module11_exclusions import run_module11
from model.growth import run_module2, run_module3, run_module4, N_CPU


def step(n: int, label: str) -> None:
    print(f"\n{'='*60}\n  STEP {n}: {label}\n{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="MYTIGATE pipeline")
    p.add_argument("--max-rdep",    type=float, default=2.0,
                   help="Max collector depth range [m] (default 2)")
    p.add_argument("--iloop",       type=float, default=0.7,
                   help="Loop interval [m] (default 0.7)")
    p.add_argument("--n-scenarios", type=int,   default=N_SCENARIOS,
                   help="Monte Carlo scenarios (default 500, use 50 for debug)")
    p.add_argument("--n-workers",   type=int,   default=N_CPU,
                   help="Parallel workers for M1 and M2")
    p.add_argument("--skip-env",    action="store_true",
                   help="Skip CMEMS sampling if env columns already present")
    p.add_argument("--scenario",    action="store_true",
                   help="Apply scenario deltas to env fields")
    p.add_argument("--delta-temp",  type=float, default=0.0,
                   help="Temperature offset [°C]")
    p.add_argument("--delta-sal",   type=float, default=0.0,
                   help="Salinity offset [psu]")
    p.add_argument("--chla-mult",   type=float, default=1.0,
                   help="ChlA multiplier (1.0 = no change)")
    p.add_argument("--precompute-scenarios", action="store_true",
                   help="After baseline: download CMIP6 data and build all "
                        "SSP × period scenario gpkg files (Steps 9-11). "
                        "Each file is skipped if already present on disk.")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip Steps 1-8 (baseline run). Use with "
                        "--precompute-scenarios to run Steps 9-11 only, "
                        "e.g. after baseline.gpkg already exists.")
    return p.parse_args()


def run_pipeline(
    max_rdep: float          = 2.0,
    iloop: float             = 0.7,
    n_scenarios: int         = N_SCENARIOS,
    n_workers: int           = N_CPU,
    skip_env: bool           = False,
    scenario: bool           = False,
    delta_temp: float        = 0.0,
    delta_sal: float         = 0.0,
    chla_mult: float         = 1.0,
    precompute_scenarios: bool = False,
    skip_baseline: bool      = False,
) -> gpd.GeoDataFrame | None:
    """
    Run full pipeline. Returns complete GeoDataFrame with all columns.
    Saved to BASELINE_GPKG — the app loads this file at startup.

    precompute_scenarios : if True, run Steps 9-11 after baseline to build
                           all CMIP6 scenario gpkg files. Each file is skipped
                           individually if already present on disk.
    skip_baseline        : if True, skip Steps 1-8 entirely. Requires
                           BASELINE_GPKG and GRID_GPKG to already exist.
                           Use with precompute_scenarios=True.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if skip_baseline:
        if not BASELINE_GPKG.exists():
            raise FileNotFoundError(
                f"--skip-baseline requires baseline.gpkg to exist: {BASELINE_GPKG}\n"
                "Run without --skip-baseline first."
            )
        print(f"\n  --skip-baseline: loading existing baseline → {BASELINE_GPKG}")
        grid = gpd.read_file(BASELINE_GPKG)
        print(f"  Loaded {len(grid):,} cells, {len(grid.columns)} columns")
        if precompute_scenarios:
            _run_precompute_scenarios(grid)
        return grid

    # ── STEP 1: Grid ──────────────────────────────────────────────────────────
    step(1, "Build / load 1km UTM32N grid")
    grid = load_or_build_grid()
    print(f"  Cells: {len(grid):,}  CRS: {grid.crs}")

    # ── STEP 2: Bathymetry ────────────────────────────────────────────────────
    step(2, "Bathymetry (EMODnet)")
    if "bathymetry_m" not in grid.columns:
        grid = sample_bathymetry(grid, BATHY_PATH)
    else:
        print("  Already present — skipping")
    viable = (grid["bathymetry_m"].fillna(0) >= 4).sum()
    print(f"  Valid: {grid['bathymetry_m'].notna().sum():,}  Viable (≥4m): {viable:,}")

    # ── STEP 3: Module 1 — Environmental fields ───────────────────────────────
    step(3, "Module 1 — CMEMS Temp / Sal / ChlA / Flow (Baltic + NWS)")
    if skip_env and "temp_mean_01" in grid.columns:
        print("  Env columns present — skipping (--skip-env)")
    else:
        grid = run_module1(
            grid,
            delta_temp     = delta_temp if scenario else 0.0,
            delta_sal      = delta_sal  if scenario else 0.0,
            chla_mult      = chla_mult  if scenario else 1.0,
            n_workers      = n_workers,
            force_resample = False,
        )

    # ── STEP 4: Module 11 — Exclusions and conflict layers ───────────────────
    step(4, "Module 11 — Exclusion masks and conflict gradients")
    grid = run_module11(grid)
    print(f"  Excluded: {grid['excluded'].sum():,} ({100*grid['excluded'].mean():.1f}%)")
    print(f"  Available: {(~grid['excluded']).sum():,}")

    # Save base grid (env + exclusions) — used by app for fast scenario recompute
    grid.to_file(GRID_GPKG, driver="GPKG")
    print(f"\n  Base grid saved → {GRID_GPKG}")

    # ── STEP 5: Module 2 — Mussel growth ─────────────────────────────────────
    step(5, f"Module 2 — Mussel growth ({n_scenarios} MC scenarios)")
    grid = run_module2(grid, n_scenarios=n_scenarios, n_workers=n_workers)

    # ── STEP 6: Module 3 — Farm upscaling ────────────────────────────────────
    step(6, f"Module 3 — Farm upscaling (max_rdep={max_rdep}m, iloop={iloop}m)")

    # Compute rdep from bathymetry (baseline mode — no user polygon yet)
    bathy = grid["bathymetry_m"].fillna(0).values
    rdep  = np.where(bathy - 2 <= max_rdep, np.maximum(bathy - 2, 0), max_rdep)
    rdep  = np.where(bathy >= 4, rdep, 0.0)
    grid["rdep"] = rdep

    grid = run_module3(grid)

    # ── STEP 7: Module 4 — Food limitation ───────────────────────────────────
    step(7, "Module 4 — Food limitation (informational)")
    grid = run_module4(grid)

    # ── STEP 8: Save ─────────────────────────────────────────────────────────
    step(8, "Save")
    grid.to_file(BASELINE_GPKG, driver="GPKG")
    print(f"  Saved → {BASELINE_GPKG}")
    print(f"  Rows: {len(grid):,}  Columns: {len(grid.columns)}")

    # Summary stats
    m = f"{PRIMARY_HARVEST:02d}"
    if f"N_red_{m}" in grid.columns:
        nr = grid[f"N_red_{m}"]
        pos = nr[nr > 0]
        print(f"\n  N-reduction November:")
        print(f"    Max:    {nr.max():.1f} tN/farm")
        print(f"    Top 10%: ≥ {nr.quantile(0.9):.1f} tN/farm")
        print(f"    Mean:   {pos.mean():.1f} tN/farm (non-zero cells)")
        print(f"    Viable: {len(pos):,} cells")

    if precompute_scenarios:
        _run_precompute_scenarios(grid)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Start app:  python app/app.py")
    print(f"  Open:       http://localhost:8050")
    print(f"{'='*60}\n")

    return grid


def _run_precompute_scenarios(grid: gpd.GeoDataFrame) -> None:
    """
    Steps 9-11: Download CMIP6 data and build all scenario gpkg files.
    Each step is individually skip-aware — safe to re-run after interruption.
    """
    from utils.download_cds import run_download_cds
    from utils.download_esgf import run_download_esgf
    from utils.climate_projections import run_climate_projections

    step(9, "Download CDS data (tos + sos) — skips if already present")
    run_download_cds()

    step(10, "Download ESGF data (uo + vo + chl) — skips if already present")
    run_download_esgf()

    step(11, "Build CMIP6 scenario gpkg files (all SSP × period) — skips per-file")
    paths = run_climate_projections(grid, period="both")

    print(f"\n  Scenario files produced:")
    for ssp_key, periods in paths.items():
        for period_key, var_paths in periods.items():
            for var, p in var_paths.items():
                status = "OK" if p.exists() else "MISSING"
                print(f"    [{status}] {p.name}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        max_rdep             = args.max_rdep,
        iloop                = args.iloop,
        n_scenarios          = args.n_scenarios,
        n_workers            = args.n_workers,
        skip_env             = args.skip_env,
        scenario             = args.scenario,
        delta_temp           = args.delta_temp,
        delta_sal            = args.delta_sal,
        chla_mult            = args.chla_mult,
        precompute_scenarios = args.precompute_scenarios,
        skip_baseline        = args.skip_baseline,
    )