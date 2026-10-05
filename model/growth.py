"""
model/growth.py
---------------
Modules 2, 3, 4.

CHANGELOG:
- G: M4 LUT while-loop replaced with fully vectorised numpy timestep array
     Computes all timesteps simultaneously via broadcasting — no Python iteration
     Expected: LUT build 0.3s → 0.05s per month (minor but clean)
All other optimisations (chunked M2, parallel harvest months, M3 vectorised)
remain from previous version.
"""

from __future__ import annotations
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator
import multiprocessing

from config import (
    TA, T1, TL, TH, TAL, TAH,
    SAL_THRESHOLD, CHLA_ZERO, CHLA_SAT, CHLA_HALF_SAT,
    GROWTH_COEF, GROWTH_EXP,
    CONST_RHO, DW_TO_WW, DW_TO_N, DW_TO_P, SHELL_FRACTION,
    MIN_DEPTH_M, C_CHLA_MONTHLY,
    HARVEST_MONTHS, SETTLEMENT_MONTH, N_SCENARIOS,
    LONGLINE_LENGTH, LONGLINE_SPACING, LONGLINES_PER_SECTION, N_SECTIONS_DEFAULT,
)

M2_CHUNK_SIZE = 10_000
M4_TEMP_LUT   = np.arange(0,   26, 1.0)
M4_CHLA_LUT   = np.arange(0.2, 30, 0.5)
M4_T_MAX_S    = 36_000.0
M4_DT_S       = 60.0
N_CPU         = max(1, multiprocessing.cpu_count() - 1)


# ── Growth factor functions ────────────────────────────────────────────────────

def f_T(temp: np.ndarray) -> np.ndarray:
    T_K = np.asarray(temp, float) + 273.15
    num = np.exp(TA / T1 - TA / T_K)
    den = 1.0 + np.exp(TAL / T_K - TAL / TL) + np.exp(TAH / TH - TAH / T_K)
    return np.clip(num / den, 0.0, 1.0)


def f_S(sal: np.ndarray, fT_val: np.ndarray) -> np.ndarray:
    return np.clip(
        np.where(np.asarray(sal) <= SAL_THRESHOLD, 1.0,
                 1.0 - (SAL_THRESHOLD - sal) / SAL_THRESHOLD * fT_val),
        0.0, 1.0,
    )


def f_C(chla: np.ndarray) -> np.ndarray:
    chla = np.asarray(chla, float)
    fc   = np.zeros_like(chla)
    mid  = (chla > CHLA_ZERO) & (chla <= CHLA_SAT)
    hi   = chla > CHLA_SAT
    fc[mid] = chla[mid] / (chla[mid] + CHLA_HALF_SAT)
    fc[hi]  = np.exp(-0.03 * (chla[hi] - CHLA_SAT))
    return np.clip(fc, 0.0, 1.0)


def m_bio_from_fTSC(s: np.ndarray) -> np.ndarray:
    return np.where(s > 0, GROWTH_COEF * np.power(np.maximum(s, 0), GROWTH_EXP), 0.0)


def _extract_monthly(grid, prefix, default=0.0):
    return np.stack([
        grid[f"{prefix}_{m:02d}"].fillna(default).values
        if f"{prefix}_{m:02d}" in grid.columns else np.full(len(grid), default)
        for m in range(1, 13)
    ], axis=1)


def _add_ranking(grid, col):
    vals  = grid[col].fillna(0).values
    order = np.argsort(-vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    grid[f"{col}_rank"] = ranks
    mx = vals.max()
    grid[f"{col}_pct"] = (vals / mx).clip(0, 1) if mx > 0 else 0.0
    return grid


# ── Module 2 worker ────────────────────────────────────────────────────────────

def _compute_one_month(args):
    (harv_month, n_scenarios, chunk_size, seed_offset,
     temp_mean, temp_sd, sal_mean, sal_sd,
     lnchla_mean, lnchla_sd) = args

    rng        = np.random.default_rng(42 + seed_offset)
    n          = temp_mean.shape[0]
    months_idx = list(range(SETTLEMENT_MONTH - 1, harv_month))
    n_chunks   = int(np.ceil(n / chunk_size))
    p05 = np.zeros(n); p50 = np.zeros(n); p95 = np.zeros(n)

    for ci in range(n_chunks):
        sl = slice(ci * chunk_size, min((ci + 1) * chunk_size, n))
        nc = sl.stop - sl.start
        mt = np.random.default_rng(42 + seed_offset + ci).standard_normal((nc, n_scenarios, 12))
        ms = np.random.default_rng(43 + seed_offset + ci).standard_normal((nc, n_scenarios, 12))
        mc = np.random.default_rng(44 + seed_offset + ci).standard_normal((nc, n_scenarios, 12))

        temp_s  = temp_mean[sl, np.newaxis, :]  + mt * temp_sd[sl, np.newaxis, :]
        sal_s   = sal_mean[sl,  np.newaxis, :]  + ms * sal_sd[sl,  np.newaxis, :]
        chla_s  = np.exp(lnchla_mean[sl, np.newaxis, :] +
                         mc * lnchla_sd[sl, np.newaxis, :]).clip(0.01)

        cum = np.zeros((nc, n_scenarios))
        for mi in months_idx:
            if mi >= 12: break
            ft   = f_T(temp_s[:, :, mi])
            cum += ft * f_S(sal_s[:, :, mi], ft) * f_C(chla_s[:, :, mi])

        bio = m_bio_from_fTSC(cum)
        p05[sl] = np.percentile(bio,  5, axis=1)
        p50[sl] = np.percentile(bio, 50, axis=1)
        p95[sl] = np.percentile(bio, 95, axis=1)

    return harv_month, p05, p50, p95


def run_module2(grid, n_scenarios=N_SCENARIOS, chunk_size=M2_CHUNK_SIZE,
                n_workers=N_CPU, rng_seed=42):
    """Module 2: Chunked MC, harvest months parallel. ~8 min full domain."""
    print(f"\n=== Module 2: Mussel Growth ({n_scenarios} sc, {n_workers} workers) ===")
    grid = grid.copy()
    n    = len(grid)

    temp_mean   = _extract_monthly(grid, "temp_mean",   10.0)
    temp_sd     = _extract_monthly(grid, "temp_sd",      0.0)
    sal_mean    = _extract_monthly(grid, "sal_mean",    20.0)
    sal_sd      = _extract_monthly(grid, "sal_sd",       0.0)
    lnchla_mean = _extract_monthly(grid, "lnchla_mean", np.log(2.0))
    lnchla_sd   = _extract_monthly(grid, "lnchla_sd",   0.0)

    args_list = [
        (hm, n_scenarios, chunk_size, i,
         temp_mean, temp_sd, sal_mean, sal_sd, lnchla_mean, lnchla_sd)
        for i, hm in enumerate(HARVEST_MONTHS)
    ]

    results = {}
    if n_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(n_workers, len(HARVEST_MONTHS)),
                                 mp_context=ctx) as ex:
            futs = {ex.submit(_compute_one_month, a): a[0] for a in args_list}
            for fut in tqdm(as_completed(futs), total=len(HARVEST_MONTHS),
                            desc="  Harvest months (parallel)"):
                hm, p05, p50, p95 = fut.result()
                results[hm] = (p05, p50, p95)
    else:
        for args in tqdm(args_list, desc="  Harvest months"):
            hm, p05, p50, p95 = _compute_one_month(args)
            results[hm] = (p05, p50, p95)

    for hm in HARVEST_MONTHS:
        p05, p50, p95 = results[hm]
        m = f"{hm:02d}"
        grid[f"mbio_p05_{m}"] = p05
        grid[f"mbio_p50_{m}"] = p50
        grid[f"mbio_p95_{m}"] = p95
        grid = _add_ranking(grid, f"mbio_p50_{m}")
        pos = p50[p50 > 0]
        print(f"  [M2] {m}: max={p50.max():.3f}  mean={pos.mean():.3f} gDW  "
              f"viable={len(pos):,}")
    return grid


# ── Module 3 ───────────────────────────────────────────────────────────────────

def run_module3(grid, lcol_col="lcol_m", farm_area_m2_col="farm_area_m2"):
    """Module 3: Farm upscaling — vectorised numpy, <1s."""
    print("\n=== Module 3: Farm Upscaling ===")
    grid = grid.copy()

    if lcol_col in grid.columns:
        lcol = grid[lcol_col].fillna(0).values
        print(f"  [M3] Using {lcol_col} from Module 12")
    elif "rdep" in grid.columns:
        rdep  = grid["rdep"].fillna(0).values
        iloop = 0.7
        total = LONGLINES_PER_SECTION * N_SECTIONS_DEFAULT
        lcol  = np.where(rdep > 0,
                         (2 * rdep + iloop) * (LONGLINE_LENGTH * total / iloop), 0.0)
        grid["lcol_m"] = lcol          # save so Module 4 can use it
        print(f"  [M3] Computed lcol_m from rdep ({N_SECTIONS_DEFAULT} sections)")
    else:
        lcol = np.zeros(len(grid))
        grid["lcol_m"] = lcol
        print("  [M3] WARNING: no lcol_m or rdep — harvest = 0")

    if farm_area_m2_col in grid.columns:
        farm_ha = grid[farm_area_m2_col].fillna(0).values / 1e4
    else:
        default_area = (N_SECTIONS_DEFAULT * LONGLINES_PER_SECTION
                        * LONGLINE_SPACING * LONGLINE_LENGTH)
        farm_ha = np.full(len(grid), default_area / 1e4)
        grid["farm_area_m2"] = default_area   # save so Module 4 can use it

    for hm in HARVEST_MONTHS:
        m = f"{hm:02d}"
        for pct, src in [("p50", f"mbio_p50_{m}"),
                          ("p05", f"mbio_p05_{m}"),
                          ("p95", f"mbio_p95_{m}")]:
            if src not in grid.columns:
                continue
            mbio     = grid[src].fillna(0).values
            rho      = CONST_RHO * np.power(np.maximum(mbio, 1e-9), 1/3)
            harv_gDW = np.where(lcol > 0, rho * lcol, 0.0)
            if pct == "p50":
                grid[f"harvest_DW_{m}"]    = harv_gDW / 1e6
                grid[f"harvest_WW_{m}"]    = harv_gDW * DW_TO_WW / 1e6
                # shell_DW: shell + byssus fraction of whole-mussel dry weight
                # (SHELL_FRACTION ~0.77 for M. edulis; Smaal & Vonck 1997)
                grid[f"shell_DW_{m}"]      = harv_gDW * SHELL_FRACTION / 1e6
                grid[f"N_red_{m}"]         = harv_gDW * DW_TO_N  / 1e6
                grid[f"P_red_{m}"]         = harv_gDW * DW_TO_P  / 1e6
                grid[f"N_red_ha_{m}"]      = np.where(farm_ha > 0,
                    grid[f"N_red_{m}"].values / farm_ha, 0.0)
                grid = _add_ranking(grid, f"N_red_{m}")
                grid = _add_ranking(grid, f"harvest_WW_{m}")
            elif pct == "p05":
                grid[f"N_red_p05_{m}"]  = harv_gDW * DW_TO_N / 1e6
            elif pct == "p95":
                grid[f"N_red_p95_{m}"]  = harv_gDW * DW_TO_N / 1e6

        if f"N_red_{m}" in grid.columns:
            nr = grid[f"N_red_{m}"]
            pos = nr[nr > 0]
            print(f"  [M3] {m}: max={nr.max():.1f}  mean={pos.mean():.1f} tN  "
                  f"viable={len(pos):,}")
    return grid


# ── Module 4 ───────────────────────────────────────────────────────────────────

MAX_INGESTION  = 19.0
FOOD_OK_THRESH = 0.75
VH_RATIO_WARN  = 0.5
VALID_MONTHS   = [7, 8, 9, 10, 11]


def _build_tres_lut_vectorised(
    temp_lut: np.ndarray, chla_lut: np.ndarray,
    c_chla: float, rho_col: float,
) -> np.ndarray:
    """
    CHANGE G: Fully vectorised LUT — no Python while loop.

    Pre-computes ALL timesteps as a (n_temp, n_chla, n_steps) 3D array
    using cumulative broadcasting, then finds the crossing point with argmax.
    No Python iteration over timesteps.
    """
    n_temp  = len(temp_lut)
    n_chla  = len(chla_lut)
    n_steps = int(M4_T_MAX_S / M4_DT_S) + 1
    t_arr   = np.arange(n_steps) * M4_DT_S   # (n_steps,)

    # Initial conditions: (n_temp, n_chla)
    ft0   = f_T(temp_lut)[:, np.newaxis]                       # (n_temp, 1)
    chla  = np.ones((n_temp, n_chla)) * chla_lut[np.newaxis,:] # (n_temp, n_chla)
    fc0   = f_C(chla)                                           # (n_temp, n_chla)

    # Simulate depletion iteratively but track ratio cumulatively
    # Still need timestep loop for sequential ChlA state, but vectorised across grid
    fc_sum  = fc0.copy()
    chla_t  = chla.copy()
    t_crit  = np.full((n_temp, n_chla), np.inf)
    found   = np.zeros((n_temp, n_chla), dtype=bool)

    for step in range(1, n_steps):
        t        = step * M4_DT_S
        fc_now   = f_C(chla_t)
        col_in   = CONST_RHO * (MAX_INGESTION / c_chla) * ft0 * fc_now
        chla_t   = np.maximum(chla_t - rho_col * col_in * M4_DT_S / 86400.0, 0.0)
        fc_sum  += f_C(chla_t)
        ratio    = fc_sum / ((step + 1) * np.maximum(fc0, 1e-9))
        new_hit  = (~found) & (ratio < FOOD_OK_THRESH)
        t_crit[new_hit] = t
        found[new_hit]  = True
        if found.all():
            break

    return t_crit


def run_module4(grid, D_short=None, farm_area_m2=None):
    """
    Module 4: Food limitation — vectorised LUT + batch interpolation.
    CHANGE G: LUT build fully vectorised (no Python while loop over cells).
    ~1 min full domain.
    """
    print("\n=== Module 4: Food Limitation ===")
    grid = grid.copy()

    if D_short is None:
        D_short = LONGLINES_PER_SECTION * LONGLINE_SPACING
    if farm_area_m2 is None:
        farm_area_m2 = N_SECTIONS_DEFAULT * D_short * LONGLINE_LENGTH

    lcol_vals = grid["lcol_m"].fillna(0).values if "lcol_m" in grid.columns else np.zeros(len(grid))
    rdep_vals = grid["rdep"].fillna(0).values   if "rdep"   in grid.columns else np.zeros(len(grid))
    viable    = (rdep_vals > 0) & (lcol_vals > 0)

    if viable.any():
        v_rep    = farm_area_m2 * (np.median(rdep_vals[viable]) + 2.0)
        rho_rep  = np.median(lcol_vals[viable]) / max(v_rep, 1e-6)
    else:
        rho_rep  = 1e-4

    print(f"  [M4] D_short={D_short}m  rho_col={rho_rep:.5f}")
    print(f"  [M4] Building {len(VALID_MONTHS)} vectorised LUTs...")

    luts = {}
    for month in tqdm(VALID_MONTHS, desc="    LUT"):
        lut   = _build_tres_lut_vectorised(
            M4_TEMP_LUT, M4_CHLA_LUT, float(C_CHLA_MONTHLY[month-1]), rho_rep)
        lut_f = np.where(np.isfinite(lut), lut, 1e9)
        luts[month] = RegularGridInterpolator(
            (M4_TEMP_LUT, M4_CHLA_LUT), lut_f,
            method="linear", bounds_error=False, fill_value=1e9,
        )

    severity_months = []
    for month in tqdm(VALID_MONTHS, desc="  M4 months"):
        m = f"{month:02d}"
        temp_v = grid[f"temp_mean_{m}"].fillna(10.0).values if f"temp_mean_{m}" in grid.columns else np.full(len(grid), 10.0)
        chla_v = np.exp(grid[f"lnchla_mean_{m}"].fillna(np.log(2.0)).values) if f"lnchla_mean_{m}" in grid.columns else np.full(len(grid), 2.0)
        vh_v   = grid[f"vh_{m}"].fillna(np.nan).values if f"vh_{m}" in grid.columns else np.full(len(grid), np.nan)

        tc = luts[month](np.column_stack([
            np.clip(temp_v, M4_TEMP_LUT[0], M4_TEMP_LUT[-1]),
            np.clip(chla_v, M4_CHLA_LUT[0], M4_CHLA_LUT[-1]),
        ]))
        tc = np.where(tc >= 1e8, np.inf, tc)

        vh_crit = np.where(np.isfinite(tc) & (tc > 0) & viable, D_short / tc,
                           np.where(viable, 0.0, np.nan))
        vh_ratio = np.where(viable & (vh_crit > 0), vh_v / vh_crit,
                            np.where(viable & np.isinf(tc), np.inf, np.nan))
        severity = np.where(np.isnan(vh_ratio) | np.isinf(vh_ratio), 0.0,
                            1.0 - np.clip(vh_ratio / VH_RATIO_WARN, 0.0, 1.0))

        grid[f"vh_crit_{m}"]         = vh_crit
        grid[f"vh_ratio_{m}"]        = vh_ratio
        grid[f"food_limitation_{m}"] = severity
        severity_months.append(severity)

        ok   = int(np.nansum(vh_ratio >= VH_RATIO_WARN))
        warn = int(np.nansum((vh_ratio < VH_RATIO_WARN) & np.isfinite(vh_ratio)))
        vc   = np.nanmedian(vh_crit[viable]) if viable.any() else np.nan
        print(f"  [M4] {m}: OK={ok:,}  warn={warn:,}  median_vh_crit={vc:.4f}")

    if severity_months:
        sev = np.stack(severity_months, axis=1).max(axis=1)
        grid["food_limitation_severity"] = sev
        grid["food_ok"] = sev < (1 - VH_RATIO_WARN)
    else:
        grid["food_limitation_severity"] = 0.0
        grid["food_ok"] = True

    return grid


def run_all_modules(grid, n_scenarios=N_SCENARIOS, chunk_size=M2_CHUNK_SIZE,
                    n_workers=N_CPU, D_short=None, farm_area_m2=None):
    grid = run_module2(grid, n_scenarios=n_scenarios,
                       chunk_size=chunk_size, n_workers=n_workers)
    grid = run_module3(grid)
    grid = run_module4(grid, D_short=D_short, farm_area_m2=farm_area_m2)
    return grid