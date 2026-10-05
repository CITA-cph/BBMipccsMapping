"""
config_scenario.py
------------------
Scenario configuration for testing CMIP6 climate projections.

Imports everything from config.py and overrides the 4 env .gpkg paths
to point at a specific SSP × period combination. Import this instead of
config.py when running scenario notebooks or tests.

Usage:
    from config_scenario import *               # uses SSP2-4.5 end-of-century
    from config_scenario import scenario_gpkg  # path helper

    # Override SSP/period at runtime:
    import config_scenario as cfg
    cfg.ACTIVE_SSP    = "ssp3_7_0"
    cfg.ACTIVE_PERIOD = "mid"
    paths = cfg.scenario_gpkg(cfg.ACTIVE_SSP, cfg.ACTIVE_PERIOD)
"""

from config import *  # noqa: F401,F403 — re-export all baseline constants

from pathlib import Path

# ── Active scenario (edit before running notebook) ────────────────────────────
ACTIVE_SSP    = "ssp2_4_5"   # ssp1_2_6 | ssp2_4_5 | ssp3_7_0 | ssp5_8_5
ACTIVE_PERIOD = "end"        # mid (2041-2060) | end (2071-2100)

# ── SSP metadata (mirrored from climate_projections.py) ───────────────────────
SSP_SCENARIOS = {
    "ssp1_2_6": "SSP1-2.6  (~1.8°C)",
    "ssp2_4_5": "SSP2-4.5  (~2.7°C)",
    "ssp3_7_0": "SSP3-7.0  (~3.6°C)",
    "ssp5_8_5": "SSP5-8.5  (~4.4°C)",
}

PERIODS = {
    "mid": ("2041-01-01", "2060-12-31"),
    "end": ("2071-01-01", "2100-12-31"),
}


def scenario_gpkg(ssp: str, period: str, var: str | None = None) -> Path | dict:
    """
    Return the path(s) for precomputed scenario gpkg files.

    Parameters
    ----------
    ssp    : 'ssp1_2_6' | 'ssp2_4_5' | 'ssp3_7_0' | 'ssp5_8_5'
    period : 'mid' | 'end'
    var    : 'temp' | 'sal' | 'chla' | 'flow' | None (returns all 4 as dict)

    Returns
    -------
    Path if var is specified, else dict[var → Path]

    Raises
    ------
    FileNotFoundError if the requested file does not exist on disk.
    """
    if ssp not in SSP_SCENARIOS:
        raise ValueError(f"Unknown SSP '{ssp}'. Valid: {list(SSP_SCENARIOS)}")
    if period not in PERIODS:
        raise ValueError(f"Unknown period '{period}'. Valid: {list(PERIODS)}")

    out_dir = PROCESSED / ssp / period  # noqa: F405 — PROCESSED from config.*
    suffix  = f"{ssp}_{period}"

    all_paths = {
        "temp": out_dir / f"grid_env_temp_{suffix}.gpkg",
        "sal":  out_dir / f"grid_env_sal_{suffix}.gpkg",
        "chla": out_dir / f"grid_env_chla_{suffix}.gpkg",
        "flow": out_dir / f"grid_env_flow_{suffix}.gpkg",
    }

    if var is not None:
        if var not in all_paths:
            raise ValueError(f"Unknown var '{var}'. Valid: {list(all_paths)}")
        p = all_paths[var]
        if not p.exists():
            raise FileNotFoundError(
                f"Scenario file not found: {p}\n"
                "Run: python pipeline.py --precompute-scenarios"
            )
        return p

    missing = [str(p) for p in all_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Scenario files missing for {ssp} / {period}:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nRun: python pipeline.py --precompute-scenarios"
        )
    return all_paths


# ── Override baseline env paths with active scenario ─────────────────────────
# These replace the GRID_ENV_*_GPKG values imported from config.py.
# Module 2 loads these paths — it does not know it is receiving scenario data.
def _set_active_scenario(ssp: str, period: str) -> None:
    """Point the module-level GRID_ENV_* paths at the active scenario."""
    global GRID_ENV_TEMP_GPKG, GRID_ENV_SAL_GPKG  # noqa: F405
    global GRID_ENV_CHLA_GPKG, GRID_ENV_FLOW_GPKG  # noqa: F405
    paths = scenario_gpkg(ssp, period)
    GRID_ENV_TEMP_GPKG = paths["temp"]   # noqa: F405
    GRID_ENV_SAL_GPKG  = paths["sal"]    # noqa: F405
    GRID_ENV_CHLA_GPKG = paths["chla"]   # noqa: F405
    GRID_ENV_FLOW_GPKG = paths["flow"]   # noqa: F405
    print(f"[config_scenario] Active: {SSP_SCENARIOS[ssp]} / {period}")
    for var, p in paths.items():
        print(f"  {var:6s} → {p.name}")
