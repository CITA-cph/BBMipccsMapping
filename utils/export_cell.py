# ── Export Cell — paste into scenario_test.ipynb ──────────────────────────────
# Exports two CSVs per active grid (baseline, and scenario if active):
#   mytigate_env_{label}.csv     — M1 environmental fields (all 12 months)
#   mytigate_results_{label}.csv — M2/M3/M4 outputs (harvest months only)
# Files saved next to the notebook. Join to Rhino grid on cell_id.

from pathlib import Path
import pandas as pd
import numpy as np

EXPORT_DIR = Path('.')   # change to a subfolder if preferred e.g. Path('exports')
EXPORT_DIR.mkdir(exist_ok=True)

HARVEST_MONTHS_STR = [f'{h:02d}' for h in [8, 9, 10, 11, 12]]
ALL_MONTHS_STR     = [f'{i:02d}' for i in range(1, 13)]

def _env_cols(grid):
    """All M1 env columns present in grid."""
    cols = ['cell_id', 'lon', 'lat', 'bathymetry_m']
    for prefix in ['temp_mean_', 'sal_mean_', 'lnchla_mean_', 'vh_']:
        cols += [f'{prefix}{mk}' for mk in ALL_MONTHS_STR
                 if f'{prefix}{mk}' in grid.columns]
    return [c for c in cols if c in grid.columns]

def _result_cols(grid):
    """All M2/M3/M4 result columns present in grid."""
    cols = ['cell_id', 'lon', 'lat', 'excluded', 'conflict_score', 'rdep']
    for prefix in [
        'harvest_WW_', 'harvest_DW_', 'shell_DW_',
        'N_red_', 'N_red_ha_',
        'P_red_',
        'mbio_p50_',
        'vh_ratio_', 'food_limitation_',
    ]:
        cols += [f'{prefix}{mk}' for mk in HARVEST_MONTHS_STR
                 if f'{prefix}{mk}' in grid.columns]
    return [c for c in cols if c in grid.columns]

def export_grid(grid, label):
    """Export env + results CSVs for one grid. Returns (env_path, results_path)."""
    # Strip geometry if GeoDataFrame
    if hasattr(grid, 'drop'):
        df = pd.DataFrame(grid.drop(columns='geometry', errors='ignore'))
    else:
        df = pd.DataFrame(grid)

    env_path     = EXPORT_DIR / f'mytigate_env_{label}.csv'
    results_path = EXPORT_DIR / f'mytigate_results_{label}.csv'

    env_df     = df[_env_cols(grid)]
    results_df = df[_result_cols(grid)]

    env_df.to_csv(env_path,     index=False, float_format='%.6f')
    results_df.to_csv(results_path, index=False, float_format='%.4f')

    print(f'\n[{label}]')
    print(f'  env     → {env_path.name}  '
          f'({len(env_df):,} rows × {len(env_df.columns)} cols, '
          f'{env_path.stat().st_size/1024:.0f} KB)')
    print(f'  results → {results_path.name}  '
          f'({len(results_df):,} rows × {len(results_df.columns)} cols, '
          f'{results_path.stat().st_size/1024:.0f} KB)')
    return env_path, results_path

# ── Run export ────────────────────────────────────────────────────────────────
print('=== Exporting CSVs ===')

export_grid(grid_baseline_m4, 'baseline')

if SCENARIO_ACTIVE and grid_scenario_m4 is not None:
    scen_label = f'{ACTIVE_SSP}_{ACTIVE_PERIOD}'
    export_grid(grid_scenario_m4, scen_label)
    print(f'\nScenario label: {SSP_SCENARIOS[ACTIVE_SSP]} / {ACTIVE_PERIOD}')
else:
    print('\nScenario not active — baseline only exported')

print('\n=== Export complete ===')
