# MYTIGATE — Handover Document v4

**Last updated:** October 2026  
**Status:** Pipeline validated end-to-end. Scenario pipeline validated. Webapp UI not yet built.  
**Science basis:** Holbach et al. (2020), *Science of the Total Environment* 736, 139624.

---

## 1. What This Project Does

MYTIGATE is a spatial decision-support tool for mussel mitigation farming in Danish and western Baltic coastal waters. It answers: *where should mussel longline farms be placed to maximise nitrogen removal from eutrophied water bodies, and how productive will those farms be?*

The tool computes, for every 1km² sea cell in the domain:
- Mussel biomass growth potential (gDW per individual mussel)
- Farm-scale harvest (tonnes wet weight, dry weight, shell dry weight)
- Nitrogen and phosphorus removal (tonnes per farm per harvest)
- Food limitation risk (is there enough flow to feed the mussels?)
- Exclusion status (is this cell legally or physically off-limits?)
- Conflict gradients (shipping, fishing, harbour proximity)

It supports two scenario modes:
- **Interactive delta mode**: user adjusts ΔTemp / ΔSal / ChlA× sliders at runtime
- **CMIP6 scenario mode**: physically grounded SSP projections (four scenarios × two time horizons) precomputed as drop-in replacement grids

---

## 2. Repository Structure

```
project_root/
├── config.py                      # All paths, constants, model parameters
├── config_scenario.py             # Scenario config — imports config.py, adds helpers
├── pipeline.py                    # Full preprocessing pipeline (Steps 1–11)
│
├── utils/
│   ├── grid.py                    # 1km UTM32N grid builder
│   ├── raster_sampler.py          # Module 1: CMEMS env sampling + blending
│   ├── download_cds.py            # CMIP6 CDS downloader (tos, sos)
│   ├── download_esgf.py           # CMIP6 ESGF downloader (uo, vo, chl)
│   └── climate_projections.py     # CMIP6 delta-change processor + scenario loader
│
├── model/
│   ├── module11_exclusions.py     # Exclusion masks + conflict gradients
│   ├── module12_farm.py           # Farm discretisation from user polygon
│   ├── growth.py                  # Modules 2, 3, 4: growth, upscaling, food limitation
│   └── site_selector.py          # Site ranking and selection algorithm
│
├── app/
│   ├── app.py                     # Dash entry point
│   ├── layout.py                  # UI layout (NEEDS REWRITE — old architecture)
│   ├── callbacks.py               # Dash callbacks (NEEDS REWRITE — old architecture)
│   └── map_utils.py               # Plotly Scattermapbox builder
│
├── notebooks/
│   ├── pipeline_debug.ipynb       # Self-contained step-by-step pipeline debug
│   └── scenario_test.ipynb        # Self-contained scenario validation notebook
│
└── data/
    ├── raw/
    │   ├── bathymetry/            # emodnet_dtm.tif — REQUIRED, never missing
    │   ├── hydrodynamics/         # CMEMS NetCDF files (see Section 5)
    │   └── conflict_layers/       # Shapefiles (see Section 6)
    └── processed/
        ├── grid_base.gpkg         # 1km grid + bathy + exclusions + env cols
        ├── baseline.gpkg          # Full M2+M3+M4 output — app loads this
        ├── grid_env_temp.gpkg     # Baseline temp cache (12 mean + 12 sd cols)
        ├── grid_env_sal.gpkg      # Baseline salinity cache
        ├── grid_env_chla.gpkg     # Baseline ln(ChlA) cache
        ├── grid_env_flow.gpkg     # Baseline flow cache (vh, uo, vo)
        └── cmip6_raw/             # Raw CMIP6 NetCDF files (see Section 8)
            ├── historical_cds/
            ├── historical_esgf_uo/
            ├── historical_esgf_vo/
            ├── historical_esgf_chl/
            ├── ssp126_cds/ ... ssp585_cds/
            ├── ssp1_2_6_esgf_uo/ ... ssp5_8_5_esgf_chl/
            └── ssp1_2_6/mid/ ssp1_2_6/end/ ... (scenario gpkg outputs)
```

---

## 3. Grid Definition

```python
GRID_CRS  = "EPSG:32632"   # UTM Zone 32N
CELL_SIZE = 1000            # 1km × 1km
X_MIN, X_MAX = 442_000, 752_000
Y_MIN, Y_MAX = 6_054_000, 6_432_000
# Approx coverage: 54.6–58.0°N, 7.8–13.1°E
# ~61,060 cells total after land clipping
```

The grid is built once by `utils/grid.py` and cached as `grid_base.gpkg`. Land cells are removed using the EEA coastline shapefile. Sea cells get `lon`/`lat` centroid coordinates in WGS84 for mapping and interpolation.

**Important:** the grid bounds were extended beyond Holbach 2020 (which used X_MIN=400,000) to include the North Sea / Jutland west coast, Limfjord, Wadden Sea, and Skagerrak. Bornholm is excluded by design (outside domain).

---

## 4. Pipeline — Steps 1–11

Run with: `python pipeline.py [flags]`

```
python pipeline.py                            # full baseline (~20 min)
python pipeline.py --n-scenarios 50           # fast debug (~5 min)
python pipeline.py --skip-env                 # skip CMEMS if cached
python pipeline.py --scenario --delta-temp 1.8 --delta-sal -0.5 --chla-mult 1.1
python pipeline.py --precompute-scenarios --n-scenarios 50   # baseline + CMIP6
python pipeline.py --precompute-scenarios --skip-baseline    # CMIP6 only (needs baseline.gpkg)
```

### Steps 1–8: Baseline

| Step | Module | Output | Skip logic |
|------|--------|--------|------------|
| 1 | Grid | `grid_base.gpkg` | Loads from cache if exists |
| 2 | Bathymetry | `bathymetry_m` column | Skips if column present; **errors if TIF missing** |
| 3 | Module 1 — CMEMS env | 4 `grid_env_*.gpkg` files | Per-variable cache check |
| 4 | Module 11 — Exclusions | `excluded`, `conflict_*` columns | Always runs |
| — | Save base grid | `grid_base.gpkg` | Overwrites |
| 5 | Module 2 — Growth MC | `mbio_p05/p50/p95_{mm}` | Always runs |
| 6 | Module 3 — Farm upscaling | `harvest_*`, `N_red_*`, `shell_DW_*` | Always runs |
| 7 | Module 4 — Food limitation | `vh_ratio_*`, `food_limitation_*` | Always runs |
| 8 | Save | `baseline.gpkg` | Overwrites |

### Steps 9–11: CMIP6 Scenarios (--precompute-scenarios only)

| Step | Action | Skip logic |
|------|--------|------------|
| 9 | `download_cds.py` — tos + sos from CDS | Per-variable/experiment skip |
| 10 | `download_esgf.py` — uo + vo + chl from DKRZ | Per-folder skip |
| 11 | `climate_projections.py` — build scenario gpkgs | Per-file skip (all 4 must exist) |

---

## 5. Module 1 — Environmental Fields (CMEMS)

**File:** `utils/raster_sampler.py`  
**Function:** `run_module1(grid, delta_temp, delta_sal, chla_mult, delta_flow, force_resample)`

### Data sources

| Variable | CMEMS product | Folder | Coverage |
|----------|--------------|--------|----------|
| Temp (thetao) | Baltic physics | `cmems_phy/` | lon 9–17°E |
| Salinity (so) | Baltic physics | `cmems_phy/` | lon 9–17°E |
| ChlA (chl) | Baltic BGC | `cmems_bgc/` | lon 9–17°E |
| Flow (uo, vo) | Baltic physics | `cmems_flow/` | lon 9–17°E |
| Temp | NWS physics | `cmems_nws_phy/` | lon 5–10°E |
| Salinity | NWS physics | `cmems_nws_phy/` | lon 5–10°E |
| ChlA | NWS BGC | `cmems_nws_bgc/` | lon 5–10°E |
| Flow | NWS physics | `cmems_nws_flow/` | lon 5–10°E |

### Baltic/NWS blending

The two CMEMS products overlap around 8.5–10°E. A **distance-weighted linear blend** is applied:
- Scalar fields (temp, sal, lnchla): blend zone 8.5–10.0°E
- Flow fields (uo, vo, vh): wider blend zone 7.5–11.5°E + Gaussian spatial smoothing (σ=3 cells) to suppress the seam in M4 critical-flow maps

### Caching

Four separate GeoPackage files are written after baseline M1, one per variable group:
```
grid_env_temp.gpkg   — temp_mean_01..12, temp_sd_01..12
grid_env_sal.gpkg    — sal_mean_01..12, sal_sd_01..12
grid_env_chla.gpkg   — lnchla_mean_01..12, lnchla_sd_01..12
grid_env_flow.gpkg   — vh_01..12, uo_01..12, vo_01..12
```

On subsequent runs, each variable is loaded from cache independently. Only variables that are perturbed (by scenario flags) are resampled from CMEMS.

### Units

| Variable | Stored unit | Notes |
|----------|------------|-------|
| temp_mean_* | °C | CMEMS native |
| sal_mean_* | psu | CMEMS native |
| lnchla_mean_* | ln(µg/L) | Natural log; ChlA in µg/L before log |
| vh_*, uo_*, vo_* | m/s | CMEMS native |

**Critical:** ChlA is stored as `ln(ChlA)` throughout the pipeline. The log floor is 0.01 µg/L (matching `CHLA_ZERO` in config). CMIP6 chl data arrives in kg/m³ and must be multiplied by 1e6 before the log transform.

### Silent default warnings

Several data gaps produce verbose WARNING prints but do not stop the pipeline:
- Missing CMEMS NetCDF files → placeholder values (vh=0.05 m/s, temp=10°C, sal=20°C, lnchla=ln(2))
- Missing exclusion shapefiles → that layer returns False (not excluded) for all cells
- Missing shipping TIF → conflict_shipping = 0 everywhere

**Bathymetry is the only hard error:** if `emodnet_dtm.tif` is missing the pipeline raises `FileNotFoundError` immediately.

---

## 6. Module 11 — Exclusions and Conflicts

**File:** `model/module11_exclusions.py`

### Hard exclusions (binary — cells removed from farm siting)

| Layer | Source | Effect |
|-------|--------|--------|
| Depth < 4m | Bathymetry | `excl_shallow = True` |
| Marine Protected Areas | HELCOM MPAs 2019 | `excl_mpa = True` |
| Military areas | EMODnet | `excl_military = True` |
| Wind farms | EMODnet | `excl_wind_farms = True` |
| Pipelines | EMODnet | `excl_pipelines = True` |
| Cables | EMODnet (power + telecom) | `excl_cables = True` |

Any excluded cell gets `excluded = True`. Excluded cells score zero in site ranking.

### Conflict gradients (soft 0–1 — display only, do not block siting)

| Column | Description |
|--------|-------------|
| `conflict_shipping` | Normalised route density from EMODnet shipping TIF |
| `conflict_fishing` | Normalised subsurface swept area ratio |
| `conflict_harbours` | Distance to nearest major port (inverted, normalised) |
| `conflict_score` | Mean of the three above |

Conflict gradients can apply a penalty in site ranking via `conflict_weight` parameter in `site_selector.py`.

### Optimisations

- All polygon layers processed concurrently (ThreadPoolExecutor)
- Spatial intersections via geopandas `sindex.query()` — no Python per-cell loops
- Shipping raster sampled at cell centroids (point sampling, not zonal stats) — ~30s vs ~15 min

---

## 7. Modules 2, 3, 4 — Growth, Upscaling, Food Limitation

**File:** `model/growth.py`  
**Science:** Holbach 2020, Eqs 3–6, 8–10, 11–17

### Module 2 — Mussel Growth (DEB-derived empirical model)

Three growth performance factors, each 0–1:

```
fT = Arrhenius temperature response (peaks ~15°C, drops at extremes)
     Parameters: TA=5800K, T1=289K, TL=275K, TH=296K, TAL=45430K, TAH=31376K

fS = Salinity penalty below 16.2 psu
     fS = 1                          if Sal > 16.2
     fS = 1 - (16.2 - Sal)/16.2 × fT  if Sal ≤ 16.2

fC = Michaelis-Menten ChlA uptake
     fC = 0                    if ChlA ≤ 0.2 µg/L
     fC = ChlA/(ChlA + 0.8)   if 0.2 < ChlA ≤ 20 µg/L
     fC = exp(-0.03×(ChlA-20)) if ChlA > 20 µg/L  (saturation penalty)

fTSC = fT × fS × fC   (integrated monthly, July settlement → harvest month)

mbio [gDW] = 0.0190 × (Σ fTSC)^2.71   (Holbach 2020 Eq. 18)
```

**Monte Carlo:** 500 scenarios (50 for debug) draw from normal distributions of Temp, Sal, ln(ChlA) using their spatial SD fields. Output is p05/p50/p95 of mbio across scenarios.

**Harvest months:** Aug, Sep, Oct, Nov, Dec (columns suffixed `_08` through `_12`). Settlement is always July. The model is validated only for the first growth season — spring harvest results are unreliable.

### Module 3 — Farm Upscaling

```
ρ_mbio [gDW/m collector] = 1269 × mbio^(1/3)   (Holbach 2020 Eq. 9)
Harvest [gDW]            = ρ_mbio × l_col        (Holbach 2020 Eq. 10)

l_col [m] = (2×rdep + iloop) × (longline_length × n_longlines / iloop)
rdep       = min(bathymetry - 2, max_rdep)  if bathymetry ≥ 4m, else 0
```

**Conversion factors (applied to whole-mussel DW including shell + byssus):**

| Output | Factor | Notes |
|--------|--------|-------|
| harvest_WW [t] | DW × 9.68 | Whole wet weight |
| harvest_DW [t] | DW / 1e6 | Unit conversion from gDW |
| shell_DW [t] | DW × 0.77 | Shell fraction (Smaal & Vonck 1997) |
| N_red [tN] | DW × 0.14 | 14% N content |
| P_red [tP] | DW × 0.008 | 0.8% P content |

**Validated internal consistency check:**  
`harvest_WW / 9.68 = harvest_DW`  
`harvest_DW × 0.77 = shell_DW`  
`harvest_DW × 0.14 = N_red`  
`harvest_DW × 0.008 = P_red`  
All pass.

### Module 4 — Food Limitation

Estimates the minimum flow velocity (m/s) needed to avoid significant within-farm ChlA depletion. Output `vh_ratio = actual_flow / critical_flow`. Values < 0.5 indicate likely food limitation and model overestimation.

Built as a LUT (Lookup Table) over Temp × ChlA space for each month, using the C:ChlA monthly ratios from Jacobsen & Markager (2016):

```python
C_CHLA_MONTHLY = [32,32,32,30,24,23,22,23,32,43,53,61]  # Jan–Dec
```

M4 is informational only — it does not block farm siting but can be used as a filter.

---

## 8. CMIP6 Climate Scenarios

**Files:** `utils/download_cds.py`, `utils/download_esgf.py`, `utils/climate_projections.py`  
**Config:** `config_scenario.py`

### Overview

Four SSP scenarios × two time periods = 8 scenario sets, each producing 4 `.gpkg` files:

| SSP | Label | ~Warming |
|-----|-------|---------|
| ssp1_2_6 | SSP1-2.6 | +1.8°C |
| ssp2_4_5 | SSP2-4.5 | +2.7°C |
| ssp3_7_0 | SSP3-7.0 | +3.6°C |
| ssp5_8_5 | SSP5-8.5 | +4.4°C |

| Period | Range |
|--------|-------|
| mid | 2041–2060 |
| end | 2071–2100 |

### Data sources

| Variable | Source | Model | Format |
|----------|--------|-------|--------|
| SST (tos) | CDS `projections-cmip6` | MPI-ESM1-2-LR | ~1° global, monthly |
| Salinity (sos) | CDS `projections-cmip6` | MPI-ESM1-2-LR | ~1° global, monthly |
| U-current (uo) | ESGF / DKRZ | MPI-ESM1-2-LR r1i1p1f1 Omon | ~0.4° native ocean grid |
| V-current (vo) | ESGF / DKRZ | MPI-ESM1-2-LR r1i1p1f1 Omon | ~0.4° native ocean grid |
| Chlorophyll (chl) | ESGF / DKRZ | MPI-ESM1-2-LR r1i1p1f1 Omon | ~0.4° native ocean grid |

**CMIP6 chl units: kg/m³.** Must be multiplied by 1e6 to convert to µg/L before the log transform. This conversion is applied in `_monthly_interp(unit_scale=1e6)` for all chl calls. Failure to apply this conversion produces zero deltas (all values clamped to the 0.01 µg/L floor before taking log).

### Delta-change method

```
projected = CMEMS_baseline + (CMIP6_future − CMIP6_historical)
```

The CMIP6 historical period (1985–2014) is subtracted to remove model bias. Only the change signal is added to the CMEMS observations. The CMEMS `.gpkg` files are never modified.

For lnchla: `delta = ln(chl_future × 1e6) − ln(chl_hist × 1e6)` — the 1e6 cancels in the subtraction, but it must be applied before the log floor clamp.

### Interpolation methods

- **CDS (tos, sos):** regular lat/lon grid → bilinear (`RegularGridInterpolator`)
- **ESGF (uo, vo, chl):** curvilinear native ocean grid → inverse-distance weighted k=4 nearest neighbours (`cKDTree`)

### SD columns

SD columns are carried forward unchanged from the CMEMS baseline. Future variability is assumed equal to observed variability. This is a known simplification — flag in any uncertainty analysis.

### Output file structure

```
data/processed/
  ssp2_4_5/
    end/
      grid_env_temp_ssp2_4_5_end.gpkg   ← temp_mean_01..12 + temp_sd_01..12
      grid_env_sal_ssp2_4_5_end.gpkg    ← sal_mean_01..12 + sal_sd_01..12
      grid_env_chla_ssp2_4_5_end.gpkg   ← lnchla_mean_01..12 + lnchla_sd_01..12
      grid_env_flow_ssp2_4_5_end.gpkg   ← vh_01..12, uo_01..12, vo_01..12
```

These are drop-in replacements for the baseline `grid_env_*.gpkg` files. Module 2 receives them identically — it does not know whether it is receiving baseline or projected data.

### Loading a scenario

```python
from utils.climate_projections import load_scenario_env
grid_scenario = load_scenario_env(grid, ssp='ssp2_4_5', period='end')
# grid_scenario has all env columns replaced; bathy, exclusions unchanged
# Pass directly to run_module2()
```

### Two scenario modes — kept separate

| Mode | Trigger | Use case |
|------|---------|---------|
| Interactive delta | `run_module1(delta_temp=x, delta_sal=y, chla_mult=z)` | Sensitivity analysis, user-defined |
| CMIP6 precomputed | `load_scenario_env(grid, ssp, period)` | Physically grounded projections |

These are mutually exclusive at runtime. Do not mix them. Back-calculating CMIP6 deltas into slider values is not implemented and not straightforward.

---

## 9. Module 12 — Farm Discretisation

**File:** `model/module12_farm.py`

Triggered when a user draws a polygon on the map. Takes a WGS84 polygon, projects to UTM32N, intersects with the grid, determines dominant current direction from uo/vo, computes farm geometry parameters (D_short, D_long, n_sections, l_col), and returns farm cells ready for M2–M4.

Default farm geometry (Holbach 2020, Section 2.5):
```python
LONGLINE_LENGTH       = 200   # m — length of each longline
LONGLINE_SPACING      = 8     # m — between longlines
LONGLINES_PER_SECTION = 30    # longlines per section
N_SECTIONS_DEFAULT    = 3
# D_short = 30 × 8 = 240m (across-current dimension)
# D_long  = 200m (along-current)
# Farm area per section = 240 × 200 = 48,000 m²
```

A mock polygon in Vejle Fjord is defined in `config.py` (`MOCK_FARM_COORDS_LONLAT`) for notebook testing.

---

## 10. Site Selector

**File:** `model/site_selector.py`

Ranks all viable cells by a chosen criterion (default: `N_red`) with optional conflict penalty, then selects until a target is met (number of farms or total N-reduction). Farm area is computed dynamically from `grid["farm_area_m2"]` — no hardcoded `FARM_AREA_HA` constant.

```python
grid, summary = rank_and_select(
    grid,
    harvest_month=11,
    criterion='N_red',
    target_mode='n_farms',
    target_value=10,
    conflict_weight=1.0,
    food_filter=True,
)
```

---

## 11. Map Utilities

**File:** `app/map_utils.py`

Builds Plotly `Scattermapbox` figures. Key notes:
- Map center and zoom are computed dynamically from grid data bounds — not hardcoded
- Marker size = 6px (increased from 4 to reduce edge gaps at target zoom)
- Mapbox token is used; Carto token (`CARTO_TOKEN`) is in config but not yet wired to the app (noted for webapp rewrite)
- `empty_map()` uses static fallback center (lat=56.3, lon=10.4, zoom=7.0)

---

## 12. Webapp Status

**Entry point:** `app/app.py` → `http://localhost:8050`

`layout.py` and `callbacks.py` are on the old pre-polygon architecture and need a full rewrite. The new architecture requires:

- Sidebar: mode toggle (baseline/CMIP6 scenario/interactive delta), farm geometry params, scenario selectors, harvest month selector
- Map area: split baseline/scenario view or single map
- Right panel: seasonal table, farm summary (N-red, harvest, shell_DW), conflict summary, export buttons
- Polygon drawing via `dash-leaflet` DrawControl (currently using `MOCK_FARM_COORDS_LONLAT` placeholder)
- Export: two CSV files per active grid (env grid + model results) via download callback

The Dash app loads `baseline.gpkg` at startup. Scenario recompute is fast (M2+M3+M4 only, ~1–2 min) because env data is already loaded.

---

## 13. Export Format (Rhino/Grasshopper)

Two CSV files are exported per active grid, joined to the Rhino grid on `cell_id`:

**`mytigate_env_{label}.csv`**
```
cell_id, lon, lat, bathymetry_m
temp_mean_01..12  (°C)
sal_mean_01..12   (psu)
lnchla_mean_01..12  (ln(µg/L))
vh_01..12  (m/s)
```

**`mytigate_results_{label}.csv`**
```
cell_id, lon, lat, excluded, conflict_score, rdep
harvest_WW_08..12  (t)
harvest_DW_08..12  (t)
shell_DW_08..12    (t)
N_red_08..12       (tN)
N_red_ha_08..12    (tN/ha)
P_red_08..12       (tP)
mbio_p50_08..12    (gDW/mussel)
vh_ratio_08..12    (dimensionless)
food_limitation_08..12  (0–1 severity)
```

Label is `baseline` or `ssp2_4_5_end` etc. Export cell is in `notebooks/export_cell.py` — paste as a cell at the end of `scenario_test.ipynb`.

---

## 14. Known Issues and Limitations

| Issue | Severity | Status |
|-------|----------|--------|
| Limfjord/Wadden Sea: CMEMS has no coverage → placeholder env values used | Medium | Flagged with WARNING print; cells not masked. May have missing data |
| Missing exclusion shapefiles silently return "not excluded" | Medium | WARNING print only; regulatory consequence if MPA file missing |
| Missing flow NetCDF silently uses vh=0.05 m/s placeholder | Medium | WARNING print; M4 and M12 farm orientation unreliable |
| SD columns unchanged in CMIP6 scenarios | Low | Known simplification; documented |
| Spring harvest M2 (Jan–May) overestimates biomass | Low | Model only valid for first growth season; only Aug–Dec used |
| Carto API token unused in app | Low | Noted for webapp rewrite |
| Bornholm excluded by grid domain bounds | By design | Documented |
| ESGF node: LLNL shut down July 2025 | Info | Use DKRZ node only |

---

## 15. How to Start Fresh on a New Machine

```bash
# 1. Clone repo, set up conda environment
conda create -n mapping python=3.11
conda activate mapping
pip install geopandas xarray rasterio rasterstats plotly dash dash-bootstrap-components \
            scipy numpy pandas tqdm cdsapi fiona cf-xarray

# 2. Edit config.py — set RAW path if data is outside project folder

# 3. Run baseline pipeline (fast debug)
python pipeline.py --n-scenarios 50

# 4. Verify in notebook
jupyter notebook notebooks/pipeline_debug.ipynb

# 5. Precompute CMIP6 scenarios (requires CDS API key + DKRZ account)
python pipeline.py --precompute-scenarios --skip-baseline

# 6. Test scenarios
jupyter notebook notebooks/scenario_test.ipynb

# 7. Start app (once layout.py + callbacks.py are rewritten)
python app/app.py
```

---

## 16. File Inventory (All Modified/Created This Session)

| File | Role | Status |
|------|------|--------|
| `config.py` | Central config | Updated: new grid bounds, SHELL_FRACTION, 4 env cache paths |
| `config_scenario.py` | Scenario config | New |
| `pipeline.py` | Main pipeline | Updated: --precompute-scenarios, --skip-baseline, Steps 9–11 |
| `utils/raster_sampler.py` | M1 CMEMS sampling | Updated: per-variable caching, Baltic/NWS blend, bathy error |
| `utils/climate_projections.py` | CMIP6 processing | Updated: unit_scale fix, load_scenario_env, per-file skip |
| `model/growth.py` | M2/M3/M4 | Updated: shell_DW output column |
| `model/site_selector.py` | Site ranking | Fixed: FARM_AREA_HA ImportError removed |
| `app/map_utils.py` | Map builder | Updated: dynamic center/zoom |
| `notebooks/scenario_test.ipynb` | Scenario validation | New |
| `notebooks/export_cell.py` | CSV export | New |
