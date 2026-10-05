# MYTIGATE_futures — Handover v5
**Mussel mitigation farm site selection · Baltic and Atlantic Denmark**

Science basis: Holbach et al. (2020), *Science of the Total Environment* 736, 139624  
Stack: Python 3.11 · GeoPandas · Xarray · Dash/Plotly · CMEMS NetCDF · CMIP6

---

## 1. Project Concept

Mussel (*Mytilus edulis*) longline farms can act as **mitigation instruments** in eutrophied coastal waters by bioextracting nitrogen (N) and phosphorus (P) while producing harvestable biomass. MYTIGATE_futures is a spatial decision-support tool that:

1. Samples environmental conditions (temperature, salinity, chlorophyll-a, flow) from CMEMS reanalysis across a 1 km² grid covering the western Baltic Sea and Danish coastal waters
2. Runs a Dynamic Energy Budget (DEB)-derived mussel growth model (Monte Carlo, 50–500 scenarios per cell)
3. Upscales individual mussel biomass to farm-level harvest and nutrient removal
4. Applies a food limitation filter based on current speed vs critical filtration rate
5. Identifies and ranks viable farm sites while excluding areas with spatial conflicts (MPAs, military zones, shipping lanes, cables, pipelines, wind farms)
6. Allows users to draw a custom farm polygon on an interactive map and compute expected production for that specific site
7. Supports climate scenario analysis via CMIP6 delta-change projections (SSP1-2.6 through SSP5-8.5)

---

## 2. Repository Structure

```
project_root/
├── config.py                     # Central config — paths, grid constants, model params
├── config_scenario.py            # Scenario config — SSP labels, scenario gpkg paths
├── pipeline.py                   # Full preprocessing pipeline (Steps 1–11)
│
├── utils/
│   ├── grid.py                   # 1km UTM32N grid builder
│   ├── raster_sampler.py         # Module 1: CMEMS env sampling + Baltic/NWS blending
│   ├── download_cds.py           # CMIP6 CDS downloader (tos, sos)
│   ├── download_esgf.py          # CMIP6 ESGF downloader (uo, vo, chl)
│   └── climate_projections.py    # CMIP6 delta-change processor + scenario loader
│
├── model/
│   ├── module12_farm.py          # M12: farm polygon discretisation + geometry
│   ├── growth.py                 # M2: DEB growth (MC); M3: upscaling; M4: food limitation
│   └── site_selector.py         # Site ranking, selection, and cell report
│
├── app/
│   ├── app.py                    # Dash entry point (host=127.0.0.1, port=8050)
│   ├── layout.py                 # UI layout — 4-step sidebar flow
│   ├── callbacks.py              # All Dash callbacks
│   └── map_utils.py              # Plotly Scattermap builders
│
├── notebooks/
│   ├── pipeline_debug.ipynb      # Step-by-step debug — REFERENCE for app logic
│   └── scenario_test.ipynb       # CMIP6 scenario validation
│
└── data/
    ├── raw/
    │   ├── bathymetry/           # emodnet_dtm.tif (REQUIRED)
    │   ├── hydrodynamics/        # CMEMS Baltic + NWS NetCDF files
    │   └── conflict_layers/      # MPA, military, shipping, cables, etc.
    └── processed/
        ├── grid_base.gpkg        # 1km UTM32N grid (no env data)
        ├── baseline.gpkg         # Full M1+M2+M3+M4 output — primary app file
        ├── grid_env_temp.gpkg    # Per-variable env caches (scenario-aware)
        ├── grid_env_sal.gpkg
        ├── grid_env_chla.gpkg
        ├── grid_env_flow.gpkg
        └── ssp*/mid|end/         # 32 precomputed CMIP6 scenario gpkg files
```

---

## 3. Grid Definition

```python
GRID_CRS  = "EPSG:32632"    # UTM Zone 32N
CELL_SIZE = 1_000            # 1km × 1km
X_MIN, X_MAX = 442_000, 752_000
Y_MIN, Y_MAX = 6_054_000, 6_432_000
# ~61,060 cells · coverage: 54.6–58°N · 7.8–13.1°E
```

Each cell stores env variables as monthly means/SDs for 12 months. Key columns: `cell_id`, `lon`, `lat`, `geometry`, `bathymetry_m`, `excluded`, `conflict_score`, then per-month env and result columns (see §6).

---

## 4. Pipeline Steps (pipeline.py)

| Step | Description | Output |
|------|-------------|--------|
| 1 | Build 1km UTM32N grid | `grid_base.gpkg` |
| 2 | Sample bathymetry from EMODnet DTM | adds `bathymetry_m` |
| 3 | M1: Sample CMEMS env + Baltic/NWS blend | adds env columns |
| 4 | M11: Exclusion masks + conflict gradients | adds `excluded`, `conflict_score` |
| 5 | M2: DEB growth (MC) | adds `mbio_p05/p50/p95_MM` |
| 6 | Compute rdep from bathymetry | adds `rdep` |
| 7 | M3: Farm upscaling | adds `harvest_*`, `N_red_*`, `P_red_*` |
| 8 | M4: Food limitation | adds `vh_ratio_*`, `food_limitation_*` |
| 9 | Save `baseline.gpkg` | primary app data file |
| 10–11 | (Optional) CMIP6 download + scenario precompute | `ssp*/mid\|end/*.gpkg` |

Run with:
```bash
python pipeline.py                           # baseline only
python pipeline.py --precompute-scenarios    # + all 32 CMIP6 scenarios
python pipeline.py --skip-env               # skip CMEMS resampling (use cache)
python pipeline.py --skip-baseline          # skip baseline, only scenarios
```

---

## 5. Module Equations (from Holbach 2020)

### Module 1 — Environmental Sampling (raster_sampler.py)
CMEMS reanalysis sampled per 1km cell, 12 monthly means. Baltic (lon > 10°E) and North Sea/NWS (lon < 8.5°E) products blended 8.5–10°E. Flow: Gaussian smooth σ=3 cells. ChlA stored as `lnchla_mean` (ln µg/L); ESGF units kg/m³ × 1e6 to convert to µg/L before log.

### Module 2 — DEB Mussel Growth (growth.py)
Monte Carlo growth model with 3 limiting factors applied multiplicatively each month from settlement (July) to harvest:

**Temperature limitation** (Holbach Eq. 3, Arrhenius):
```
fT = exp(TA/T1 − TA/T_K) / [1 + exp(TAL/T_K − TAL/TL) + exp(TAH/TH − TAH/T_K)]
```
Parameters: TA=5800, T1=289K, TL=275K, TH=296K, TAL=45430, TAH=31376

**Salinity limitation** (Holbach Eq. 4):
```
fS = 1 − (SAL_THRESHOLD − S) / SAL_THRESHOLD × fT    if S ≤ SAL_THRESHOLD (16.2 psu)
fS = 1                                                  if S > SAL_THRESHOLD
```

**Chlorophyll-a / food limitation** (Holbach Eq. 5):
```
fC = ChlA / (ChlA + CHLA_HALF_SAT)    if 0.2 < ChlA ≤ 20 µg/L
fC = exp(−0.03 × (ChlA − 20))         if ChlA > 20 µg/L
fC = 0                                  if ChlA ≤ 0.2 µg/L
```
CHLA_HALF_SAT = 0.8 µg/L

**Cumulative growth sum** (July→harvest month):
```
Σ = Σ_months fT × fS × fC
```

**Individual mussel dry weight** (Holbach Eq. 6):
```
m_bio [gDW] = 0.0190 × Σ^2.71
```
Output: `mbio_p05`, `mbio_p50`, `mbio_p95` [gDW] per harvest month per cell.

### Module 3 — Farm Upscaling (growth.py)
Mussel packing density on longlines:
```
ρ_bio [g/m] = 1269 × m_bio^(1/3)
```

Collector length per cell (from M12 geometry):
```
l_col [m] = (2 × rdep + iloop) × (longline_length × total_lines / iloop)
```

Farm-level harvest:
```
harvest_DW [t] = ρ_bio × l_col / 1e6
harvest_WW [t] = harvest_DW × 9.68           (DW→fresh weight)
shell_DW   [t] = harvest_DW × 0.77           (shell fraction, Smaal & Vonck 1997)
N_red      [t] = harvest_DW × 0.14           (14% N content)
P_red      [t] = harvest_DW × 0.008          (0.8% P content)
N_red_ha       = N_red / farm_area_ha
```

### Module 4 — Food Limitation (growth.py)
Vectorised LUT approach. For each (temp, ChlA) pair, finds time `t_crit` when filtration reduces food by 25% below ambient. Critical current speed:
```
vh_crit [m/s] = D_short / t_crit
vh_ratio       = vh / vh_crit
```
- `vh_ratio ≥ 0.5` → OK (sufficient flushing)
- `vh_ratio < 0.5` → food limitation warning
- `rho_col = median(lcol) / farm_volume` — farm density parameter

### Module 11 — Exclusion + Conflicts (module11_exclusions.py)
**Hard exclusions** (cell unusable): depth < 4m, MPAs, military zones, wind farms, pipelines, submarine cables.

**Soft conflict gradients** (0–1 score): shipping lanes (from EMODnet route density raster), fishing grounds (subsurface swept area ratio), harbour proximity (inverse distance). Combined as `conflict_score`.

### Module 12 — Farm Discretisation (module12_farm.py)
User draws polygon (lon/lat). M12:
1. Converts polygon to UTM32N via pyproj
2. Finds grid cells intersecting polygon via STRtree spatial index
3. Filters excluded cells and depth < 4m
4. Computes dominant current direction from mean uo/vo at harvest month
5. Farm orientation = `arctan2(uo, vo)` (aligned with current)
6. Farm geometry:
   ```
   D_short = longlines_per_section × longline_spacing   # 30×8 = 240m (current direction)
   D_long  = longline_length                             # 200m (across current)
   section_area = D_short × D_long                       # 48,000 m²
   n_sections   = polygon_utm.area / section_area        # scales with drawn polygon
   farm_area    = n_sections × section_area
   ```
7. Returns `farm_cells` GeoDataFrame (grid cells with env conditions) + geometry dict

**Key design decision:** Grid cells inside the polygon determine environmental conditions for M2/M3/M4. The polygon area determines farm geometry (n_sections, lcol). A small polygon = few sections = small farm. A large polygon = many sections = megafarm. This is intentional — the tool supports any farm scale.

---

## 6. Column Naming Convention

All time-varying columns are suffixed `_MM` where MM = two-digit month (01–12).

| Column pattern | Source | Description |
|----------------|--------|-------------|
| `temp_mean_MM` | M1 | Monthly mean temperature [°C] |
| `temp_sd_MM` | M1 | Monthly SD temperature |
| `sal_mean_MM` | M1 | Monthly mean salinity [psu] |
| `sal_sd_MM` | M1 | Monthly SD salinity |
| `lnchla_mean_MM` | M1 | Monthly mean ln(ChlA) [ln µg/L] |
| `lnchla_sd_MM` | M1 | Monthly SD ln(ChlA) |
| `vh_MM` | M1 | Monthly mean current speed [m/s] |
| `uo_MM`, `vo_MM` | M1 | Monthly mean current components [m/s] |
| `bathymetry_m` | M2 | Water depth [m] (static) |
| `excluded` | M11 | Hard exclusion flag (bool) |
| `conflict_score` | M11 | Soft conflict gradient 0–1 |
| `rdep` | pipeline | Collector depth [m] |
| `lcol_m` | M12 | Total collector length per cell [m] |
| `mbio_p05_MM` | M2 | 5th pct mussel dry weight [gDW] |
| `mbio_p50_MM` | M2 | Median mussel dry weight [gDW] |
| `mbio_p95_MM` | M2 | 95th pct mussel dry weight [gDW] |
| `harvest_DW_MM` | M3 | Farm harvest dry weight [t] |
| `harvest_WW_MM` | M3 | Farm harvest fresh weight [t] |
| `shell_DW_MM` | M3 | Shell dry weight [t] |
| `N_red_MM` | M3 | N-reduction [tN] |
| `P_red_MM` | M3 | P-reduction [tP] |
| `N_red_ha_MM` | M3 | N-reduction per ha [tN/ha] |
| `N_red_p05_MM` | M3 | N-reduction 5th pct [tN] |
| `N_red_p95_MM` | M3 | N-reduction 95th pct [tN] |
| `vh_ratio_MM` | M4 | Current speed ratio (vh/vh_crit) |
| `vh_crit_MM` | M4 | Critical current speed [m/s] |
| `food_limitation_MM` | M4 | Food limitation severity 0–1 |
| `food_ok` | M4 | Majority months OK (bool) |

---

## 7. Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| Bathymetry | EMODnet DTM | `emodnet_dtm.tif`, EPSG:4326, ~115m resolution |
| Temperature, salinity | CMEMS Baltic Physics Reanalysis | `cmems_phy/` |
| Chlorophyll-a | CMEMS Baltic BGC Reanalysis | `cmems_bgc/` |
| Currents (uo, vo) | CMEMS Baltic Physics Reanalysis | `cmems_flow/` |
| NWS temp, sal | CMEMS NWS Atlantic Physics | `cmems_nws_phy/` — fills Jutland/Limfjord gap |
| NWS ChlA | CMEMS NWS BGC | `cmems_nws_bgc/` |
| NWS currents | CMEMS NWS Physics | `cmems_nws_flow/` |
| CMIP6 tos, sos | CDS (Copernicus) | `download_cds.py` |
| CMIP6 uo, vo, chl | ESGF | `download_esgf.py` — chl in kg/m³, ×1e6 for µg/L |
| MPAs | HELCOM 2019 | `HELCOM_MPAs_2019_2.shp` |
| Military zones | EMODnet HA | `militaryareaspolyPolygon.shp` |
| Wind farms | EMODnet HA | `windfarmspolyPolygon.shp` |
| Shipping routes | EMODnet HA EMSA | Route density raster TIF |
| Fishing intensity | EMODnet HA | Subsurface swept area ratio polygon |
| Harbours | EMODnet HA | Port goods traffic points |
| Pipelines | EMODnet HA | `pipelinesLine.shp` |
| Cables | EMODnet HA | Power + Telecommunication cables |
| Coastline | EEA v3035 | `EEA_Coastline_20170228.shp` |

---

## 8. CMIP6 Climate Scenarios

Delta-change method: `projected = CMEMS_baseline + (CMIP6_future − CMIP6_historical)`

| SSP | Label | Approx. warming |
|-----|-------|----------------|
| ssp1_2_6 | SSP1-2.6 | +1.8°C |
| ssp2_4_5 | SSP2-4.5 | +2.7°C |
| ssp3_7_0 | SSP3-7.0 | +3.6°C |
| ssp5_8_5 | SSP5-8.5 | +4.4°C |

Periods: `mid` (2041–2060), `end` (2071–2100). → 8 combinations × 4 variables = 32 scenario files.

---

## 9. App Architecture (MYTIGATE_futures Dash App)

### Entry point
`app/app.py` — runs at `http://127.0.0.1:8050`. Host must be `127.0.0.1` (not `0.0.0.0`) for local Windows operation.

### Layout (app/layout.py)
Dark-themed 3-panel layout:
- **Left sidebar (340px)** — 4-step workflow controls
- **Centre** — interactive Plotly Scattermap (`carto-positron`, `scrollZoom=True`, `drawclosedpath` enabled)
- **Right panel (320px)** — farm summary, clicked cell report, baseline vs scenario comparison

### 4-Step Sidebar Flow

**Step 0 — Scenario mode** (always visible at top)
- Radio: Baseline / CMIP6 scenario / Custom delta (slow)
- CMIP6 panel: SSP dropdown + Period dropdown + "Load scenario" button
- Delta panel: ΔTemp slider, ΔSal slider, ChlA× slider + "Run scenario" button

**Step 1 — Explore environment**
- Colour layer dropdown: No color / Exclusion zones / Temp / Sal / ChlA / Flow / Bathymetry
- Month dropdown (Aug–Dec)
- Startup default: plain basemap (no colour)
- "Exclusion zones" option shows red (excluded) / grey (available) dot map

**Step 2 — Draw farm & compute**
- Instructions: "Click the heart/lasso icon (♡), click and drag to draw polygon"
- Zoom warning: red (< 9), orange (9–10), green (≥ 10)
- Zoom guardrail: blocks Compute if zoom < 10
- Max collector depth slider: 2m / 4m / 6m / 8m (snapping)
- Loop interval slider: 0.7m / 1.0m / 1.5m / 2.0m (snapping)
- Compute button → M12 → M2 → M3 → M4 on farm_cells (right panel only)
- Map continues to show full baseline grid

**Step 3 — Visualise results**
- Result layer dropdown (unlocks after compute): N_red / Harvest WW / Shell DW / mbio p50 / vh_ratio / conflict_score
- Scenario mode (if active): Baseline ↔ Scenario map toggle appears

**Step 4 — Export**
- Env grid baseline CSV
- Results baseline CSV
- Farm summary CSV (geometry + aggregated results from right panel)
- Farm cell summary CSV (per-cell results for selected harvest month)
- Scenario exports (appear when scenario loaded)

### Map behaviour
- `uirevision="static"` — viewport never resets on figure update
- `preserve_viewport=True` on all re-renders — no zoom jitter
- Only `empty_map()` sets initial center/zoom (54.6–58°N, 7.8–13.1°E)
- Drawn polygon: paper-anchored shape stored in `store-drawn-shape`, cleared after compute
- Farm boundary: geo-anchored `Scattermap` line trace (union of farm cell geometries → WGS84), scales with zoom
- Map toggle (Baseline/Scenario): appears only after scenario is loaded AND compute done
- Crop toggle (Farm area/Full grid): appears after compute done

### Polygon drawing → coordinate conversion
Plotly `newshape` stores drawn paths as SVG in paper coordinates (0–1 fraction of figure). Conversion to lon/lat uses `map._derived` from `relayoutData`, which contains the actual tile bounds:
```python
lon = lon_min + px × (lon_max - lon_min)
lat = lat_min + py × (lat_max - lat_min)
```
Bounds stored in `store-map-viewport` whenever map pans/zooms.

### Server-side storage (app.server._*)
```python
_baseline       # full grid GeoDataFrame (baseline.gpkg) — map display
_base_env_grid  # grid_base.gpkg — used for scenario delta recompute
_scenario       # full grid GeoDataFrame (scenario env) — scenario map display
_farm_baseline  # farm_cells after M12+M2+M3+M4 — right panel summary
_farm_scenario  # farm_cells after scenario M12+M2+M3+M4 — right panel
_m12_result     # dict from discretise_farm() — farm geometry
_scenario_label # string label for active scenario
_drawn_shapes   # list of Plotly shape dicts — cleared after compute
```

### Critical architectural rule (matches pipeline_debug.ipynb exactly)
```
M12 on farm_cells     → farm geometry params (D_short, farm_area_m2, lcol_m)
M2 on farm_cells      → mbio per cell (for right panel)
M3 on farm_cells      → harvest/N_red per cell (for right panel)
M4 on farm_cells      → food limitation (for right panel)
MAP always uses _baseline (full grid from pipeline.py)
Farm cells are shown as a cyan boundary outline, NOT as coloured dots
```
**Never recompute M2/M3/M4 on the full grid in the app** — baseline.gpkg already has it from pipeline.py.

---

## 10. Debug Notebook (pipeline_debug.ipynb)

The notebook is the **authoritative reference** for app logic. It runs the exact same call sequence as the app callbacks. When debugging the app, compare callback behaviour to the corresponding notebook cell.

Key cells:
- Cell 1: grid loading
- Cell 2: M1 env sampling
- Cell 3: M11 exclusions
- Cell 4: M2 growth on full grid (produces spatial visualisation)
- Cell 5: M3 upscaling on full grid
- Cell 6: M4 food limitation on full grid
- Cell 7: M12 on MOCK_FARM_COORDS_LONLAT
- Cell 8: M2+M3+M4 on farm_cells → farm summary (right panel equivalent)
- Cells 9+: validation, surrounding cell analysis, zero-cell check

Mock farm in config (Vejle Fjord, ~154 ha):
```python
MOCK_FARM_COORDS_LONLAT = [
    (9.789647581800251,  55.57062803111441),
    (9.809943162740261,  55.57034022734398),
    (9.808489188666039,  55.561383235752025),
    (9.789311316885234,  55.56123503890327),
]
```

---

## 11. Setup for Students

Students receive precomputed data via OneDrive — they do NOT run the pipeline.

```bash
conda create -n mapping python=3.11
conda install -c conda-forge geopandas rasterio fiona pyogrio contextily
pip install -r requirements.txt
```

```
requirements.txt:
dash>=2.18
dash-bootstrap-components>=1.6
plotly>=6.0
xarray
netCDF4
scipy
tqdm
pyproj
shapely
```

Run app: `python app/app.py` → open `http://localhost:8050`

---

## 12. Known Issues and Design Notes

**Coordinate conversion accuracy:** Paper→lon/lat conversion uses `map._derived` tile bounds stored on pan/zoom. Pan or zoom the map before drawing to ensure bounds are captured. Accuracy improves at higher zoom levels.

**Farm scale:** The tool is scale-agnostic. Small polygon = small farm (~19 ha standard). Large polygon = megafarm (~2000 ha). n_sections scales linearly with `polygon_utm.area / 48,000 m²`.

**Limfjord/Wadden Sea:** CMEMS coverage gaps in these areas produce placeholder env values. A WARNING is printed; cells are not masked.

**CMIP6 SDs:** Standard deviation columns in CMIP6 scenario grids are unchanged from baseline (known simplification — delta-change only shifts means).

**M4 rho_col:** Computed from median lcol and farm_area_m2. For very small farms (few viable cells) this may be unreliable. Always check `median_vh_crit` in terminal output.

**Baltic/NWS blending zone:** 8.5–10°E for scalars; 7.5–11.5°E + Gaussian smooth (σ=3) for flow. Artefacts possible near blend boundaries.

**pycache:** After updating any `.py` file, delete `__pycache__` folders in `model/`, `app/`, `utils/` to force recompile. Python may load stale `.pyc` files otherwise.

---

## 13. File Checkpoint — v5

All files in `app/` and `model/` as of this handover:

| File | Location | Lines | Description |
|------|----------|-------|-------------|
| `app.py` | `app/` | 36 | Entry point, host=127.0.0.1:8050 |
| `layout.py` | `app/` | 563 | Full UI layout, 4-step sidebar |
| `callbacks.py` | `app/` | 924 | All Dash callbacks |
| `map_utils.py` | `app/` | 304 | Plotly Scattermap builders |
| `config.py` | root | 132 | All constants and paths |
| `config_scenario.py` | root | 111 | SSP labels, scenario paths |
| `pipeline.py` | root | 250 | Full pipeline Steps 1–11 |
| `raster_sampler.py` | `utils/` | 504 | Module 1: CMEMS sampling |
| `climate_projections.py` | `utils/` | 495 | CMIP6 delta-change |
| `growth.py` | `model/` | 378 | M2, M3, M4 |
| `module12_farm.py` | `model/` | 178 | M12 farm discretisation |
| `site_selector.py` | `model/` | 137 | Site ranking + cell report |
| `export_cell.py` | root | 77 | Notebook export helper |
| `pipeline_debug.ipynb` | `notebooks/` | — | Reference debug notebook |
| `scenario_test.ipynb` | `notebooks/` | — | CMIP6 validation notebook |
