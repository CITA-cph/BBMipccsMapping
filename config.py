"""
config.py
---------
Central configuration: all paths, grid constants, model parameters.

ARCHITECTURE CHANGE: Removed SCENARIO_PRESETS and scenario_gpkg().
Scenarios are now runtime multipliers applied to Module 1 outputs,
not precomputed files. See utils/raster_sampler.py run_module1().

Edit RAW path (line 17) to point at your data folder before running.
"""

from pathlib import Path
import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Data directories (edit RAW if your data is outside the project folder) ────
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW_BATHY    = RAW / "bathymetry"
RAW_HYDRO    = RAW / "hydrodynamics"
RAW_CONFLICT = RAW / "conflict_layers"

# ── Bathymetry ─────────────────────────────────────────────────────────────────
BATHY_PATH = RAW_BATHY / "emodnet_dtm.tif"

# ── CMEMS Baltic products (lon 9–17°E) ────────────────────────────────────────
CMEMS_PHY_DIR  = RAW_HYDRO / "cmems_phy"       # thetao, so
CMEMS_BGC_DIR  = RAW_HYDRO / "cmems_bgc"       # chl
CMEMS_FLOW_DIR = RAW_HYDRO / "cmems_flow"      # uo, vo

# ── CMEMS NWS products (lon 5–10°E, fills Jutland/Limfjord) ───────────────────
CMEMS_NWS_PHY_DIR  = RAW_HYDRO / "cmems_nws_phy"
CMEMS_NWS_BGC_DIR  = RAW_HYDRO / "cmems_nws_bgc"
CMEMS_NWS_FLOW_DIR = RAW_HYDRO / "cmems_nws_flow"

# ── Conflict / exclusion layers (exact subfolder names, nothing renamed) ───────
COASTLINE_SHP    = RAW_CONFLICT / "eea_v_3035_100_k_coastline-poly_p_1995-2017_v03_r00" / "EEA_Coastline_20170228.shp"
MPA_SHP          = RAW_CONFLICT / "HELCOM_MPAs_2019_2" / "HELCOM_MPAs_2019_2.shp"
MILITARY_SHP     = RAW_CONFLICT / "Military Areas (Polygons)" / "militaryareaspolyPolygon.shp"
WINDFARMS_SHP    = RAW_CONFLICT / "Wind Farms (polygons)" / "windfarmspolyPolygon.shp"
FISHING_SHP      = RAW_CONFLICT / "Average Subsurface Swept Area Ratio" / "fishingsubsurfacePolygon.shp"
HARBOURS_SHP     = RAW_CONFLICT / "Main ports (goods traffic 1997-2025)" / "portgoodsPoint.shp"
PIPELINES_SHP    = RAW_CONFLICT / "Pipelines" / "pipelinesLine.shp"
CABLES_DIRS      = [
    RAW_CONFLICT / "Power Cables",
    RAW_CONFLICT / "Telecommunication Cables",
]
SHIPPING_TIF_DIR = RAW_CONFLICT / "EMODnet_HA_EMSA_Route_Density_all_Yearly"

# ── Grid (UTM32N, 1km cells) ───────────────────────────────────────────────────
# Extended beyond Holbach 2020 (X_MIN was 400,000) to include:
#   North Sea / Jutland west coast, Limfjord, Wadden Sea, Skagerrak
GRID_CRS  = "EPSG:32632"
CELL_SIZE = 1_000           # metres — 1km x 1km
X_MIN, X_MAX = 442_000, 752_000
Y_MIN, Y_MAX = 6_054_000, 6_432_000

# ── Processed file paths ───────────────────────────────────────────────────────
GRID_GPKG          = PROCESSED / "grid_base.gpkg"
BASELINE_GPKG      = PROCESSED / "baseline.gpkg"

# ── Per-variable env caches (scenario-aware) ───────────────────────────────────
# Each file holds cell_id + that variable's mean/sd columns + geometry.
# Scenario recompute loads only the caches for unchanged variables,
# resamples CMEMS only for variables that are perturbed.
# Flow is also scenario-variable (wind-driven circulation may change under SSP).
GRID_ENV_TEMP_GPKG = PROCESSED / "grid_env_temp.gpkg"
GRID_ENV_SAL_GPKG  = PROCESSED / "grid_env_sal.gpkg"
GRID_ENV_CHLA_GPKG = PROCESSED / "grid_env_chla.gpkg"
GRID_ENV_FLOW_GPKG = PROCESSED / "grid_env_flow.gpkg"

# ── DEB growth model parameters (Holbach 2020, Eqs 3–6) ───────────────────────
TA, T1        = 5800, 289
TL, TH        = 275, 296
TAL, TAH      = 45430, 31376
SAL_THRESHOLD = 16.2
CHLA_ZERO     = 0.2
CHLA_SAT      = 20.0
CHLA_HALF_SAT = 0.8
GROWTH_COEF   = 0.0190
GROWTH_EXP    = 2.71
CONST_RHO     = 1269        # biomass density fitted constant
DW_TO_WW      = 9.68        # dry weight → fresh weight
DW_TO_N       = 0.14        # 14% N content
DW_TO_P       = 0.008       # 0.8% P content
# Shell fraction of whole-mussel dry weight (tissue + shell + byssus).
# ~0.77 for Mytilus edulis in Baltic conditions (Smaal & Vonck 1997;
# Taylor et al. 2019). Used to derive shell_DW for downstream applications.
SHELL_FRACTION = 0.77
MIN_DEPTH_M   = 4.0         # minimum viable farm depth [m]

# ── Farm geometry parameters (exposed — user can override in UI) ───────────────
# Defaults from Holbach 2020 Section 2.5, based on operational Danish farms.
# ARCHITECTURE CHANGE: previously hardcoded as FARM_AREA_HA=18.8, D_SHORT=250,
# D_LONG=750. Now computed dynamically in model/module12_farm.py from these.
LONGLINE_LENGTH       = 200   # m  — length of each longline
LONGLINE_SPACING      = 8     # m  — distance between longlines
LONGLINES_PER_SECTION = 30    # number of longlines per section
N_SECTIONS_DEFAULT    = 3     # default sections for standard farm

# Derived geometry (computed here for reference, recomputed dynamically in M12)
# D_short = LONGLINES_PER_SECTION * LONGLINE_SPACING  = 240m  (current direction)
# D_long  = LONGLINE_LENGTH                            = 200m  (across current)
# FARM_AREA = N_SECTIONS * D_short * D_long            = 3 * 240 * 200 = 144,000 m²

# ── C:ChlA monthly ratios (Jacobsen & Markager 2016) ─────────────────────────
C_CHLA_MONTHLY = np.array([32,32,32,30,24,23,22,23,32,43,53,61])  # Jan–Dec

# ── Harvest and growth season ──────────────────────────────────────────────────
HARVEST_MONTHS   = [8, 9, 10, 11, 12]   # Aug–Dec
PRIMARY_HARVEST  = 11                    # November default
SETTLEMENT_MONTH = 7                     # July

# ── Monte Carlo ────────────────────────────────────────────────────────────────
N_SCENARIOS = 50   # reduce to 50 for fast debug runs

# ── Mock farm polygon (WGS84 lon/lat) for notebook testing ────────────────────
# Located in Vejle Fjord area, Denmark
# ARCHITECTURE CHANGE: replaces hardcoded farm geometry assumptions
MOCK_FARM_COORDS_LONLAT = [
    (9.789647581800251,  55.57062803111441),
    (9.809943162740261,  55.57034022734398),
    (9.808489188666039,  55.561383235752025),
    (9.789311316885234,  55.56123503890327),
]

MAPBOX_TOKEN = "pk.eyJ1IjoiZ2Fiem9saW5hIiwiYSI6ImNtdTN4MnVkazA3YjAyd3F5OTBoNXJ6emsifQ.QuwLxbCjgeoIuFPTB0Sj8w"
CARTO_TOKEN = "cb1_3n4m_1_5d09007e890f8c38523b49bd"