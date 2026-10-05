"""
utils/download_cds.py
---------------------
Downloads sea_surface_temperature (tos) and sea_surface_salinity (sos)
from Copernicus Climate Data Store (CDS) for:
  - historical  (1985–2014)
  - ssp126, ssp245, ssp370, ssp585  (2015–2100)

Output: data/processed/cmip6_raw/<experiment>_cds/ containing .nc files

Run standalone:
    python utils/download_cds.py
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED

try:
    import cdsapi
except ImportError:
    raise ImportError("cdsapi not installed — run: pip install cdsapi")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_CDS = "mpi_esm1_2_lr"
RAW_DIR   = PROCESSED / "cmip6_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "historical": ("1985", "2014"),
    "ssp126":     ("2015", "2100"),
    "ssp245":     ("2015", "2100"),
    "ssp370":     ("2015", "2100"),
    "ssp585":     ("2015", "2100"),
}


VARIABLES = {
    "sea_surface_temperature": "tos",
    "sea_surface_salinity":    "sos",
}


def download_cds_var(experiment: str, year_start: str, year_end: str,
                     cds_var: str, short_name: str) -> None:
    """Download one variable for one experiment. Skips if already present."""
    out_dir = RAW_DIR / f"{experiment}_cds"
    out_dir.mkdir(exist_ok=True)

    # Skip if this variable already exists in the folder
    existing = list(out_dir.glob(f"{short_name}_*.nc"))
    if existing:
        print(f"  [CDS] skip {experiment}/{short_name} ({len(existing)} files already exist)")
        return

    out_zip = RAW_DIR / f"{experiment}_{short_name}_cds.zip"
    print(f"  [CDS] Downloading {experiment}/{short_name} ({year_start}–{year_end})...")
    years  = [str(y) for y in range(int(year_start), int(year_end) + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    client = cdsapi.Client()
    client.retrieve(
        "projections-cmip6",
        {
            "download_format":     "zip",
            "data_format":         "netcdf_legacy",
            "temporal_resolution": "monthly",
            "experiment":          experiment,
            "variable":            cds_var,
            "model":               MODEL_CDS,
            "year":                years,
            "month":               months,
        },
        str(out_zip),
    )
    with zipfile.ZipFile(out_zip, "r") as z:
        z.extractall(out_dir)
    nc_files = list(out_dir.glob(f"{short_name}_*.nc"))
    print(f"  [CDS] Extracted {len(nc_files)} {short_name} files → {out_dir.name}/")
    out_zip.unlink()  # remove zip after extraction


def run_download_cds() -> None:
    print("=== CDS Download: tos + sos (separate requests) ===")
    for experiment, (y_start, y_end) in EXPERIMENTS.items():
        for cds_var, short_name in VARIABLES.items():
            download_cds_var(experiment, y_start, y_end, cds_var, short_name)
    print("=== CDS downloads complete ===")
    print(f"Files in: {RAW_DIR}")


if __name__ == "__main__":
    run_download_cds()