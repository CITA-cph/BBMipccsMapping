"""
utils/download_esgf.py
----------------------
Runs the DKRZ wget script to download uo, vo, chl from ESGF,
then sorts the downloaded .nc files into the correct subfolders.

Expected wget script locations:
    utils/wget_script_2026-10-4_18-15-26.sh
    utils/wget_script_2026-10-4_18-15-50.sh
    utils/wget_script_2026-10-4_18-16-19.sh

Output folders under data/processed/cmip6_raw/:
    historical_esgf_uo/   historical_esgf_vo/   historical_esgf_chl/
    ssp1_2_6_esgf_uo/     ssp1_2_6_esgf_vo/     ssp1_2_6_esgf_chl/
    ssp2_4_5_esgf_uo/     ssp2_4_5_esgf_vo/     ssp2_4_5_esgf_chl/
    ssp3_7_0_esgf_uo/     ssp3_7_0_esgf_vo/     ssp3_7_0_esgf_chl/
    ssp5_8_5_esgf_uo/     ssp5_8_5_esgf_vo/     ssp5_8_5_esgf_chl/

Run standalone:
    python utils/download_esgf.py
"""

import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DIR     = PROCESSED / "cmip6_raw"
WGET_SCRIPTS = [
    Path(__file__).resolve().parent / "wget_script_2026-10-4_19-14-15.sh"
]
WGET_DIR = RAW_DIR / "wget_downloads"   # wget runs here, files land here

# Maps experiment name (as it appears in filename) → (variable → dest folder)
SORT_MAP = {
    "historical": {
        "uo":  RAW_DIR / "historical_esgf_uo",
        "vo":  RAW_DIR / "historical_esgf_vo",
        "chl": RAW_DIR / "historical_esgf_chl",
    },
    "ssp126": {
        "uo":  RAW_DIR / "ssp1_2_6_esgf_uo",
        "vo":  RAW_DIR / "ssp1_2_6_esgf_vo",
        "chl": RAW_DIR / "ssp1_2_6_esgf_chl",
    },
    "ssp245": {
        "uo":  RAW_DIR / "ssp2_4_5_esgf_uo",
        "vo":  RAW_DIR / "ssp2_4_5_esgf_vo",
        "chl": RAW_DIR / "ssp2_4_5_esgf_chl",
    },
    "ssp370": {
        "uo":  RAW_DIR / "ssp3_7_0_esgf_uo",
        "vo":  RAW_DIR / "ssp3_7_0_esgf_vo",
        "chl": RAW_DIR / "ssp3_7_0_esgf_chl",
    },
    "ssp585": {
        "uo":  RAW_DIR / "ssp5_8_5_esgf_uo",
        "vo":  RAW_DIR / "ssp5_8_5_esgf_vo",
        "chl": RAW_DIR / "ssp5_8_5_esgf_chl",
    },
}


def already_downloaded() -> bool:
    """
    Check if all expected destination folders have .nc files.
    Also returns True if files are sitting unsorted in wget_downloads
    (sort_files will handle them without re-downloading).
    """
    all_sorted = all(
        dest.exists() and list(dest.glob("*.nc"))
        for exp_map in SORT_MAP.values()
        for dest in exp_map.values()
    )
    if all_sorted:
        return True
    # Files may be downloaded but unsorted — check wget_downloads
    unsorted = list(WGET_DIR.glob("*.nc")) if WGET_DIR.exists() else []
    if unsorted:
        print(f"  [ESGF] {len(unsorted)} files in wget_downloads — skipping download, will sort")
        return True
    return False


def _find_bash() -> str:
    """Find Git Bash executable."""
    bash_paths = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        "bash",
    ]
    for bp in bash_paths:
        try:
            subprocess.run([bp, "--version"], capture_output=True, check=True)
            return bp
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("bash not found. Install Git for Windows.")


def run_wget() -> None:
    """Run all DKRZ wget scripts via Git Bash."""
    missing = [s for s in WGET_SCRIPTS if not s.exists()]
    if missing:
        raise FileNotFoundError(
            f"wget scripts not found:\n" + "\n".join(str(s) for s in missing)
        )

    WGET_DIR.mkdir(parents=True, exist_ok=True)
    bash = _find_bash()

    # Check wget is available
    wget_check = subprocess.run(
        [bash, "-c", "which wget || echo MISSING"],
        capture_output=True, text=True,
    )
    if "MISSING" in wget_check.stdout:
        raise RuntimeError(
            "wget not found in Git Bash.\n"
            "Run in Git Bash: curl -L https://eternallybored.org/misc/wget/1.21.4/64/wget.exe -o /usr/bin/wget.exe"
        )

    DKRZ_USER = "user"
    DKRZ_PASS = "pass"
    posix_dir = WGET_DIR.as_posix()

    # Run all scripts in parallel for ~3x speed
    print(f"  [ESGF] Running {len(WGET_SCRIPTS)} scripts in parallel...")
    processes = []
    for script in WGET_SCRIPTS:
        posix_script = script.as_posix()
        # Pass credentials via printf pipe to avoid interactive prompt
        cmd = (
            f"printf '{DKRZ_USER}\\n{DKRZ_PASS}\\n' | "
            f"bash '{posix_script}' -H "
            f"--no-check-certificate "
            f"-d '{posix_dir}'"
        )
        p = subprocess.Popen([bash, "-c", cmd], cwd=posix_dir)
        processes.append((script.name, p))

    for name, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"  [ESGF] WARNING: {name} exited with code {p.returncode}")
        else:
            print(f"  [ESGF] {name} complete")


def sort_files() -> None:
    """Sort downloaded .nc files from WGET_DIR into correct subfolders."""
    nc_files = list(WGET_DIR.glob("*.nc"))
    print(f"  [SORT] Found {len(nc_files)} .nc files to sort")

    unsorted = []
    for nc_file in nc_files:
        name = nc_file.name
        sorted_ = False
        for exp, var_map in SORT_MAP.items():
            if exp in name:
                for var, dest in var_map.items():
                    if name.startswith(var + "_"):
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(nc_file), dest / name)
                        print(f"  [SORT] {name} → {dest.name}/")
                        sorted_ = True
                        break
            if sorted_:
                break
        if not sorted_:
            unsorted.append(name)

    if unsorted:
        print(f"  [SORT] {len(unsorted)} files not sorted (not needed):")
        for f in unsorted[:5]:
            print(f"    {f}")
        if len(unsorted) > 5:
            print(f"    ... and {len(unsorted)-5} more")


def run_download_esgf() -> None:
    print("=== ESGF Download: uo + vo + chl ===")

    # Always sort if there are unsorted files in wget_downloads
    unsorted = list(WGET_DIR.glob("*.nc")) if WGET_DIR.exists() else []
    if unsorted:
        print(f"  [ESGF] {len(unsorted)} unsorted files found — sorting now...")
        sort_files()

    # Check if all destination folders are populated
    all_sorted = all(
        dest.exists() and list(dest.glob("*.nc"))
        for exp_map in SORT_MAP.values()
        for dest in exp_map.values()
    )
    if all_sorted:
        print("  [ESGF] All folders populated — done")
        return

    # Still missing files — run wget
    run_wget()
    sort_files()

    print("=== ESGF downloads complete ===")
    for exp, var_map in SORT_MAP.items():
        for var, dest in var_map.items():
            n = len(list(dest.glob("*.nc"))) if dest.exists() else 0
            print(f"  {dest.name}: {n} files")


if __name__ == "__main__":
    run_download_esgf()