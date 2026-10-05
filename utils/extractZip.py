import zipfile
from pathlib import Path

RAW_DIR = Path(r"C:\Users\garo\OneDrive - Det Kongelige Akademi\AP\03_teaching\0_CiA\2627\SEM1_workshop2\code_claude\data\processed\cmip6_raw")

for zip_path in RAW_DIR.glob("*_cds.zip"):
    out_dir = RAW_DIR / zip_path.stem
    out_dir.mkdir(exist_ok=True)
    print(f"Extracting {zip_path.name} → {out_dir.name}/")
    with zipfile.ZipFile(zip_path, "r") as z:
        print(f"  Contents: {z.namelist()}")
        z.extractall(out_dir)
    print(f"  Done: {list(out_dir.glob('*.nc'))}")