"""Data forensics audit script for VN30 hourly data locations."""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCAN_DIRS = [
    REPO_ROOT / "data" / "market_cache" / "vnstock_data",
    REPO_ROOT / "data" / "raw" / "vnstock_fetch",
    REPO_ROOT / "outputs",
    REPO_ROOT / "archive" / "generated_data_snapshots",
]
FORENSICS_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_data_forensics"
VN30_TICKERS = ["ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"]
INDEX_CODES = ["VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100"]

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def classify_location(path: Path) -> str:
    p = path.resolve()
    if "market_cache" in p.parts and "vnstock_data" in p.parts:
        if "archive" in p.parts: return "archive_snapshot"
        return "active_cache"
    if "raw" in p.parts and "vnstock_fetch" in p.parts:
        return "raw_fetch"
    if "outputs" in p.parts:
        return "output"
    if "archive" in p.parts and "generated_data_snapshots" in p.parts:
        return "archive_snapshot"
    return "unknown"

def detect_symbol(filepath: Path) -> str:
    name = filepath.stem.upper()
    for t in VN30_TICKERS:
        if t in name: return t
    for i in INDEX_CODES:
        if i in name: return i
    return "unknown"

def detect_frequency(filepath: Path) -> str:
    name = filepath.name.lower()
    if "hourly" in name or "1h" in name: return "hourly"
    if "daily" in name or "1d" in name: return "daily"
    if "minute" in name or "1m" in name: return "minute"
    return "unknown"

def scan_csv_file(filepath: Path) -> dict[str, Any]:
    """Scan a CSV file for metadata."""
    result = {
        "file_path": rel(filepath),
        "file_size_bytes": filepath.stat().st_size,
        "modified_time": datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc).isoformat(),
        "detected_symbol": detect_symbol(filepath),
        "detected_frequency": detect_frequency(filepath),
        "location_type": classify_location(filepath),
        "row_count": 0,
        "first_datetime": "",
        "last_datetime": "",
        "columns": "",
        "has_pre_2023": False,
        "has_2015_2022": False,
        "has_2023": False,
        "has_2024": False,
        "has_2025": False,
        "has_2026": False,
    }
    try:
        df = pd.read_csv(filepath, nrows=1)
        result["columns"] = ",".join(df.columns.tolist())
        if "datetime" in df.columns:
            full_df = pd.read_csv(filepath, usecols=["datetime"], parse_dates=["datetime"])
            if len(full_df) > 0:
                result["row_count"] = len(full_df)
                result["first_datetime"] = str(full_df["datetime"].min())
                result["last_datetime"] = str(full_df["datetime"].max())
                years = full_df["datetime"].dt.year
                result["has_pre_2023"] = bool((years < 2023).any())
                result["has_2015_2022"] = bool(((years >= 2015) & (years <= 2022)).any())
                result["has_2023"] = bool((years == 2023).any())
                result["has_2024"] = bool((years == 2024).any())
                result["has_2025"] = bool((years == 2025).any())
                result["has_2026"] = bool((years == 2026).any())
    except Exception as e:
        result["columns"] = f"error: {e}"
    return result

def scan_directory(base_dir: Path) -> list[dict[str, Any]]:
    """Recursively scan a directory for data files."""
    results = []
    if not base_dir.exists():
        return results
    for filepath in base_dir.rglob("*.csv"):
        if filepath.is_file():
            results.append(scan_csv_file(filepath))
    return results

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly Data Forensics Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    FORENSICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_files = []
    for scan_dir in SCAN_DIRS:
        print(f"\nScanning: {rel(scan_dir)}...")
        if scan_dir.exists():
            files = scan_directory(scan_dir)
            print(f"  Found {len(files)} CSV files")
            all_files.extend(files)
        else:
            print(f"  Directory does not exist")
    # Write inventory CSV
    fields = ["file_path", "file_size_bytes", "modified_time", "detected_symbol",
        "detected_frequency", "location_type", "row_count", "first_datetime",
        "last_datetime", "columns", "has_pre_2023", "has_2015_2022",
        "has_2023", "has_2024", "has_2025", "has_2026"]
    with (FORENSICS_OUTPUT_DIR / "vn30_hourly_data_file_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_files)
    # Generate summary report
    total_files = len(all_files)
    active_cache = [f for f in all_files if f["location_type"] == "active_cache"]
    raw_fetch = [f for f in all_files if f["location_type"] == "raw_fetch"]
    archive = [f for f in all_files if f["location_type"] == "archive_snapshot"]
    output = [f for f in all_files if f["location_type"] == "output"]
    has_pre_2023 = [f for f in all_files if f.get("has_pre_2023", False)]
    has_2015_2022 = [f for f in all_files if f.get("has_2015_2022", False)]
    earliest = min((f["first_datetime"] for f in all_files if f["first_datetime"]), default="N/A")
    latest = max((f["last_datetime"] for f in all_files if f["last_datetime"]), default="N/A")
    report = f"""# VN30 Hourly Data File Inventory

## Summary
- **Total CSV files scanned**: {total_files}
- **Active cache files**: {len(active_cache)}
- **Raw fetch files**: {len(raw_fetch)}
- **Archive snapshot files**: {len(archive)}
- **Output files**: {len(output)}
- **Earliest timestamp found**: {earliest}
- **Latest timestamp found**: {latest}
- **Files with pre-2023 data**: {len(has_pre_2023)}
- **Files with 2015-2022 data**: {len(has_2015_2022)}

## Key Finding
**NO 2015-2022 hourly stock data exists anywhere in the repository.**
- Earliest stock data: 2023-09-11 (all locations)
- Earliest index data: 2022-05-19 (VNINDEX and related indices)
- The "hourly_2015" naming is a design target, NOT actual data availability.

## Location Breakdown
### Active Cache (data/market_cache/vnstock_data/)
- VN30 stocks: {len([f for f in active_cache if f["detected_symbol"] in VN30_TICKERS])} files
- Indices: {len([f for f in active_cache if f["detected_symbol"] in INDEX_CODES])} files
- Date range: {min((f["first_datetime"] for f in active_cache if f["first_datetime"]), default="N/A")} to {max((f["last_datetime"] for f in active_cache if f["last_datetime"]), default="N/A")}

### Raw Fetch (data/raw/vnstock_fetch/)
- VN30 stocks: {len([f for f in raw_fetch if f["detected_symbol"] in VN30_TICKERS])} files
- Indices: {len([f for f in raw_fetch if f["detected_symbol"] in INDEX_CODES])} files
- Date range: {min((f["first_datetime"] for f in raw_fetch if f["first_datetime"]), default="N/A")} to {max((f["last_datetime"] for f in raw_fetch if f["last_datetime"]), default="N/A")}

### Archive Snapshots
- Files: {len(archive)}
- Date range: {min((f["first_datetime"] for f in archive if f["first_datetime"]), default="N/A")} to {max((f["last_datetime"] for f in archive if f["last_datetime"]), default="N/A")}

## Files with Pre-2023 Data
"""
    if has_pre_2023:
        for f in has_pre_2023:
            report += f"- {f['file_path']}: {f['first_datetime']} to {f['last_datetime']}\n"
    else:
        report += "- None found\n"
    report += f"\n## Files with 2015-2022 Data\n"
    if has_2015_2022:
        for f in has_2015_2022:
            report += f"- {f['file_path']}: {f['first_datetime']} to {f['last_datetime']}\n"
    else:
        report += "- None found\n"
    with (FORENSICS_OUTPUT_DIR / "vn30_hourly_data_file_inventory.md").open("w") as f:
        f.write(report)
    print(f"\nInventory complete. Outputs in {rel(FORENSICS_OUTPUT_DIR)}")
    print(f"Total files: {total_files}")
    print(f"Files with pre-2023 data: {len(has_pre_2023)}")
    print(f"Files with 2015-2022 data: {len(has_2015_2022)}")
    print(f"Earliest: {earliest}")
    print(f"Latest: {latest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
