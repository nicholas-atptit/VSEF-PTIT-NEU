"""Validate VN30 daily 2015 data readiness."""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015"
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31")
EVAL_START = pd.Timestamp("2025-01-01")

def read_universe() -> list[str]:
    tickers = []
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker", "")).strip().upper()
            if t: tickers.append(t)
    return tickers

def validate_ticker(ticker: str) -> dict:
    path = CACHE_ROOT / f"{ticker}.csv"
    result = {"ticker": ticker, "exists": path.exists(), "rows": 0, "first_date": "", "last_date": "",
        "train_rows": 0, "val_rows": 0, "eval_rows": 0, "usable": False, "error": ""}
    if not path.exists():
        result["error"] = "file missing"
        return result
    try:
        df = pd.read_csv(path, parse_dates=["datetime"])
        result["rows"] = len(df)
        result["first_date"] = str(df["datetime"].min())
        result["last_date"] = str(df["datetime"].max())
        result["train_rows"] = int((df["datetime"] <= TRAIN_END).sum())
        result["val_rows"] = int(((df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)).sum())
        result["eval_rows"] = int((df["datetime"] >= EVAL_START).sum())
        result["usable"] = result["rows"] >= 100 and result["train_rows"] >= 50 and result["eval_rows"] >= 10
    except Exception as e:
        result["error"] = str(e)
    return result

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Data Readiness Validation")
    print("=" * 60)
    tickers = read_universe()
    print(f"Validating {len(tickers)} tickers...")
    results = [validate_ticker(t) for t in tickers]
    usable = [r for r in results if r["usable"]]
    missing = [r for r in results if not r["exists"]]
    print(f"\nUsable: {len(usable)}/{len(results)}")
    print(f"Missing: {len(missing)}/{len(results)}")
    for r in results:
        status = "USABLE" if r["usable"] else ("MISSING" if not r["exists"] else "INSUFFICIENT")
        print(f"  {r['ticker']}: {status} - {r['rows']} rows (train={r['train_rows']}, val={r['val_rows']}, eval={r['eval_rows']})")
    # Write readiness report
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    readiness_csv = REPORT_ROOT / "vn30_daily_2015_readiness.csv"
    fields = ["ticker", "exists", "rows", "first_date", "last_date", "train_rows", "val_rows", "eval_rows", "usable", "error"]
    with readiness_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    all_ready = len(usable) == len(tickers)
    manifest = {"tickers_total": len(tickers), "tickers_usable": len(usable), "tickers_missing": len(missing),
        "all_ready": all_ready, "validated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    with (REPORT_ROOT / "vn30_daily_2015_readiness.json").open("w") as f: json.dump(manifest, f, indent=2)
    print(f"\nAll ready: {all_ready}")
    return 0 if all_ready else 1

if __name__ == "__main__":
    raise SystemExit(main())
