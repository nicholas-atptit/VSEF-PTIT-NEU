"""Build VN30 daily 2015 readiness manifest."""
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

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Readiness Manifest")
    print("=" * 60)
    tickers = read_universe()
    ticker_details = []
    for ticker in tickers:
        path = CACHE_ROOT / f"{ticker}.csv"
        detail = {"ticker": ticker, "exists": path.exists(), "rows": 0, "first_date": "", "last_date": "",
            "train_rows": 0, "val_rows": 0, "eval_rows": 0, "usable": False}
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["datetime"])
                detail["rows"] = len(df)
                detail["first_date"] = str(df["datetime"].min())
                detail["last_date"] = str(df["datetime"].max())
                detail["train_rows"] = int((df["datetime"] <= TRAIN_END).sum())
                detail["val_rows"] = int(((df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)).sum())
                detail["eval_rows"] = int((df["datetime"] >= EVAL_START).sum())
                detail["usable"] = detail["rows"] >= 100 and detail["train_rows"] >= 50 and detail["eval_rows"] >= 10
            except: pass
        ticker_details.append(detail)
    usable = [d for d in ticker_details if d["usable"]]
    manifest = {
        "track": "vn30_daily_2015",
        "universe": "VN30 January 2025 review",
        "frequency": "daily",
        "tickers_total": len(tickers),
        "tickers_usable": len(usable),
        "all_ready": len(usable) == len(tickers),
        "train_end": str(TRAIN_END),
        "val_start": str(VAL_START),
        "val_end": str(VAL_END),
        "eval_start": str(EVAL_START),
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tickers": ticker_details,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with (REPORT_ROOT / "vn30_daily_2015_readiness_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Total tickers: {len(tickers)}")
    print(f"Usable: {len(usable)}")
    print(f"All ready: {manifest['all_ready']}")
    print(f"Manifest: {REPORT_ROOT / 'vn30_daily_2015_readiness_manifest.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
