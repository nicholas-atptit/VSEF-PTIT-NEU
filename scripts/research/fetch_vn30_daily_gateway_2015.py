"""Fetch VN30 stock daily OHLCV from 2015 via provider gateway."""
from __future__ import annotations
import argparse, csv, sys, time
from datetime import date, datetime
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.providers.vn_price_gateway import fetch_price_history
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName

CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015" / "fetch"
SOURCES = (SourceName.KBS, SourceName.VCI)

def read_universe() -> list[str]:
    tickers = []
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker", "")).strip().upper()
            if t: tickers.append(t)
    return tickers

def cache_path(ticker: str) -> Path:
    return CACHE_ROOT / f"{ticker}.csv"

def fetch_ticker_daily(ticker: str, start: str = "2015-01-01", end: str = "2026-12-31") -> dict:
    """Fetch daily data for a single ticker."""
    result = {"ticker": ticker, "rows": 0, "first_datetime": "", "last_datetime": "", "status": "failed", "error": ""}
    try:
        req = FetchRequest(
            symbol=ticker,
            asset_type=AssetType.STOCK,
            start=start,
            end=end,
            frequency=Frequency.DAILY,
            preferred_sources=SOURCES,
        )
        resp = fetch_price_history(req)
        if resp.data is not None and len(resp.data) > 0:
            df = resp.data.copy()
            df["ticker"] = ticker
            df["frequency"] = "1D"
            if "datetime" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "datetime"})
            cols = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
            for c in ["provider", "source"]:
                if c not in df.columns: df[c] = getattr(resp, c, "")
            df = df[[c for c in cols + ["provider", "source", "frequency"] if c in df.columns]]
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            df = df[(df["close"] > 0) & (df["volume"] >= 0)]
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path(ticker), index=False)
            result.update({
                "rows": len(df),
                "first_datetime": str(df["datetime"].min()),
                "last_datetime": str(df["datetime"].max()),
                "status": "success",
            })
        else:
            result["error"] = "empty response"
    except Exception as e:
        result["error"] = str(e)
    return result

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Gateway Fetch")
    print("=" * 60)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    args = parser.parse_args()
    started = time.monotonic()
    tickers = [args.ticker] if args.ticker else read_universe()
    print(f"Fetching {len(tickers)} tickers from {args.start} to {args.end}")
    results = []
    for i, ticker in enumerate(tickers):
        if time.monotonic() - started > args.max_runtime_seconds:
            print(f"Runtime cap reached at {ticker}")
            break
        print(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
        r = fetch_ticker_daily(ticker, args.start, args.end)
        results.append(r)
        print(f"  {r['status']}: {r['rows']} rows ({r['first_datetime']} to {r['last_datetime']})")
        time.sleep(0.5)
    # Write summary
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = REPORT_ROOT / "vn30_daily_2015_fetch_summary.csv"
    fields = ["ticker", "rows", "first_datetime", "last_datetime", "status", "error"]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    success = [r for r in results if r["status"] == "success"]
    print(f"\nDone. {len(success)}/{len(results)} tickers fetched successfully.")
    total_rows = sum(r["rows"] for r in success)
    print(f"Total rows: {total_rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
