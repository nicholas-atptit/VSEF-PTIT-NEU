"""Fetch latest fundamental ratios from vnstock_data and cache them locally."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from src.data.adapters.vnstock_adapter import VnstockAdapter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def fetch_top_fundamentals(limit: int = 300) -> None:
    data_dir = "data/daily_market_split_data"
    files = sorted(glob.glob(f"{data_dir}/*.csv"), key=lambda path: os.path.getsize(path), reverse=True)[:limit]
    tickers = [Path(path).stem.upper() for path in files]

    if not tickers:
        print("No cached ticker universe found under data/daily_market_split_data.")
        return

    adapter = VnstockAdapter(symbol_list=tickers)
    results: list[pd.DataFrame] = []

    for ticker in tqdm(tickers, desc="Fetching finance.ratio"):
        frame = adapter.get_financial_ratios(ticker)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local["ticker"] = ticker
        results.append(local)

    if not results:
        print("No fundamental data retrieved from vnstock_data Finance.ratio.")
        return

    master = pd.concat(results, ignore_index=True)
    if isinstance(master.columns, pd.MultiIndex):
        master.columns = ["_".join(map(str, col)).strip("_") for col in master.columns.values]

    os.makedirs("data", exist_ok=True)
    master.to_csv("data/fundamentals_latest.csv", index=False)
    print(f"Saved fundamentals for {master['ticker'].nunique()} tickers to data/fundamentals_latest.csv")


if __name__ == "__main__":
    fetch_top_fundamentals(limit=300)
