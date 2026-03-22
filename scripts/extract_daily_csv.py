"""Extract DAILY stock data per ticker for multi-timeframe ML training."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
import datetime as dt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from vnstock import Vnstock
from src.utils.logging import setup_logging, get_logger
from vnstock_data import Listing

setup_logging()
logger = get_logger("daily_extractor")


async def fetch_daily_data(ticker: str, days: int = 730) -> pd.DataFrame | None:
    """Fetch daily history for a ticker."""
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days)
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1D"
        )
        return df
    except Exception as e:
        logger.debug("daily_fetch_error", ticker=ticker, error=str(e))
        return None


async def main():
    parser = argparse.ArgumentParser("Extract daily stock data per ticker")
    parser.add_argument("--tickers", nargs="+", help="Specific symbols")
    parser.add_argument("--all", action="store_true", help="All market symbols")
    parser.add_argument("--days", type=int, default=1825, help="Lookback days (default 5 years)")
    parser.add_argument("--output", type=str, default="data/daily_market_split")
    args = parser.parse_args()

    symbols = []
    if args.tickers:
        symbols = [s.upper() for s in args.tickers]
    elif args.all:
        try:
            df_listing = Listing(source='vnd').all_symbols()
            ticker_col = 'ticker' if 'ticker' in df_listing.columns else df_listing.columns[0]
            symbols = df_listing[ticker_col].tolist()
        except Exception as e:
            logger.error("failed_to_get_symbols", error=str(e))
            return

    if not symbols:
        logger.error("no_symbols")
        return

    output_dir_base = args.output.replace(".csv", "")
    if not output_dir_base.endswith("_data"):
        output_dir_base = f"{output_dir_base}_data"
    output_dir = Path(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("starting_daily_extraction", count=len(symbols), output_dir=str(output_dir))
    print(f"🚀 Bắt đầu tải {len(symbols)} mã dữ liệu Daily ({args.days} ngày)...")

    count_files = 0
    total_rows = 0
    for i, ticker in enumerate(symbols):
        if i % 10 == 0:
            logger.info("progress", current=i, total=len(symbols), files_saved=count_files)
            if i > 0:
                print(f"  ⏳ {i}/{len(symbols)} – Đã lưu {count_files} files...")

        df = await fetch_daily_data(ticker, args.days)
        if df is not None and not df.empty:
            df["ticker"] = ticker
            file_path = output_dir / f"{ticker}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            count_files += 1
            total_rows += len(df)

        await asyncio.sleep(0.2)  # 300 req/min

    logger.info("finished", directory=str(output_dir), files=count_files, total_rows=total_rows)
    print(f"\n✅ Xong! Đã xuất {total_rows} dòng vào {count_files} file CSV Daily tại: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
