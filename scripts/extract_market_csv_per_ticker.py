"""Extract hourly stock data in CSV format with decimal precision."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.historical.hourly_service import fetch_hourly_data, format_row_to_text
from src.utils.logging import setup_logging, get_logger
from vnstock_data import Listing

setup_logging()
logger = get_logger("csv_extractor")

async def main():
    parser = argparse.ArgumentParser(description="Extract CSV hourly market data")
    parser.add_argument("--tickers", nargs="+", help="Specific symbols")
    parser.add_argument("--all", action="store_true", help="All market symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days")
    parser.add_argument("--output", type=str, default="stock_hourly_market_data.csv")
    
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

    # Create output directory for individual CSVs
    output_dir_base = args.output.replace(".jsonl", "").replace(".csv", "")
    if not output_dir_base.endswith("_data"):
        output_dir_base = f"{output_dir_base}_data"
    output_dir = Path(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("starting_csv_extraction_split", count=len(symbols), output_dir=str(output_dir))
    
    count_files = 0
    total_rows = 0
    for i, ticker in enumerate(symbols):
        if i % 10 == 0:
            logger.info("progress", current=i, total=len(symbols), files_saved=count_files)
        
        df = await fetch_hourly_data(ticker, args.days)
        if df is not None and not df.empty:
            # Prepare data
            df["ticker"] = ticker
            # Add LLM text column just in case
            df["llm_text"] = df.apply(lambda row: format_row_to_text(ticker, row), axis=1)
            
            # Save individual file
            file_path = output_dir / f"{ticker}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            count_files += 1
            total_rows += len(df)
        
        await asyncio.sleep(0.2) # Throttling 300 req/min

    logger.info("finished", directory=str(output_dir), files=count_files, total_rows=total_rows)
    print(f"\n✅ Ngon lành! Đã xuất {total_rows} dòng vào {count_files} file CSV tại thư mục: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
