"""Extract hourly stock data into a SINGLE unified CSV file."""

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
logger = get_logger("csv_single_extractor")

async def main():
    parser = argparse.ArgumentParser(description="Extract a SINGLE CSV hourly market file")
    parser.add_argument("--tickers", nargs="+", help="Specific symbols")
    parser.add_argument("--all", action="store_true", help="All market symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days")
    parser.add_argument("--output", type=str, default="stock_hourly_unified_data.csv")
    
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

    logger.info("starting_unified_csv_extraction", count=len(symbols))
    
    all_data = []
    for i, ticker in enumerate(symbols):
        if i % 10 == 0:
            logger.info("progress", current=i, total=len(symbols), collected=len(all_data))
        
        df = await fetch_hourly_data(ticker, args.days)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                rec = row.to_dict()
                rec["ticker"] = ticker
                rec["llm_text"] = format_row_to_text(ticker, row)
                all_data.append(rec)
        
        await asyncio.sleep(0.2) # Throttling 300 req/min

    if all_data:
        logger.info("exporting_to_csv", path=args.output, count=len(all_data))
        df_final = pd.DataFrame(all_data)
        df_final.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n✅ Ngon lành! Đã gộp {len(all_data)} dòng vào 1 file CSV duy nhất: {args.output}")
    else:
        logger.warning("no_data_collected")

if __name__ == "__main__":
    asyncio.run(main())
