"""Extract hourly stock data in JSONL format optimized for LLM training."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.historical.hourly_service import fetch_hourly_data, format_row_to_text
from src.utils.logging import setup_logging, get_logger
from vnstock_data import Listing

setup_logging()
logger = get_logger("jsonl_extractor")

async def main():
    parser = argparse.ArgumentParser(description="Extract JSONL for LLM training")
    parser.add_argument("--tickers", nargs="+", help="Specific symbols")
    parser.add_argument("--all", action="store_true", help="All market symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days")
    parser.add_argument("--output", type=str, default="stock_hourly_training_data.jsonl")
    
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

    logger.info("starting_jsonl_extraction", count=len(symbols))
    
    count_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, ticker in enumerate(symbols):
            if i % 10 == 0:
                logger.info("progress", current=i, total=len(symbols), written=count_written)
            
            df = await fetch_hourly_data(ticker, args.days)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    text = format_row_to_text(ticker, row)
                    entry = {
                        "text": text,
                        "metadata": {
                            "ticker": ticker,
                            "timestamp": str(row.get("time", "")),
                            "interval": "1H"
                        }
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count_written += 1
            
            await asyncio.sleep(0.2) # Throttling 300 req/min

    logger.info("finished", file=args.output, count=count_written)
    print(f"\n✅ Lụm lúa! Đã xuất {count_written} dòng vào file JSONL: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
