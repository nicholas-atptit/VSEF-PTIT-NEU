import os
import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import get_settings
from src.utils.logging import get_logger
import glob
from datetime import datetime

logger = get_logger(__name__)

def import_local_data():
    settings = get_settings()
    # Use sync URL for faster batch processing with psycopg2
    engine = create_engine(settings.timescale_sync_url)
    
    daily_path = "data/daily_market_split_data/*.csv"
    hourly_path = "data/hourly_market_split_data/*.csv"
    
    daily_files = glob.glob(daily_path)
    hourly_files = glob.glob(hourly_path)
    
    total_files = len(daily_files) + len(hourly_files)
    logger.info("starting_local_import", total_files=total_files)
    
    # 1. Daily Data Import
    for i, file_path in enumerate(daily_files):
        ticker = os.path.basename(file_path).replace(".csv", "")
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Normalize columns: time -> timestamp
            df = df.rename(columns={'time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['ticker'] = ticker
            df['timeframe'] = '1d'
            df['source'] = 'local_import'
            df['exchange'] = 'HOSE' # Default
            
            # Ensure all required columns exist
            required = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'timeframe', 'source', 'exchange']
            df = df[required]
            
            # Efficient insert
            df.to_sql('raw_prices', engine, if_exists='append', index=False, method='multi', chunksize=1000)
            
            if i % 100 == 0:
                logger.info("imported_daily", progress=f"{i}/{len(daily_files)}", ticker=ticker)
        except Exception as e:
            logger.error("failed_to_import_daily", ticker=ticker, error=str(e))

    # 2. Hourly Data Import
    for i, file_path in enumerate(hourly_files):
        ticker = os.path.basename(file_path).replace(".csv", "")
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            df = df.rename(columns={'time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['ticker'] = ticker
            df['timeframe'] = '1h'
            df['source'] = 'local_import'
            df['exchange'] = 'HOSE'
            
            # Check for llm_text column (store in metadata if table supports it, otherwise skip)
            required = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'timeframe', 'source', 'exchange']
            df_prices = df[required]
            
            df_prices.to_sql('raw_prices', engine, if_exists='append', index=False, method='multi', chunksize=1000)
            
            if i % 100 == 0:
                logger.info("imported_hourly", progress=f"{i}/{len(hourly_files)}", ticker=ticker)
        except Exception as e:
            logger.error("failed_to_import_hourly", ticker=ticker, error=str(e))

    logger.info("local_import_complete")

if __name__ == "__main__":
    import_local_data()
