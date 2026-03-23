from sqlalchemy import create_engine, text
from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

def sync_prices():
    settings = get_settings()
    engine = create_engine(settings.timescale_sync_url)
    
    query = """
    INSERT INTO adjusted_prices (
        timestamp, ticker, open, high, low, close, volume, 
        timeframe, exchange, source, adjustment_factor
    )
    SELECT 
        timestamp, ticker, open, high, low, close, volume, 
        timeframe, exchange, source, 1.0
    FROM raw_prices
    ON CONFLICT (timestamp, ticker) DO NOTHING;
    """
    
    try:
        with engine.connect() as conn:
            print("Syncing raw_prices to adjusted_prices...")
            res = conn.execute(text(query))
            conn.commit()
            print(f"Sync complete. Rows affected: {res.rowcount if hasattr(res, 'rowcount') else 'unknown'}")
    except Exception as e:
        print(f"Error syncing prices: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    sync_prices()
