import sys
from pathlib import Path
import datetime as dt

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.ml.data_loader import load_ohlcv_from_db
from config.settings import get_settings

def main():
    ticker = "VGI"
    settings = get_settings()
    print(f"Timescale Sync URL: {settings.timescale_sync_url}")
    
    try:
        df = load_ohlcv_from_db(ticker)
        if df is None:
            print("❌ load_ohlcv_from_db returned None")
        elif df.empty:
            print("❌ load_ohlcv_from_db returned EMPTY DataFrame")
        else:
            print(f"✅ load_ohlcv_from_db returned {len(df)} rows")
            print(f"Last date: {df['date'].iloc[-1]}")
            print(f"Last close: {df['close'].iloc[-1]}")
    except Exception as e:
        print(f"❌ Error calling load_ohlcv_from_db: {e}")

if __name__ == "__main__":
    main()
