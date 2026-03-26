from vnstock import Vnstock
import pandas as pd
import os

def fetch_all_sectors():
    v = Vnstock()
    print("🚀 Fetching all industry sector mappings...")
    try:
        # One-call to get all symbols mapped to industries
        df = v.stock(symbol='HPG').listing.symbols_by_industries()
        if df is not None and not df.empty:
            # Rename for consistency
            df = df.rename(columns={'symbol': 'ticker', 'industry_name': 'industry'})
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/ticker_sectors.csv', index=False)
            print(f"✅ Saved sector info for {len(df)} tickers to data/ticker_sectors.csv")
        else:
            print("❌ No sector data returned.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fetch_all_sectors()
