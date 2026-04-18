"""Fetch all sector/industry mappings using vnstock_data (canonical provider)."""

import os

from src.data.adapters.vnstock_adapter import VnstockAdapter

def fetch_all_sectors():
    print("Fetching all industry sector mappings via vnstock_data...")
    try:
        df = VnstockAdapter().get_symbols_by_industries()
        if df is not None and not df.empty:
            # Keep industry columns if available
            industry_cols = [c for c in df.columns if "industry" in c.lower() or "sector" in c.lower() or "ticker" in c.lower()]
            if industry_cols:
                df_sectors = df[industry_cols].copy()
            else:
                df_sectors = df.copy()

            os.makedirs("data", exist_ok=True)
            df_sectors.to_csv("data/ticker_sectors.csv", index=False)
            print(f"Saved sector info for {len(df_sectors)} tickers to data/ticker_sectors.csv")
        else:
            print("No sector data returned.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_all_sectors()
