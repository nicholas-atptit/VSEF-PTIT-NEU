import pandas as pd
from vnstock import Vnstock
import os
import time
import glob
from pathlib import Path
from tqdm import tqdm

def fetch_top_fundamentals(limit=300):
    v = Vnstock()
    data_dir = "data/daily_market_split_data"
    # Find all CSVs and sort by size (proxy for market interest/history)
    files = sorted(glob.glob(f"{data_dir}/*.csv"), key=lambda x: os.path.getsize(x), reverse=True)[:limit]
    tickers = [Path(f).stem for f in files]
    
    print(f"🚀 Fetching fundamentals for top {len(tickers)} tickers...")
    results = []
    
    for ticker in tqdm(tickers):
        try:
            # Fetch quarterly ratios from TCBS or VCI (VCI is more reliable in some versions)
            df = v.stock(symbol=ticker, source='VCI').finance.ratio(period='quarterly', lang='vi')
            if df is not None and not df.empty:
                # Take latest 4 quarters to get recent trend
                latest = df.iloc[-4:].copy()
                latest['ticker'] = ticker
                # Track the quarter/year if possible
                if 'Meta' in latest.columns and 'CP' in latest['Meta'].columns:
                     pass # Multi-index handling done later
                results.append(latest)
            time.sleep(0.3) # Rate limit to prevent IP block
        except Exception:
            continue
            
    if results:
        # Flatten Multi-index and combine
        master = pd.concat(results)
        if isinstance(master.columns, pd.MultiIndex):
            master.columns = ['_'.join(col).strip() for col in master.columns.values]
        
        # Save to data/
        os.makedirs('data', exist_ok=True)
        master.to_csv('data/fundamentals_latest.csv', index=False)
        print(f"✅ Saved fundamentals for {len(results)} tickers to data/fundamentals_latest.csv")
    else:
        print("❌ No fundamental data retrieved.")

if __name__ == "__main__":
    fetch_top_fundamentals(limit=300)
