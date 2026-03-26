import pandas as pd
import glob
import os
from pathlib import Path

def compute_sector_proxies():
    print("🚀 Computing Sector Overlays...")
    sector_path = 'data/ticker_sectors.csv'
    if not os.path.exists(sector_path):
        print("❌ ticker_sectors.csv not found.")
        return
    
    sectors_df = pd.read_csv(sector_path)
    data_dir = "data/daily_market_split_data"
    
    all_data = []
    # Use top 500 tickers for sector indices (others might be too noisy/illiquid)
    files = sorted(glob.glob(f"{data_dir}/*.csv"), key=lambda x: os.path.getsize(x), reverse=True)[:500]
    
    for f in files:
        ticker = Path(f).stem
        if ticker not in sectors_df['ticker'].values:
            continue
        
        industry = sectors_df[sectors_df['ticker'] == ticker]['industry'].values[0]
        df = pd.read_csv(f)
        if 'time' in df.columns: df = df.rename(columns={'time': 'date'})
        
        df = df.sort_values('date')
        df['ret'] = df['close'].pct_change()
        df['industry'] = industry
        all_data.append(df[['date', 'industry', 'ret']])

    if not all_data:
        print("❌ No data for sectors.")
        return

    master = pd.concat(all_data)
    # Group by Date AND Industry
    sector_indices = master.groupby(['date', 'industry'])['ret'].mean().reset_index()
    sector_indices.to_csv('data/sector_proxies.csv', index=False)
    print(f"✅ Saved sector proxies to data/sector_proxies.csv ({len(sector_indices)} rows)")

if __name__ == "__main__":
    compute_sector_proxies()
