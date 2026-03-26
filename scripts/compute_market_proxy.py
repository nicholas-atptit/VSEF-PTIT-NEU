import pandas as pd
import glob
from pathlib import Path
import os

def compute_proxy():
    print("🚀 Computing Market Proxy (Top 200 Tickers)...")
    data_dir = "data/daily_market_split_data"
    files = sorted(glob.glob(f"{data_dir}/*.csv"), key=lambda x: os.path.getsize(x), reverse=True)[:200]
    
    all_returns = []
    for f in files:
        df = pd.read_csv(f)
        if 'time' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'time': 'date'})
        if 'date' not in df.columns or 'close' not in df.columns:
            continue
        
        df = df.sort_values('date')
        df['m_ret'] = df['close'].pct_change()
        all_returns.append(df[['date', 'm_ret']])
    
    if not all_returns:
        print("❌ No data found for proxy.")
        return

    master = pd.concat(all_returns)
    proxy = master.groupby('date')['m_ret'].mean().reset_index()
    proxy.to_csv('data/market_proxy.csv', index=False)
    print(f"✅ Market Proxy saved to data/market_proxy.csv ({len(proxy)} rows)")

if __name__ == "__main__":
    compute_proxy()
