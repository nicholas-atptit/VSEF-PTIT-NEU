import pandas as pd
import os

def clean_fundamentals():
    print("🚀 Cleaning and Date-mapping Fundamentals...")
    fpath = 'data/fundamentals_latest.csv'
    if not os.path.exists(fpath):
        print("❌ fundamentals_latest.csv not found.")
        return
    
    df = pd.read_csv(fpath)
    
    # Map Year and Quarter to End-of-Quarter Date
    # Columns expected: Meta_Năm, Meta_Kỳ
    def get_date(row):
        year = int(row['Meta_Năm'])
        q = str(row['Meta_Kỳ'])
        if q == '1': return f"{year}-03-31"
        if q == '2': return f"{year}-06-30"
        if q == '3': return f"{year}-09-30"
        if q == '4': return f"{year}-12-31"
        return f"{year}-01-01"

    df['date'] = df.apply(get_date, axis=1)
    
    # Feature columns to keep (ROE, P/E, P/B, Debt/Equity)
    # We use these because they are universal
    cols_to_keep = ['ticker', 'date']
    
    # Look for specific keywords in cleaned column names
    price_cols = [c for c in df.columns if any(k in c.lower() for k in ['roe', 'p/e', 'p/b', 'nợ', 'biên', 'lợi nhuận'])]
    cols_to_keep.extend(price_cols)
    
    clean_df = df[cols_to_keep].copy()
    
    # Ensure all features are numeric
    for col in price_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    clean_df.dropna(subset=['ticker', 'date'], inplace=True)
    clean_df.to_csv('data/fundamentals_clean.csv', index=False)
    print(f"✅ Saved clean fundamentals to data/fundamentals_clean.csv ({len(clean_df)} rows)")

if __name__ == "__main__":
    clean_fundamentals()
