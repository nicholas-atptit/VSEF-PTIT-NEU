import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Thêm src vào path để import
sys.path.append(os.getcwd())

from src.ml.data_loader import VN100DataLoader, apply_context_features
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer import DualModelTrainer
from scripts.train_ml_tickers import build_daily_features, build_hourly_features

def load_market_proxy():
    p = Path("data/market_proxy.csv")
    return pd.read_csv(p) if p.exists() else None

def load_sector_proxies():
    p = Path("data/sector_proxies.csv")
    return pd.read_csv(p) if p.exists() else None

def load_ticker_sectors():
    p = Path("data/ticker_sectors.csv")
    return pd.read_csv(p) if p.exists() else None

def main():
    parser = argparse.ArgumentParser(description="High-Fidelity Training vs Inference Parity Check")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker to check (e.g. FPT)")
    parser.add_argument("--days", type=int, default=300, help="Lookback days for inference dataset")
    parser.add_argument("--join-market", action="store_true", help="Join market features")
    parser.add_argument("--join-sectors", action="store_true", help="Join sector features")
    parser.add_argument("--join-fundamentals", action="store_true", help="Join fundamentals")
    parser.add_argument("--join-sentiment", action="store_true", help="Join sentiment")
    parser.add_argument("--strict", action="store_true", help="Fail if ANY extra/missing features")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root_dir = Path(".")
    daily_dir = Path("data/daily_market_split_data")
    
    print(f"=== PARITY CHECK: {ticker} ===")
    
    # 1. TRAINING PATH RECONSTRUCTION
    print(f"\n[1/3] Reconstructing Training Path...")
    fpath = daily_dir / f"{ticker}.csv"
    if not fpath.exists():
        print(f"[ERR]: No raw data found for {ticker}")
        sys.exit(1)
        
    raw_df = pd.read_csv(fpath)
    if 'time' in raw_df.columns and 'date' not in raw_df.columns:
        raw_df = raw_df.rename(columns={'time': 'date'})
        
    # Step A: Context Merges (Market, Sector) - MUST COME BEFORE technical features for parity
    m_df = load_market_proxy() if args.join_market else None
    s_df = load_sector_proxies() if args.join_sectors else None
    t_sectors = load_ticker_sectors() if args.join_sectors else None
    
    train_df = apply_context_features(
        raw_df.copy(), ticker, 
        market_df=m_df, 
        sector_df=s_df, 
        ticker_sectors=t_sectors
    )

    # Step B: Daily Features (Kalman, Technicals, d_ prefixing)
    train_df = build_daily_features(train_df)
    train_df['date'] = pd.to_datetime(train_df['date']).dt.date
    
    # Step C: Fundamentals
    if args.join_fundamentals:
        fund_path = Path("data/fundamentals_clean.csv")
        if fund_path.exists():
            fdf = pd.read_csv(fund_path)
            tfund = fdf[fdf['ticker'] == ticker].copy()
            if not tfund.empty and 'date' in tfund.columns:
                train_df = train_df.merge(tfund.drop(columns=['ticker']), on='date', how='left').ffill()
            else:
                print(f"⚠️ Skipping fundamentals merge for training path (no 'date' column found in fundamentals)")

    # Step D: Subset to feature_cols.joblib if available
    selected_cols = None
    for contract_name in ["feature_cols.joblib", "feature_cols_binary.joblib", "feature_cols_short.joblib"]:
        p = root_dir / "models" / ticker / contract_name
        if p.exists():
            selected_cols = joblib.load(p)
            print(f"[OK] Loaded trained feature contract from: {p}")
            break

    last_row_train = train_df.iloc[-1]
    last_date = train_df['date'].iloc[-1]
    print(f"Training row built. Date: {last_date}")

    # 2. INFERENCE PATH
    print(f"\n[2/3] Executing Inference Path...")
    loader = VN100DataLoader()
    infer_df = loader.build_inference_dataset(
        tickers=[ticker],
        lookback_days=args.days,
        join_market=args.join_market,
        join_fundamentals=args.join_fundamentals,
        join_sentiment=args.join_sentiment,
        join_sectors=args.join_sectors
    )
    
    # Calculate Features through the DualModelTrainer's inference path
    trainer = DualModelTrainer()
    # We use our specific ticker
    # This also performs d_ delta computation now via FeatureEngineer.transform
    infer_df = trainer.compute_features_for_ticker(ticker, infer_df[infer_df['ticker']==ticker].copy())
    
    # Normalize date for comparison
    infer_df['date'] = pd.to_datetime(infer_df['date']).dt.date
    
    if infer_df.empty:
        print("[ERR]: Inference feature computation returned empty DF")
        sys.exit(1)
        
    last_row_infer = infer_df.iloc[-1]
    print(f"Inference row built. Date: {last_row_infer['date']}")

    # 3. RIGOROUS COMPARISON
    print(f"\n[3/3] Comparing Feature Vectors...")
    
    # Clean and normalize strings
    train_cols_all = {str(c).strip() for c in last_row_train.index}
    infer_cols_all = {str(c).strip() for c in last_row_infer.index}
    
    # Metadata removal
    for meta in ['ticker', 'date']:
        train_cols_all.discard(meta)
        infer_cols_all.discard(meta)
        
    if selected_cols:
        target_cols = [str(c).strip() for c in selected_cols if str(c).strip() in train_cols_all]
        print(f"Target contract features: {len(target_cols)}")
    # ALIGN BY DATE: Use the latest date from the training path as our anchor
    # since training drops data at the end (due to future-looking targets).
    common_date = last_row_train['date']
    if common_date not in infer_df['date'].values:
        print(f"[ERR]: Training date {common_date} not found in Inference path!")
        available_dates = sorted(infer_df['date'].unique())
        print(f"       Inference dates range: {available_dates[0]} -> {available_dates[-1]}")
        sys.exit(1)
        
    match_row_infer = infer_df[infer_df['date'] == common_date].iloc[0]
    print(f"Comparing features for Date: {common_date}")
    
    # Filter to contract if requested
    target_cols = selected_cols if selected_cols else [c for c in train_df.columns if c not in ["date", "ticker"]]
    print(f"Target contract features: {len(target_cols)}")
    
    mismatches = []
    for col in target_cols:
        if col not in match_row_infer.index:
            mismatches.append((col, "MISSING", "INFER"))
            continue
            
        v_t = last_row_train[col]
        v_i = match_row_infer[col]
        
        # Numeric comparison
        try:
            # Normalize to float for comparison
            if not np.isclose(float(v_t), float(v_i), rtol=1e-5, atol=1e-7, equal_nan=True):
                mismatches.append((col, v_t, v_i))
        except (ValueError, TypeError):
            # String or other comparison
            if str(v_t).strip() != str(v_i).strip():
                mismatches.append((col, v_t, v_i))

    if mismatches:
        print(f"[ERR]: {len(mismatches)} VALUE MISMATCHES found!")
        print(f"{'Feature':<35} | {'Train Value':<15} | {'Infer Value':<15} | {'Diff':<15}")
        for col, vt, vi in mismatches[:50]:
            try:
                diff = abs(float(vt) - float(vi))
                print(f"{col:<35} | {vt:<15.6f} | {vi:<15.6f} | {diff:<15.8f}")
            except:
                print(f"{col:<35} | {vt:<15} | {vi:<15} | N/A")
        fail = True
    
    print(f"\n{'='*80}")
    if fail:
        print("[FAIL] PARITY CHECK: FAILED")
        sys.exit(1)
    else:
        print("[OK] PARITY CHECK: SUCCESS (Contract matched)")
        if args.strict and (missing_in_infer or extra_in_infer):
            print("[WARN] STRICT FAIL: Extra features exist.")
            sys.exit(1)

if __name__ == "__main__":
    main()
