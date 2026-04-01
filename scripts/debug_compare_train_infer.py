"""Parity check script to verify Train vs Infer feature contract consistency.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.ml.feature_engineering import FeatureEngineer
from src.ml.data_loader import VN100DataLoader
from scripts.train_ml_tickers import build_daily_features
from src.ml.trainer import DualModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Check Train-Infer Feature Parity")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker to check")
    parser.add_argument("--days", type=int, default=300, help="Lookback days")
    parser.add_argument("--join-market", action="store_true", help="Join market data")
    parser.add_argument("--join-fundamentals", action="store_true", help="Join fundamental data")
    parser.add_argument("--join-sentiment", action="store_true", help="Join sentiment data")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Checking parity for {ticker}...")

    # 1. TRAIN-STYLE PATH
    loader = VN100DataLoader(prefer_source="csv")
    raw_df = loader._load_single(ticker)
    if raw_df.empty:
        print(f"[ERR] No data found for {ticker}")
        sys.exit(1)
        
    train_feat_df = build_daily_features(raw_df.copy())
    print(f"Train feat DF shape: {train_feat_df.shape}")
    if train_feat_df.empty:
        print("[ERR] Train feat DF is empty!")
        sys.exit(1)

    # Extract last row, excluding target columns
    feature_cols_train = [c for c in train_feat_df.columns if c.startswith('d_')]
    last_row_train = train_feat_df[feature_cols_train].iloc[-1]

    # 2. INFER-STYLE PATH
    infer_df = loader.build_inference_dataset(
        tickers=[ticker],
        lookback_days=args.days,
        join_market=args.join_market,
        join_fundamentals=args.join_fundamentals,
        join_sentiment=args.join_sentiment
    )
    
    trainer = DualModelTrainer()
    # Mock model loading to get feature_cols if possible, or just compute all
    try:
        trainer._ensure_models_loaded(ticker)
        has_model = True
    except:
        has_model = False
        print(f"[WARN] No trained model found for {ticker}, will compare all generated features.")

    infer_feat_df = trainer.compute_features_for_ticker(ticker, infer_df[infer_df['ticker']==ticker].copy())
    print(f"Infer feat DF shape: {infer_feat_df.shape}")
    if infer_feat_df.empty:
        print("[ERR] Infer feat DF is empty!")
        sys.exit(1)
    last_row_infer = infer_feat_df.iloc[-1]

    # 3. COMPARISON
    print("\n--- Feature Set Comparison ---")
    
    train_cols = set(last_row_train.index)
    infer_cols = set(last_row_infer.index)
    
    common_cols = train_cols.intersection(infer_cols)
    only_train = train_cols - infer_cols
    only_infer = infer_cols - train_cols
    
    print(f"Common features: {len(common_cols)}")
    if only_train:
        print(f"[ERR] Missing in Infer: {sorted(list(only_train))}")
    if only_infer:
        print(f"[WARN] Extra in Infer: {sorted(list(only_infer))}")

    # Value comparison
    mismatches = []
    for col in sorted(list(common_cols)):
        v_train = last_row_train[col]
        v_infer = last_row_infer[col]
        
        if not np.isclose(float(v_train), float(v_infer), rtol=1e-5, atol=1e-8, equal_nan=True):
            mismatches.append((col, v_train, v_infer))

    if mismatches:
        print(f"\n[ERR] Value Mismatches found ({len(mismatches)}):")
        print(f"{'Feature':<30} | {'Train Value':<15} | {'Infer Value':<15}")
        print("-" * 65)
        for col, vt, vi in mismatches[:20]:
            print(f"{col:<30} | {vt:<15.6f} | {vi:<15.6f}")
        if len(mismatches) > 20:
            print(f"... and {len(mismatches)-20} more.")
        sys.exit(1)
    else:
        if not only_train:
            print("\n[OK] PARITY SUCCESS: All common features match perfectly!")
            sys.exit(0)
        else:
            print("\n[WARN] PARITY INCOMPLETE: Common features match, but some train features are missing in infer.")
            sys.exit(1)

if __name__ == "__main__":
    main()
