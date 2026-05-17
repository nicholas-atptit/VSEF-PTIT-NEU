"""Null/permutation test for VN30 hourly 2015 top-k 75% result."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
EXISTING_OUTPUTS = REPO_ROOT / "outputs" / "vn30_hourly_2015_topk_ranking_experiments"
VERIFICATION_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_topk_verification"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SEED = 42
K = 10
N_PERMUTATIONS = 1000
LGBM_PARAMS = {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1}

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def fmt_pct(v: Any) -> str:
    try:
        n = float(v)
        if not math.isfinite(n): return ""
        return f"{n * 100:.2f}%"
    except: return ""

def load_universe_tickers() -> list[str]:
    tickers = []
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker", "")).strip().upper()
            if t: tickers.append(t)
    return tickers

def load_stock_data(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = STOCK_CACHE_DIR / f"{ticker}.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["ticker"] = ticker
        for col in OHLCV_COLUMNS: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["datetime"] + OHLCV_COLUMNS)
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]
        frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)

def build_features_minimal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feat_cols = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 10, 20): prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20): prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
        for window in (5, 10, 20, 60):
            min_p = max(3, min(window, window // 2))
            prepared.loc[idx, f"rolling_return_mean_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"rolling_return_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
            sma = close.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"close_sma_ratio_{window}"] = close / sma - 1.0
            prepared.loc[idx, f"momentum_{window}"] = close / close.shift(window) - 1.0
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.loc[(loss == 0.0) & (gain > 0.0)] = 100.0
        rsi.loc[(loss == 0.0) & (gain == 0.0)] = 50.0
        prepared.loc[idx, "rsi_14"] = rsi
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        prepared.loc[idx, "macd"] = macd
        prepared.loc[idx, "macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        prepared.loc[idx, "macd_hist"] = prepared.loc[idx, "macd"] - prepared.loc[idx, "macd_signal"]
        prepared.loc[idx, "volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
        vol_ma = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma - 1.0
        prepared.loc[idx, "high_low_range"] = (group["high"] - group["low"]) / close
        prepared.loc[idx, "open_close_spread"] = (close - group["open"]) / group["open"].replace(0.0, np.nan)
        prepared.loc[idx, "close_position_in_range"] = (close - group["low"]) / (group["high"] - group["low"]).replace(0.0, np.nan)
        for lag in (1, 2, 4, 8, 20): prepared.loc[idx, f"lag_ret_{lag}"] = close.pct_change(periods=lag, fill_method=None).shift(1)
        for window in (4, 8, 20, 40):
            min_p = max(2, window // 4)
            prepared.loc[idx, f"roll_ret_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_change_{window}"] = volume.pct_change(periods=1, fill_method=None).rolling(window, min_periods=min_p).mean()
    feat_cols = ["return_1","return_2","return_3","return_5","return_10","return_20",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20",
        "rolling_return_mean_5","rolling_return_vol_5","close_sma_ratio_5","momentum_5",
        "rolling_return_mean_10","rolling_return_vol_10","close_sma_ratio_10","momentum_10",
        "rolling_return_mean_20","rolling_return_vol_20","close_sma_ratio_20","momentum_20",
        "rolling_return_mean_60","rolling_return_vol_60","close_sma_ratio_60","momentum_60",
        "rsi_14","macd","macd_signal","macd_hist","volume_change_1","volume_shock_20",
        "high_low_range","open_close_spread","close_position_in_range",
        "lag_ret_1","lag_ret_2","lag_ret_4","lag_ret_8","lag_ret_20",
        "roll_ret_4","roll_ret_8","roll_ret_20","roll_ret_40",
        "roll_vol_8","roll_vol_20","roll_vol_40",
        "roll_vol_change_8","roll_vol_change_20","roll_vol_change_40"]
    time_cols = ["day_of_week","day_of_month","month","quarter","hour","minute"]
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["datetime"].dt.day.astype(float)
    prepared["month"] = prepared["datetime"].dt.month.astype(float)
    prepared["quarter"] = prepared["datetime"].dt.quarter.astype(float)
    prepared["hour"] = prepared["datetime"].dt.hour.astype(float)
    prepared["minute"] = prepared["datetime"].dt.minute.astype(float)
    all_cols = list(set(feat_cols + time_cols))
    existing = [c for c in all_cols if c in prepared.columns]
    prepared[existing] = prepared[existing].replace([np.inf, -np.inf], np.nan)
    return prepared, existing

def compute_future_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    returns = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        ret = (future_close - group["close"]) / group["close"]
        returns.append(pd.Series(ret.values, index=idx, name="future_return"))
    if returns: return pd.concat(returns)
    return pd.Series(dtype=float, name="future_return")

def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(X, y)

def compute_precision_at_k(subset_df: pd.DataFrame, future_returns: pd.Series, scores: np.ndarray, k: int) -> float:
    """Compute overall precision@k."""
    df_temp = subset_df.copy()
    df_temp["score"] = scores
    df_temp["future_return"] = future_returns.values
    
    total_hits = 0
    total_events = 0
    for dt, group in df_temp.groupby("datetime"):
        if len(group) < k: continue
        true_top_k_idx = group.nlargest(k, "future_return").index
        pred_top_k_idx = group.nlargest(k, "score").index
        hits = len(set(pred_top_k_idx) & set(true_top_k_idx))
        total_hits += hits
        total_events += k
    
    return total_hits / total_events if total_events > 0 else 0.0

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Top-K 75% Null/Permutation Test")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    
    VERIFICATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading stock data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    print(f"   {len(tickers)} tickers, {len(stock_df)} rows")
    
    print("\n2. Building features...")
    feature_df, feat_cols = build_features_minimal(stock_df)
    print(f"   {len(feat_cols)} features")
    
    # Focus on h=120 (the actual best)
    horizon = 120
    print(f"\n3. Computing future returns for h={horizon}...")
    future_returns = compute_future_returns(feature_df, horizon)
    
    # Split data
    eval_mask = feature_df["datetime"] >= EVAL_START
    eval_df = feature_df[eval_mask].copy()
    eval_fr = future_returns.reindex(eval_df.index).dropna()
    eval_df = eval_df.reindex(eval_fr.index)
    
    train_mask = feature_df["datetime"] <= TRAIN_END
    train_df = feature_df[train_mask].copy()
    train_fr = future_returns.reindex(train_df.index).dropna()
    
    # Create binary labels
    train_labels = pd.Series(0, index=train_fr.index)
    for dt, group in feature_df.reindex(train_fr.index).groupby("datetime"):
        if len(group) < 2: continue
        median_ret = future_returns.loc[group.index].median()
        above_mask = future_returns.loc[group.index] > median_ret
        train_labels.loc[above_mask[above_mask].index] = 1
    
    fcp = [c for c in feat_cols if c in feature_df.columns]
    tX = feature_df.reindex(train_fr.index)[fcp].fillna(0)
    ty = train_labels.astype(int)
    
    # Train model
    print(f"\n4. Training LightGBM for h={horizon}...")
    model = train_lightgbm(tX, ty)
    
    # Get predictions
    eX = feature_df.reindex(eval_fr.index)[fcp].fillna(0)
    eval_proba = model.predict_proba(eX)[:, 1]
    
    # Add future_return to eval_df for groupby operations
    eval_df = eval_df.copy()
    eval_df["future_return"] = eval_fr.values
    
    # Observed precision@10
    print(f"\n5. Computing observed precision@10...")
    observed_precision = compute_precision_at_k(eval_df, eval_fr, eval_proba, K)
    print(f"   Observed precision@10: {fmt_pct(observed_precision)}")
    
    # Null test 1: Random top-k baseline
    print(f"\n6. Running random top-k baseline ({N_PERMUTATIONS} seeds)...")
    np.random.seed(SEED)
    random_precisions = []
    for seed in range(N_PERMUTATIONS):
        np.random.seed(seed)
        total_hits = 0
        total_events = 0
        for dt, group in eval_df.groupby("datetime"):
            if len(group) < K: continue
            true_top_k_idx = group.nlargest(K, "future_return").index
            random_idx = np.random.choice(group.index, size=K, replace=False)
            hits = len(set(random_idx) & set(true_top_k_idx))
            total_hits += hits
            total_events += K
        random_precisions.append(total_hits / total_events if total_events > 0 else 0.0)
    
    random_mean = np.mean(random_precisions)
    random_std = np.std(random_precisions)
    random_p_value = np.sum(np.array(random_precisions) >= observed_precision) / len(random_precisions)
    
    print(f"   Random mean: {fmt_pct(random_mean)}")
    print(f"   Random std: {fmt_pct(random_std)}")
    print(f"   Empirical p-value: {random_p_value:.6f}")
    
    # Null test 2: Shuffle model scores within each timestamp
    print(f"\n7. Running score shuffle permutation test ({N_PERMUTATIONS} permutations)...")
    score_shuffle_precisions = []
    # Build index to position mapping
    idx_to_pos = {idx: i for i, idx in enumerate(eval_fr.index)}
    
    for perm in range(N_PERMUTATIONS):
        np.random.seed(SEED + perm + 1000)
        shuffled_scores = eval_proba.copy()
        
        # Shuffle scores within each timestamp group
        for dt, group in eval_df.groupby("datetime"):
            if len(group) < K: continue
            idx_list = [i for i in group.index if i in idx_to_pos]
            if len(idx_list) == 0: continue
            positions = [idx_to_pos[i] for i in idx_list]
            original_scores = eval_proba[positions].copy()
            np.random.shuffle(original_scores)
            for i, pos in enumerate(positions):
                shuffled_scores[pos] = original_scores[i]
        
        prec = compute_precision_at_k(eval_df, eval_fr, shuffled_scores, K)
        score_shuffle_precisions.append(prec)
        
        if (perm + 1) % 200 == 0:
            print(f"   Progress: {perm + 1}/{N_PERMUTATIONS}")
    
    score_shuffle_mean = np.mean(score_shuffle_precisions)
    score_shuffle_std = np.std(score_shuffle_precisions)
    score_shuffle_p_value = np.sum(np.array(score_shuffle_precisions) >= observed_precision) / len(score_shuffle_precisions)
    
    print(f"   Score shuffle mean: {fmt_pct(score_shuffle_mean)}")
    print(f"   Score shuffle std: {fmt_pct(score_shuffle_std)}")
    print(f"   Empirical p-value: {score_shuffle_p_value:.6f}")
    
    # Null test 3: Shuffle true labels within each timestamp
    print(f"\n8. Running label shuffle permutation test ({N_PERMUTATIONS} permutations)...")
    label_shuffle_precisions = []
    true_returns = eval_fr.values.copy()
    for perm in range(N_PERMUTATIONS):
        np.random.seed(SEED + perm + 2000)
        shuffled_returns = true_returns.copy()
        # Shuffle within each timestamp
        for dt, group in eval_df.groupby("datetime"):
            if len(group) < K: continue
            idx_list = list(group.index)
            pos_in_returns = [list(eval_fr.index).index(i) for i in idx_list if i in eval_fr.index]
            if len(pos_in_returns) == 0: continue
            original_returns = true_returns[pos_in_returns]
            np.random.shuffle(original_returns)
            for i, pos in enumerate(pos_in_returns):
                shuffled_returns[pos] = original_returns[i]
        
        # Compute precision with shuffled labels
        df_temp = eval_df.copy()
        df_temp["score"] = eval_proba
        df_temp["future_return"] = shuffled_returns
        
        total_hits = 0
        total_events = 0
        for dt, group in df_temp.groupby("datetime"):
            if len(group) < K: continue
            true_top_k_idx = group.nlargest(K, "future_return").index
            pred_top_k_idx = group.nlargest(K, "score").index
            hits = len(set(pred_top_k_idx) & set(true_top_k_idx))
            total_hits += hits
            total_events += K
        
        prec = total_hits / total_events if total_events > 0 else 0.0
        label_shuffle_precisions.append(prec)
        
        if (perm + 1) % 200 == 0:
            print(f"   Progress: {perm + 1}/{N_PERMUTATIONS}")
    
    label_shuffle_mean = np.mean(label_shuffle_precisions)
    label_shuffle_std = np.std(label_shuffle_precisions)
    label_shuffle_p_value = np.sum(np.array(label_shuffle_precisions) >= observed_precision) / len(label_shuffle_precisions)
    
    print(f"   Label shuffle mean: {fmt_pct(label_shuffle_mean)}")
    print(f"   Label shuffle std: {fmt_pct(label_shuffle_std)}")
    print(f"   Empirical p-value: {label_shuffle_p_value:.6f}")
    
    # Save results
    print(f"\n9. Saving null test results...")
    null_results = []
    for i in range(N_PERMUTATIONS):
        null_results.append({
            "permutation": i + 1,
            "random_precision": random_precisions[i],
            "score_shuffle_precision": score_shuffle_precisions[i],
            "label_shuffle_precision": label_shuffle_precisions[i],
        })
    
    null_df = pd.DataFrame(null_results)
    null_df.to_csv(VERIFICATION_OUTPUT_DIR / "topk_75_null_test.csv", index=False)
    
    # Generate null test report
    report = f"""# VN30 Hourly 2015 - Top-K 75% Null/Permutation Test Report

## Observed Result
- Observed precision@10: {fmt_pct(observed_precision)}
- Horizon: h={horizon}
- Model: LightGBM
- Events: {len(eval_df.groupby('datetime'))}

## Null Test 1: Random Top-K Baseline
- Random mean: {fmt_pct(random_mean)}
- Random std: {fmt_pct(random_std)}
- Random min: {fmt_pct(np.min(random_precisions))}
- Random max: {fmt_pct(np.max(random_precisions))}
- Empirical p-value: {random_p_value:.6f}
- Significantly above random: {'YES' if random_p_value < 0.05 else 'NO'}

## Null Test 2: Score Shuffle Permutation
- Score shuffle mean: {fmt_pct(score_shuffle_mean)}
- Score shuffle std: {fmt_pct(score_shuffle_std)}
- Empirical p-value: {score_shuffle_p_value:.6f}
- Significantly above shuffled scores: {'YES' if score_shuffle_p_value < 0.05 else 'NO'}

## Null Test 3: Label Shuffle Permutation
- Label shuffle mean: {fmt_pct(label_shuffle_mean)}
- Label shuffle std: {fmt_pct(label_shuffle_std)}
- Empirical p-value: {label_shuffle_p_value:.6f}
- Significantly above shuffled labels: {'YES' if label_shuffle_p_value < 0.05 else 'NO'}

## Conclusion
- The observed precision@10 is {'statistically significantly' if random_p_value < 0.05 else 'NOT statistically significantly'} above random selection.
- The model scores contain {'meaningful' if score_shuffle_p_value < 0.05 else 'NO'} predictive signal beyond random.
- The result is {'robust to' if label_shuffle_p_value < 0.05 else 'NOT robust to'} label permutation.

## Interpretation
{'The result shows genuine predictive signal above random chance.' if random_p_value < 0.05 else 'The result may be due to random chance.'}
{'The model has learned meaningful patterns.' if score_shuffle_p_value < 0.05 else 'The model scores may not contain meaningful signal.'}
"""
    
    with (VERIFICATION_OUTPUT_DIR / "topk_75_null_test.md").open("w") as f:
        f.write(report)
    
    print(f"\nNull test complete. Outputs in {rel(VERIFICATION_OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
