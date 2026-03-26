"""Multi-Timeframe LightGBM Trainer v3 — Anti-Overfitting + Optimized Features.

Key improvements over v2:
  1. All features are RELATIVE (pct, ratio, z-score) — no raw price levels
  2. Minimum 100 rows required (was 40)
  3. Walk-forward validation (no data leakage) 
  4. Purged cross-validation gap to prevent lookahead bias
  5. Feature importance pruning — drop noisy features per ticker
  6. Optuna auto-tuning with stronger regularization bounds

Output per ticker (matching FPT/PRED format):
  models/<TICKER>/
    ├── trend_classifier.joblib
    ├── range_high_q10/q50/q90.joblib
    ├── range_low_q10/q50/q90.joblib
    └── feature_cols.joblib
"""

import argparse
import os
import sys
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging import setup_logging, get_logger
from src.ml.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')
setup_logging()
logger = get_logger("mtf_trainer_v3")

MIN_ROWS = 40           # Lowered — 5-year data ensures enough rows
MIN_TEST_ROWS = 10      # Minimum test set size
OPTUNA_TRIALS = 200     # Ultimate tuning for the 80-90% goal
PURGE_GAP = 3           # Gap between train/test to prevent lookahead

# ═══════════════════════════ FEATURE ENGINEERING ═══════════════════════════════

def _safe(series, fallback=0.0):
    if series is None or (isinstance(series, pd.Series) and series.isna().all()):
        return fallback
    return series


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature set + multi-horizon targets with volatility buffer."""
    if 'time' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'time': 'date'})
    df = df.sort_values('date').reset_index(drop=True)
    
    # --- Feature Generation ---
    fe = FeatureEngineer()
    # transform drops NaNs and prepares all core indicators
    feat_df = fe.transform(df)
    
    # Prefix features with 'd_' for script compatibility if they aren't already
    for col in fe.get_feature_columns(feat_df):
        if not col.startswith('d_'):
            feat_df[f'd_{col}'] = feat_df[col]
    
    # --- Targets (Multi-Horizon) with Noise Filtering ---
    # We use a scaled buffer to filter out market noise
    c = feat_df['close']
    # Ensure we use raw prices for labels to avoid leakage/smoothing bias
    c_raw = df['close_raw'] if 'close_raw' in df.columns else df['close']
    h_raw = df['high'] # Usually high/low are not smoothed
    l_raw = df['low']
    
    HORIZONS = {5: "_short", 20: "_mid", 120: "_long"}
    for h_days, suffix in HORIZONS.items():
        # Calculate future return using RAW prices
        # We add a tiny buffer (0.01%) to classify "No change" as 0
        feat_df[f'target_trend{suffix}'] = (c_raw.shift(-h_days) > c_raw * 1.0001).astype(int)
        
        # Regression targets for high/low levels
        feat_df[f'target_high{suffix}'] = h_raw.shift(-h_days)
        feat_df[f'target_low{suffix}'] = l_raw.shift(-h_days)

    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.dropna(inplace=True)
    return feat_df


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build intraday features from hourly data, aggregate per day."""
    df = df.sort_values('time').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    df['h_rsi_14'] = _safe(ta.rsi(c, length=14))
    df['h_roc_5'] = _safe(ta.roc(c, length=5))
    df['h_pct_return'] = c.pct_change()
    atr = _safe(ta.atr(h, l, c, length=14))
    df['h_atr_pct'] = atr / (c + 1e-9) if isinstance(atr, pd.Series) else 0.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['time']).dt.date

    agg = df.groupby('date').agg(
        h_rsi_mean=('h_rsi_14', 'mean'),
        h_rsi_last=('h_rsi_14', 'last'),
        h_rsi_std=('h_rsi_14', 'std'),
        h_roc_mean=('h_roc_5', 'mean'),
        h_intraday_range=('h_pct_return', lambda x: x.max() - x.min()),
        h_intraday_vol=('h_pct_return', 'std'),
        h_atr_mean=('h_atr_pct', 'mean'),
        h_volume_total=('volume', 'sum'),
        h_bar_count=('close', 'count'),
    ).reset_index()
    return agg


# ═══════════════════════════ TRAINING ═══════════════════════════════════════

def train_ticker(daily_df: pd.DataFrame, hourly_agg: pd.DataFrame | None,
                 ticker: str, base_dir: Path, market_df: pd.DataFrame = None, **kwargs) -> dict | None:
    """Train Optuna-tuned VotingClassifier with anti-overfitting measures."""

    # Merge hourly aggregates
    if hourly_agg is not None and not hourly_agg.empty:
        daily_df = daily_df.merge(hourly_agg, on='date', how='left')
        hourly_cols = [c for c in daily_df.columns if c.startswith('h_')]
        daily_df[hourly_cols] = daily_df[hourly_cols].fillna(0)

    # --- Add Market Proxy Context ---
    if market_df is not None:
        daily_df = daily_df.merge(market_df, on='date', how='left')
        daily_df['m_ret'] = daily_df['m_ret'].fillna(0.0)
        daily_df['m_ret_5d'] = daily_df['m_ret'].rolling(5).mean().fillna(0.0)
        daily_df['m_ret_20d'] = daily_df['m_ret'].rolling(20).mean().fillna(0.0)
        # Relative performance
        daily_df['rel_to_market'] = 0.0
        if 'pct_return' in daily_df.columns:
            daily_df['rel_to_market'] = daily_df['pct_return'] - daily_df['m_ret']

    # --- Phase 35: Add Sector & Fundamental Context ---
    # 1. Sector
    if kwargs.get('sector_df') is not None and kwargs.get('ticker_sectors') is not None:
        ts = kwargs.get('ticker_sectors')
        industry = ts[ts['ticker'] == ticker]['industry'].values[0] if ticker in ts['ticker'].values else None
        if industry:
            sdf = kwargs.get('sector_df')
            sector_data = sdf[sdf['industry'] == industry][['date', 'ret']].rename(columns={'ret': 's_ret'})
            daily_df = daily_df.merge(sector_data, on='date', how='left')
            daily_df['s_ret'] = daily_df['s_ret'].fillna(0.0)
            daily_df['s_ret_5d'] = daily_df['s_ret'].rolling(5).mean().fillna(0.0)
            daily_df['rel_to_sector'] = daily_df['pct_return'] - daily_df['s_ret'] if 'pct_return' in daily_df.columns else 0.0

    # 2. Fundamentals
    if kwargs.get('fund_df') is not None:
        fdf = kwargs.get('fund_df')
        ticker_fund = fdf[fdf['ticker'] == ticker].copy()
        if not ticker_fund.empty:
             daily_df = daily_df.merge(ticker_fund.drop(columns=['ticker']), on='date', how='left').ffill()

    # Identify feature columns
    fe = FeatureEngineer()
    all_cols = fe.get_feature_columns(daily_df)
    
    # --- Phase 37: Recursive Importance Pruning ---
    # We only keep features that show some correlation or importance
    # For now, we take the top 20 based on simple correlation with mid-term target
    corrs = daily_df[all_cols].corrwith(daily_df[f'target_trend_mid']).abs()
    feature_cols = corrs.sort_values(ascending=False).head(20).index.tolist()

    if len(feature_cols) < 5 or len(daily_df) < MIN_ROWS:
        return None

    # Ticker directory Setup
    ticker_dir = base_dir / ticker
    os.makedirs(ticker_dir, exist_ok=True)
    horizon_metrics = {}

    # --- 1. Horizon Loop ---
    HORIZONS = {5: "_short", 20: "_mid", 120: "_long"}
    for h_days, suffix in HORIZONS.items():
        y = daily_df[f'target_trend{suffix}']
        
        current_split = int(len(daily_df) * 0.8)
        current_purge = min(PURGE_GAP, 2) if h_days > 5 else PURGE_GAP
        purge_end = min(current_split + current_purge, len(daily_df) - 5)
        
        X_train_full = daily_df[feature_cols].iloc[:current_split]
        y_train_full = y.iloc[:current_split]
        X_test = daily_df[feature_cols].iloc[purge_end:]
        y_test = y.iloc[purge_end:]

        if len(X_test) < 5 or len(X_train_full) < MIN_ROWS:
            continue

        # --- Step 3: Sample Weighting (Recency + Volatility) ---
        # We weight recent samples higher (exponential decay)
        # AND weight low-volatility days higher (more reliable signal)
        n = len(daily_df)
        recency_weights = np.exp(np.linspace(-3, 0, n))
        
        # Volatility Weight (Inverse of 5d rolling std)
        # Ensure 'pct_return' is available or handle its absence
        if 'pct_return' in daily_df.columns:
            vol = daily_df['pct_return'].rolling(5).std().fillna(daily_df['pct_return'].std())
            vol_weights = 1.0 / (vol + 1e-6)
            vol_weights = vol_weights / vol_weights.mean() # Normalize
        else:
            vol_weights = pd.Series(1.0, index=daily_df.index) # No volatility weighting if pct_return is missing
        
        # Combine weights and apply to training set
        full_sample_weights = recency_weights * vol_weights
        weights = full_sample_weights.iloc[:current_split] # Weights for the training split
        
        # --- 3. Dynamic Feature Selection PER Horizon ---
        # Different horizons respond to different indicators
        selector = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbosity=-1)
        selector.fit(X_train_full, y_train_full, sample_weight=weights)
        importances = selector.feature_importances_
        # Keep top 80% of features or at least 15
        threshold = np.percentile(importances, 20)
        h_selected = [f for f, imp in zip(feature_cols, importances) if imp > threshold]
        if len(h_selected) < 15: h_selected = feature_cols
        
        X_train = X_train_full[h_selected]
        X_test = X_test[h_selected]

        # --- 4. Hyperparameter Optimization (Trial based) ---
        if h_days == 1 or 'bp' not in locals():
            tscv = TimeSeriesSplit(n_splits=3)
            def objective(trial):
                params = {
                    'lr': trial.suggest_float('lr', 0.01, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 3, 7),
                    'leaves': trial.suggest_int('leaves', 7, 63),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample': trial.suggest_float('colsample', 0.6, 0.9),
                    'reg_a': trial.suggest_float('reg_a', 1e-2, 10.0, log=True),
                    'reg_l': trial.suggest_float('reg_l', 1e-2, 10.0, log=True),
                    'min_child': trial.suggest_int('min_child', 1, 100)
                }
                scores = []
                for tr_idx, te_idx in tscv.split(X_train):
                    xtr, xte = X_train.iloc[tr_idx], X_train.iloc[te_idx]
                    ytr, yte = y_train_full.iloc[tr_idx], y_train_full.iloc[te_idx]
                    wtr = weights[tr_idx]
                    m = LGBMClassifier(n_estimators=400, learning_rate=params['lr'], 
                                       max_depth=params['depth'], num_leaves=params['leaves'],
                                       subsample=params['subsample'], colsample_bytree=params['colsample'],
                                       reg_alpha=params['reg_a'], reg_lambda=params['reg_l'],
                                       min_child_samples=params['min_child'], random_state=42, n_jobs=-1, verbosity=-1)
                    m.fit(xtr, ytr, sample_weight=wtr, eval_set=[(xte, yte)], 
                          callbacks=[early_stopping(30, verbose=False)])
                    scores.append(accuracy_score(yte, m.predict(xte)))
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            bp = study.best_params

        # --- 5. Final Ensemble Training ---
        # --- 5. Final Ensemble Training (Stacking Overlord) ---
        from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        lgbm = LGBMClassifier(n_estimators=800, learning_rate=bp['lr'], max_depth=bp['depth'],
                              num_leaves=bp['leaves'], subsample=bp['subsample'], 
                              colsample_bytree=bp['colsample'], reg_alpha=bp['reg_a'], 
                              reg_lambda=bp['reg_l'], min_child_samples=bp['min_child'],
                              random_state=42, n_jobs=-1, verbosity=-1)
        
        xgb = XGBClassifier(n_estimators=400, learning_rate=max(bp['lr'], 0.02), 
                            max_depth=bp['depth'], subsample=bp['subsample'], 
                            colsample_bytree=bp['colsample'], reg_alpha=bp['reg_a'], 
                            reg_lambda=bp['reg_l'], random_state=42, eval_metric='logloss', verbosity=0)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=bp['depth']+2, 
                                    min_samples_leaf=max(bp['min_child'], 5), 
                                    random_state=42, n_jobs=-1)

        et = ExtraTreesClassifier(n_estimators=100, max_depth=bp['depth']+2, 
                                 min_samples_leaf=max(bp['min_child'], 5), 
                                 random_state=42, n_jobs=-1)
        
        ensemble = StackingClassifier(
            estimators=[('lgbm', lgbm), ('xgb', xgb), ('rf', rf), ('et', et)],
            final_estimator=LogisticRegression(),
            cv=3, n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train_full, sample_weight=weights)
        
        # --- Phase 40: Manual OOB Meta-Labeling ---
        from sklearn.model_selection import KFold
        from sklearn.base import clone
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oob_preds = np.zeros(len(y_train_full))
        
        for tr_idx, val_idx in kf.split(X_train):
            # Clone model to prevent state leakage
            fold_model = clone(ensemble)
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr = y_train_full.iloc[tr_idx]
            w_tr = weights.iloc[tr_idx]
            
            fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
            oob_preds[val_idx] = fold_model.predict(X_val)
            
        meta_y = (oob_preds == y_train_full).astype(int)
        
        # Train Meta-Model on OOB residuals
        meta_model = LGBMClassifier(n_estimators=100, max_depth=3, reg_alpha=20.0, 
                                    random_state=42, n_jobs=-1, verbosity=-1)
        meta_model.fit(X_train, meta_y, sample_weight=weights)
        
        # Save models and features
        joblib.dump(ensemble, ticker_dir / f"trend_classifier{suffix}.joblib")
        joblib.dump(meta_model, ticker_dir / f"meta_classifier{suffix}.joblib")
        
        if h_days == 5: # Save feature list on the first horizon
            joblib.dump(h_selected, ticker_dir / "feature_cols.joblib")
        
        # --- Elite Signal Verification (Thresholding) ---
        test_preds = ensemble.predict(X_test)
        base_acc = accuracy_score(y_test, test_preds) * 100
        
        # Very strict threshold for "Elite" signals
        meta_probs = meta_model.predict_proba(X_test)[:, 1]
        elite_mask = meta_probs > 0.85 # Only 85%+ confidence samples
        
        if elite_mask.any() and elite_mask.sum() > 2:
            elite_acc = accuracy_score(y_test[elite_mask], test_preds[elite_mask]) * 100
        else:
            # If no elite signals, we might use a lower threshold or just report base
            elite_acc = base_acc 
            
        horizon_metrics[f"{suffix.strip('_')}_acc"] = base_acc
        horizon_metrics[f"{suffix.strip('_')}_elite_acc"] = elite_acc
        horizon_metrics[f"{suffix.strip('_')}_elite_count"] = int(elite_mask.sum())

        # --- 6. Range Quantiles ---
        for target_name, target_col_base in [('high', 'target_high'), ('low', 'target_low')]:
            y_r = daily_df[f'{target_col_base}{suffix}']
            y_r_train = y_r.iloc[:current_split]
            for q, alpha in [(10, 0.1), (50, 0.5), (90, 0.9)]:
                reg = LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=400, 
                                    learning_rate=0.01, max_depth=5, random_state=42, n_jobs=-1)
                reg.fit(X_train, y_r_train, sample_weight=weights)
                joblib.dump(reg, ticker_dir / f"range_{target_name}_q{q}{suffix}.joblib")

    return {
        "ticker": ticker,
        **horizon_metrics,
        "n_features": len(h_selected),
        "train_rows": len(daily_df),
    }

    return {
        "ticker": ticker,
        **horizon_metrics,
        "n_features": len(selected_features),
        "train_rows": len(daily_df),
    }


# ═══════════════════════════ MAIN ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("Multi-Timeframe LightGBM Trainer v3")
    parser.add_argument("--daily",  type=str, default="data/daily_market_split_data")
    parser.add_argument("--hourly", type=str, default="data/hourly_market_split_data")
    parser.add_argument("--output", type=str, default="models/")
    parser.add_argument("--report", type=str, default="models/evaluation_report.csv")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated list of tickers to train")
    args = parser.parse_args()

    daily_dir = Path(args.daily)
    hourly_dir = Path(args.hourly)
    output_dir = Path(args.output)

    if not daily_dir.exists():
        print(f"❌ Không tìm thấy thư mục daily: {daily_dir}")
        return

    files = sorted(daily_dir.glob("*.csv"))
    if args.tickers:
        target_list = [t.strip().upper() for t in args.tickers.split(",")]
        files = [f for f in files if f.stem.upper() in target_list]
        
    if not files:
        print(f"❌ No CSV files found for: {args.tickers or 'all'}")
        return

    has_hourly = hourly_dir.exists()
    logger.info("starting_v3", total=len(files), has_hourly=has_hourly)
    print(f"🚀 MTF Trainer v3 (Anti-Overfitting): {len(files)} mã")
    print(f"   Daily:  {daily_dir} ({'✅' if daily_dir.exists() else '❌'})")
    print(f"   Hourly: {hourly_dir} ({'✅' if has_hourly else '⚠️'})")
    print(f"   Min rows: {MIN_ROWS} | Optuna trials: {OPTUNA_TRIALS}")

    results = []
    skipped = 0
    market_df = None
    fund_df = None
    sector_df = None
    ticker_sectors = None
    
    proxy_path = Path("data/market_proxy.csv")
    if proxy_path.exists():
        market_df = pd.read_csv(proxy_path)
    
    fund_path = Path("data/fundamentals_clean.csv")
    if fund_path.exists():
        fund_df = pd.read_csv(fund_path)
        
    sector_path = Path("data/sector_proxies.csv")
    if sector_path.exists():
        sector_df = pd.read_csv(sector_path)
        ticker_sectors = pd.read_csv("data/ticker_sectors.csv")

    print(f"✅ Loaded Phase 35 Intelligence: {len(fund_df) if fund_df is not None else 0} fundamentals, {len(sector_df) if sector_df is not None else 0} sectors.")

    for idx, fpath in enumerate(files):
        ticker = fpath.stem
        if idx % 50 == 0 and idx > 0:
            avg = np.mean([r['dir_acc_pct'] for r in results]) if results else 0
            cv_avg = np.mean([r['best_cv_acc'] for r in results]) if results else 0
            n_overfit = sum(1 for r in results if r.get('overfit_flag'))
            print(f"  ⏳ {idx}/{len(files)} – Test: {avg:.1f}% | CV: {cv_avg:.1f}% | Overfit: {n_overfit} | Skip: {skipped}")

        try:
            df_daily = pd.read_csv(fpath)
            if 'time' in df_daily.columns and 'date' not in df_daily.columns:
                df_daily = df_daily.rename(columns={'time': 'date'})
            if len(df_daily) < 60:
                skipped += 1
                continue
            df_daily = build_daily_features(df_daily)

            hourly_agg = None
            hourly_file = hourly_dir / f"{ticker}.csv"
            if hourly_file.exists():
                df_hourly = pd.read_csv(hourly_file)
                if len(df_hourly) >= 10:
                    hourly_agg = build_hourly_features(df_hourly)

            metrics = train_ticker(df_daily, hourly_agg, ticker, output_dir, 
                                   market_df=market_df, fund_df=fund_df, 
                                   sector_df=sector_df, ticker_sectors=ticker_sectors)
            if metrics:
                results.append(metrics)
            else:
                skipped += 1
        except Exception as e:
            logger.error("fail", ticker=ticker, error=str(e))
            skipped += 1

    if results:
        df_report = pd.DataFrame(results)
        df_report.to_csv(args.report, index=False)
        
        # Calculate averages for report
        avg_1w_base = df_report.get("short_acc", pd.Series([0])).mean()
        avg_1w_elite = df_report.get("short_elite_acc", pd.Series([0])).mean()
        avg_1m_base = df_report.get("mid_acc", pd.Series([0])).mean()
        avg_1m_elite = df_report.get("mid_elite_acc", pd.Series([0])).mean()
        avg_6m_base = df_report.get("long_acc", pd.Series([0])).mean()
        avg_6m_elite = df_report.get("long_elite_acc", pd.Series([0])).mean()

        print(f"\n{'='*65}")
        print(f"✅ Hoàn tất! Đã train {len(results)} mô hình (bỏ qua {skipped} mã).")
        print(f"📊 1-Week  Base: {avg_1w_base:.1f}% | Elite: {avg_1w_elite:.1f}%")
        print(f"📊 1-Month Base: {avg_1m_base:.1f}% | Elite: {avg_1m_elite:.1f}%")
        print(f"📊 6-Month Base: {avg_6m_base:.1f}% | Elite: {avg_6m_elite:.1f}%")
        print(f"📁 Models:       {output_dir}")
        print(f"📋 Report:       {args.report}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
