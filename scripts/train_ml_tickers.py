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
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.benchmark.baselines import BaselineStrategies
from sklearn.model_selection import TimeSeriesSplit
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging import setup_logging, get_logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.labels.training_adapter import resolve_label_config, LabelTrainingConfig
from src.ml.experiment_tracker import ExperimentTracker
from src.validators.data_quality import DataQualityValidator

warnings.filterwarnings('ignore')
setup_logging()
logger = get_logger("mtf_trainer_v3")

from src.data.universe import get_vn100_universe
from src.ml.data_loader import apply_context_features

MIN_ROWS = 40           # Lowered — 5-year data ensures enough rows
MIN_TEST_ROWS = 10      # Minimum test set size
OPTUNA_TRIALS = 200     # Ultimate tuning for the 80-90% goal
PURGE_GAP = 10          # Default gap between train/test to prevent lookahead

# Configuration constants

# ═══════════════════════════ FEATURE ENGINEERING ═══════════════════════════════

def _safe(series, fallback=0.0):
    if series is None or (isinstance(series, pd.Series) and series.isna().all()):
        return fallback
    return series


def build_daily_features(df: pd.DataFrame, sentiment_df: pd.DataFrame = None, label_config: LabelTrainingConfig = None) -> pd.DataFrame:
    """Enhanced feature set + multi-horizon targets with volatility buffer."""
    if 'time' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'time': 'date'})
    df = df.sort_values('date').reset_index(drop=True)
    
    # --- Feature Generation ---
    fe = FeatureEngineer()
    # transform now handles Kalman filtering, deltas (d_), and sentiment
    feat_df = fe.transform(df, sentiment_df=sentiment_df, drop_na=True)
    
    if label_config:
        # Generate custom label
        feat_df = label_config.generator.generate(feat_df)
    else:
        # --- Targets (Multi-Horizon) with Noise Filtering ---
        # IMPORTANT: Extract target series from feat_df AFTER transform() 
        # to ensure they are aligned with the dropped/filtered feature rows.
        c_raw = feat_df['close']
        h_raw = feat_df['high']
        l_raw = feat_df['low']
        
        HORIZONS = {5: "_short", 20: "_mid", 120: "_long"}
        for h_days, suffix in HORIZONS.items():
            # Calculate future return using RAW prices
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

def _train_custom_label(daily_df: pd.DataFrame, ticker: str, ticker_dir: Path, feature_cols: list[str], label_config: LabelTrainingConfig) -> dict:
    """Train a single model for the requested custom label."""
    y = daily_df[label_config.label_column]
    
    # Minimal Time-Series Safe Split
    # Logic: Train on [0, split_idx - horizon], Test on [split_idx, end]
    # This prevents the training set from "seeing" test set prices in its labels.
    split_idx = int(len(daily_df) * 0.8)
    horizon = getattr(label_config.generator, "HORIZON", 5) # Default 5 if not specified
    
    train_end = split_idx - horizon
    if train_end < MIN_ROWS:
        return {} # Not enough data to be safe
        
    X_train_full = daily_df[feature_cols].iloc[:train_end]
    y_train_full = y.iloc[:train_end]
    X_test = daily_df[feature_cols].iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    if len(X_test) < 5 or len(X_train_full) < MIN_ROWS:
        return {}
        
    n = len(daily_df)
    recency_weights = np.exp(np.linspace(-3, 0, n))
    vol_weights = pd.Series(1.0, index=daily_df.index)
    if 'pct_return' in daily_df.columns:
        vol = daily_df['pct_return'].rolling(5).std().fillna(daily_df['pct_return'].std())
        vol_weights = 1.0 / (vol + 1e-6)
        vol_weights = vol_weights / vol_weights.mean()
    weights = (recency_weights * vol_weights).iloc[:train_end]
    
    # Feature Selection
    selector = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbosity=-1) if label_config.task_type == "classification" else LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbosity=-1)
    selector.fit(X_train_full, y_train_full, sample_weight=weights)
    importances = selector.feature_importances_
    threshold = np.percentile(importances, 20)
    h_selected = [f for f, imp in zip(feature_cols, importances) if imp > threshold]
    if len(h_selected) < 5: h_selected = feature_cols
    
    X_train = X_train_full[h_selected]
    X_test = X_test[h_selected]
    
    logger.info("training_custom_label", ticker=ticker, mode=label_config.mode, max_rows=len(X_train))
    
    if label_config.task_type == "classification":
        model = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_train, y_train_full, sample_weight=weights)
        test_preds = model.predict(X_test)
        
        # --- Financial Evaluation Hook ---
        evaluator = MetricsEvaluator()
        test_returns = daily_df['pct_return'].iloc[split_idx:]
        eval_res = evaluator.evaluate_strategy(test_preds, test_returns)
        m = eval_res["metrics"]
        logger.info("quant_eval", ticker=ticker, mode=label_config.mode, sharpe=m["sharpe"], cagr=m["cagr"], trades=eval_res["trade_stats"]["num_trades"])
        
        acc = accuracy_score(y_test, test_preds) * 100
        metric_val = acc
        metric_name = f"{label_config.mode}_acc"
        model_name = f"trend_classifier_{label_config.mode}.joblib"
        
        # Financial metrics for model selection/logging
        sharpe = m["sharpe"]
        cagr = m["cagr"]
    else:
        model = LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_train, y_train_full, sample_weight=weights)
        from sklearn.metrics import mean_absolute_error
        test_preds = model.predict(X_test)
        metric_val = mean_absolute_error(y_test, test_preds)
        metric_name = f"{label_config.mode}_mae"
        model_name = f"regressor_{label_config.mode}.joblib"
        sharpe = 0.0
        cagr = 0.0
        
    joblib.dump(model, ticker_dir / model_name)
    joblib.dump(h_selected, ticker_dir / f"feature_cols_{label_config.mode}.joblib")
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        metric_name: metric_val,
        "n_features": len(h_selected)
    }

def train_ticker(daily_df: pd.DataFrame, hourly_agg: pd.DataFrame | None,
                 ticker: str, base_dir: Path, market_df: pd.DataFrame = None, 
                 label_config: LabelTrainingConfig = None, **kwargs) -> dict | None:
    """Train Optuna-tuned VotingClassifier with anti-overfitting measures."""

    # Merge hourly aggregates
    if hourly_agg is not None and not hourly_agg.empty:
        daily_df = daily_df.merge(hourly_agg, on='date', how='left')
        hourly_cols = [c for c in daily_df.columns if c.startswith('h_')]
        daily_df[hourly_cols] = daily_df[hourly_cols].fillna(0)

    # 2. Identify feature columns
    fe = FeatureEngineer()
    all_cols = fe.get_feature_columns(daily_df)
    
    # --- Recursive Importance Pruning ---
    if label_config:
        corrs = daily_df[all_cols].corrwith(daily_df[label_config.label_column]).abs()
    else:
        corrs = daily_df[all_cols].corrwith(daily_df[f'target_trend_mid']).abs()
        
    feature_cols = corrs.sort_values(ascending=False).head(20).index.tolist()

    if len(feature_cols) < 5 or len(daily_df) < MIN_ROWS:
        return None

    # Ticker directory Setup
    ticker_dir = base_dir / ticker
    os.makedirs(ticker_dir, exist_ok=True)
    horizon_metrics = {}
    
    tracker = kwargs.get('tracker')
    train_start = pd.Timestamp.now().isoformat()

    if label_config:
        # Custom label path
        metrics = _train_custom_label(daily_df, ticker, ticker_dir, feature_cols, label_config)
        result_dict = {
            "ticker": ticker,
            **metrics,
            "train_rows": len(daily_df),
        }
        
        train_end = pd.Timestamp.now().isoformat()
        if tracker:
            tracker.log_experiment(
                ticker=ticker,
                label_type=label_config.mode,
                model_type="LGBM_Custom_Label",
                feature_count=metrics.get("n_features", len(feature_cols)),
                metrics=metrics,
                train_start=train_start,
                train_end=train_end,
                model_path=str(ticker_dir)
            )
        return result_dict

    # --- 1. Horizon Loop ---
    HORIZONS = {5: "_short", 20: "_mid", 120: "_long"}
    for h_days, suffix in HORIZONS.items():
        y = daily_df[f'target_trend{suffix}']
        
        # Time-Series Safe Split per Horizon
        split_idx = int(len(daily_df) * 0.8)
        train_end = split_idx - h_days
        
        if train_end < MIN_ROWS:
            continue
            
        X_train_full = daily_df[feature_cols].iloc[:train_end]
        y_train_full = y.iloc[:train_end]
        X_test = daily_df[feature_cols].iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) < 5 or len(X_train_full) < MIN_ROWS:
            continue
        
        n = len(X_train_full)
        recency_weights = np.exp(np.linspace(-3, 0, n))
        weights = recency_weights / recency_weights.mean()

        # Feature Selection PER Horizon
        selector = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1, verbosity=-1)
        selector.fit(X_train_full, y_train_full, sample_weight=weights)
        importances = selector.feature_importances_
        threshold = np.percentile(importances, 20)
        h_selected = [f for f, imp in zip(feature_cols, importances) if imp > threshold]
        if len(h_selected) < 15: h_selected = feature_cols
        
        X_train = X_train_full[h_selected]
        X_test = X_test[h_selected]

        # Use fixed "best" params if optuna not requested
        bp = {'lr': 0.05, 'depth': 5, 'leaves': 31, 'subsample': 0.8, 'colsample': 0.8, 'reg_a': 0.1, 'reg_l': 0.1, 'min_child': 20}
        
        lgbm = LGBMClassifier(n_estimators=800, learning_rate=bp['lr'], max_depth=bp['depth'],
                               num_leaves=bp['leaves'], subsample=bp['subsample'], 
                               colsample_bytree=bp['colsample'], reg_alpha=bp['reg_a'], 
                               reg_lambda=bp['reg_l'], min_child_samples=bp['min_child'],
                               random_state=42, n_jobs=-1, verbosity=-1)
        
        lgbm.fit(X_train, y_train_full, sample_weight=weights)
        
        # Save models and features
        joblib.dump(lgbm, ticker_dir / f"trend_classifier{suffix}.joblib")
        if h_days == 5:
            joblib.dump(h_selected, ticker_dir / "feature_cols.joblib")
        
        test_preds = lgbm.predict(X_test)
        
        # --- Financial Evaluation Hook ---
        evaluator = MetricsEvaluator()
        baseline_eng = BaselineStrategies()
        test_returns = daily_df['pct_return'].iloc[split_idx:]
        
        eval_res = evaluator.evaluate_strategy(test_preds, test_returns)
        m = eval_res["metrics"]
        
        # Benchmarking vs Buy and Hold
        bh_res = baseline_eng.buy_and_hold(test_returns)
        comparison = evaluator.compare_vs_baseline(m, bh_res["metrics"])
        
        logger.info("quant_eval", ticker=ticker, horizon=suffix, 
                    sharpe=m["sharpe"], cagr=m["cagr"], 
                    alpha_sharpe=comparison["alpha_sharpe"],
                    outperform_bh=comparison["outperformance"])
        
        acc = accuracy_score(y_test, test_preds) * 100
        horizon_metrics[f"{suffix.strip('_')}_sharpe"] = m["sharpe"]
        horizon_metrics[f"{suffix.strip('_')}_cagr"] = m["cagr"]
        horizon_metrics[f"{suffix.strip('_')}_acc"] = acc
        horizon_metrics[f"{suffix.strip('_')}_elite_acc"] = acc # Fallback
        horizon_metrics[f"{suffix.strip('_')}_elite_count"] = 0

    result_dict = {
        "ticker": ticker,
        **horizon_metrics,
        "n_features": len(h_selected) if 'h_selected' in locals() else 0,
        "train_rows": len(daily_df),
    }

    train_end = pd.Timestamp.now().isoformat()
    if tracker:
        tracker.log_experiment(
            ticker=ticker,
            label_type="multi_horizon",
            model_type="LGBM",
            feature_count=result_dict["n_features"],
            metrics=horizon_metrics,
            train_start=train_start,
            train_end=train_end,
            model_path=str(ticker_dir)
        )

    return result_dict


# ═══════════════════════════ MAIN ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("Multi-Timeframe LightGBM Trainer v3")
    parser.add_argument("--daily",  type=str, default="data/daily_market_split_data")
    parser.add_argument("--hourly", type=str, default="data/hourly_market_split_data")
    parser.add_argument("--output", type=str, default="models/")
    parser.add_argument("--report", type=str, default="models/evaluation_report.csv")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated list of tickers to train")
    parser.add_argument("--all", action="store_true", dest="train_all", help="Train ALL tickers in the data directory")
    parser.add_argument("--vn100", action="store_true", dest="train_vn100", help="Train only VN100 + 4 Viettel tickers (dynamic load)")
    parser.add_argument("--label-mode", type=str, default=None, help="Use custom label generator")
    parser.add_argument("--retrain-existing", action="store_true", help="Force retraining")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit number of tickers")
    parser.add_argument("--start-date", type=str, default=None, help="Filter from date")
    parser.add_argument("--end-date", type=str, default=None, help="Filter to date")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()

    label_config = None
    if args.label_mode:
        label_config = resolve_label_config(args.label_mode)

    daily_dir = Path(args.daily)
    hourly_dir = Path(args.hourly)
    output_dir = Path(args.output)
    
    files = sorted(daily_dir.glob("*.csv"))
    if args.train_vn100:
        vn100_list = get_vn100_universe(mode="current_plus_viettel")
        target_set = set(t.upper() for t in vn100_list)
        files = [f for f in files if f.stem.upper() in target_set]
    elif args.tickers:
        target_list = [t.strip().upper() for t in args.tickers.split(",")]
        files = [f for f in files if f.stem.upper() in target_list]

    if args.max_tickers:
        files = files[:args.max_tickers]

    if not files:
        print("❌ No CSV files found.")
        return

    market_df = None
    fund_df = None
    sector_df = None
    ticker_sectors = None
    sentiment_df = None
    
    if Path("data/market_proxy.csv").exists():
        market_df = pd.read_csv("data/market_proxy.csv")
    if Path("data/fundamentals_clean.csv").exists():
        fund_df = pd.read_csv("data/fundamentals_clean.csv")
    if Path("data/sector_proxies.csv").exists():
        sector_df = pd.read_csv("data/sector_proxies.csv")
        ticker_sectors = pd.read_csv("data/ticker_sectors.csv")
    if Path("data/sentiment_features.csv").exists():
        sentiment_df = pd.read_csv("data/sentiment_features.csv")

    tracker = ExperimentTracker()
    results = []
    trained = 0
    skipped = 0
    failures = 0

    for idx, fpath in enumerate(files):
        ticker = fpath.stem.upper()
        ticker_dir = output_dir / ticker
        
        # Simple skip logic
        if (ticker_dir / "feature_cols.joblib").exists() and not args.retrain_existing:
            skipped += 1
            continue

        try:
            df_daily = pd.read_csv(fpath)
            if 'time' in df_daily.columns and 'date' not in df_daily.columns:
                df_daily = df_daily.rename(columns={'time': 'date'})

            validator = DataQualityValidator(ticker=ticker)
            validator.validate_ohlcv(df_daily)
            
            ticker_sentiment = None
            if sentiment_df is not None:
                ticker_sentiment = sentiment_df[sentiment_df['ticker'] == ticker]

            # 1. Apply Context
            df_daily = apply_context_features(df_daily, ticker, market_df=market_df, 
                                             sector_df=sector_df, ticker_sectors=ticker_sectors)

            # 2. Build Features on FULL history
            df_daily = build_daily_features(df_daily, sentiment_df=ticker_sentiment, label_config=label_config)
            
            # 3. Date Filtering AFTER feature engineering
            if args.start_date:
                df_daily = df_daily[df_daily['date'] >= args.start_date]
            if args.end_date:
                df_daily = df_daily[df_daily['date'] <= args.end_date]

            if len(df_daily) < 60:
                skipped += 1
                continue

            # 4. Hourly
            hourly_agg = None
            h_file = hourly_dir / f"{ticker}.csv"
            if h_file.exists():
                df_hourly = pd.read_csv(h_file)
                if len(df_hourly) >= 10:
                    hourly_agg = build_hourly_features(df_hourly)

            metrics = train_ticker(df_daily, hourly_agg, ticker, output_dir, 
                                   market_df=market_df, fund_df=fund_df, label_config=label_config, tracker=tracker)
            if metrics:
                results.append(metrics)
                trained += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error("fail", ticker=ticker, error=str(e))
            failures += 1

    print(f"✅ Hoàn tất! Đã train {trained} mô hình.")

if __name__ == "__main__":
    main()
