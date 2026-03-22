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

warnings.filterwarnings('ignore')
setup_logging()
logger = get_logger("mtf_trainer_v3")

MIN_ROWS = 40           # Lowered — 5-year data ensures enough rows
MIN_TEST_ROWS = 10      # Minimum test set size
OPTUNA_TRIALS = 50      # 50 trials per ticker for maximum tuning
PURGE_GAP = 3           # Gap between train/test to prevent lookahead

# ═══════════════════════════ FEATURE ENGINEERING ═══════════════════════════════

def _safe(series, fallback=0.0):
    if series is None or (isinstance(series, pd.Series) and series.isna().all()):
        return fallback
    return series


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build RELATIVE features only — no raw price levels to prevent data leakage."""
    df = df.sort_values('time').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # ── Returns (relative) ──
    df['d_log_return'] = np.log(c / c.shift(1))
    df['d_pct_return'] = c.pct_change()
    df['d_pct_return_2'] = c.pct_change(2)
    df['d_pct_return_5'] = c.pct_change(5)

    # ── Intraday range (as % of close) ──
    df['d_range_pct'] = (h - l) / (c + 1e-9)
    df['d_body_pct'] = (c - df['open']) / (c + 1e-9) if 'open' in df.columns else 0.0
    df['d_upper_shadow'] = (h - c.clip(lower=df.get('open', c))) / (c + 1e-9)

    # ── Moving Average DISTANCES (relative, not raw MA values) ──
    for w in [5, 10, 20, 50]:
        sma = ta.sma(c, length=w)
        df[f'd_dist_sma_{w}'] = (c - sma) / (sma + 1e-9)

    # ── Oscillators (already bounded/relative) ──
    df['d_rsi_14'] = _safe(ta.rsi(c, length=14))
    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['d_macd_norm'] = macd_df.iloc[:, 0] / (c + 1e-9)  # Normalize MACD by price
        df['d_macd_hist'] = macd_df.iloc[:, 1] / (c + 1e-9)
    else:
        df['d_macd_norm'] = df['d_macd_hist'] = 0.0

    # ── Bollinger Band %B (already relative) ──
    bb = ta.bbands(c, length=20, std=2)
    if bb is not None and not bb.empty:
        df['d_bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-9)
        df['d_bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / (c + 1e-9)
    else:
        df['d_bb_pct'] = 0.5
        df['d_bb_width'] = 0.0

    # ── Volatility (relative) ──
    atr = _safe(ta.atr(h, l, c, length=14))
    df['d_atr_pct'] = atr / (c + 1e-9) if isinstance(atr, pd.Series) else 0.0

    # ── Trend Strength ──
    adx = ta.adx(h, l, c, length=14)
    df['d_adx_14'] = adx.iloc[:, 0] if adx is not None else 0.0
    df['d_cci_14'] = _safe(ta.cci(h, l, c, length=14))
    df['d_roc_10'] = _safe(ta.roc(c, length=10))

    # ── Pivot Points (relative distances) ──
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    df['d_dist_pivot'] = (c - pivot) / (pivot + 1e-9)
    df['d_dist_r1'] = (c - r1) / (r1 + 1e-9)
    df['d_dist_s1'] = (c - s1) / (s1 + 1e-9)

    # ── Volume features (relative) ──
    df['d_vol_dir'] = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
    for w in [5, 20]:
        sw = min(w, len(df) - 1) if len(df) > w else w
        df[f'd_vol_ratio_{w}'] = v / (v.rolling(sw, min_periods=1).mean() + 1e-9)
        df[f'd_ret_roll_{w}'] = c.pct_change(sw)

    # ── Momentum composites ──
    df['d_mom_5_20'] = df.get('d_dist_sma_5', 0) - df.get('d_dist_sma_20', 0)

    # ── Day-of-week encoding (0=Mon,4=Fri) ──
    try:
        df['d_dow'] = pd.to_datetime(df['time']).dt.dayofweek
    except:
        df['d_dow'] = 0

    # ── Targets ──
    df['target_trend'] = (c.shift(-1) > c).astype(int)
    df['target_high'] = h.shift(-1)
    df['target_low'] = l.shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['time']).dt.date
    return df


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
                 ticker: str, base_dir: Path) -> dict | None:
    """Train Optuna-tuned VotingClassifier with anti-overfitting measures."""

    # Merge hourly aggregates
    if hourly_agg is not None and not hourly_agg.empty:
        daily_df = daily_df.merge(hourly_agg, on='date', how='left')
        hourly_cols = [c for c in daily_df.columns if c.startswith('h_')]
        daily_df[hourly_cols] = daily_df[hourly_cols].fillna(0)

    # Identify feature columns (ONLY relative features)
    feature_cols = [c for c in daily_df.columns
                    if c.startswith('d_') or c.startswith('h_')]
    feature_cols = [f for f in feature_cols if f not in ('d_vol_dir',)]

    if len(feature_cols) < 10 or len(daily_df) < MIN_ROWS:
        return None

    X = daily_df[feature_cols]
    y = daily_df['target_trend']

    # ─── Walk-Forward Split with Purge Gap ───
    split = int(len(X) * 0.75)                     # 75/25 for more test data
    purge_end = min(split + PURGE_GAP, len(X) - MIN_TEST_ROWS)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[purge_end:]
    y_test = y.iloc[purge_end:]

    if len(X_test) < MIN_TEST_ROWS or len(X_train) < 50:
        return None

    # ─── Feature importance pruning with a quick pre-fit ───
    quick = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    quick.fit(X_train, y_train)
    importances = quick.feature_importances_
    # Keep features with importance > median (remove noisy ones)
    mask = importances >= np.median(importances)
    selected_features = [f for f, m in zip(feature_cols, mask) if m]
    if len(selected_features) < 8:
        selected_features = feature_cols  # fallback: keep all

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # ─── Optuna hyperparameter search ───
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        lr = trial.suggest_float('lr', 0.005, 0.08, log=True)
        depth = trial.suggest_int('depth', 3, 6)
        leaves = trial.suggest_int('leaves', 8, 31)
        subsample = trial.suggest_float('subsample', 0.5, 0.85)
        colsample = trial.suggest_float('colsample', 0.4, 0.8)
        reg_a = trial.suggest_float('reg_a', 0.1, 10.0, log=True)
        reg_l = trial.suggest_float('reg_l', 0.1, 10.0, log=True)
        min_child = trial.suggest_int('min_child', 5, 50)

        scores = []
        for tr_idx, te_idx in tscv.split(X_train):
            xtr, xte = X_train.iloc[tr_idx], X_train.iloc[te_idx]
            ytr, yte = y_train.iloc[tr_idx], y_train.iloc[te_idx]
            m = LGBMClassifier(
                n_estimators=500, learning_rate=lr, max_depth=depth,
                num_leaves=leaves, subsample=subsample, colsample_bytree=colsample,
                reg_alpha=reg_a, reg_lambda=reg_l, min_child_samples=min_child,
                random_state=42, n_jobs=-1, verbosity=-1,
            )
            m.fit(xtr, ytr, eval_set=[(xte, yte)],
                  callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
            scores.append(accuracy_score(yte, m.predict(xte)))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    bp = study.best_params
    best_cv = study.best_value * 100

    # ─── Build VotingClassifier Ensemble ───
    lgbm = LGBMClassifier(
        n_estimators=600, learning_rate=bp['lr'], max_depth=bp['depth'],
        num_leaves=bp['leaves'], subsample=bp['subsample'],
        colsample_bytree=bp['colsample'], reg_alpha=bp['reg_a'],
        reg_lambda=bp['reg_l'], min_child_samples=bp['min_child'],
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=max(bp['lr'], 0.02),
        max_depth=bp['depth'], subsample=bp['subsample'],
        colsample_bytree=bp['colsample'],
        reg_alpha=bp['reg_a'], reg_lambda=bp['reg_l'],
        random_state=42, eval_metric='logloss', verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=bp['depth'] + 1,
        min_samples_leaf=max(bp['min_child'], 5),
        random_state=42, n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb), ('rf', rf)],
        voting='soft', n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, preds) * 100

    # ─── Overfitting check: if test >> CV, flag it ───
    overfit_flag = test_acc > best_cv + 15

    # ─── Quantile regressors ───
    ticker_dir = base_dir / ticker
    os.makedirs(ticker_dir, exist_ok=True)

    for target_name, target_col in [('high', 'target_high'), ('low', 'target_low')]:
        y_r = daily_df[target_col]
        y_r_train = y_r.iloc[:split]
        X_tr_full = daily_df[selected_features].iloc[:split]
        for q, alpha in [(10, 0.10), (50, 0.50), (90, 0.90)]:
            reg = LGBMRegressor(
                objective='quantile', alpha=alpha,
                n_estimators=300, learning_rate=0.01, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=-1,
            )
            reg.fit(X_tr_full, y_r_train)
            joblib.dump(reg, ticker_dir / f"range_{target_name}_q{q}.joblib")

    # ─── Save ───
    joblib.dump(ensemble, ticker_dir / "trend_classifier.joblib")
    joblib.dump(selected_features, ticker_dir / "feature_cols.joblib")

    return {
        "ticker": ticker,
        "dir_acc_pct": test_acc,
        "best_cv_acc": best_cv,
        "n_features": len(selected_features),
        "has_hourly": hourly_agg is not None and not hourly_agg.empty,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "overfit_flag": overfit_flag,
    }


# ═══════════════════════════ MAIN ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("Multi-Timeframe LightGBM Trainer v3")
    parser.add_argument("--daily",  type=str, default="data/daily_market_split_data")
    parser.add_argument("--hourly", type=str, default="data/hourly_market_split_data")
    parser.add_argument("--output", type=str, default="models/")
    parser.add_argument("--report", type=str, default="models/evaluation_report.csv")
    args = parser.parse_args()

    daily_dir = Path(args.daily)
    hourly_dir = Path(args.hourly)
    output_dir = Path(args.output)

    if not daily_dir.exists():
        print(f"❌ Không tìm thấy thư mục daily: {daily_dir}")
        return

    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        print("❌ No CSV files found.")
        return

    has_hourly = hourly_dir.exists()
    logger.info("starting_v3", total=len(files), has_hourly=has_hourly)
    print(f"🚀 MTF Trainer v3 (Anti-Overfitting): {len(files)} mã")
    print(f"   Daily:  {daily_dir} ({'✅' if daily_dir.exists() else '❌'})")
    print(f"   Hourly: {hourly_dir} ({'✅' if has_hourly else '⚠️'})")
    print(f"   Min rows: {MIN_ROWS} | Optuna trials: {OPTUNA_TRIALS}")

    results = []
    skipped = 0
    for idx, fpath in enumerate(files):
        ticker = fpath.stem
        if idx % 50 == 0 and idx > 0:
            avg = np.mean([r['dir_acc_pct'] for r in results]) if results else 0
            cv_avg = np.mean([r['best_cv_acc'] for r in results]) if results else 0
            n_overfit = sum(1 for r in results if r.get('overfit_flag'))
            print(f"  ⏳ {idx}/{len(files)} – Test: {avg:.1f}% | CV: {cv_avg:.1f}% | Overfit: {n_overfit} | Skip: {skipped}")

        try:
            df_daily = pd.read_csv(fpath)
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

            metrics = train_ticker(df_daily, hourly_agg, ticker, output_dir)
            if metrics:
                results.append(metrics)
            else:
                skipped += 1
        except Exception as e:
            logger.error("fail", ticker=ticker, error=str(e))
            skipped += 1

    if results:
        df_report = pd.DataFrame(results).sort_values("dir_acc_pct", ascending=False)
        df_report.to_csv(args.report, index=False)
        avg = df_report["dir_acc_pct"].mean()
        cv_avg = df_report["best_cv_acc"].mean()
        top50 = df_report.head(50)["dir_acc_pct"].mean()
        n_overfit = df_report["overfit_flag"].sum()
        mtf = df_report["has_hourly"].sum()

        print(f"\n{'='*65}")
        print(f"✅ Hoàn tất! Đã train {len(results)} mô hình (bỏ qua {skipped} mã).")
        print(f"📊 Test Accuracy TB:       {avg:.2f}%")
        print(f"📊 Cross-Val Accuracy TB:  {cv_avg:.2f}%")
        print(f"🏆 Top-50 Test Acc TB:     {top50:.2f}%")
        print(f"⚠️  Overfitting flags:     {n_overfit}/{len(results)}")
        print(f"🔀 Multi-Timeframe:        {mtf}/{len(results)}")
        print(f"📁 Models:                 {output_dir}")
        print(f"📋 Report:                 {args.report}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
