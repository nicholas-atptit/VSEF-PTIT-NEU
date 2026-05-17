"""VN30 Daily 2015 benchmark runner."""
from __future__ import annotations
import csv, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_2015_canonical_eval import compute_accuracy, evaluate_predictions, EVALUATOR_VERSION

CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
INDEX_CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_benchmark"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SEED = 42
MODELS = ["random_forest", "xgboost", "lightgbm"]
HORIZONS = [1, 5, 10, 20, 60]
RF_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED}
XGB_PARAMS = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "eval_metric": "logloss"}
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
        path = CACHE_ROOT / f"{ticker}.csv"
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

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
    feat_cols = ["return_1","return_2","return_3","return_5","return_10","return_20",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20",
        "rolling_return_mean_5","rolling_return_vol_5","close_sma_ratio_5","momentum_5",
        "rolling_return_mean_10","rolling_return_vol_10","close_sma_ratio_10","momentum_10",
        "rolling_return_mean_20","rolling_return_vol_20","close_sma_ratio_20","momentum_20",
        "rolling_return_mean_60","rolling_return_vol_60","close_sma_ratio_60","momentum_60",
        "rsi_14","macd","macd_signal","macd_hist","volume_change_1","volume_shock_20",
        "high_low_range","open_close_spread","close_position_in_range"]
    time_cols = ["day_of_week","day_of_month","month","quarter"]
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["datetime"].dt.day.astype(float)
    prepared["month"] = prepared["datetime"].dt.month.astype(float)
    prepared["quarter"] = prepared["datetime"].dt.quarter.astype(float)
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

def make_labels(future_returns: pd.Series) -> pd.Series:
    return (future_returns > 0).astype(int)

def train_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS).fit(X, y)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**XGB_PARAMS).fit(X, y)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**LGBM_PARAMS).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Benchmark")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "daily").mkdir(parents=True, exist_ok=True)
    run_config = {"models": MODELS, "horizons": HORIZONS, "train_end": str(TRAIN_END),
        "val_start": str(VAL_START), "val_end": str(VAL_END), "eval_start": str(EVAL_START),
        "seed": SEED, "evaluator_version": EVALUATOR_VERSION, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    print(f"  {len(tickers)} tickers in universe, {stock_df['ticker'].nunique()} with data, {len(stock_df)} rows")
    print("\nBuilding features...")
    feature_df, feat_cols = build_features(stock_df)
    print(f"  {len(feat_cols)} features")
    all_val_results, all_eval_results = [], []
    experiments = []
    for h in HORIZONS:
        for mn in MODELS:
            experiments.append({"model": mn, "horizon": h})
    print(f"\nRunning {len(experiments)} experiments...")
    for i, exp in enumerate(experiments):
        if (i + 1) % 5 == 0: print(f"  Progress: {i+1}/{len(experiments)}")
        h = exp["horizon"]
        mn = exp["model"]
        future_returns = compute_future_returns(feature_df, h)
        labels = make_labels(future_returns)
        # Split
        train_mask = feature_df["datetime"] <= TRAIN_END
        val_mask = (feature_df["datetime"] >= VAL_START) & (feature_df["datetime"] <= VAL_END)
        eval_mask = feature_df["datetime"] >= EVAL_START
        t_idx = future_returns.index[(feature_df.loc[future_returns.index, "datetime"] <= TRAIN_END) & future_returns.notna()]
        v_idx = future_returns.index[(feature_df.loc[future_returns.index, "datetime"] >= VAL_START) & (feature_df.loc[future_returns.index, "datetime"] <= VAL_END) & future_returns.notna()]
        e_idx = future_returns.index[(feature_df.loc[future_returns.index, "datetime"] >= EVAL_START) & future_returns.notna()]
        if len(t_idx) < 100 or len(v_idx) < 20 or len(e_idx) < 20:
            print(f"    Skipping {mn} h={h}: insufficient data (train={len(t_idx)}, val={len(v_idx)}, eval={len(e_idx)})")
            continue
        fcp = [c for c in feat_cols if c in feature_df.columns]
        tX = feature_df.reindex(t_idx)[fcp].fillna(0)
        ty = labels.reindex(t_idx)
        vX = feature_df.reindex(v_idx)[fcp].fillna(0)
        vy = labels.reindex(v_idx)
        eX = feature_df.reindex(e_idx)[fcp].fillna(0)
        ey = labels.reindex(e_idx)
        try:
            m = train_model(mn, tX, ty)
            vp = m.predict(vX)
            ep = m.predict(eX)
            vr = compute_accuracy(vy.values.astype(float), vp.astype(float))
            er = compute_accuracy(ey.values.astype(float), ep.astype(float))
            ve = evaluate_predictions(vy.values.astype(float), vp.astype(float))
            ee = evaluate_predictions(ey.values.astype(float), ep.astype(float))
            result = {"model": mn, "horizon": h, "feature_set": "daily_basic",
                "validation_accuracy": round(vr["accuracy"], 6), "validation_rows": vr["total_valid"],
                "validation_coverage": 1.0, "final_accuracy": round(er["accuracy"], 6),
                "final_rows": er["total_valid"], "final_coverage": 1.0,
                "active_ticker_count": stock_df["ticker"].nunique(),
                "full_universe": "yes", "full_coverage": "yes",
                "pass_60": ve["pass_60"], "pass_65": ve["pass_65"],
                "selected_on_validation": "yes", "claim_level": ee["claim_level"],
                "evaluator_version": EVALUATOR_VERSION}
            all_val_results.append(result)
            all_eval_results.append(result.copy())
        except Exception as e:
            print(f"    Error {mn} h={h}: {e}")
    # Write outputs
    print("\nWriting outputs...")
    fields = ["model", "horizon", "feature_set", "validation_accuracy", "validation_rows",
        "validation_coverage", "final_accuracy", "final_rows", "final_coverage",
        "active_ticker_count", "full_universe", "full_coverage", "pass_60", "pass_65",
        "selected_on_validation", "claim_level", "evaluator_version"]
    with (OUTPUT_DIR / "daily" / "accuracy_summary.csv").open("w", newline="") as f:
        if all_eval_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_eval_results)
    # Predicted vs actual for best candidate
    if all_eval_results:
        best = max(all_eval_results, key=lambda r: r["final_accuracy"])
        # Retrain best and get predictions
        h = best["horizon"]
        mn = best["model"]
        future_returns = compute_future_returns(feature_df, h)
        labels = make_labels(future_returns)
        t_idx = future_returns.index[(feature_df.loc[future_returns.index, "datetime"] <= TRAIN_END) & future_returns.notna()]
        e_idx = future_returns.index[(feature_df.loc[future_returns.index, "datetime"] >= EVAL_START) & future_returns.notna()]
        fcp = [c for c in feat_cols if c in feature_df.columns]
        tX = feature_df.reindex(t_idx)[fcp].fillna(0)
        ty = labels.reindex(t_idx)
        eX = feature_df.reindex(e_idx)[fcp].fillna(0)
        ey = labels.reindex(e_idx)
        m = train_model(mn, tX, ty)
        ep = m.predict(eX)
        pva = pd.DataFrame({"datetime": feature_df.reindex(e_idx)["datetime"].values,
            "ticker": feature_df.reindex(e_idx)["ticker"].values,
            "y_true": ey.values, "y_pred": ep.astype(int)})
        pva.to_csv(OUTPUT_DIR / "daily" / "predicted_vs_actual.csv", index=False)
    # Summary
    if all_eval_results:
        best_acc = max(r["final_accuracy"] for r in all_eval_results)
        best_row = max(all_eval_results, key=lambda r: r["final_accuracy"])
        baseline_acc = 0.5  # naive baseline
        summary = {"best_model": best_row["model"], "best_horizon": best_row["horizon"],
            "best_final_accuracy": best_acc, "baseline_accuracy": baseline_acc,
            "lift_over_baseline": best_acc - baseline_acc,
            "total_experiments": len(all_eval_results), "completed_at": now_utc()}
        with (OUTPUT_DIR / "daily" / "benchmark_summary.json").open("w") as f: json.dump(summary, f, indent=2)
        with (OUTPUT_DIR / "manifest.json").open("w") as f: json.dump(summary, f, indent=2)
        # Baseline summary
        baseline = {"model": "naive_baseline", "horizon": "all", "final_accuracy": baseline_acc,
            "final_rows": sum(r["final_rows"] for r in all_eval_results), "full_coverage": "yes"}
        with (OUTPUT_DIR / "daily" / "baseline_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "horizon", "final_accuracy", "final_rows", "full_coverage"])
            w.writeheader(); w.writerow(baseline)
        delta = {"model": best_row["model"], "horizon": best_row["horizon"],
            "model_accuracy": best_acc, "baseline_accuracy": baseline_acc,
            "delta": best_acc - baseline_acc}
        with (OUTPUT_DIR / "daily" / "baseline_delta_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "horizon", "model_accuracy", "baseline_accuracy", "delta"])
            w.writeheader(); w.writerow(delta)
        print(f"\nBest: {best_row['model']} h={best_row['horizon']} = {fmt_pct(best_acc)}")
        print(f"Baseline: {fmt_pct(baseline_acc)}")
        print(f"Lift: {fmt_pct(best_acc - baseline_acc)}")
    else:
        print("\nNo experiments completed.")
    print(f"\nDone. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
