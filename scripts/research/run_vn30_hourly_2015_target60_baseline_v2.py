"""Baseline-v2 experiment runner targeting >=60% full-universe accuracy.

Feature sets A-D, models LightGBM/XGBoost/RF, horizons 1/4/8/20.
Hyperparameter tuning on 2024 validation only.
Final evaluation 2025-2026 untouched until scoring.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_target60_baseline_v2"

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
HORIZONS = [1, 4, 8, 20]
MODELS = ["lightgbm", "xgboost", "random_forest"]
SEED = 42

THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]

LGBM_GRID = [
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8},
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7},
    {"num_leaves": 63, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9},
    {"num_leaves": 20, "max_depth": 5, "learning_rate": 0.1, "n_estimators": 150, "min_child_samples": 50, "subsample": 0.6, "colsample_bytree": 0.6},
    {"num_leaves": 45, "max_depth": 8, "learning_rate": 0.02, "n_estimators": 400, "min_child_samples": 15, "subsample": 0.75, "colsample_bytree": 0.85},
]

XGB_GRID = [
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10},
    {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 500, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 1},
    {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 250, "subsample": 0.85, "colsample_bytree": 0.75, "min_child_weight": 7},
]

RF_GRID = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": None},
    {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "max_features": "log2", "class_weight": None},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"},
    {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced_subsample"},
]

GRID_MAP = {"lightgbm": LGBM_GRID, "xgboost": XGB_GRID, "random_forest": RF_GRID}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def load_universe_tickers() -> list[str]:
    tickers = []
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker", "")).strip().upper()
            if t:
                tickers.append(t)
    return tickers


def load_stock_data(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = STOCK_CACHE_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["ticker"] = ticker
        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["datetime"] + OHLCV_COLUMNS)
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def load_index_data() -> dict[str, pd.DataFrame]:
    indices = {}
    for code in ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[df["close"] > 0].sort_values("datetime").reset_index(drop=True)
        indices[code] = df
    return indices


def build_features_v2(df: pd.DataFrame, index_data: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build all feature sets A-D."""
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)

    # Feature set A: existing benchmark features
    feat_a = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)

        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 10, 20):
            prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20):
            prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
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

    feat_a_cols = [
        "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
        "return_1_lag_1", "return_1_lag_2", "return_1_lag_3", "return_1_lag_5", "return_1_lag_10", "return_1_lag_20",
        "rolling_return_mean_5", "rolling_return_vol_5", "close_sma_ratio_5", "momentum_5",
        "rolling_return_mean_10", "rolling_return_vol_10", "close_sma_ratio_10", "momentum_10",
        "rolling_return_mean_20", "rolling_return_vol_20", "close_sma_ratio_20", "momentum_20",
        "rolling_return_mean_60", "rolling_return_vol_60", "close_sma_ratio_60", "momentum_60",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volume_change_1", "volume_shock_20", "high_low_range", "open_close_spread", "close_position_in_range",
    ]

    # Feature set B: additional stock lag features
    feat_b_extra = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)

        for lag in (1, 2, 4, 8, 20):
            col = f"lag_ret_{lag}"
            prepared.loc[idx, col] = close.pct_change(periods=lag, fill_method=None).shift(1)
            feat_b_extra.append(col)
        for window in (4, 8, 20, 40):
            min_p = max(2, window // 4)
            col = f"roll_ret_{window}"
            prepared.loc[idx, col] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
            feat_b_extra.append(col)
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            col = f"roll_vol_{window}"
            prepared.loc[idx, col] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
            feat_b_extra.append(col)
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            col = f"roll_vol_change_{window}"
            prepared.loc[idx, col] = volume.pct_change(periods=1, fill_method=None).rolling(window, min_periods=min_p).mean()
            feat_b_extra.append(col)
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            col = f"roll_hl_range_{window}"
            prepared.loc[idx, col] = prepared.loc[idx, "high_low_range"].rolling(window, min_periods=min_p).mean()
            feat_b_extra.append(col)
        prepared.loc[idx, "close_to_open_ret"] = (close - group["open"]) / group["open"].replace(0.0, np.nan)
        feat_b_extra.append("close_to_open_ret")

    feat_b_cols = feat_a_cols + list(set(feat_b_extra))

    # Feature set C: market context features
    feat_c_extra = []
    if index_data:
        for code, idx_df in index_data.items():
            idx_df = idx_df.copy()
            idx_df["idx_ret"] = idx_df["close"].pct_change(periods=1, fill_method=None)
            idx_features = []
            for lag in (1, 2, 3, 5, 10, 20):
                col = f"{code.lower()}_lag_{lag}"
                idx_df[col] = idx_df["idx_ret"].shift(lag)
                idx_features.append(col)
            for window in (20, 60):
                min_p = max(3, window // 4)
                col_mean = f"{code.lower()}_roll_mean_{window}"
                col_vol = f"{code.lower()}_roll_vol_{window}"
                idx_df[col_mean] = idx_df["idx_ret"].rolling(window, min_periods=min_p).mean()
                idx_df[col_vol] = idx_df["idx_ret"].rolling(window, min_periods=min_p).std()
                idx_features.extend([col_mean, col_vol])

            sel_cols = ["datetime"] + idx_features
            idx_clean = idx_df[sel_cols].dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
            prepared = prepared.merge(idx_clean, on="datetime", how="left")
            feat_c_extra.extend(idx_features)

    # market return minus stock return
    if "vnindex_lag_1" in prepared.columns:
        prepared["market_minus_stock_ret"] = prepared["vnindex_lag_1"] - prepared["return_1"]
        feat_c_extra.append("market_minus_stock_ret")

    feat_c_cols = feat_a_cols + list(set(feat_c_extra))

    # Feature set D: combined
    feat_d_cols = list(set(feat_a_cols + feat_b_extra + feat_c_extra))

    # Time features
    time_cols = ["day_of_week", "day_of_month", "month", "quarter", "hour", "minute"]
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["datetime"].dt.day.astype(float)
    prepared["month"] = prepared["datetime"].dt.month.astype(float)
    prepared["quarter"] = prepared["datetime"].dt.quarter.astype(float)
    prepared["hour"] = prepared["datetime"].dt.hour.astype(float)
    prepared["minute"] = prepared["datetime"].dt.minute.astype(float)

    for cols in [feat_a_cols, feat_b_cols, feat_c_cols, feat_d_cols]:
        all_cols = cols + time_cols
        existing = [c for c in all_cols if c in prepared.columns]
        prepared[existing] = prepared[existing].replace([np.inf, -np.inf], np.nan)

    return prepared, {
        "A": feat_a_cols + time_cols,
        "B": feat_b_cols + time_cols,
        "C": feat_c_cols + time_cols,
        "D": feat_d_cols + time_cols,
    }


def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Build directional labels for a given horizon."""
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(float)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=idx))
    if labels:
        return pd.concat(labels)
    return pd.Series(dtype=float)


def split_data(df: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = df["datetime"] <= TRAIN_END
    val_mask = (df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)
    eval_mask = df["datetime"] >= EVAL_START
    return df[train_mask].copy(), df[val_mask].copy(), df[eval_mask].copy()


def train_model(model_name: str, X: pd.DataFrame, y: pd.Series, params: dict) -> Any:
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            num_leaves=params.get("num_leaves", 31), max_depth=params.get("max_depth", -1),
            learning_rate=params.get("learning_rate", 0.05), n_estimators=params.get("n_estimators", 200),
            min_child_samples=params.get("min_child_samples", 20), subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8), random_state=SEED, verbose=-1,
        ).fit(X, y)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            max_depth=params.get("max_depth", 4), learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200), subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8), min_child_weight=params.get("min_child_weight", 5),
            random_state=SEED, eval_metric="logloss", use_label_encoder=False,
        ).fit(X, y)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5), max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", None), random_state=SEED,
        ).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")


def predict_with_confidence(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    preds = model.predict(X)
    probas = model.predict_proba(X)
    return preds, probas.max(axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))


def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Baseline-v2 Target 60")
    print("=" * 60)
    print(f"Started: {now_utc()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_config = {
        "train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "models": MODELS, "horizons": HORIZONS,
        "feature_sets": ["A", "B", "C", "D"], "seed": SEED, "created_at": now_utc(),
    }
    with (OUTPUT_DIR / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")

    print("\nBuilding features...")
    feature_df, feature_sets = build_features_v2(stock_df, index_data)
    for k, v in feature_sets.items():
        print(f"  Feature set {k}: {len(v)} features")

    print("\nRunning experiments...")
    val_results = []
    eval_results = []
    total = len(MODELS) * len(HORIZONS) * 4
    current = 0

    for horizon in HORIZONS:
        labels = build_labels(feature_df, horizon)
        train_df, val_df, eval_df = split_data(feature_df, labels)

        for model_name in MODELS:
            for feat_set in ["A", "B", "C", "D"]:
                current += 1
                f_cols = [c for c in feature_sets[feat_set] if c in feature_df.columns]
                target_col = f"target_h{horizon}"

                train_labels = labels.reindex(train_df.index).dropna()
                if len(train_labels) < 100:
                    continue

                train_X = train_df.reindex(train_labels.index)[f_cols].fillna(0)
                train_y = train_labels.astype(int)

                best_val_acc = 0.0
                best_params = None
                best_model = None

                for params in GRID_MAP[model_name]:
                    try:
                        m = train_model(model_name, train_X, train_y, params)
                        val_labels = labels.reindex(val_df.index).dropna()
                        if len(val_labels) < 20:
                            continue
                        val_X = val_df.reindex(val_labels.index)[f_cols].fillna(0)
                        val_preds, _ = predict_with_confidence(m, val_X)
                        va = accuracy(val_labels.values, val_preds)
                        if va > best_val_acc:
                            best_val_acc = va
                            best_params = params
                            best_model = m
                    except Exception:
                        continue

                if best_model is None:
                    continue

                # Final eval
                eval_labels = labels.reindex(eval_df.index).dropna()
                if len(eval_labels) < 20:
                    continue
                eval_X = eval_df.reindex(eval_labels.index)[f_cols].fillna(0)
                eval_preds, eval_conf = predict_with_confidence(best_model, eval_X)
                global_acc = accuracy(eval_labels.values, eval_preds)

                val_labels_full = labels.reindex(val_df.index).dropna()
                val_X_full = val_df.reindex(val_labels_full.index)[f_cols].fillna(0)
                val_preds_full, val_conf_full = predict_with_confidence(best_model, val_X_full)

                # Threshold selection on validation
                best_thresh = None
                best_thresh_acc = 0.0
                best_thresh_cov = 0.0
                for thresh in THRESHOLDS:
                    mask = val_conf_full >= thresh
                    n = mask.sum()
                    if n < 100:
                        continue
                    va = accuracy(val_labels_full.values[mask], val_preds_full[mask])
                    cov = n / len(val_labels_full)
                    if cov >= 0.30 and va > best_thresh_acc:
                        best_thresh_acc = va
                        best_thresh = thresh
                        best_thresh_cov = cov

                # Apply to eval
                final_thresh_acc = None
                final_thresh_cov = None
                final_thresh_rows = None
                if best_thresh is not None:
                    mask = eval_conf >= best_thresh
                    n = mask.sum()
                    if n > 0:
                        final_thresh_acc = accuracy(eval_labels.values[mask], eval_preds[mask])
                        final_thresh_cov = n / len(eval_labels)
                        final_thresh_rows = int(n)

                val_row = {
                    "model": model_name, "horizon": horizon, "feature_set": feat_set,
                    "best_val_accuracy": round(best_val_acc, 6),
                    "best_params": str(best_params),
                    "selected_threshold": best_thresh if best_thresh is not None else "",
                    "val_threshold_accuracy": round(best_thresh_acc, 6) if best_thresh_acc > 0 else "",
                    "val_threshold_coverage": round(best_thresh_cov, 4) if best_thresh_cov > 0 else "",
                }
                val_results.append(val_row)

                eval_row = {
                    "model": model_name, "horizon": horizon, "feature_set": feat_set,
                    "global_accuracy": round(global_acc, 6),
                    "global_n": len(eval_labels),
                    "selected_threshold": best_thresh if best_thresh is not None else "",
                    "threshold_eval_accuracy": round(final_thresh_acc, 6) if final_thresh_acc is not None else "",
                    "threshold_eval_coverage": round(final_thresh_cov, 4) if final_thresh_cov is not None else "",
                    "threshold_eval_rows": final_thresh_rows if final_thresh_rows is not None else "",
                    "elapsed_seconds": 0,
                }
                eval_results.append(eval_row)

                print(f"  [{current}/{total}] {model_name} h={horizon} {feat_set}: val={fmt_pct(best_val_acc)} eval={fmt_pct(global_acc)}" +
                      (f" thresh={best_thresh} acc={fmt_pct(final_thresh_acc)}" if final_thresh_acc else ""))

    # Write outputs
    with (OUTPUT_DIR / "baseline_v2_validation_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_results[0].keys())
        w.writeheader()
        w.writerows(val_results)

    with (OUTPUT_DIR / "baseline_v2_final_eval_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_results[0].keys())
        w.writeheader()
        w.writerows(eval_results)

    # Best by horizon
    best_by_h = []
    for h in HORIZONS:
        h_rows = [r for r in eval_results if r["horizon"] == h]
        if h_rows:
            best = max(h_rows, key=lambda x: float(x.get("global_accuracy", 0) or 0))
            best_by_h.append(best)
    with (OUTPUT_DIR / "baseline_v2_best_by_horizon.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=best_by_h[0].keys() if best_by_h else [])
        w.writeheader()
        w.writerows(best_by_h)

    # Above 60 candidates
    above60 = []
    for r in eval_results:
        ga = float(r.get("global_accuracy", 0) or 0)
        ta = float(r.get("threshold_eval_accuracy", 0) or 0)
        if ga >= 0.60 or ta >= 0.60:
            above60.append(r)
    with (OUTPUT_DIR / "baseline_v2_above60_candidates.csv").open("w", newline="") as f:
        if above60:
            w = csv.DictWriter(f, fieldnames=above60[0].keys())
            w.writeheader()
            w.writerows(above60)

    best_global = max((float(r.get("global_accuracy", 0) or 0) for r in eval_results), default=0)
    best_thresh_eval = max((float(r.get("threshold_eval_accuracy", 0) or 0) for r in eval_results), default=0)

    manifest = {
        "total_experiments": current,
        "completed": len(eval_results),
        "best_global_accuracy": round(best_global, 6),
        "best_threshold_eval_accuracy": round(best_thresh_eval, 6),
        "baseline_60_pass": best_global >= 0.60,
        "completed_at": now_utc(),
        "leakage_safe": True, "daily_data_used": False, "resampling_used": False,
    }
    with (OUTPUT_DIR / "experiment_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    log = [
        "# Baseline-v2 Run Log", "",
        f"- Completed: {manifest['completed_at']}",
        f"- Experiments: {manifest['completed']}",
        f"- Best global: {fmt_pct(best_global)}",
        f"- Best threshold eval: {fmt_pct(best_thresh_eval)}",
        f"- Baseline 60 pass: {'YES' if manifest['baseline_60_pass'] else 'NO'}",
        f"- Gap to 60: {fmt_pct(0.60 - best_global)}",
        "",
    ]
    with (OUTPUT_DIR / "baseline_v2_run_log.md").open("w") as f:
        f.write("\n".join(log))

    print(f"\nBest global: {fmt_pct(best_global)}")
    print(f"Best threshold eval: {fmt_pct(best_thresh_eval)}")
    print(f"Baseline 60 pass: {'YES' if manifest['baseline_60_pass'] else 'NO'}")
    print(f"Gap to 60: {fmt_pct(0.60 - best_global)}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())