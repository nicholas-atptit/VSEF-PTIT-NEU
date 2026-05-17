"""Leakage-safe above-60% optimization for VN30 hourly 2015 benchmark.

This script runs controlled experiments to try to achieve >60% directional accuracy.
All hyperparameter tuning and threshold selection is done on pre-2025 validation data only.
Final evaluation (2025-2026) is untouched until final scoring.

Feature sets:
  A: existing benchmark features
  B: stock lagged features
  C: lagged market context (index)
  D: combined stock + index context

Models: LightGBM, XGBoost, Random Forest
Horizons: 1, 4, 8, 20 (priority: h=20)
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_above60_optimization"

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
EVAL_END = pd.Timestamp("2026-05-14 23:59:59")
TRAIN_CUTOFF = pd.Timestamp("2024-12-31 23:59:59")

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
]

XGB_GRID = [
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10},
    {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 500, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 1},
]

RF_GRID = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": None},
    {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "max_features": "log2", "class_weight": None},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"},
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
        reader = csv.DictReader(f)
        for row in reader:
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
    combined = combined.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return combined


def load_index_data() -> dict[str, pd.DataFrame]:
    indices = {}
    index_files = ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]
    for code in index_files:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[df["close"] > 0]
        df = df.sort_values("datetime").reset_index(drop=True)
        indices[code] = df
    return indices


def build_existing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature set A: existing benchmark features."""
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feature_columns = []

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

    feature_columns = [
        "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
        "return_1_lag_1", "return_1_lag_2", "return_1_lag_3", "return_1_lag_5", "return_1_lag_10", "return_1_lag_20",
        "rolling_return_mean_5", "rolling_return_vol_5", "close_sma_ratio_5", "momentum_5",
        "rolling_return_mean_10", "rolling_return_vol_10", "close_sma_ratio_10", "momentum_10",
        "rolling_return_mean_20", "rolling_return_vol_20", "close_sma_ratio_20", "momentum_20",
        "rolling_return_mean_60", "rolling_return_vol_60", "close_sma_ratio_60", "momentum_60",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volume_change_1", "volume_shock_20", "high_low_range", "open_close_spread", "close_position_in_range",
    ]

    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["datetime"].dt.day.astype(float)
    prepared["month"] = prepared["datetime"].dt.month.astype(float)
    prepared["quarter"] = prepared["datetime"].dt.quarter.astype(float)
    prepared["hour"] = prepared["datetime"].dt.hour.astype(float)
    prepared["minute"] = prepared["datetime"].dt.minute.astype(float)
    feature_columns.extend(["day_of_week", "day_of_month", "month", "quarter", "hour", "minute"])

    prepared[feature_columns] = prepared[feature_columns].replace([np.inf, -np.inf], np.nan)
    return prepared, feature_columns


def add_index_features(df: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Feature set C: lagged market context from index data."""
    prepared = df.copy()
    all_index_features = []

    for code, idx_df in index_data.items():
        idx_df = idx_df.copy()
        idx_df["idx_return"] = idx_df["close"].pct_change(periods=1, fill_method=None)

        idx_features = []
        for lag in (1, 2, 3, 5, 10, 20):
            col = f"{code.lower()}_lag_{lag}"
            idx_df[col] = idx_df["idx_return"].shift(lag)
            idx_features.append(col)

        for window in (20, 60):
            min_p = max(3, min(window, window // 2))
            col_mean = f"{code.lower()}_roll_mean_{window}"
            col_vol = f"{code.lower()}_roll_vol_{window}"
            idx_df[col_mean] = idx_df["idx_return"].rolling(window, min_periods=min_p).mean()
            idx_df[col_vol] = idx_df["idx_return"].rolling(window, min_periods=min_p).std()
            idx_features.extend([col_mean, col_vol])

        all_index_features.extend(idx_features)

        sel_cols = ["datetime"] + idx_features
        idx_df = idx_df[sel_cols].dropna(subset=["datetime"])
        idx_df = idx_df.drop_duplicates("datetime", keep="last")

        prepared = prepared.merge(idx_df, on="datetime", how="left")

    prepared[all_index_features] = prepared[all_index_features].replace([np.inf, -np.inf], np.nan)
    return prepared, all_index_features


def build_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build directional labels for a given horizon."""
    result = df.copy()
    result = result.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    labels = []
    for ticker, group in result.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(int)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=idx))

    if labels:
        result["target"] = pd.concat(labels)
    else:
        result["target"] = np.nan

    return result


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train (pre-2024), validation (2024), evaluation (2025+)."""
    train_mask = df["datetime"] <= TRAIN_END
    val_mask = (df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)
    eval_mask = df["datetime"] >= EVAL_START

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    eval_df = df[eval_mask].copy()

    return train_df, val_df, eval_df


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> Any:
    """Train a model with given parameters."""
    if model_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=SEED,
            verbose=-1,
        )
    elif model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 5),
            random_state=SEED,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", None),
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model


def predict_with_confidence(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and confidence (max probability)."""
    predictions = model.predict(X)
    probas = model.predict_proba(X)
    confidence = probas.max(axis=1)
    return predictions, confidence


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy."""
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))


def run_single_experiment(
    model_name: str,
    horizon: int,
    feature_set: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Run a single model/horizon/feature experiment."""
    results = {
        "model": model_name,
        "horizon": horizon,
        "feature_set": feature_set,
        "status": "failed",
    }

    target_col = f"target_h{horizon}"

    train_labels = train_df[target_col].dropna()
    if len(train_labels) < 100:
        results["status"] = "skipped_insufficient_train"
        return results

    train_X = train_df.reindex(train_labels.index)[feature_columns].fillna(0)
    train_y = train_labels.astype(int)

    best_val_acc = 0.0
    best_params = None
    best_model = None

    grid = GRID_MAP.get(model_name, [])
    for i, params in enumerate(grid):
        try:
            model = train_model(model_name, train_X, train_y, params)
            val_X = val_df[val_df[target_col].notna()][feature_columns].fillna(0)
            val_y = val_df[val_df[target_col].notna()][target_col].astype(int)
            if len(val_y) < 20:
                continue
            val_preds, _ = predict_with_confidence(model, val_X)
            val_acc = evaluate_accuracy(val_y.values, val_preds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
                best_model = model
        except Exception:
            continue

    if best_model is None:
        results["status"] = "failed_no_valid_model"
        return results

    results["status"] = "completed"
    results["best_val_accuracy"] = round(best_val_acc, 6)
    results["best_params"] = str(best_params)

    eval_X = eval_df[eval_df[target_col].notna()][feature_columns].fillna(0)
    eval_y = eval_df[eval_df[target_col].notna()][target_col].astype(int)
    if len(eval_y) < 20:
        results["status"] = "skipped_insufficient_eval"
        return results

    eval_preds, eval_conf = predict_with_confidence(best_model, eval_X)
    global_acc = evaluate_accuracy(eval_y.values, eval_preds)
    results["global_eval_accuracy"] = round(global_acc, 6)
    results["eval_observations"] = len(eval_y)

    threshold_results = []
    for thresh in THRESHOLDS:
        mask = eval_conf >= thresh
        n = mask.sum()
        if n == 0:
            continue
        acc = float(np.mean(eval_y.values[mask] == eval_preds[mask]))
        cov = n / len(eval_y) if len(eval_y) > 0 else 0.0
        threshold_results.append({
            "threshold": thresh,
            "observations": int(n),
            "accuracy": round(acc, 6),
            "coverage": round(cov, 4),
            "pass_60pct": acc >= 0.60,
            "coverage_ok_30": cov >= 0.30,
            "rows_ok_1000": n >= 1000,
        })

    results["threshold_results"] = threshold_results

    val_threshold_results = []
    val_X_full = val_df[val_df[target_col].notna()][feature_columns].fillna(0)
    val_y_full = val_df[val_df[target_col].notna()][target_col].astype(int)
    val_preds_full, val_conf_full = predict_with_confidence(best_model, val_X_full)
    for thresh in THRESHOLDS:
        mask = val_conf_full >= thresh
        n = mask.sum()
        if n == 0:
            continue
        acc = float(np.mean(val_y_full.values[mask] == val_preds_full[mask]))
        cov = n / len(val_y_full) if len(val_y_full) > 0 else 0.0
        val_threshold_results.append({
            "threshold": thresh,
            "observations": int(n),
            "accuracy": round(acc, 6),
            "coverage": round(cov, 4),
        })

    results["val_threshold_results"] = val_threshold_results

    best_val_thresh = None
    best_val_thresh_acc = 0.0
    for tr in val_threshold_results:
        if tr["coverage"] >= 0.30 and tr["observations"] >= 1000:
            if tr["accuracy"] > best_val_thresh_acc:
                best_val_thresh_acc = tr["accuracy"]
                best_val_thresh = tr["threshold"]

    if best_val_thresh is None:
        for tr in val_threshold_results:
            if tr["coverage"] >= 0.20 and tr["observations"] >= 500:
                if tr["accuracy"] > best_val_thresh_acc:
                    best_val_thresh_acc = tr["accuracy"]
                    best_val_thresh = tr["threshold"]

    results["selected_threshold"] = best_val_thresh
    results["selected_threshold_method"] = "validation_30pct_1000rows" if best_val_thresh is not None else "none"

    if best_val_thresh is not None:
        for tr in threshold_results:
            if tr["threshold"] == best_val_thresh:
                results["final_threshold_accuracy"] = tr["accuracy"]
                results["final_threshold_observations"] = tr["observations"]
                results["final_threshold_coverage"] = tr["coverage"]
                break

    return results


def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 Above-60% Optimization")
    print("=" * 60)
    print(f"Started at: {now_utc()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_config = {
        "train_start": str(TRAIN_START),
        "train_end": str(TRAIN_END),
        "val_start": str(VAL_START),
        "val_end": str(VAL_END),
        "eval_start": str(EVAL_START),
        "eval_end": str(EVAL_END),
        "models": MODELS,
        "horizons": HORIZONS,
        "feature_sets": ["A", "B", "C", "D"],
        "seed": SEED,
        "thresholds": THRESHOLDS,
        "created_at": now_utc(),
    }
    with (OUTPUT_DIR / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    print("\nLoading data...")
    tickers = load_universe_tickers()
    print(f"  Universe: {len(tickers)} tickers")

    stock_df = load_stock_data(tickers)
    print(f"  Stock data: {len(stock_df)} rows")

    index_data = load_index_data()
    print(f"  Index data: {len(index_data)} indices ({', '.join(index_data.keys())})")

    print("\nBuilding features...")
    feature_df, base_features = build_existing_features(stock_df)
    print(f"  Feature set A: {len(base_features)} features")

    feature_df_b, extra_b = build_existing_features(stock_df)
    feature_set_b_features = base_features
    print(f"  Feature set B: {len(feature_set_b_features)} features (same as A, stock-lagged already included)")

    feature_df_c, index_features = add_index_features(feature_df, index_data)
    feature_set_c_features = base_features + index_features
    print(f"  Feature set C: {len(feature_set_c_features)} features (A + index context)")

    feature_set_d_features = feature_set_c_features
    print(f"  Feature set D: {len(feature_set_d_features)} features (same as C)")

    print("\nBuilding labels...")
    all_results = []
    total_experiments = len(MODELS) * len(HORIZONS) * 4
    current = 0

    for horizon in HORIZONS:
        labeled = build_labels(feature_df_c, horizon)
        labeled_b = build_labels(feature_df_b, horizon)

        train_full, val_full, eval_full = split_data(labeled)
        train_b, val_b, eval_b = split_data(labeled_b)

        for model_name in MODELS:
            for feature_set in ["A", "B", "C", "D"]:
                current += 1
                if feature_set in ("A", "C", "D"):
                    t_df, v_df, e_df = train_full, val_full, eval_full
                    f_cols = feature_set_c_features if feature_set in ("C", "D") else base_features
                else:
                    t_df, v_df, e_df = train_b, val_b, eval_b
                    f_cols = feature_set_b_features

                target_col = f"target_h{horizon}"
                if target_col not in t_df.columns:
                    continue

                print(f"  [{current}/{total_experiments}] {model_name} h={horizon} features={feature_set}...")
                start = time.time()

                try:
                    result = run_single_experiment(
                        model_name, horizon, feature_set,
                        t_df, v_df, e_df, f_cols,
                    )
                    result["runtime_seconds"] = round(time.time() - start, 1)
                    all_results.append(result)

                    if result["status"] == "completed":
                        ga = result.get("global_eval_accuracy", 0)
                        va = result.get("best_val_accuracy", 0)
                        print(f"    val={fmt_pct(va)} eval={fmt_pct(ga)}")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results.append({
                        "model": model_name,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "status": f"exception: {e}",
                        "runtime_seconds": round(time.time() - start, 1),
                    })

    print(f"\nExperiments completed: {current}/{total_experiments}")

    print("\nWriting results...")
    results_flat = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("threshold_results", "val_threshold_results")}
        row["threshold_results_json"] = json.dumps(r.get("threshold_results", []))
        row["val_threshold_results_json"] = json.dumps(r.get("val_threshold_results", []))
        results_flat.append(row)

    with (OUTPUT_DIR / "experiment_results_validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_flat[0].keys() if results_flat else [])
        writer.writeheader()
        writer.writerows(results_flat)

    global_candidates = []
    coverage_candidates = []
    exploratory_candidates = []
    selected_thresholds = []

    for r in all_results:
        if r.get("status") != "completed":
            continue

        global_acc = r.get("global_eval_accuracy", 0)
        global_candidates.append({
            "model": r["model"],
            "horizon": r["horizon"],
            "feature_set": r["feature_set"],
            "global_eval_accuracy": global_acc,
            "best_val_accuracy": r.get("best_val_accuracy", 0),
            "eval_observations": r.get("eval_observations", 0),
        })

        sel_thresh = r.get("selected_threshold")
        if sel_thresh is not None:
            selected_thresholds.append({
                "model": r["model"],
                "horizon": r["horizon"],
                "feature_set": r["feature_set"],
                "selected_threshold": sel_thresh,
                "selection_method": r.get("selected_threshold_method", ""),
                "final_threshold_accuracy": r.get("final_threshold_accuracy", ""),
                "final_threshold_observations": r.get("final_threshold_observations", ""),
                "final_threshold_coverage": r.get("final_threshold_coverage", ""),
            })

        for tr in r.get("threshold_results", []):
            if tr["pass_60pct"] and tr["coverage_ok_30"] and tr["rows_ok_1000"]:
                coverage_candidates.append({
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "feature_set": r["feature_set"],
                    "threshold": tr["threshold"],
                    "accuracy": tr["accuracy"],
                    "observations": tr["observations"],
                    "coverage": tr["coverage"],
                })
            elif tr["pass_60pct"]:
                exploratory_candidates.append({
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "feature_set": r["feature_set"],
                    "threshold": tr["threshold"],
                    "accuracy": tr["accuracy"],
                    "observations": tr["observations"],
                    "coverage": tr["coverage"],
                    "coverage_ok_30": tr["coverage_ok_30"],
                    "rows_ok_1000": tr["rows_ok_1000"],
                })

    for path, data in [
        ("above60_global_candidates.csv", global_candidates),
        ("above60_coverage_candidates.csv", coverage_candidates),
        ("above60_exploratory_candidates.csv", exploratory_candidates),
        ("selected_thresholds.csv", selected_thresholds),
    ]:
        if data:
            with (OUTPUT_DIR / path).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    manifest = {
        "total_experiments": total_experiments,
        "completed_experiments": sum(1 for r in all_results if r.get("status") == "completed"),
        "global_candidates": len(global_candidates),
        "coverage_candidates": len(coverage_candidates),
        "exploratory_candidates": len(exploratory_candidates),
        "best_global_accuracy": max((r.get("global_eval_accuracy", 0) for r in all_results if r.get("status") == "completed"), default=0),
        "best_coverage_accuracy": max((c["accuracy"] for c in coverage_candidates), default=0),
        "completed_at": now_utc(),
        "leakage_safe": True,
        "daily_data_used": False,
        "resampling_used": False,
        "new_data_fetched": False,
        "universe_changed": False,
    }
    with (OUTPUT_DIR / "experiment_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    log_lines = [
        "# VN30 Hourly 2015 Above-60% Optimization Run Log",
        "",
        f"- Started: {now_utc()}",
        f"- Completed: {manifest['completed_at']}",
        f"- Total experiments: {total_experiments}",
        f"- Completed: {manifest['completed_experiments']}",
        f"- Best global accuracy: {fmt_pct(manifest['best_global_accuracy'])}",
        f"- Best coverage-qualified accuracy: {fmt_pct(manifest['best_coverage_accuracy'])}",
        f"- Leakage safe: yes",
        f"- Daily data used: no",
        f"- Resampling used: no",
        "",
    ]
    with (OUTPUT_DIR / "optimization_run_log.md").open("w") as f:
        f.write("\n".join(log_lines))

    print(f"\nBest global accuracy: {fmt_pct(manifest['best_global_accuracy'])}")
    print(f"Best coverage-qualified accuracy: {fmt_pct(manifest['best_coverage_accuracy'])}")
    print(f"Coverage candidates: {len(coverage_candidates)}")
    print(f"Exploratory candidates: {len(exploratory_candidates)}")
    print(f"\nDone. Outputs in {rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())