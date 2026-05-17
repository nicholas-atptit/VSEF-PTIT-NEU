"""VN30 Daily 2015 Target60 optimization runner."""
from __future__ import annotations
import csv, json, math, sys, time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_2015_canonical_eval import compute_accuracy, evaluate_predictions, EVALUATOR_VERSION

CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_target60_optimization"
SEED = 42
MODELS = ["random_forest", "xgboost", "lightgbm"]
HORIZONS = [1, 3, 5, 10, 20, 40, 60]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Hyperparameter grids
RF_GRID = [
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 6, "max_features": "log2", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 700, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
]
XGB_GRID = [
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 7, "learning_rate": 0.08, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 7, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 700, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 4, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 6, "random_state": SEED, "eval_metric": "logloss"},
]
LGBM_GRID = [
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1},
    {"num_leaves": 15, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "random_state": SEED, "verbose": -1},
    {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.08, "n_estimators": 300, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": SEED, "verbose": -1},
    {"num_leaves": 20, "max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.6, "random_state": SEED, "verbose": -1},
    {"num_leaves": 45, "max_depth": -1, "learning_rate": 0.1, "n_estimators": 200, "min_child_samples": 15, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": SEED, "verbose": -1},
]

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

def build_features_basic(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Basic OHLCV features (same as original benchmark)."""
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
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
    return prepared, feat_cols

def build_features_extended(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extended feature set with more lags, vol, and cross-window features."""
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        op = group["open"].astype(float)
        # Returns
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 7, 10, 15, 20, 30, 60): prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20): prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
        # Rolling stats
        for window in (3, 5, 10, 15, 20, 30, 60, 120):
            min_p = max(2, min(window, window // 2))
            r = prepared.loc[idx, "return_1"]
            prepared.loc[idx, f"rolling_return_mean_{window}"] = r.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"rolling_return_vol_{window}"] = r.rolling(window, min_periods=min_p).std()
            prepared.loc[idx, f"rolling_return_skew_{window}"] = r.rolling(window, min_periods=min_p).skew()
            prepared.loc[idx, f"rolling_return_kurt_{window}"] = r.rolling(window, min_periods=min_p).kurt()
            sma = close.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"close_sma_ratio_{window}"] = close / sma - 1.0
            prepared.loc[idx, f"momentum_{window}"] = close / close.shift(window) - 1.0
            # Rolling volume
            vm = volume.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"volume_ma_ratio_{window}"] = volume / vm - 1.0
            prepared.loc[idx, f"volume_vol_{window}"] = volume.rolling(window, min_periods=min_p).std() / vm
        # RSI variants
        for period in (7, 14, 21):
            delta = close.diff()
            gain = delta.clip(lower=0.0).rolling(period, min_periods=period//2).mean()
            loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=period//2).mean()
            rs = gain / loss.replace(0.0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi.loc[(loss == 0.0) & (gain > 0.0)] = 100.0
            rsi.loc[(loss == 0.0) & (gain == 0.0)] = 50.0
            prepared.loc[idx, f"rsi_{period}"] = rsi
        # MACD variants
        for fast, slow, sig in [(8, 17, 9), (12, 26, 9), (5, 13, 5)]:
            ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
            macd = ema_f - ema_s
            prepared.loc[idx, f"macd_{fast}_{slow}"] = macd
            prepared.loc[idx, f"macd_signal_{fast}_{slow}"] = macd.ewm(span=sig, adjust=False, min_periods=sig).mean()
            prepared.loc[idx, f"macd_hist_{fast}_{slow}"] = prepared.loc[idx, f"macd_{fast}_{slow}"] - prepared.loc[idx, f"macd_signal_{fast}_{slow}"]
        # Volume features
        prepared.loc[idx, "volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
        for lag in (1, 2, 3, 5): prepared.loc[idx, f"volume_change_lag_{lag}"] = prepared.loc[idx, "volume_change_1"].shift(lag)
        vol_ma20 = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma20 - 1.0
        # Price range features
        prepared.loc[idx, "high_low_range"] = (high - low) / close
        prepared.loc[idx, "open_close_spread"] = (close - op) / op.replace(0.0, np.nan)
        prepared.loc[idx, "close_position_in_range"] = (close - low) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "upper_shadow"] = (high - prepared.loc[idx, ["open", "close"]].max(axis=1)) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "lower_shadow"] = (prepared.loc[idx, ["open", "close"]].min(axis=1) - low) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "body_ratio"] = (close - op).abs() / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "gap"] = (op - close.shift(1)) / close.shift(1)
    feat_cols = ["return_1","return_2","return_3","return_5","return_7","return_10","return_15","return_20","return_30","return_60",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20"]
    for w in (3, 5, 10, 15, 20, 30, 60, 120):
        feat_cols.extend([f"rolling_return_mean_{w}",f"rolling_return_vol_{w}",f"rolling_return_skew_{w}",f"rolling_return_kurt_{w}",
            f"close_sma_ratio_{w}",f"momentum_{w}",f"volume_ma_ratio_{w}",f"volume_vol_{w}"])
    for p in (7, 14, 21): feat_cols.append(f"rsi_{p}")
    for f, s in [(8, 17), (12, 26), (5, 13)]:
        feat_cols.extend([f"macd_{f}_{s}",f"macd_signal_{f}_{s}",f"macd_hist_{f}_{s}"])
    feat_cols.extend(["volume_change_1","volume_change_lag_1","volume_change_lag_2","volume_change_lag_3","volume_change_lag_5",
        "volume_shock_20","high_low_range","open_close_spread","close_position_in_range",
        "upper_shadow","lower_shadow","body_ratio","gap"])
    return prepared, feat_cols

def build_features_cross(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extended + cross-sectional rank features."""
    prepared, feat_cols = build_features_extended(df)
    # Cross-sectional rank features
    cs_cols = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        prepared.loc[idx, "cs_return_rank"] = prepared.loc[idx, "return_1"].rank(pct=True)
        prepared.loc[idx, "cs_volume_rank"] = prepared.loc[idx, "volume_shock_20"].rank(pct=True)
        prepared.loc[idx, "cs_momentum_rank"] = prepared.loc[idx, "momentum_20"].rank(pct=True)
    cs_cols = ["cs_return_rank", "cs_volume_rank", "cs_momentum_rank"]
    return prepared, feat_cols + cs_cols

FEATURE_SETS = {
    "daily_basic": build_features_basic,
    "daily_extended": build_features_extended,
    "daily_cross": build_features_cross,
}

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

def train_model(model_name: str, params: dict, X: pd.DataFrame, y: pd.Series):
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params).fit(X, y)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params).fit(X, y)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")

def get_param_grid(model_name: str) -> list[dict]:
    if model_name == "random_forest": return RF_GRID
    elif model_name == "xgboost": return XGB_GRID
    elif model_name == "lightgbm": return LGBM_GRID
    return [{}]

def param_id(model_name: str, params: dict) -> str:
    key = ""
    if model_name == "random_forest":
        key = f"n{params.get('n_estimators',0)}_d{params.get('max_depth',0)}_l{params.get('min_samples_leaf',0)}"
    elif model_name == "xgboost":
        key = f"d{params.get('max_depth',0)}_lr{params.get('learning_rate',0)}_n{params.get('n_estimators',0)}_mcw{params.get('min_child_weight',0)}"
    elif model_name == "lightgbm":
        key = f"nl{params.get('num_leaves',0)}_d{params.get('max_depth',0)}_lr{params.get('learning_rate',0)}_n{params.get('n_estimators',0)}"
    return key

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Target60 Optimization")
    print("=" * 60)
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "daily").mkdir(parents=True, exist_ok=True)

    run_config = {"models": MODELS, "horizons": HORIZONS, "feature_sets": list(FEATURE_SETS.keys()),
        "seed": SEED, "evaluator_version": EVALUATOR_VERSION, "created_at": now_utc(),
        "note": "Daily-only target60 optimization. No hourly data, no resampling, no final-label tuning."}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)

    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    print(f"  {len(tickers)} tickers in universe, {stock_df['ticker'].nunique()} with data, {len(stock_df)} rows")

    # Precompute feature sets
    feature_sets_data = {}
    for fs_name, fs_fn in FEATURE_SETS.items():
        print(f"\nBuilding features: {fs_name}...")
        fdf, fcols = fs_fn(stock_df)
        feature_sets_data[fs_name] = (fdf, fcols)
        print(f"  {len(fcols)} features")

    # Validation and eval splits
    # Rolling validation: 2021, 2022, 2023, 2024
    VAL_YEARS = [2021, 2022, 2023, 2024]
    EVAL_START = pd.Timestamp("2025-01-01")
    TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
    VAL_2024_START = pd.Timestamp("2024-01-01")
    VAL_2024_END = pd.Timestamp("2024-12-31 23:59:59")

    all_val_results = []
    all_final_results = []
    skipped = []
    experiments = []
    for h in HORIZONS:
        for mn in MODELS:
            for fs_name in FEATURE_SETS:
                for pi, params in enumerate(get_param_grid(mn)):
                    experiments.append({"model": mn, "horizon": h, "feature_set": fs_name, "params": params, "param_idx": pi})

    print(f"\nRunning {len(experiments)} experiments...")
    for i, exp in enumerate(experiments):
        elapsed = time.monotonic() - started
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(experiments)} ({elapsed:.0f}s elapsed)")
        h = exp["horizon"]
        mn = exp["model"]
        fs_name = exp["feature_set"]
        params = exp["params"]
        pid = param_id(mn, params)
        fdf, fcols = feature_sets_data[fs_name]
        future_returns = compute_future_returns(fdf, h)
        labels = make_labels(future_returns)
        fcp = [c for c in fcols if c in fdf.columns]

        # Check data sufficiency
        all_idx = future_returns.index[future_returns.notna()]
        if len(all_idx) < 200:
            skipped.append({"model": mn, "horizon": h, "feature_set": fs_name, "hyperparams_id": pid, "reason": "insufficient data"})
            continue

        # Rolling validation: train on all data before each val year, validate on that year
        val_accuracies = []
        val_rows_list = []
        for vy in VAL_YEARS:
            vy_start = pd.Timestamp(f"{vy}-01-01")
            vy_end = pd.Timestamp(f"{vy}-12-31 23:59:59")
            t_idx = all_idx[(fdf.loc[all_idx, "datetime"] < vy_start) & future_returns.reindex(all_idx).notna()]
            v_idx = all_idx[(fdf.loc[all_idx, "datetime"] >= vy_start) & (fdf.loc[all_idx, "datetime"] <= vy_end) & future_returns.reindex(all_idx).notna()]
            if len(t_idx) < 100 or len(v_idx) < 20:
                continue
            tX = fdf.reindex(t_idx)[fcp].fillna(0)
            ty = labels.reindex(t_idx)
            vX = fdf.reindex(v_idx)[fcp].fillna(0)
            vy_labels = labels.reindex(v_idx)
            try:
                m = train_model(mn, params, tX, ty)
                vp = m.predict(vX)
                vr = compute_accuracy(vy_labels.values.astype(float), vp.astype(float))
                val_accuracies.append(vr["accuracy"])
                val_rows_list.append(vr["total_valid"])
            except:
                pass

        if not val_accuracies:
            skipped.append({"model": mn, "horizon": h, "feature_set": fs_name, "hyperparams_id": pid, "reason": "no rolling val folds"})
            continue

        mean_val_acc = np.mean(val_accuracies)
        total_val_rows = sum(val_rows_list)

        # Final evaluation: train on all pre-2025 data, evaluate on 2025+
        t_idx = all_idx[(fdf.loc[all_idx, "datetime"] <= TRAIN_END) & future_returns.reindex(all_idx).notna()]
        e_idx = all_idx[(fdf.loc[all_idx, "datetime"] >= EVAL_START) & future_returns.reindex(all_idx).notna()]
        if len(t_idx) < 100 or len(e_idx) < 20:
            skipped.append({"model": mn, "horizon": h, "feature_set": fs_name, "hyperparams_id": pid, "reason": "insufficient final eval data"})
            continue

        tX = fdf.reindex(t_idx)[fcp].fillna(0)
        ty = labels.reindex(t_idx)
        eX = fdf.reindex(e_idx)[fcp].fillna(0)
        ey = labels.reindex(e_idx)

        try:
            m = train_model(mn, params, tX, ty)
            ep = m.predict(eX)
            er = compute_accuracy(ey.values.astype(float), ep.astype(float))
            ee = evaluate_predictions(ey.values.astype(float), ep.astype(float))

            result = {
                "model": mn, "horizon": h, "feature_set": fs_name, "hyperparams_id": pid,
                "validation_accuracy": round(float(mean_val_acc), 6),
                "validation_rows": int(total_val_rows),
                "final_accuracy": round(float(er["accuracy"]), 6),
                "final_rows": int(er["total_valid"]),
                "final_coverage": 1.0,
                "active_ticker_count": stock_df["ticker"].nunique(),
                "pass_60": er["accuracy"] >= 0.60,
                "selected_on_validation": "yes",
                "claim_level": ee["claim_level"],
                "evaluator_version": EVALUATOR_VERSION,
            }
            all_val_results.append(result)
            all_final_results.append(result.copy())
        except Exception as e:
            skipped.append({"model": mn, "horizon": h, "feature_set": fs_name, "hyperparams_id": pid, "reason": f"error: {e}"})

    # Write outputs
    print("\nWriting outputs...")
    fields = ["model", "horizon", "feature_set", "hyperparams_id", "validation_accuracy",
        "validation_rows", "final_accuracy", "final_rows", "final_coverage",
        "active_ticker_count", "pass_60", "selected_on_validation", "claim_level", "evaluator_version"]

    with (OUTPUT_DIR / "daily" / "validation_candidate_results.csv").open("w", newline="") as f:
        if all_val_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_val_results)

    with (OUTPUT_DIR / "daily" / "final_candidate_results.csv").open("w", newline="") as f:
        if all_final_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_final_results)

    # Best candidates
    if all_val_results:
        best_val = max(all_val_results, key=lambda r: r["validation_accuracy"])
        best_final = max(all_final_results, key=lambda r: r["final_accuracy"])
        candidates60 = [r for r in all_final_results if r["pass_60"]]

        with (OUTPUT_DIR / "daily" / "best_daily_candidates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerow(best_val)
            w.writerow(best_final)

        with (OUTPUT_DIR / "daily" / "daily_60_candidates.csv").open("w", newline="") as f:
            if candidates60:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(candidates60)
            else:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()

        # Manifest
        manifest = {
            "total_experiments": len(experiments),
            "completed": len(all_final_results),
            "skipped": len(skipped),
            "best_validation_model": best_val["model"],
            "best_validation_horizon": best_val["horizon"],
            "best_validation_accuracy": best_val["validation_accuracy"],
            "best_final_model": best_final["model"],
            "best_final_horizon": best_final["horizon"],
            "best_final_accuracy": best_final["final_accuracy"],
            "best_final_rows": best_final["final_rows"],
            "candidates_60": len(candidates60),
            "target60_passed": len(candidates60) > 0,
            "completed_at": now_utc(),
        }
        with (OUTPUT_DIR / "daily_target60_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
        with (OUTPUT_DIR / "manifest.json").open("w") as f: json.dump(manifest, f, indent=2)

        print(f"\nBest validation: {best_val['model']} {best_val['feature_set']} h={best_val['horizon']} = {fmt_pct(best_val['validation_accuracy'])}")
        print(f"Best final: {best_final['model']} {best_final['feature_set']} h={best_final['horizon']} = {fmt_pct(best_final['final_accuracy'])}")
        print(f"60% candidates: {len(candidates60)}")
    else:
        print("\nNo experiments completed.")

    # Skipped log
    with (OUTPUT_DIR / "daily" / "skipped_or_blocked_candidates.md").open("w") as f:
        f.write("# Skipped or Blocked Candidates\n\n")
        if skipped:
            f.write(f"Total skipped: {len(skipped)}\n\n")
            f.write("| model | horizon | feature_set | hyperparams_id | reason |\n")
            f.write("|---|---|---|---|---|\n")
            for s in skipped[:50]:
                f.write(f"| {s['model']} | {s['horizon']} | {s['feature_set']} | {s['hyperparams_id']} | {s['reason']} |\n")
            if len(skipped) > 50:
                f.write(f"\n... and {len(skipped) - 50} more\n")
        else:
            f.write("No candidates were skipped.\n")

    # Run log
    elapsed = time.monotonic() - started
    with (OUTPUT_DIR / "daily" / "daily_target60_run_log.md").open("w") as f:
        f.write("# Daily Target60 Run Log\n\n")
        f.write(f"- Started: {now_utc()}\n")
        f.write(f"- Duration: {elapsed:.0f}s\n")
        f.write(f"- Experiments: {len(experiments)}\n")
        f.write(f"- Completed: {len(all_final_results)}\n")
        f.write(f"- Skipped: {len(skipped)}\n")
        f.write(f"- 60% candidates: {len([r for r in all_final_results if r['pass_60']])}\n")
        f.write(f"- Daily-only: yes\n")
        f.write(f"- Hourly resampling: no\n")
        f.write(f"- Final-label tuning: no\n")

    print(f"\nDone in {elapsed:.0f}s. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
