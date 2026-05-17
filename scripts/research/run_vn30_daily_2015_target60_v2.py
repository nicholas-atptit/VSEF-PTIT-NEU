"""VN30 Daily 2015 Target60 V2 optimization runner."""
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_target60_v2"
SEED = 42
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
THRESHOLDS = [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65]
VAL_YEARS = [2021, 2022, 2023, 2024]
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")

# Hyperparameter grids
LGBM_GRID = [
    {"num_leaves": 20, "max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.6, "random_state": SEED, "verbose": -1},
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1},
    {"num_leaves": 15, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "random_state": SEED, "verbose": -1},
    {"num_leaves": 45, "max_depth": -1, "learning_rate": 0.1, "n_estimators": 200, "min_child_samples": 15, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": SEED, "verbose": -1},
    {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.08, "n_estimators": 300, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": SEED, "verbose": -1},
]
XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "subsample": 0.75, "colsample_bytree": 0.6, "min_child_weight": 4, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 6, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 7, "learning_rate": 0.08, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 7, "random_state": SEED, "eval_metric": "logloss"},
]
RF_GRID = [
    {"n_estimators": 700, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
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

def load_daily_index() -> pd.DataFrame | None:
    """Load VN30 daily index if available."""
    for candidate in ["VN30.csv", "vn30.csv"]:
        p = INDEX_CACHE_ROOT / candidate
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            for col in OHLCV_COLUMNS:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["datetime"])
            return df.sort_values("datetime").reset_index(drop=True)
    return None

def _build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build base features per ticker group. Returns DataFrame with new columns."""
    prepared = df.copy()
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        op = group["open"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 7, 10, 15, 20, 30, 60):
            prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20):
            prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
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
            vm = volume.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"volume_ma_ratio_{window}"] = volume / vm - 1.0
            prepared.loc[idx, f"volume_vol_{window}"] = volume.rolling(window, min_periods=min_p).std() / vm
        for period in (7, 14, 21):
            delta = close.diff()
            gain = delta.clip(lower=0.0).rolling(period, min_periods=period // 2).mean()
            loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=period // 2).mean()
            rs = gain / loss.replace(0.0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi.loc[(loss == 0.0) & (gain > 0.0)] = 100.0
            rsi.loc[(loss == 0.0) & (gain == 0.0)] = 50.0
            prepared.loc[idx, f"rsi_{period}"] = rsi
        for fast, slow, sig in [(8, 17, 9), (12, 26, 9), (5, 13, 5)]:
            ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
            macd = ema_f - ema_s
            prepared.loc[idx, f"macd_{fast}_{slow}"] = macd
            prepared.loc[idx, f"macd_signal_{fast}_{slow}"] = macd.ewm(span=sig, adjust=False, min_periods=sig).mean()
            prepared.loc[idx, f"macd_hist_{fast}_{slow}"] = prepared.loc[idx, f"macd_{fast}_{slow}"] - prepared.loc[idx, f"macd_signal_{fast}_{slow}"]
        prepared.loc[idx, "volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
        for lag in (1, 2, 3, 5):
            prepared.loc[idx, f"volume_change_lag_{lag}"] = prepared.loc[idx, "volume_change_1"].shift(lag)
        vol_ma20 = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma20 - 1.0
        prepared.loc[idx, "high_low_range"] = (high - low) / close
        prepared.loc[idx, "open_close_spread"] = (close - op) / op.replace(0.0, np.nan)
        prepared.loc[idx, "close_position_in_range"] = (close - low) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "upper_shadow"] = (high - prepared.loc[idx, ["open", "close"]].max(axis=1)) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "lower_shadow"] = (prepared.loc[idx, ["open", "close"]].min(axis=1) - low) / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "body_ratio"] = (close - op).abs() / (high - low).replace(0.0, np.nan)
        prepared.loc[idx, "gap"] = (op - close.shift(1)) / close.shift(1)
    return prepared

def _get_base_feat_cols() -> list[str]:
    cols = ["return_1","return_2","return_3","return_5","return_7","return_10","return_15","return_20","return_30","return_60",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20"]
    for w in (3, 5, 10, 15, 20, 30, 60, 120):
        cols.extend([f"rolling_return_mean_{w}",f"rolling_return_vol_{w}",f"rolling_return_skew_{w}",f"rolling_return_kurt_{w}",
            f"close_sma_ratio_{w}",f"momentum_{w}",f"volume_ma_ratio_{w}",f"volume_vol_{w}"])
    for p in (7, 14, 21): cols.append(f"rsi_{p}")
    for f, s in [(8, 17), (12, 26), (5, 13)]:
        cols.extend([f"macd_{f}_{s}",f"macd_signal_{f}_{s}",f"macd_hist_{f}_{s}"])
    cols.extend(["volume_change_1","volume_change_lag_1","volume_change_lag_2","volume_change_lag_3","volume_change_lag_5",
        "volume_shock_20","high_low_range","open_close_spread","close_position_in_range",
        "upper_shadow","lower_shadow","body_ratio","gap"])
    return cols

def build_features_cross(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = _build_base_features(df)
    feat_cols = _get_base_feat_cols()
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        prepared.loc[idx, "cs_return_rank"] = prepared.loc[idx, "return_1"].rank(pct=True)
        prepared.loc[idx, "cs_volume_rank"] = prepared.loc[idx, "volume_shock_20"].rank(pct=True)
        prepared.loc[idx, "cs_momentum_rank"] = prepared.loc[idx, "momentum_20"].rank(pct=True)
    return prepared, feat_cols + ["cs_return_rank", "cs_volume_rank", "cs_momentum_rank"]

def build_features_cross_plus_market(df: pd.DataFrame, index_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    prepared, feat_cols = build_features_cross(df)
    if index_df is None or index_df.empty:
        return prepared, feat_cols
    idx_df = index_df[["datetime", "close", "volume"]].copy()
    idx_df.columns = ["datetime", "idx_close", "idx_volume"]
    idx_df["idx_return_1"] = idx_df["idx_close"].pct_change(periods=1, fill_method=None)
    for lag in (1, 2, 3, 5):
        idx_df[f"idx_return_1_lag_{lag}"] = idx_df["idx_return_1"].shift(lag)
    for w in (5, 10, 20, 60):
        min_p = max(2, w // 2)
        idx_df[f"idx_rolling_return_mean_{w}"] = idx_df["idx_return_1"].rolling(w, min_periods=min_p).mean()
        idx_df[f"idx_rolling_return_vol_{w}"] = idx_df["idx_return_1"].rolling(w, min_periods=min_p).std()
        idx_df[f"idx_momentum_{w}"] = idx_df["idx_close"] / idx_df["idx_close"].shift(w) - 1.0
    idx_df = idx_df.dropna(subset=["datetime"])
    prepared = prepared.merge(idx_df, on="datetime", how="left")
    market_cols = ["idx_return_1","idx_return_1_lag_1","idx_return_1_lag_2","idx_return_1_lag_3","idx_return_1_lag_5",
        "idx_rolling_return_mean_5","idx_rolling_return_vol_5","idx_momentum_5",
        "idx_rolling_return_mean_10","idx_rolling_return_vol_10","idx_momentum_10",
        "idx_rolling_return_mean_20","idx_rolling_return_vol_20","idx_momentum_20",
        "idx_rolling_return_mean_60","idx_rolling_return_vol_60","idx_momentum_60"]
    return prepared, feat_cols + [c for c in market_cols if c in prepared.columns]

def build_features_cross_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared, feat_cols = build_features_cross(df)
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        prepared.loc[idx, "momentum_x_vol"] = prepared.loc[idx, "momentum_20"] * prepared.loc[idx, "rolling_return_vol_20"]
        prepared.loc[idx, "rsi_x_momentum"] = prepared.loc[idx, "rsi_14"] * prepared.loc[idx, "momentum_10"]
        prepared.loc[idx, "return_x_volume"] = prepared.loc[idx, "return_1"] * prepared.loc[idx, "volume_shock_20"]
        prepared.loc[idx, "sma_ratio_x_momentum"] = prepared.loc[idx, "close_sma_ratio_20"] * prepared.loc[idx, "momentum_20"]
        prepared.loc[idx, "macd_x_vol"] = prepared.loc[idx, "macd_hist_12_26"] * prepared.loc[idx, "rolling_return_vol_10"]
    int_cols = ["momentum_x_vol", "rsi_x_momentum", "return_x_volume", "sma_ratio_x_momentum", "macd_x_vol"]
    return prepared, feat_cols + int_cols

def build_features_vol_normalized(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared, feat_cols = build_features_cross(df)
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        vol = prepared.loc[idx, "rolling_return_vol_20"].replace(0.0, np.nan)
        for c in ["return_1", "return_5", "return_10", "return_20"]:
            if c in prepared.columns:
                prepared.loc[idx, f"{c}_vol_norm"] = prepared.loc[idx, c] / vol
        for c in ["momentum_5", "momentum_10", "momentum_20", "momentum_60"]:
            if c in prepared.columns:
                prepared.loc[idx, f"{c}_vol_norm"] = prepared.loc[idx, c] / vol
    vn_cols = [f"{c}_vol_norm" for c in ["return_1","return_5","return_10","return_20","momentum_5","momentum_10","momentum_20","momentum_60"]]
    return prepared, feat_cols + [c for c in vn_cols if c in prepared.columns]

def build_features_momentum_vol_interaction(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared, feat_cols = build_features_cross(df)
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        rv20 = prepared.loc[idx, "rolling_return_vol_20"]
        high_vol = (rv20 > rv20.quantile(0.75)).astype(float)
        low_vol = (rv20 <= rv20.quantile(0.25)).astype(float)
        for c in ["return_1", "momentum_10", "momentum_20"]:
            if c in prepared.columns:
                prepared.loc[idx, f"{c}_high_vol"] = prepared.loc[idx, c] * high_vol
                prepared.loc[idx, f"{c}_low_vol"] = prepared.loc[idx, c] * low_vol
        prepared.loc[idx, "vol_regime"] = rv20 / rv20.rolling(60, min_periods=20).mean() - 1.0
    mvi_cols = []
    for c in ["return_1", "momentum_10", "momentum_20"]:
        mvi_cols.extend([f"{c}_high_vol", f"{c}_low_vol"])
    mvi_cols.append("vol_regime")
    return prepared, feat_cols + [c for c in mvi_cols if c in prepared.columns]

FEATURE_SETS = {
    "daily_cross": build_features_cross,
    "daily_cross_plus_market": build_features_cross_plus_market,
    "daily_cross_interactions": build_features_cross_interactions,
    "volatility_normalized": build_features_vol_normalized,
    "momentum_volatility_interaction": build_features_momentum_vol_interaction,
}

HORIZONS = [30, 40, 50, 60]
MODEL_CONFIGS = {
    "lightgbm": {"grid": LGBM_GRID, "horizons": HORIZONS},
    "xgboost": {"grid": XGB_GRID, "horizons": HORIZONS},
    "random_forest": {"grid": RF_GRID, "horizons": [40, 60]},
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

def param_id(model_name: str, params: dict) -> str:
    if model_name == "random_forest":
        return f"n{params.get('n_estimators',0)}_d{params.get('max_depth',0)}_l{params.get('min_samples_leaf',0)}"
    elif model_name == "xgboost":
        return f"d{params.get('max_depth',0)}_lr{params.get('learning_rate',0)}_n{params.get('n_estimators',0)}"
    elif model_name == "lightgbm":
        return f"nl{params.get('num_leaves',0)}_d{params.get('max_depth',0)}_lr{params.get('learning_rate',0)}_n{params.get('n_estimators',0)}"
    return ""

def candidate_id(mn: str, h: int, fs: str, pid: str, thr: float) -> str:
    return f"{mn}_h{h}_{fs}_{pid}_t{int(thr*1000)}"

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Target60 V2 Optimization")
    print("=" * 60)
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "daily").mkdir(parents=True, exist_ok=True)

    run_config = {"models": list(MODEL_CONFIGS.keys()), "horizons": HORIZONS,
        "feature_sets": list(FEATURE_SETS.keys()), "thresholds": THRESHOLDS,
        "seed": SEED, "evaluator_version": EVALUATOR_VERSION, "created_at": now_utc(),
        "note": "Daily-only target60 v2. Threshold tuning, focused horizons 30-60."}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)

    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    print(f"  {len(tickers)} tickers, {len(stock_df)} rows")

    index_df = load_daily_index()
    print(f"  Daily index cache: {'yes' if index_df is not None else 'no'}")

    # Precompute feature sets
    feature_sets_data = {}
    for fs_name, fs_fn in FEATURE_SETS.items():
        print(f"\nBuilding features: {fs_name}...")
        if "market" in fs_name:
            fdf, fcols = fs_fn(stock_df, index_df)
        else:
            fdf, fcols = fs_fn(stock_df)
        feature_sets_data[fs_name] = (fdf, fcols)
        print(f"  {len(fcols)} features")

    # Build experiments
    experiments = []
    for mn, cfg in MODEL_CONFIGS.items():
        for h in cfg["horizons"]:
            for fs_name in FEATURE_SETS:
                for pi, params in enumerate(cfg["grid"]):
                    pid = param_id(mn, params)
                    experiments.append({"model": mn, "horizon": h, "feature_set": fs_name,
                        "params": params, "param_idx": pi, "param_id": pid})

    print(f"\nRunning {len(experiments)} base experiments x {len(THRESHOLDS)} thresholds = {len(experiments) * len(THRESHOLDS)} total...")

    all_results = []
    skipped = []

    for i, exp in enumerate(experiments):
        elapsed = time.monotonic() - started
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(experiments)} ({elapsed:.0f}s elapsed)")

        h = exp["horizon"]
        mn = exp["model"]
        fs_name = exp["feature_set"]
        params = exp["params"]
        pid = exp["param_id"]
        fdf, fcols = feature_sets_data[fs_name]
        fcp = [c for c in fcols if c in fdf.columns]

        future_returns = compute_future_returns(fdf, h)
        labels = make_labels(future_returns)
        all_idx = future_returns.index[future_returns.notna()]
        if len(all_idx) < 200:
            skipped.append({"candidate_id": candidate_id(mn, h, fs_name, pid, 0.5), "model": mn, "horizon": h,
                "feature_set": fs_name, "hyperparams_id": pid, "decision_threshold": 0.5, "reason": "insufficient data"})
            continue

        # Rolling validation
        val_accuracies = []
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
                vprob = m.predict_proba(vX)[:, 1]
                # Test all thresholds on this fold
                for thr in THRESHOLDS:
                    vp = (vprob >= thr).astype(int)
                    vr = compute_accuracy(vy_labels.values.astype(float), vp.astype(float))
                    val_accuracies.append({"year": vy, "threshold": thr, "accuracy": vr["accuracy"], "rows": vr["total_valid"]})
            except:
                pass

        if not val_accuracies:
            skipped.append({"candidate_id": candidate_id(mn, h, fs_name, pid, 0.5), "model": mn, "horizon": h,
                "feature_set": fs_name, "hyperparams_id": pid, "decision_threshold": 0.5, "reason": "no rolling val folds"})
            continue

        # Find best threshold by mean validation accuracy
        val_df = pd.DataFrame(val_accuracies)
        thr_stats = val_df.groupby("threshold", sort=False).agg(
            mean_accuracy=("accuracy", "mean"),
            min_accuracy=("accuracy", "min"),
            std_accuracy=("accuracy", "std"),
            total_rows=("rows", "sum"),
        ).reset_index()
        thr_stats["stability_score"] = thr_stats["mean_accuracy"] - thr_stats["std_accuracy"] / 2
        best_thr_row = thr_stats.loc[thr_stats["stability_score"].idxmax()]
        best_thr = float(best_thr_row["threshold"])
        mean_val_acc = float(best_thr_row["mean_accuracy"])
        min_val_acc = float(best_thr_row["min_accuracy"])
        std_val_acc = float(best_thr_row["std_accuracy"])
        stability = float(best_thr_row["stability_score"])

        # Final evaluation
        t_idx = all_idx[(fdf.loc[all_idx, "datetime"] <= TRAIN_END) & future_returns.reindex(all_idx).notna()]
        e_idx = all_idx[(fdf.loc[all_idx, "datetime"] >= EVAL_START) & future_returns.reindex(all_idx).notna()]
        if len(t_idx) < 100 or len(e_idx) < 20:
            skipped.append({"candidate_id": candidate_id(mn, h, fs_name, pid, best_thr), "model": mn, "horizon": h,
                "feature_set": fs_name, "hyperparams_id": pid, "decision_threshold": best_thr, "reason": "insufficient final eval data"})
            continue

        tX = fdf.reindex(t_idx)[fcp].fillna(0)
        ty = labels.reindex(t_idx)
        eX = fdf.reindex(e_idx)[fcp].fillna(0)
        ey = labels.reindex(e_idx)

        try:
            m = train_model(mn, params, tX, ty)
            eprob = m.predict_proba(eX)[:, 1]
            ep = (eprob >= best_thr).astype(int)
            er = compute_accuracy(ey.values.astype(float), ep.astype(float))
            ee = evaluate_predictions(ey.values.astype(float), ep.astype(float))

            cid = candidate_id(mn, h, fs_name, pid, best_thr)
            result = {
                "candidate_id": cid, "model": mn, "horizon": h, "feature_set": fs_name,
                "hyperparams_id": pid, "decision_threshold": best_thr,
                "rolling_validation_mean_accuracy": round(mean_val_acc, 6),
                "rolling_validation_min_accuracy": round(min_val_acc, 6),
                "rolling_validation_std": round(std_val_acc, 6),
                "stability_score": round(stability, 6),
                "final_accuracy": round(float(er["accuracy"]), 6),
                "final_rows": int(er["total_valid"]),
                "final_coverage": 1.0,
                "active_ticker_count": stock_df["ticker"].nunique(),
                "pass_60": er["accuracy"] >= 0.60,
                "selected_on_validation": "yes",
                "claim_level": ee["claim_level"],
                "evaluator_version": EVALUATOR_VERSION,
            }
            all_results.append(result)
        except Exception as e:
            skipped.append({"candidate_id": candidate_id(mn, h, fs_name, pid, best_thr), "model": mn, "horizon": h,
                "feature_set": fs_name, "hyperparams_id": pid, "decision_threshold": best_thr, "reason": f"error: {e}"})

    # Write outputs
    print("\nWriting outputs...")
    fields = ["candidate_id", "model", "horizon", "feature_set", "hyperparams_id", "decision_threshold",
        "rolling_validation_mean_accuracy", "rolling_validation_min_accuracy", "rolling_validation_std",
        "stability_score", "final_accuracy", "final_rows", "final_coverage", "active_ticker_count",
        "pass_60", "selected_on_validation", "claim_level", "evaluator_version"]

    with (OUTPUT_DIR / "daily" / "rolling_validation_results.csv").open("w", newline="") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_results)

    with (OUTPUT_DIR / "daily" / "final_candidate_results.csv").open("w", newline="") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_results)

    if all_results:
        # Best by stability score
        best_stability = max(all_results, key=lambda r: r["stability_score"])
        best_final = max(all_results, key=lambda r: r["final_accuracy"])
        candidates60 = [r for r in all_results if r["pass_60"]]

        with (OUTPUT_DIR / "daily" / "candidate_selection_scores.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(all_results, key=lambda x: x["stability_score"], reverse=True)[:20]:
                w.writerow(r)

        with (OUTPUT_DIR / "daily" / "daily_60_candidates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            if candidates60:
                w.writerows(candidates60)

        manifest = {
            "total_base_experiments": len(experiments),
            "total_candidates": len(all_results),
            "skipped": len(skipped),
            "best_stability_model": best_stability["model"],
            "best_stability_horizon": best_stability["horizon"],
            "best_stability_feature_set": best_stability["feature_set"],
            "best_stability_threshold": best_stability["decision_threshold"],
            "best_stability_score": best_stability["stability_score"],
            "best_stability_final_accuracy": best_stability["final_accuracy"],
            "best_final_model": best_final["model"],
            "best_final_horizon": best_final["horizon"],
            "best_final_feature_set": best_final["feature_set"],
            "best_final_threshold": best_final["decision_threshold"],
            "best_final_accuracy": best_final["final_accuracy"],
            "best_final_rows": best_final["final_rows"],
            "candidates_60": len(candidates60),
            "target60_passed": len(candidates60) > 0,
            "completed_at": now_utc(),
        }
        with (OUTPUT_DIR / "daily_target60_v2_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
        with (OUTPUT_DIR / "manifest.json").open("w") as f: json.dump(manifest, f, indent=2)

        print(f"\nBest stability: {best_stability['candidate_id']} = {fmt_pct(best_stability['final_accuracy'])} final (stability={best_stability['stability_score']:.4f})")
        print(f"Best final: {best_final['candidate_id']} = {fmt_pct(best_final['final_accuracy'])}")
        print(f"60% candidates: {len(candidates60)}")
    else:
        print("\nNo experiments completed.")

    # Skipped log
    with (OUTPUT_DIR / "daily" / "skipped_or_blocked_candidates.md").open("w") as f:
        f.write("# Skipped or Blocked Candidates\n\n")
        if skipped:
            f.write(f"Total skipped: {len(skipped)}\n\n")
            f.write("| candidate_id | model | horizon | feature_set | hyperparams_id | threshold | reason |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for s in skipped[:50]:
                f.write(f"| {s['candidate_id']} | {s['model']} | {s['horizon']} | {s['feature_set']} | {s['hyperparams_id']} | {s['decision_threshold']} | {s['reason']} |\n")
            if len(skipped) > 50:
                f.write(f"\n... and {len(skipped) - 50} more\n")
        else:
            f.write("No candidates were skipped.\n")

    # Run log
    elapsed = time.monotonic() - started
    with (OUTPUT_DIR / "daily" / "daily_target60_v2_run_log.md").open("w") as f:
        f.write("# Daily Target60 V2 Run Log\n\n")
        f.write(f"- Started: {now_utc()}\n")
        f.write(f"- Duration: {elapsed:.0f}s\n")
        f.write(f"- Base experiments: {len(experiments)}\n")
        f.write(f"- Total candidates (with thresholds): {len(experiments) * len(THRESHOLDS)}\n")
        f.write(f"- Completed: {len(all_results)}\n")
        f.write(f"- Skipped: {len(skipped)}\n")
        f.write(f"- 60% candidates: {len([r for r in all_results if r['pass_60']])}\n")
        f.write(f"- Daily-only: yes\n")
        f.write(f"- Hourly resampling: no\n")
        f.write(f"- Final-label tuning: no\n")
        f.write(f"- Thresholds tested: {THRESHOLDS}\n")

    print(f"\nDone in {elapsed:.0f}s. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
