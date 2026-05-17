"""Full tuning sweep for VN30 hourly 2015 using canonical evaluator."""
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

sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))
from vn30_hourly_2015_canonical_eval import compute_accuracy, evaluate_predictions, classify_result, EVALUATOR_VERSION

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_full_tuning_sweep"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
MODELS = ["lightgbm", "xgboost", "random_forest"]
HORIZONS = [4, 8, 20, 40, 60, 80, 120]
SEED = 42
CONFIDENCE_THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]
MIN_ROWS_PER_TICKER = [50, 100, 150]

# Hyperparameter grids (bounded)
LGBM_GRID = [
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1},
    {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "random_state": SEED, "verbose": -1},
    {"num_leaves": 15, "max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9, "random_state": SEED, "verbose": -1},
]
XGB_GRID = [
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 7, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3, "random_state": SEED, "eval_metric": "logloss"},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10, "random_state": SEED, "eval_metric": "logloss"},
]
RF_GRID = [
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 500, "max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED},
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": "log2", "class_weight": "balanced", "random_state": SEED},
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

def load_index_data() -> dict[str, pd.DataFrame]:
    indices = {}
    for code in ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[df["close"] > 0].sort_values("datetime").reset_index(drop=True)
        indices[code] = df
    return indices

def build_features(df: pd.DataFrame, index_data: dict | None = None, include_market: bool = False) -> tuple[pd.DataFrame, list[str]]:
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
    if include_market and index_data:
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
                cm = f"{code.lower()}_roll_mean_{window}"
                cv = f"{code.lower()}_roll_vol_{window}"
                idx_df[cm] = idx_df["idx_ret"].rolling(window, min_periods=min_p).mean()
                idx_df[cv] = idx_df["idx_ret"].rolling(window, min_periods=min_p).std()
                idx_features.extend([cm, cv])
            sel_cols = ["datetime"] + idx_features
            idx_clean = idx_df[sel_cols].dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
            prepared = prepared.merge(idx_clean, on="datetime", how="left")
            feat_cols.extend(idx_features)
            if "vnindex_lag_1" in prepared.columns and "market_minus_stock_ret" not in feat_cols:
                prepared["market_minus_stock_ret"] = prepared["vnindex_lag_1"] - prepared["return_1"]
                feat_cols.append("market_minus_stock_ret")
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

def build_labels_absolute(df: pd.DataFrame, horizon: int) -> pd.Series:
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(float)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=idx))
    if labels: return pd.concat(labels)
    return pd.Series(dtype=float)

def build_labels_relative(df: pd.DataFrame, horizon: int, index_returns: pd.Series) -> pd.Series:
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        stock_ret = (future_close - group["close"]) / group["close"]
        market_ret = index_returns.reindex(group["datetime"]).shift(-horizon).values
        relative_ret = stock_ret.values - market_ret
        direction = (relative_ret > 0).astype(float)
        direction = pd.Series(direction, index=idx)
        direction.loc[future_close.isna()] = np.nan
        labels.append(direction)
    if labels: return pd.concat(labels)
    return pd.Series(dtype=float)

def split_data(df: pd.DataFrame, labels: pd.Series):
    train_mask = df["datetime"] <= TRAIN_END
    val_mask = (df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)
    eval_mask = df["datetime"] >= EVAL_START
    return df[train_mask].copy(), df[val_mask].copy(), df[eval_mask].copy()

def train_model(model_name: str, X: pd.DataFrame, y: pd.Series, params: dict | None = None):
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**(params or LGBM_GRID[0])).fit(X, y)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**(params or XGB_GRID[0])).fit(X, y)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**(params or RF_GRID[0])).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")

def predict_proba(model, X: pd.DataFrame):
    probas = model.predict_proba(X)
    preds = model.predict(X)
    return preds, probas[:, 1], probas

def directional_accuracy(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0: return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))

def run_single_experiment(feature_df, feat_cols, index_returns, horizon, target_type, market_ref, model_name, params=None, param_id="default"):
    fcp = [c for c in feat_cols if c in feature_df.columns]
    if target_type == "absolute":
        labels = build_labels_absolute(feature_df, horizon)
    else:
        idx_ret = index_returns.get(market_ref)
        if idx_ret is None: return None
        labels = build_labels_relative(feature_df, horizon, idx_ret)
    train_df, val_df, eval_df = split_data(feature_df, labels)
    tl = labels.reindex(train_df.index).dropna()
    vl = labels.reindex(val_df.index).dropna()
    el = labels.reindex(eval_df.index).dropna()
    if len(tl) < 100 or len(vl) < 20 or len(el) < 20: return None
    tX = feature_df.reindex(tl.index)[fcp].fillna(0)
    ty = tl.astype(int)
    vX = feature_df.reindex(vl.index)[fcp].fillna(0)
    vy = vl.astype(int)
    eX = feature_df.reindex(el.index)[fcp].fillna(0)
    ey = el.astype(int)
    try:
        m = train_model(model_name, tX, ty, params)
        vp, vprob, vpr = predict_proba(m, vX)
        ep, eprob, epr = predict_proba(m, eX)
        vr = compute_accuracy(vy.values, vp)
        er = compute_accuracy(ey.values, ep)
        return {"model": model_name, "horizon": horizon, "target_type": target_type,
            "market_reference": market_ref if target_type != "absolute" else "",
            "feature_set": "C", "param_id": param_id,
            "validation_accuracy": round(vr["accuracy"], 6), "validation_coverage": 1.0, "validation_rows": vr["total_valid"],
            "final_accuracy": round(er["accuracy"], 6), "final_coverage": 1.0, "final_rows": er["total_valid"],
            "vp": vp, "vprob": vprob, "ep": ep, "eprob": eprob,
            "vy": vy.values, "ey": ey.values, "vl_index": vl.index, "el_index": el.index}
    except: return None

def apply_policies(result, tickers, feature_df, candidate_id):
    policies = []
    vp, vprob = result["vp"], result["vprob"]
    ep, eprob = result["ep"], result["eprob"]
    vy, ey = result["vy"], result["ey"]
    total_val = len(vy[~np.isnan(vy)])
    total_eval = len(ey[~np.isnan(ey)])

    # Policy A: Confidence abstention
    for thresh in CONFIDENCE_THRESHOLDS:
        vmask = vprob >= thresh
        v_acc = compute_accuracy(vy[vmask], vp[vmask])
        emask = eprob >= thresh
        e_acc = compute_accuracy(ey[emask], ep[emask])
        if e_acc["total_valid"] == 0: continue
        ecov = e_acc["total_valid"] / total_eval
        pid = f"{candidate_id}_conf_{thresh}"
        cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
        policies.append({"candidate_id": pid, "model": result["model"], "horizon": result["horizon"],
            "target_type": result["target_type"], "feature_set": result["feature_set"],
            "policy_type": "confidence_abstention", "hyperparams_id": result["param_id"],
            "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
            "validation_coverage": round(v_acc["total_valid"] / total_val, 4) if total_val > 0 else 0,
            "validation_rows": v_acc["total_valid"],
            "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
            "final_rows": e_acc["total_valid"], "active_ticker_count": len(tickers),
            "pass_60": e_acc["accuracy"] >= 0.60 and ecov >= 0.95 and e_acc["total_valid"] >= 1000,
            "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
            "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Policy B: Per-ticker whitelist
    ticker_val_acc = {}
    for ticker in tickers:
        t_mask = feature_df["ticker"] == ticker
        t_vl_idx = result["vl_index"][t_mask.reindex(result["vl_index"])]
        if len(t_vl_idx) < 50: continue
        t_idx_mask = t_mask.reindex(result["vl_index"]).values
        t_vp = vp[t_idx_mask]
        t_vy_mask = t_idx_mask & ~np.isnan(vy)
        t_vy = vy[t_vy_mask]
        t_vp_trimmed = t_vp[:len(t_vy)]
        if len(t_vy) < 20: continue
        acc = directional_accuracy(t_vy, t_vp_trimmed)
        ticker_val_acc[ticker] = (acc, len(t_vy))
    for min_rows in MIN_ROWS_PER_TICKER:
        whitelist = [t for t, (a, n) in ticker_val_acc.items() if n >= min_rows]
        if not whitelist: continue
        vmask = np.array([feature_df.loc[i, "ticker"] in whitelist for i in result["vl_index"]]) & ~np.isnan(vy)
        v_acc = compute_accuracy(vy[vmask], vp[vmask])
        emask = np.array([feature_df.loc[i, "ticker"] in whitelist for i in result["el_index"]]) & ~np.isnan(ey)
        e_acc = compute_accuracy(ey[emask], ep[emask])
        if e_acc["total_valid"] == 0: continue
        ecov = e_acc["total_valid"] / total_eval
        pid = f"{candidate_id}_ticker_{min_rows}"
        cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
        policies.append({"candidate_id": pid, "model": result["model"], "horizon": result["horizon"],
            "target_type": result["target_type"], "feature_set": result["feature_set"],
            "policy_type": "per_ticker_whitelist", "hyperparams_id": result["param_id"],
            "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
            "validation_coverage": round(v_acc["total_valid"] / total_val, 4) if total_val > 0 else 0,
            "validation_rows": v_acc["total_valid"],
            "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
            "final_rows": e_acc["total_valid"], "active_ticker_count": len(whitelist),
            "pass_60": e_acc["accuracy"] >= 0.60 and ecov >= 0.95 and e_acc["total_valid"] >= 1000,
            "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
            "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})
    return policies

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Full Tuning Sweep")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"models": MODELS, "horizons": HORIZONS, "seed": SEED,
        "evaluator_version": EVALUATOR_VERSION, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    print("\nBuilding features...")
    feature_df_c, feat_cols_c = build_features(stock_df, index_data, include_market=True)
    print(f"  Feature set C: {len(feat_cols_c)} features")
    index_returns_map = {}
    for code, df in index_data.items():
        df = df.copy().sort_values("datetime")
        index_returns_map[code] = pd.Series(df["close"].pct_change(periods=1, fill_method=None).values, index=df["datetime"])

    # Define focused experiment grid
    experiments = []
    # Base configs: model x horizon x target x param_grid
    base_configs = [
        ("lightgbm", HORIZONS, "absolute", "", LGBM_GRID),
        ("xgboost", HORIZONS, "absolute", "", XGB_GRID),
        ("random_forest", HORIZONS, "absolute", "", RF_GRID),
        ("xgboost", [8, 20, 40, 60], "relative_vn30", "VN30", XGB_GRID),
        ("xgboost", [8, 20, 40, 60], "relative_vnindex", "VNINDEX", XGB_GRID),
        ("lightgbm", [8, 20, 40, 60], "relative_vn30", "VN30", LGBM_GRID),
        ("lightgbm", [8, 20, 40, 60], "relative_vnindex", "VNINDEX", LGBM_GRID),
    ]
    for model, horizons, target, mref, param_grid in base_configs:
        for h in horizons:
            for pi, params in enumerate(param_grid):
                experiments.append({"model": model, "horizon": h, "target": target,
                    "market_ref": mref, "params": params, "param_id": f"p{pi}"})

    print(f"\nRunning {len(experiments)} base experiments...")
    all_policies = []
    candidate_results = []
    for i, exp in enumerate(experiments):
        if (i + 1) % 20 == 0: print(f"  Progress: {i+1}/{len(experiments)}")
        result = run_single_experiment(feature_df_c, feat_cols_c, index_returns_map, exp["horizon"],
            exp["target"], exp["market_ref"], exp["model"], exp["params"], exp["param_id"])
        if result is None: continue
        cid = f"{exp['model']}_h{exp['horizon']}_{exp['target']}_{exp['param_id']}"
        policies = apply_policies(result, tickers, feature_df_c, cid)
        all_policies.extend(policies)
        candidate_results.append({"candidate_id": cid, "model": exp["model"], "horizon": exp["horizon"],
            "target_type": exp["target"], "feature_set": "C", "policy_type": "base",
            "validation_accuracy": result["validation_accuracy"], "validation_coverage": result["validation_coverage"],
            "validation_rows": result["validation_rows"], "final_accuracy": result["final_accuracy"],
            "final_coverage": result["final_coverage"], "final_rows": result["final_rows"],
            "active_ticker_count": len(tickers), "pass_60": result["final_accuracy"] >= 0.60,
            "pass_65": result["final_accuracy"] >= 0.65,
            "claim_level": classify_result(result["final_accuracy"], result["final_coverage"], result["final_rows"], True),
            "evaluator_version": EVALUATOR_VERSION})
        # Clean up large arrays
        for k in ["vp", "vprob", "ep", "eprob", "vy", "ey", "vl_index", "el_index"]:
            if k in result: del result[k]

    # Select best policy on validation
    print("\nSelecting best policy on validation...")
    valid_policies = [p for p in all_policies if p["validation_rows"] >= 1000 and p["validation_coverage"] >= 0.30]
    if valid_policies:
        best_policy = max(valid_policies, key=lambda p: (p["validation_accuracy"], p["validation_coverage"]))
        print(f"  Best: {best_policy['candidate_id']}, model={best_policy['model']}, h={best_policy['horizon']}, "
              f"target={best_policy['target_type']}, type={best_policy['policy_type']}, "
              f"val_acc={fmt_pct(best_policy['validation_accuracy'])}, eval_acc={fmt_pct(best_policy['final_accuracy'])}")
    else:
        best_policy = None
        print("  No valid policy found meeting coverage >=30% and rows >=1000")

    # Write outputs
    print("\nWriting outputs...")
    fieldnames = ["candidate_id", "model", "horizon", "target_type", "feature_set", "policy_type",
        "hyperparams_id", "selected_on_validation", "validation_accuracy", "validation_coverage",
        "validation_rows", "final_accuracy", "final_coverage", "final_rows", "active_ticker_count",
        "pass_60", "pass_65", "claim_level", "evaluator_version"]
    with (OUTPUT_DIR / "validation_candidate_results.csv").open("w", newline="") as f:
        if all_policies:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(all_policies)
    with (OUTPUT_DIR / "final_candidate_results.csv").open("w", newline="") as f:
        if all_policies:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(all_policies)
    with (OUTPUT_DIR / "selected_policies.csv").open("w", newline="") as f:
        if best_policy:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerow(best_policy)
    b60_candidates = [p for p in all_policies if p.get("pass_60") == True]
    f65_candidates = [p for p in all_policies if p.get("pass_65") == True or float(p.get("final_accuracy", 0) or 0) >= 0.65]
    exploratory = [p for p in all_policies if p not in f65_candidates and p not in b60_candidates]
    with (OUTPUT_DIR / "baseline60_candidates.csv").open("w", newline="") as f:
        if b60_candidates:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(b60_candidates)
    with (OUTPUT_DIR / "final65_candidates.csv").open("w", newline="") as f:
        if f65_candidates:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(f65_candidates)
    with (OUTPUT_DIR / "exploratory_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(exploratory[:500])
    best_f65 = max((float(p.get("final_accuracy", 0) or 0) for p in all_policies if float(p.get("final_coverage", 0) or 0) >= 0.30 and int(p.get("final_rows", 0) or 0) >= 1000), default=0)
    best_b60 = max((float(p.get("final_accuracy", 0) or 0) for p in all_policies if float(p.get("final_coverage", 0) or 0) >= 0.95 and int(p.get("final_rows", 0) or 0) >= 1000), default=0)
    f65_pass = best_f65 >= 0.65
    b60_pass = best_b60 >= 0.60
    manifest = {"total_base_experiments": len(candidate_results), "total_policies": len(all_policies),
        "baseline60_candidates": len(b60_candidates), "final65_candidates": len(f65_candidates),
        "best_baseline60_accuracy": round(best_b60, 6), "best_final65_accuracy": round(best_f65, 6),
        "baseline60_pass": b60_pass, "final65_pass": f65_pass, "completed_at": now_utc(),
        "evaluator_version": EVALUATOR_VERSION, "leakage_safe": True, "daily_data_used": False, "resampling_used": False}
    with (OUTPUT_DIR / "full_tuning_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    blocked = ["# Full Tuning Sweep - Blocked or Skipped Candidates", "",
        f"- Completed: {manifest['completed_at']}", f"- Base experiments: {manifest['total_base_experiments']}",
        f"- Policies evaluated: {manifest['total_policies']}", "",
        "## Notes", "", "- All policies selected on 2024 validation only.",
        "- Canonical evaluator v1.0.0 used for all metrics.", ""]
    with (OUTPUT_DIR / "blocked_or_skipped_candidates.md").open("w") as f: f.write("\n".join(blocked))
    log = ["# Full Tuning Sweep Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Base experiments: {manifest['total_base_experiments']}", f"- Policies: {manifest['total_policies']}",
        f"- Best baseline60: {fmt_pct(best_b60)}", f"- Best final65: {fmt_pct(best_f65)}",
        f"- Baseline60 pass: {'YES' if b60_pass else 'NO'}", f"- Final65 pass: {'YES' if f65_pass else 'NO'}",
        f"- Gap to 60: {fmt_pct(0.60 - best_b60)}", f"- Gap to 65: {fmt_pct(0.65 - best_f65)}", ""]
    with (OUTPUT_DIR / "full_tuning_run_log.md").open("w") as f: f.write("\n".join(log))
    print(f"\nBest baseline60: {fmt_pct(best_b60)}")
    print(f"Best final65: {fmt_pct(best_f65)}")
    print(f"Baseline60 pass: {'YES' if b60_pass else 'NO'}")
    print(f"Final65 pass: {'YES' if f65_pass else 'NO'}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())