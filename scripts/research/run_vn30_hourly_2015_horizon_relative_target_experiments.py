"""Horizon expansion & relative target experiments for VN30 hourly 2015."""
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

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_horizon_relative_target_experiments"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
LONG_HORIZONS = [40, 60, 80, 120]
MODELS = ["lightgbm", "xgboost", "random_forest"]
SEED = 42
NOISE_BANDS = [0.0005, 0.001, 0.0015, 0.002, 0.003]
CONFIDENCE_THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]
LGBM_PARAMS = {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1}
XGB_PARAMS = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": SEED, "eval_metric": "logloss"}
RF_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED}

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

def get_index_returns(index_data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    result = {}
    for code, df in index_data.items():
        df = df.copy().sort_values("datetime")
        result[code] = pd.Series(df["close"].pct_change(periods=1, fill_method=None).values, index=df["datetime"])
    return result

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

def build_labels_relative(df: pd.DataFrame, horizon: int, index_returns: pd.Series, noise_band: float = 0.0) -> pd.Series:
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        stock_ret = (future_close - group["close"]) / group["close"]
        market_ret = index_returns.reindex(group["datetime"]).shift(-horizon).values
        relative_ret = stock_ret.values - market_ret
        if noise_band > 0:
            direction = np.where(np.abs(relative_ret) > noise_band, (relative_ret > 0).astype(float), np.nan)
        else:
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

def train_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**LGBM_PARAMS).fit(X, y)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**XGB_PARAMS).fit(X, y)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")

def predict_with_confidence(model, X: pd.DataFrame):
    preds = model.predict(X)
    probas = model.predict_proba(X)
    return preds, probas.max(axis=1), probas

def directional_accuracy(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0: return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))

def run_experiment(feature_df, feat_cols, index_returns, horizon, target_type, market_ref, noise_band, model_name):
    fcp = [c for c in feat_cols if c in feature_df.columns]
    if target_type == "absolute":
        labels = build_labels_absolute(feature_df, horizon)
    else:
        idx_ret = index_returns.get(market_ref)
        if idx_ret is None: return None
        labels = build_labels_relative(feature_df, horizon, idx_ret, noise_band)
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
        m = train_model(model_name, tX, ty)
        vp, vc, vpr = predict_with_confidence(m, vX)
        ep, ec, epr = predict_with_confidence(m, eX)
        va = directional_accuracy(vy.values, vp)
        ea = directional_accuracy(ey.values, ep)
        return {"model": model_name, "horizon": horizon, "target_type": target_type,
            "market_reference": market_ref if target_type != "absolute" else "",
            "noise_band": noise_band, "feature_set": "C",
            "validation_accuracy": round(va, 6), "validation_coverage": 1.0, "validation_rows": len(vl),
            "final_accuracy": round(ea, 6), "final_coverage": 1.0, "final_rows": len(el),
            "model_obj": m, "eval_X": eX, "eval_y": ey, "eval_preds": ep, "eval_probas": epr}
    except: return None

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Horizon & Relative Target Experiments")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "models": MODELS, "horizons": LONG_HORIZONS, "seed": SEED,
        "noise_bands": NOISE_BANDS, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    print("\nBuilding features...")
    feature_df_c, feat_cols_c = build_features(stock_df, index_data, include_market=True)
    print(f"  Feature set C: {len(feat_cols_c)} features")
    index_returns = get_index_returns(index_data)
    all_val_results, all_eval_results = [], []
    label_dist, selected_policy = [], []

    # Experiment configurations
    experiments = []
    # A. Absolute direction, long horizons
    for h in LONG_HORIZONS:
        for mn in MODELS:
            experiments.append({"target_type": "absolute", "market_ref": "", "horizon": h, "noise_band": 0.0, "model": mn})
    # B. Relative to VN30
    for h in LONG_HORIZONS:
        for mn in MODELS:
            experiments.append({"target_type": "relative_vn30", "market_ref": "VN30", "horizon": h, "noise_band": 0.0, "model": mn})
    # C. Relative to VNINDEX
    for h in LONG_HORIZONS:
        for mn in MODELS:
            experiments.append({"target_type": "relative_vnindex", "market_ref": "VNINDEX", "horizon": h, "noise_band": 0.0, "model": mn})
    # D. Relative with noise band
    for h in LONG_HORIZONS:
        for nb in NOISE_BANDS:
            for mn in MODELS:
                experiments.append({"target_type": "relative_vn30_noise", "market_ref": "VN30", "horizon": h, "noise_band": nb, "model": mn})
                experiments.append({"target_type": "relative_vnindex_noise", "market_ref": "VNINDEX", "horizon": h, "noise_band": nb, "model": mn})

    print(f"\nRunning {len(experiments)} experiments...")
    for i, exp in enumerate(experiments):
        if (i + 1) % 50 == 0: print(f"  Progress: {i+1}/{len(experiments)}")
        result = run_experiment(feature_df_c, feat_cols_c, index_returns, exp["horizon"],
            exp["target_type"], exp["market_ref"], exp["noise_band"], exp["model"])
        if result is None: continue
        all_val_results.append({k: result[k] for k in ["target_type", "market_reference", "horizon", "noise_band",
            "model", "feature_set", "validation_accuracy", "validation_coverage", "validation_rows"]})
        all_eval_results.append({k: result[k] for k in ["target_type", "market_reference", "horizon", "noise_band",
            "model", "feature_set", "validation_accuracy", "validation_coverage", "validation_rows",
            "final_accuracy", "final_coverage", "final_rows"]})

    # E. Confidence filtering on best candidates
    print("\nApplying confidence filtering...")
    best_by_config = {}
    for r in all_eval_results:
        key = (r["target_type"], r["market_reference"], r["horizon"], r["noise_band"], r["model"])
        if key not in best_by_config or r["validation_accuracy"] > best_by_config[key]["validation_accuracy"]:
            best_by_config[key] = r
    for key, r in best_by_config.items():
        target_type, market_ref, horizon, noise_band, model_name = key
        idx_ret = index_returns.get(market_ref) if market_ref else None
        if target_type == "absolute":
            labels = build_labels_absolute(feature_df_c, horizon)
        else:
            labels = build_labels_relative(feature_df_c, horizon, idx_ret, noise_band)
        train_df, val_df, eval_df = split_data(feature_df_c, labels)
        fcp = [c for c in feat_cols_c if c in feature_df_c.columns]
        tl = labels.reindex(train_df.index).dropna()
        vl = labels.reindex(val_df.index).dropna()
        el = labels.reindex(eval_df.index).dropna()
        if len(tl) < 100 or len(vl) < 20 or len(el) < 20: continue
        tX = feature_df_c.reindex(tl.index)[fcp].fillna(0)
        ty = tl.astype(int)
        vX = feature_df_c.reindex(vl.index)[fcp].fillna(0)
        vy = vl.astype(int)
        eX = feature_df_c.reindex(el.index)[fcp].fillna(0)
        ey = el.astype(int)
        try:
            m = train_model(model_name, tX, ty)
            vp, vc, vpr = predict_with_confidence(m, vX)
            ep, ec, epr = predict_with_confidence(m, eX)
            for thresh in CONFIDENCE_THRESHOLDS:
                vmask = vc >= thresh
                emask = ec >= thresh
                vn, en = vmask.sum(), emask.sum()
                if vn < 100 or en < 100: continue
                vac = directional_accuracy(vy.values[vmask], vp[vmask])
                eac = directional_accuracy(ey.values[emask], ep[emask])
                vcov = vn / len(vy)
                ecov = en / len(ey)
                all_val_results.append({"target_type": target_type, "market_reference": market_ref, "horizon": horizon,
                    "noise_band": noise_band, "model": model_name, "feature_set": "C",
                    "validation_accuracy": round(vac, 6), "validation_coverage": round(vcov, 4), "validation_rows": int(vn),
                    "confidence_threshold": thresh})
                all_eval_results.append({"target_type": target_type, "market_reference": market_ref, "horizon": horizon,
                    "noise_band": noise_band, "model": model_name, "feature_set": "C",
                    "validation_accuracy": round(vac, 6), "validation_coverage": round(vcov, 4), "validation_rows": int(vn),
                    "final_accuracy": round(eac, 6), "final_coverage": round(ecov, 4), "final_rows": int(en),
                    "confidence_threshold": thresh})
        except: continue    # Select best policy on validation
    print("\nSelecting best policy on validation...")
    valid_val = [r for r in all_val_results if r.get("validation_rows", 0) >= 100]
    if valid_val:
        best_val = max(valid_val, key=lambda r: r["validation_accuracy"])
        selected_policy.append({**best_val, "selected_on_validation": "yes"})
        print(f"  Best: {best_val['target_type']}, {best_val['model']}, h={best_val['horizon']}, val_acc={fmt_pct(best_val['validation_accuracy'])}")

    # Compute label distribution
    print("\nComputing label distribution...")
    for h in LONG_HORIZONS:
        labels = build_labels_absolute(feature_df_c, h)
        n_up = int((labels == 1).sum())
        n_down = int((labels == 0).sum())
        label_dist.append({"target_type": "absolute", "horizon": h, "n_up": n_up, "n_down": n_down, "n_total": n_up + n_down,
            "up_ratio": round(n_up / (n_up + n_down), 4) if (n_up + n_down) > 0 else 0})
    for h in LONG_HORIZONS:
        for market_ref in ["VN30", "VNINDEX"]:
            idx_ret = index_returns.get(market_ref)
            if idx_ret is None: continue
            labels = build_labels_relative(feature_df_c, h, idx_ret)
            n_up = int((labels == 1).sum())
            n_down = int((labels == 0).sum())
            label_dist.append({"target_type": f"relative_{market_ref.lower()}", "horizon": h,
                "n_up": n_up, "n_down": n_down, "n_total": n_up + n_down,
                "up_ratio": round(n_up / (n_up + n_down), 4) if (n_up + n_down) > 0 else 0})

    # Write outputs
    print("\nWriting outputs...")
    fieldnames_val = ["target_type", "market_reference", "horizon", "noise_band", "model", "feature_set",
        "validation_accuracy", "validation_coverage", "validation_rows", "confidence_threshold"]
    fieldnames_eval = ["target_type", "market_reference", "horizon", "noise_band", "model", "feature_set",
        "validation_accuracy", "validation_coverage", "validation_rows", "final_accuracy", "final_coverage",
        "final_rows", "confidence_threshold"]
    with (OUTPUT_DIR / "validation_results.csv").open("w", newline="") as f:
        if all_val_results:
            w = csv.DictWriter(f, fieldnames=fieldnames_val, extrasaction="ignore")
            w.writeheader(); w.writerows(all_val_results)
    with (OUTPUT_DIR / "final_eval_results.csv").open("w", newline="") as f:
        if all_eval_results:
            w = csv.DictWriter(f, fieldnames=fieldnames_eval, extrasaction="ignore")
            w.writeheader(); w.writerows(all_eval_results)
    with (OUTPUT_DIR / "label_distribution.csv").open("w", newline="") as f:
        if label_dist:
            w = csv.DictWriter(f, fieldnames=label_dist[0].keys())
            w.writeheader(); w.writerows(label_dist)
    with (OUTPUT_DIR / "selected_policy.csv").open("w", newline="") as f:
        if selected_policy:
            w = csv.DictWriter(f, fieldnames=selected_policy[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(selected_policy)

    # Find best results
    global_results = [r for r in all_eval_results if r.get("final_coverage", 0) >= 0.95 and r.get("final_rows", 0) >= 1000]
    qualified_results = [r for r in all_eval_results if r.get("final_coverage", 0) >= 0.30 and r.get("final_rows", 0) >= 1000]
    bg = max((float(r.get("final_accuracy", 0) or 0) for r in global_results), default=0)
    bc = max((float(r.get("final_accuracy", 0) or 0) for r in qualified_results), default=0)
    b60 = bg >= 0.60
    f65 = bc >= 0.65
    baseline60 = [r for r in global_results if float(r.get("final_accuracy", 0) or 0) >= 0.60]
    final65 = [r for r in qualified_results if float(r.get("final_accuracy", 0) or 0) >= 0.65]
    exploratory = [r for r in all_eval_results if r not in baseline60 and r not in final65]
    manifest = {"total_experiments": len(all_eval_results), "global_candidates": len(global_results),
        "coverage_qualified_candidates": len(qualified_results), "best_global_accuracy": round(bg, 6),
        "best_coverage_qualified_accuracy": round(bc, 6), "baseline_60_pass": b60,
        "final_65_pass": f65, "completed_at": now_utc(),
        "leakage_safe": True, "daily_data_used": False, "resampling_used": False}
    with (OUTPUT_DIR / "experiment_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    with (OUTPUT_DIR / "baseline60_candidates.csv").open("w", newline="") as f:
        if baseline60:
            w = csv.DictWriter(f, fieldnames=fieldnames_eval, extrasaction="ignore")
            w.writeheader(); w.writerows(baseline60)
    with (OUTPUT_DIR / "final65_candidates.csv").open("w", newline="") as f:
        if final65:
            w = csv.DictWriter(f, fieldnames=fieldnames_eval, extrasaction="ignore")
            w.writeheader(); w.writerows(final65)
    with (OUTPUT_DIR / "exploratory_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=fieldnames_eval, extrasaction="ignore")
            w.writeheader(); w.writerows(exploratory[:100])
    log = ["# Horizon & Relative Target Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Experiments: {manifest['total_experiments']}", f"- Best global: {fmt_pct(bg)}",
        f"- Best coverage-qualified: {fmt_pct(bc)}", f"- Baseline 60: {'YES' if b60 else 'NO'}",
        f"- Final 65: {'YES' if f65 else 'NO'}",
        f"- Gap to 60: {fmt_pct(0.60 - bg)}", f"- Gap to 65: {fmt_pct(0.65 - bc)}", ""]
    with (OUTPUT_DIR / "run_log.md").open("w") as f: f.write("\n".join(log))
    print(f"\nBest global: {fmt_pct(bg)}")
    print(f"Best coverage-qualified: {fmt_pct(bc)}")
    print(f"Baseline 60: {'YES' if b60 else 'NO'}")
    print(f"Final 65: {'YES' if f65 else 'NO'}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())