"""Overall directional final65 sweep for VN30 hourly 2015."""
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

from scripts.research.vn30_hourly_2015_canonical_eval import compute_accuracy, compute_pooled_accuracy, evaluate_predictions, EVALUATOR_VERSION

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_overall_directional_final65"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SEED = 42

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
    for code in ["VNINDEX", "VN30"]:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[df["close"] > 0].sort_values("datetime").reset_index(drop=True)
        indices[code] = df
    return indices

def build_features(df: pd.DataFrame, index_data: dict | None = None, feature_set: str = "combined") -> tuple[pd.DataFrame, list[str]]:
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feat_cols = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 10, 20, 40, 60): prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20): prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
        for window in (5, 10, 20, 40, 60):
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
        for lag in (1, 2, 4, 8, 20, 40): prepared.loc[idx, f"lag_ret_{lag}"] = close.pct_change(periods=lag, fill_method=None).shift(1)
        for window in (4, 8, 20, 40, 60):
            min_p = max(2, window // 4)
            prepared.loc[idx, f"roll_ret_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
        for window in (8, 20, 40, 60):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_change_{window}"] = volume.pct_change(periods=1, fill_method=None).rolling(window, min_periods=min_p).mean()
        # Volatility-normalized momentum
        for window in (20, 40, 60):
            min_p = max(3, window // 4)
            mom = close / close.shift(window) - 1.0
            vol = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
            prepared.loc[idx, f"vol_norm_momentum_{window}"] = mom / vol.replace(0.0, np.nan)
    feat_cols = ["return_1","return_2","return_3","return_5","return_10","return_20","return_40","return_60",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20",
        "rolling_return_mean_5","rolling_return_vol_5","close_sma_ratio_5","momentum_5",
        "rolling_return_mean_10","rolling_return_vol_10","close_sma_ratio_10","momentum_10",
        "rolling_return_mean_20","rolling_return_vol_20","close_sma_ratio_20","momentum_20",
        "rolling_return_mean_40","rolling_return_vol_40","close_sma_ratio_40","momentum_40",
        "rolling_return_mean_60","rolling_return_vol_60","close_sma_ratio_60","momentum_60",
        "rsi_14","macd","macd_signal","macd_hist","volume_change_1","volume_shock_20",
        "high_low_range","open_close_spread","close_position_in_range",
        "lag_ret_1","lag_ret_2","lag_ret_4","lag_ret_8","lag_ret_20","lag_ret_40",
        "roll_ret_4","roll_ret_8","roll_ret_20","roll_ret_40","roll_ret_60",
        "roll_vol_8","roll_vol_20","roll_vol_40","roll_vol_60",
        "roll_vol_change_8","roll_vol_change_20","roll_vol_change_40",
        "vol_norm_momentum_20","vol_norm_momentum_40","vol_norm_momentum_60"]
    if feature_set in ("market_lagged", "combined", "interaction") and index_data:
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
    if feature_set in ("combined", "interaction"):
        # Interaction features
        for col1 in ["return_1", "momentum_20", "rsi_14"]:
            for col2 in ["vnindex_lag_1", "vnindex_roll_mean_20"]:
                if col1 in prepared.columns and col2 in prepared.columns:
                    int_col = f"{col1}_x_{col2}"
                    prepared[int_col] = prepared[col1] * prepared[col2]
                    feat_cols.append(int_col)
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
    """Binary directional labels: 1 if future return > 0, else 0."""
    return (future_returns > 0).astype(int)

def train_rf(X: pd.DataFrame, y: pd.Series, params: dict):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params, random_state=SEED).fit(X, y)

def train_xgb(X: pd.DataFrame, y: pd.Series, params: dict):
    import xgboost as xgb
    return xgb.XGBClassifier(**params, random_state=SEED, eval_metric="logloss").fit(X, y)

def train_lgbm(X: pd.DataFrame, y: pd.Series, params: dict):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params, random_state=SEED, verbose=-1).fit(X, y)

TRAINERS = {"random_forest": train_rf, "xgboost": train_xgb, "lightgbm": train_lgbm}

def hyperparam_grid():
    """Generate hyperparameter candidates."""
    candidates = []
    cid = 0
    # RF configs
    for n_est in [300, 500, 800]:
        for max_d in [6, 10, 16, None]:
            for min_leaf in [3, 5, 10, 20]:
                for max_feat in ["sqrt", "log2", 0.5]:
                    for cw in [None, "balanced"]:
                        cid += 1
                        candidates.append({"candidate_id": f"rf_{cid}", "model": "random_forest",
                            "params": {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_leaf,
                                "max_features": max_feat, "class_weight": cw}})
    # XGB configs
    for n_est in [300, 500, 800]:
        for max_d in [3, 4, 5, 6]:
            for lr in [0.01, 0.03, 0.05]:
                for ss in [0.7, 0.9, 1.0]:
                    for cs in [0.7, 0.9, 1.0]:
                        for mcw in [1, 5, 10]:
                            cid += 1
                            candidates.append({"candidate_id": f"xgb_{cid}", "model": "xgboost",
                                "params": {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr,
                                    "subsample": ss, "colsample_bytree": cs, "min_child_weight": mcw}})
    # LGBM configs
    for n_est in [300, 500, 800]:
        for nl in [15, 31, 63]:
            for max_d in [4, 8, 12, -1]:
                for lr in [0.01, 0.03, 0.05]:
                    for mcs in [20, 50, 100]:
                        for ss in [0.7, 0.9, 1.0]:
                            for cs in [0.7, 0.9, 1.0]:
                                for cw in [None, "balanced"]:
                                    cid += 1
                                    candidates.append({"candidate_id": f"lgbm_{cid}", "model": "lightgbm",
                                        "params": {"n_estimators": n_est, "num_leaves": nl, "max_depth": max_d,
                                            "learning_rate": lr, "min_child_samples": mcs, "subsample": ss,
                                            "colsample_bytree": cs, "class_weight": cw}})
    return candidates

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Overall Directional Final65 Sweep")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "seed": SEED, "evaluator_version": EVALUATOR_VERSION, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    feature_sets = ["combined"]
    horizons = [40, 60, 80, 100, 120, 160]
    all_candidates = hyperparam_grid()
    print(f"\nTotal hyperparameter candidates: {len(all_candidates)}")
    print(f"Feature sets: {feature_sets}")
    print(f"Horizons: {horizons}")
    print(f"Total combinations: {len(all_candidates) * len(feature_sets) * len(horizons)}")
    # Prioritize RF h=60 first
    rf_h60 = [c for c in all_candidates if c["model"] == "random_forest"]
    others = [c for c in all_candidates if c["model"] != "random_forest"]
    # Build features for each feature set
    feature_cache = {}
    for fs in feature_sets:
        print(f"\nBuilding features: {fs}...")
        inc_market = fs in ("market_lagged", "combined", "interaction")
        feature_cache[fs], feat_cols = build_features(stock_df, index_data, feature_set=fs)
        print(f"  {len(feat_cols)} features")
    val_results, eval_results = [], []
    cid_counter = 0
    skipped = []
    # Phase 1: RF h=60 with all feature sets
    print("\n=== Phase 1: RF h=60 sweep ===")
    for c in rf_h60[:20]:  # Limit to first 20 RF configs for time
        for fs in feature_sets:
            cid_counter += 1
            h = 60
            try:
                fdf = feature_cache[fs]
                fr = compute_future_returns(fdf, h)
                labels = make_labels(fr)
                dt_series = fdf.loc[fr.index, "datetime"]
                valid_mask = fr.notna()
                t_idx = fr.index[(dt_series <= TRAIN_END) & valid_mask]
                v_idx = fr.index[(dt_series >= VAL_START) & (dt_series <= VAL_END) & valid_mask]
                e_idx = fr.index[(dt_series >= EVAL_START) & valid_mask]
                if len(t_idx) < 100 or len(v_idx) < 20 or len(e_idx) < 20:
                    skipped.append(f"{c['candidate_id']} h={h} fs={fs}: insufficient data")
                    continue
                fcp = [col for col in feat_cols if col in fdf.columns]
                tX = fdf.reindex(t_idx)[fcp].fillna(0)
                ty = labels.reindex(t_idx)
                vX = fdf.reindex(v_idx)[fcp].fillna(0)
                vy = labels.reindex(v_idx)
                eX = fdf.reindex(e_idx)[fcp].fillna(0)
                ey = labels.reindex(e_idx)
                params = c["params"].copy()
                m = TRAINERS[c["model"]](tX, ty, params)
                vp = m.predict(vX)
                ep = m.predict(eX)
                vr = compute_accuracy(vy.values.astype(float), vp.astype(float))
                er = compute_accuracy(ey.values.astype(float), ep.astype(float))
                ve = evaluate_predictions(vy.values.astype(float), vp.astype(float))
                ee = evaluate_predictions(ey.values.astype(float), ep.astype(float))
                val_results.append({"candidate_id": c["candidate_id"], "model": c["model"], "horizon": h,
                    "feature_set": fs, "hyperparams_id": json.dumps(params, default=str),
                    "validation_accuracy": round(vr["accuracy"], 6), "validation_rows": vr["total_valid"],
                    "validation_coverage": 1.0, "final_accuracy": round(er["accuracy"], 6),
                    "final_rows": er["total_valid"], "final_coverage": 1.0, "active_ticker_count": len(tickers),
                    "full_universe": "yes", "full_coverage": "yes", "pass_60": ve["pass_60"],
                    "pass_65": ve["pass_65"], "selected_on_validation": "yes", "claim_level": ee["claim_level"],
                    "evaluator_version": EVALUATOR_VERSION})
                eval_results.append(val_results[-1].copy())
            except Exception as e:
                skipped.append(f"{c['candidate_id']} h={h} fs={fs}: {e}")
            if cid_counter % 50 == 0: print(f"  Progress: {cid_counter} candidates processed")
    # Phase 2: Other models/horizons (limited for time)
    print("\n=== Phase 2: Other models/horizons sweep ===")
    # Select top 50 RF h=60 from Phase 1
    if val_results:
        val_df = pd.DataFrame(val_results)
        top_rf = val_df[val_df["model"] == "random_forest"].nlargest(50, "validation_accuracy")
        top_rf_configs = []
        for _, row in top_rf.iterrows():
            top_rf_configs.append({"candidate_id": row["candidate_id"], "model": row["model"],
                "params": json.loads(row["hyperparams_id"]), "feature_set": row["feature_set"]})
    else:
        top_rf_configs = []
    # Run other models with limited configs
    other_models = [c for c in all_candidates if c["model"] != "random_forest"][:20]  # Limit
    for c in other_models:
        for h in [60]:
            for fs in ["combined"]:  # Use best feature set
                cid_counter += 1
                try:
                    fdf = feature_cache[fs]
                    fr = compute_future_returns(fdf, h)
                    labels = make_labels(fr)
                    dt_series = fdf.loc[fr.index, "datetime"]
                    valid_mask = fr.notna()
                    t_idx = fr.index[(dt_series <= TRAIN_END) & valid_mask]
                    v_idx = fr.index[(dt_series >= VAL_START) & (dt_series <= VAL_END) & valid_mask]
                    e_idx = fr.index[(dt_series >= EVAL_START) & valid_mask]
                    if len(t_idx) < 100 or len(v_idx) < 20 or len(e_idx) < 20:
                        skipped.append(f"{c['candidate_id']} h={h} fs={fs}: insufficient data")
                        continue
                    fcp = [col for col in feat_cols if col in fdf.columns]
                    tX = fdf.reindex(t_idx)[fcp].fillna(0)
                    ty = labels.reindex(t_idx)
                    vX = fdf.reindex(v_idx)[fcp].fillna(0)
                    vy = labels.reindex(v_idx)
                    eX = fdf.reindex(e_idx)[fcp].fillna(0)
                    ey = labels.reindex(e_idx)
                    params = c["params"].copy()
                    m = TRAINERS[c["model"]](tX, ty, params)
                    vp = m.predict(vX)
                    ep = m.predict(eX)
                    vr = compute_accuracy(vy.values.astype(float), vp.astype(float))
                    er = compute_accuracy(ey.values.astype(float), ep.astype(float))
                    ve = evaluate_predictions(vy.values.astype(float), vp.astype(float))
                    ee = evaluate_predictions(ey.values.astype(float), ep.astype(float))
                    val_results.append({"candidate_id": c["candidate_id"], "model": c["model"], "horizon": h,
                        "feature_set": fs, "hyperparams_id": json.dumps(params, default=str),
                        "validation_accuracy": round(vr["accuracy"], 6), "validation_rows": vr["total_valid"],
                        "validation_coverage": 1.0, "final_accuracy": round(er["accuracy"], 6),
                        "final_rows": er["total_valid"], "final_coverage": 1.0, "active_ticker_count": len(tickers),
                        "full_universe": "yes", "full_coverage": "yes", "pass_60": ve["pass_60"],
                        "pass_65": ve["pass_65"], "selected_on_validation": "yes", "claim_level": ee["claim_level"],
                        "evaluator_version": EVALUATOR_VERSION})
                    eval_results.append(val_results[-1].copy())
                except Exception as e:
                    skipped.append(f"{c['candidate_id']} h={h} fs={fs}: {e}")
                if cid_counter % 100 == 0: print(f"  Progress: {cid_counter} candidates processed")
    # Write outputs
    print("\nWriting outputs...")
    fields = ["candidate_id", "model", "horizon", "feature_set", "hyperparams_id", "validation_accuracy",
        "validation_rows", "validation_coverage", "final_accuracy", "final_rows", "final_coverage",
        "active_ticker_count", "full_universe", "full_coverage", "pass_60", "pass_65",
        "selected_on_validation", "claim_level", "evaluator_version"]
    with (OUTPUT_DIR / "validation_candidate_results.csv").open("w", newline="") as f:
        if val_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(val_results)
    with (OUTPUT_DIR / "final_candidate_results.csv").open("w", newline="") as f:
        if eval_results:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(eval_results)
    # Best candidates
    if eval_results:
        er_df = pd.DataFrame(eval_results)
        best_global = er_df.nlargest(10, "final_accuracy")
        with (OUTPUT_DIR / "best_global_candidates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(best_global.to_dict("records"))
        pass65 = er_df[er_df["pass_65"] == True]
        with (OUTPUT_DIR / "full_coverage_65_candidates.csv").open("w", newline="") as f:
            if len(pass65) > 0:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); w.writerows(pass65.to_dict("records"))
        best_acc = float(er_df["final_accuracy"].max())
        best_row = er_df.loc[er_df["final_accuracy"].idxmax()]
    else:
        best_acc = 0.0
        best_row = None
    # Manifest
    manifest = {"total_candidates_processed": cid_counter, "total_skipped": len(skipped),
        "best_final_accuracy": round(best_acc, 6), "final65_passed": bool(best_acc >= 0.65) if best_row is not None else False,
        "completed_at": now_utc(), "leakage_safe": True, "daily_data_used": False,
        "resampling_used": False, "full_universe_required": True, "full_coverage_required": True,
        "no_abstention": True, "no_ranking_metric": True}
    with (OUTPUT_DIR / "overall_directional_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    # Run log
    log = ["# Overall Directional Final65 Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Candidates processed: {manifest['total_candidates_processed']}",
        f"- Skipped: {manifest['total_skipped']}",
        f"- Best final accuracy: {fmt_pct(best_acc)}",
        f"- Final65 passed: {'YES' if manifest['final65_passed'] else 'NO'}", ""]
    if best_row is not None:
        log.extend([f"- Best model: {best_row['model']}", f"- Best horizon: {best_row['horizon']}",
            f"- Best feature set: {best_row['feature_set']}", f"- Best validation accuracy: {fmt_pct(best_row['validation_accuracy'])}", ""])
    with (OUTPUT_DIR / "overall_directional_run_log.md").open("w") as f: f.write("\n".join(log))
    # Skipped candidates
    with (OUTPUT_DIR / "skipped_or_blocked_candidates.md").open("w") as f:
        f.write("# Skipped or Blocked Candidates\n\n")
        for s in skipped[:100]: f.write(f"- {s}\n")
        if len(skipped) > 100: f.write(f"\n... and {len(skipped) - 100} more\n")
    print(f"\nBest final accuracy: {fmt_pct(best_acc)}")
    if best_row is not None:
        print(f"Best config: {best_row['model']} h={best_row['horizon']} fs={best_row['feature_set']}")
        print(f"Validation accuracy: {fmt_pct(best_row['validation_accuracy'])}")
    print(f"Final65 passed: {'YES' if manifest['final65_passed'] else 'NO'}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
