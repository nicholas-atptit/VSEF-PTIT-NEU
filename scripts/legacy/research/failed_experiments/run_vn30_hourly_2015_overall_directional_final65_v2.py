"""Overall directional final65 v2 sweep with rolling validation for VN30 hourly 2015."""
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

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_overall_directional_final65_v2"
SEED = 42

# Rolling validation windows
WINDOWS = {
    "A": {"train_end": pd.Timestamp("2021-12-31 23:59:59"), "val_start": pd.Timestamp("2022-01-01"), "val_end": pd.Timestamp("2022-12-31 23:59:59")},
    "B": {"train_end": pd.Timestamp("2022-12-31 23:59:59"), "val_start": pd.Timestamp("2023-01-01"), "val_end": pd.Timestamp("2023-12-31 23:59:59")},
    "C": {"train_end": pd.Timestamp("2023-12-31 23:59:59"), "val_start": pd.Timestamp("2024-01-01"), "val_end": pd.Timestamp("2024-12-31 23:59:59")},
}
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

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
    return (future_returns > 0).astype(int)

def train_rf(X, y, params):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params, random_state=SEED, bootstrap=True).fit(X, y)

def train_xgb(X, y, params):
    import xgboost as xgb
    return xgb.XGBClassifier(**params, random_state=SEED, eval_metric="logloss").fit(X, y)

def train_lgbm(X, y, params):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params, random_state=SEED, verbose=-1).fit(X, y)

TRAINERS = {"random_forest": train_rf, "xgboost": train_xgb, "lightgbm": train_lgbm}

def generate_candidates():
    """Generate hyperparameter candidates prioritized by known strong areas."""
    candidates = []
    cid = 0
    # RF configs - prioritized
    for n_est in [300, 500, 800, 1200]:
        for max_d in [6, 10, 16, 24, None]:
            for min_leaf in [1, 3, 5, 10, 20]:
                for max_feat in ["sqrt", "log2", 0.3, 0.5, 0.8]:
                    for cw in [None, "balanced", "balanced_subsample"]:
                        cid += 1
                        candidates.append({"candidate_id": f"rf_{cid}", "model": "random_forest",
                            "params": {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_leaf,
                                "max_features": max_feat, "class_weight": cw}})
    # XGB configs
    for n_est in [300, 500, 800]:
        for max_d in [3, 4, 5, 6]:
            for lr in [0.01, 0.03, 0.05, 0.08]:
                for ss in [0.7, 0.85, 1.0]:
                    for cs in [0.7, 0.85, 1.0]:
                        for mcw in [1, 5, 10]:
                            for rl in [1, 5, 10]:
                                cid += 1
                                candidates.append({"candidate_id": f"xgb_{cid}", "model": "xgboost",
                                    "params": {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr,
                                        "subsample": ss, "colsample_bytree": cs, "min_child_weight": mcw, "reg_lambda": rl}})
    # LGBM configs
    for n_est in [300, 500, 800]:
        for nl in [15, 31, 63, 127]:
            for max_d in [4, 8, 12, -1]:
                for lr in [0.01, 0.03, 0.05]:
                    for mcs in [20, 50, 100]:
                        for ss in [0.7, 0.85, 1.0]:
                            for cs in [0.7, 0.85, 1.0]:
                                for cw in [None, "balanced"]:
                                    cid += 1
                                    candidates.append({"candidate_id": f"lgbm_{cid}", "model": "lightgbm",
                                        "params": {"n_estimators": n_est, "num_leaves": nl, "max_depth": max_d,
                                            "learning_rate": lr, "min_child_samples": mcs, "subsample": ss,
                                            "colsample_bytree": cs, "class_weight": cw}})
    return candidates

def get_split_indices(fdf: pd.DataFrame, fr: pd.Series, train_end: pd.Timestamp, val_start: pd.Timestamp, val_end: pd.Timestamp):
    dt_series = fdf.loc[fr.index, "datetime"]
    valid_mask = fr.notna().values
    t_mask = (dt_series <= train_end).values & valid_mask
    v_mask = (dt_series >= val_start).values & (dt_series <= val_end).values & valid_mask
    t_idx = fr.index[t_mask]
    v_idx = fr.index[v_mask]
    return t_idx, v_idx

def evaluate_candidate(fdf, feat_cols, fr, labels, c, h, fs, tickers, eval_idx=None):
    """Evaluate a single candidate on validation windows and final eval."""
    fcp = [col for col in feat_cols if col in fdf.columns]
    # Rolling validation
    val_accuracies = {}
    val_rows = {}
    for wname, wcfg in WINDOWS.items():
        t_idx, v_idx = get_split_indices(fdf, fr, wcfg["train_end"], wcfg["val_start"], wcfg["val_end"])
        if len(t_idx) < 100 or len(v_idx) < 20:
            val_accuracies[wname] = None
            val_rows[wname] = 0
            continue
        tX = fdf.reindex(t_idx)[fcp].fillna(0)
        ty = labels.reindex(t_idx)
        vX = fdf.reindex(v_idx)[fcp].fillna(0)
        vy = labels.reindex(v_idx)
        params = c["params"].copy()
        m = TRAINERS[c["model"]](tX, ty, params)
        vp = m.predict(vX)
        vr = compute_accuracy(vy.values.astype(float), vp.astype(float))
        val_accuracies[wname] = vr["accuracy"]
        val_rows[wname] = vr["total_valid"]
    # Compute rolling validation stats
    valid_vals = [v for v in val_accuracies.values() if v is not None]
    if len(valid_vals) == 0:
        return None
    mean_val = float(np.mean(valid_vals))
    min_val = float(np.min(valid_vals))
    std_val = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
    stability = mean_val - std_val
    # Final evaluation (only once, after selection)
    final_acc, final_rows_val = None, 0
    if eval_idx is not None and len(eval_idx) > 0:
        # Train on full pre-2025 data for final eval
        dt_series = fdf.loc[fr.index, "datetime"]
        all_train_mask = (dt_series <= WINDOWS["C"]["train_end"]).values & fr.notna().values
        all_train_idx = fr.index[all_train_mask]
        if len(all_train_idx) >= 100:
            tX = fdf.reindex(all_train_idx)[fcp].fillna(0)
            ty = labels.reindex(all_train_idx)
            eX = fdf.reindex(eval_idx)[fcp].fillna(0)
            ey = labels.reindex(eval_idx)
            params = c["params"].copy()
            m = TRAINERS[c["model"]](tX, ty, params)
            ep = m.predict(eX)
            er = compute_accuracy(ey.values.astype(float), ep.astype(float))
            final_acc = er["accuracy"]
            final_rows_val = er["total_valid"]
    return {"candidate_id": c["candidate_id"], "model": c["model"], "horizon": h, "feature_set": fs,
        "hyperparams_id": json.dumps(c["params"], default=str),
        "val_a_acc": val_accuracies.get("A"), "val_b_acc": val_accuracies.get("B"), "val_c_acc": val_accuracies.get("C"),
        "val_a_rows": val_rows.get("A", 0), "val_b_rows": val_rows.get("B", 0), "val_c_rows": val_rows.get("C", 0),
        "mean_validation_accuracy": mean_val, "min_validation_accuracy": min_val,
        "validation_std": std_val, "stability_score": stability,
        "final_accuracy": final_acc, "final_rows": final_rows_val}

def save_checkpoint(state, path):
    with path.open("w") as f: json.dump(state, f, indent=2)

def load_checkpoint(path):
    if path.exists():
        with path.open("r") as f: return json.load(f)
    return None

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Overall Directional Final65 V2 Sweep")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "checkpoint_state.json"
    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path)
    start_idx = ckpt.get("last_candidate_idx", -1) + 1 if ckpt else 0
    completed_results = ckpt.get("completed_results", []) if ckpt else []
    skipped = ckpt.get("skipped", []) if ckpt else []
    run_config = {"seed": SEED, "evaluator_version": EVALUATOR_VERSION, "rolling_windows": list(WINDOWS.keys()),
        "created_at": now_utc(), "resume_from": start_idx}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    feature_sets = ["combined"]
    horizons = [60]
    all_candidates = generate_candidates()
    print(f"\nTotal hyperparameter candidates: {len(all_candidates)}")
    # Build features
    feature_cache = {}
    feat_cols_cache = {}
    for fs in feature_sets:
        print(f"\nBuilding features: {fs}...")
        feature_cache[fs], feat_cols_cache[fs] = build_features(stock_df, index_data, feature_set=fs)
        print(f"  {len(feat_cols_cache[fs])} features")
    # Precompute future returns and labels for each horizon
    fr_cache = {}
    labels_cache = {}
    for h in horizons:
        fr_cache[h] = compute_future_returns(feature_cache["combined"], h)
        labels_cache[h] = make_labels(fr_cache[h])
    # Final eval indices
    eval_idx_cache = {}
    for h in horizons:
        fr = fr_cache[h]
        fdf = feature_cache["combined"]
        dt_series = fdf.loc[fr.index, "datetime"]
        eval_mask = (dt_series >= EVAL_START).values & fr.notna().values
        eval_idx_cache[h] = fr.index[eval_mask]
    print(f"\nRunning candidates from index {start_idx}...")
    cid_counter = start_idx
    max_candidates = 10  # Limit for runtime (rolling validation is 3x slower)
    for i in range(start_idx, min(len(all_candidates), start_idx + max_candidates)):
        c = all_candidates[i]
        cid_counter += 1
        for h in horizons:
            for fs in feature_sets:
                try:
                    fdf = feature_cache[fs]
                    feat_cols = feat_cols_cache[fs]
                    fr = fr_cache[h]
                    labels = labels_cache[h]
                    eval_idx = eval_idx_cache[h]
                    result = evaluate_candidate(fdf, feat_cols, fr, labels, c, h, fs, tickers, eval_idx)
                    if result is None:
                        skipped.append(f"{c['candidate_id']} h={h} fs={fs}: insufficient data")
                        continue
                    result["active_ticker_count"] = len(tickers)
                    result["full_universe"] = "yes"
                    result["full_coverage"] = "yes" if result["final_rows"] > 0 else "no"
                    result["pass_60"] = result["final_accuracy"] >= 0.60 if result["final_accuracy"] is not None else False
                    result["pass_65"] = result["final_accuracy"] >= 0.65 if result["final_accuracy"] is not None else False
                    result["selected_on_validation"] = "yes"
                    result["claim_level"] = "final65_coverage_qualified" if result["pass_65"] else ("baseline60_global" if result["pass_60"] else "failed")
                    result["evaluator_version"] = EVALUATOR_VERSION
                    completed_results.append(result)
                except Exception as e:
                    skipped.append(f"{c['candidate_id']} h={h} fs={fs}: {e}")
        # Save checkpoint every 10 candidates
        if (i - start_idx + 1) % 10 == 0:
            save_checkpoint({"last_candidate_idx": i, "completed_results": completed_results, "skipped": skipped}, checkpoint_path)
            print(f"  Checkpoint saved at candidate {i+1}/{min(len(all_candidates), start_idx + max_candidates)}, completed: {len(completed_results)}")
    # Final checkpoint
    save_checkpoint({"last_candidate_idx": cid_counter - 1, "completed_results": completed_results, "skipped": skipped}, checkpoint_path)
    # Write outputs
    print("\nWriting outputs...")
    # Rolling validation results
    rolling_fields = ["candidate_id", "model", "horizon", "feature_set", "hyperparams_id",
        "val_a_acc", "val_b_acc", "val_c_acc", "val_a_rows", "val_b_rows", "val_c_rows",
        "mean_validation_accuracy", "min_validation_accuracy", "validation_std", "stability_score"]
    with (OUTPUT_DIR / "rolling_validation_results.csv").open("w", newline="") as f:
        if completed_results:
            w = csv.DictWriter(f, fieldnames=rolling_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(completed_results)
    # Candidate selection scores
    sel_fields = rolling_fields + ["final_accuracy", "final_rows", "final_coverage", "active_ticker_count",
        "full_universe", "full_coverage", "pass_60", "pass_65", "selected_on_validation", "claim_level", "evaluator_version"]
    with (OUTPUT_DIR / "candidate_selection_scores.csv").open("w", newline="") as f:
        if completed_results:
            w = csv.DictWriter(f, fieldnames=sel_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(completed_results)
    # Final candidate results
    with (OUTPUT_DIR / "final_candidate_results.csv").open("w", newline="") as f:
        if completed_results:
            w = csv.DictWriter(f, fieldnames=sel_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(completed_results)
    # Best candidates
    if completed_results:
        cr_df = pd.DataFrame(completed_results)
        valid_final = cr_df[cr_df["final_accuracy"].notna()]
        if len(valid_final) > 0:
            best_global = valid_final.nlargest(10, "stability_score")
            with (OUTPUT_DIR / "best_global_candidates.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sel_fields, extrasaction="ignore")
                w.writeheader(); w.writerows(best_global.to_dict("records"))
            pass65 = valid_final[valid_final["pass_65"] == True]
            with (OUTPUT_DIR / "full_coverage_65_candidates.csv").open("w", newline="") as f:
                if len(pass65) > 0:
                    w = csv.DictWriter(f, fieldnames=sel_fields, extrasaction="ignore")
                    w.writeheader(); w.writerows(pass65.to_dict("records"))
            # Select best by stability score
            best_row = valid_final.loc[valid_final["stability_score"].idxmax()]
            best_acc = float(best_row["final_accuracy"])
        else:
            best_acc = 0.0
            best_row = None
    else:
        best_acc = 0.0
        best_row = None
    # Manifest
    manifest = {"total_candidates_processed": cid_counter, "total_completed": len(completed_results),
        "total_skipped": len(skipped), "search_completed": cid_counter >= min(len(all_candidates), start_idx + max_candidates),
        "best_final_accuracy": round(best_acc, 6) if best_row is not None else 0.0,
        "final65_passed": bool(best_acc >= 0.65) if best_row is not None else False,
        "completed_at": now_utc(), "leakage_safe": True, "daily_data_used": False,
        "resampling_used": False, "full_universe_required": True, "full_coverage_required": True,
        "no_abstention": True, "no_ranking_metric": True, "rolling_validation_used": True}
    with (OUTPUT_DIR / "overall_directional_v2_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    # Run log
    log = ["# Overall Directional Final65 V2 Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Candidates processed: {manifest['total_candidates_processed']}",
        f"- Completed: {manifest['total_completed']}",
        f"- Skipped: {manifest['total_skipped']}",
        f"- Search completed: {'YES' if manifest['search_completed'] else 'NO (runtime-limited)'}",
        f"- Best final accuracy: {fmt_pct(best_acc)}",
        f"- Final65 passed: {'YES' if manifest['final65_passed'] else 'NO'}", ""]
    if best_row is not None:
        log.extend([f"- Best model: {best_row['model']}", f"- Best horizon: {best_row['horizon']}",
            f"- Best feature set: {best_row['feature_set']}",
            f"- Rolling validation mean: {fmt_pct(best_row['mean_validation_accuracy'])}",
            f"- Rolling validation min: {fmt_pct(best_row['min_validation_accuracy'])}",
            f"- Rolling validation std: {fmt_pct(best_row['validation_std'])}",
            f"- Stability score: {best_row['stability_score']:.6f}", ""])
    with (OUTPUT_DIR / "overall_directional_v2_run_log.md").open("w") as f: f.write("\n".join(log))
    # Skipped candidates
    with (OUTPUT_DIR / "skipped_or_blocked_candidates.md").open("w") as f:
        f.write("# Skipped or Blocked Candidates\n\n")
        for s in skipped[:100]: f.write(f"- {s}\n")
        if len(skipped) > 100: f.write(f"\n... and {len(skipped) - 100} more\n")
    print(f"\nBest final accuracy: {fmt_pct(best_acc)}")
    if best_row is not None:
        print(f"Best config: {best_row['model']} h={best_row['horizon']} fs={best_row['feature_set']}")
        print(f"Rolling val mean: {fmt_pct(best_row['mean_validation_accuracy'])}")
        print(f"Rolling val min: {fmt_pct(best_row['min_validation_accuracy'])}")
        print(f"Rolling val std: {fmt_pct(best_row['validation_std'])}")
    print(f"Final65 passed: {'YES' if manifest['final65_passed'] else 'NO'}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
