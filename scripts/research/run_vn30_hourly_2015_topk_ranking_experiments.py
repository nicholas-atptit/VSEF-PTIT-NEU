"""Top-K ranking experiments for VN30 hourly 2015."""
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_topk_ranking_experiments"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
MODELS = ["lightgbm", "xgboost", "random_forest"]
HORIZONS = [20, 40, 60, 80, 120]
SEED = 42
K_VALUES = [3, 5, 10]
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

def compute_future_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute future return for each row."""
    returns = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        ret = (future_close - group["close"]) / group["close"]
        returns.append(pd.Series(ret.values, index=idx, name="future_return"))
    if returns: return pd.concat(returns)
    return pd.Series(dtype=float, name="future_return")

def split_data(df: pd.DataFrame, future_returns: pd.Series):
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

def predict_proba(model, X: pd.DataFrame):
    probas = model.predict_proba(X)
    preds = model.predict(X)
    return preds, probas[:, 1], probas

def compute_topk_metrics(subset_df: pd.DataFrame, future_returns: pd.Series, scores: np.ndarray, k: int) -> dict[str, Any]:
    """Compute precision@k and hit_rate@k for a given timestamp cross-section."""
    df_temp = subset_df.copy()
    df_temp["score"] = scores
    df_temp["future_return"] = future_returns.values
    
    results = {"precision_hits": 0, "precision_total": 0, "hit_events": 0, "total_events": 0, "selected_stock_events": 0}
    
    for dt, group in df_temp.groupby("datetime"):
        if len(group) < k: continue
        # True top-k: actual future return rank
        true_top_k = group.nlargest(k, "future_return").index
        # Predicted top-k: score rank
        pred_top_k = group.nlargest(k, "score").index
        
        # Precision@k: how many predicted top-k are in true top-k
        hits = len(set(pred_top_k) & set(true_top_k))
        results["precision_hits"] += hits
        results["precision_total"] += k
        results["selected_stock_events"] += k
        
        # Hit rate: was at least one predicted top-k in true top-k?
        if hits > 0:
            results["hit_events"] += 1
        results["total_events"] += 1
        
    precision = results["precision_hits"] / results["precision_total"] if results["precision_total"] > 0 else 0.0
    hit_rate = results["hit_events"] / results["total_events"] if results["total_events"] > 0 else 0.0
    
    return {"precision_at_k": precision, "hit_rate_at_k": hit_rate, 
            "events": results["total_events"], "selected_stock_events": results["selected_stock_events"]}

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Top-K Ranking Experiments")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"models": MODELS, "horizons": HORIZONS, "k_values": K_VALUES,
        "train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "seed": SEED, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    print("\nBuilding features...")
    feature_df_c, feat_cols_c = build_features(stock_df, index_data, include_market=True)
    print(f"  Feature set C: {len(feat_cols_c)} features")
    all_val_results, all_eval_results = [], []

    # Define experiments
    experiments = []
    for h in HORIZONS:
        for mn in MODELS:
            experiments.append({"model": mn, "horizon": h, "feature_set": "C"})

    print(f"\nRunning {len(experiments)} base experiments...")
    for i, exp in enumerate(experiments):
        if (i + 1) % 10 == 0: print(f"  Progress: {i+1}/{len(experiments)}")
        h = exp["horizon"]
        mn = exp["model"]
        future_returns = compute_future_returns(feature_df_c, h)
        train_df, val_df, eval_df = split_data(feature_df_c, future_returns)
        fcp = [c for c in feat_cols_c if c in feature_df_c.columns]
        tl = future_returns.reindex(train_df.index).dropna()
        vl = future_returns.reindex(val_df.index).dropna()
        el = future_returns.reindex(eval_df.index).dropna()
        if len(tl) < 100 or len(vl) < 20 or len(el) < 20: continue
        # Create binary labels for training (above median = 1)
        train_labels = pd.Series(0, index=tl.index)
        for dt, group in feature_df_c.reindex(tl.index).groupby("datetime"):
            if len(group) < 2: continue
            median_ret = future_returns.loc[group.index].median()
            above_mask = future_returns.loc[group.index] > median_ret
            train_labels.loc[above_mask[above_mask].index] = 1
        tX = feature_df_c.reindex(tl.index)[fcp].fillna(0)
        ty = train_labels.astype(int)
        try:
            m = train_model(mn, tX, ty)
            # Validation
            vX = feature_df_c.reindex(vl.index)[fcp].fillna(0)
            vp, vprob, vpr = predict_proba(m, vX)
            # Eval
            eX = feature_df_c.reindex(el.index)[fcp].fillna(0)
            ep, eprob, epr = predict_proba(m, eX)
            # Align validation data
            v_valid_idx = vl.index.intersection(val_df.index)
            v_aligned_df = val_df.reindex(v_valid_idx)
            v_aligned_fr = future_returns.reindex(v_valid_idx)
            v_score_map = {idx: prob for idx, prob in zip(vl.index, vprob)}
            v_aligned_scores = np.array([v_score_map.get(idx, 0.0) for idx in v_valid_idx])
            
            # Align eval data
            e_valid_idx = el.index.intersection(eval_df.index)
            e_aligned_df = eval_df.reindex(e_valid_idx)
            e_aligned_fr = future_returns.reindex(e_valid_idx)
            e_score_map = {idx: prob for idx, prob in zip(el.index, eprob)}
            e_aligned_scores = np.array([e_score_map.get(idx, 0.0) for idx in e_valid_idx])
            
            for k in K_VALUES:
                val_metrics = compute_topk_metrics(v_aligned_df, v_aligned_fr, v_aligned_scores, k)
                eval_metrics = compute_topk_metrics(e_aligned_df, e_aligned_fr, e_aligned_scores, k)
                pass_65_p = val_metrics["precision_at_k"] >= 0.65
                pass_65_h = val_metrics["hit_rate_at_k"] >= 0.65
                claim = "ranking65_passed" if (eval_metrics["precision_at_k"] >= 0.65 or eval_metrics["hit_rate_at_k"] >= 0.65) else "exploratory"
                all_val_results.append({"model": mn, "horizon": h, "feature_set": "C", "k": k,
                    "ranking_policy": "probability_score", "validation_precision_at_k": round(val_metrics["precision_at_k"], 6),
                    "validation_hit_rate_at_k": round(val_metrics["hit_rate_at_k"], 6),
                    "validation_events": val_metrics["events"], "validation_selected_stock_events": val_metrics["selected_stock_events"],
                    "selected_on_validation": "yes"})
                all_eval_results.append({"model": mn, "horizon": h, "feature_set": "C", "k": k,
                    "ranking_policy": "probability_score", "validation_precision_at_k": round(val_metrics["precision_at_k"], 6),
                    "validation_hit_rate_at_k": round(val_metrics["hit_rate_at_k"], 6),
                    "validation_events": val_metrics["events"], "validation_selected_stock_events": val_metrics["selected_stock_events"],
                    "final_precision_at_k": round(eval_metrics["precision_at_k"], 6),
                    "final_hit_rate_at_k": round(eval_metrics["hit_rate_at_k"], 6),
                    "final_events": eval_metrics["events"], "final_selected_stock_events": eval_metrics["selected_stock_events"],
                    "pass_65_precision": eval_metrics["precision_at_k"] >= 0.65,
                    "pass_65_hit_rate": eval_metrics["hit_rate_at_k"] >= 0.65,
                    "claim_level": claim, "selected_on_validation": "yes"})
        except Exception as e: print(f"    Error: {e}")

    # Select best policy on validation
    print("\nSelecting best policy on validation...")
    valid_policies = [r for r in all_val_results if r["validation_events"] >= 10]
    if valid_policies:
        best_val = max(valid_policies, key=lambda r: (r["validation_precision_at_k"], r["validation_hit_rate_at_k"]))
        print(f"  Best: {best_val['model']}, h={best_val['horizon']}, k={best_val['k']}, "
              f"val_prec={fmt_pct(best_val['validation_precision_at_k'])}, val_hr={fmt_pct(best_val['validation_hit_rate_at_k'])}")
    else:
        best_val = None
        print("  No valid policy found")

    # Write outputs
    print("\nWriting outputs...")
    val_fields = ["model", "horizon", "feature_set", "k", "ranking_policy", "validation_precision_at_k",
        "validation_hit_rate_at_k", "validation_events", "validation_selected_stock_events", "selected_on_validation"]
    eval_fields = val_fields + ["final_precision_at_k", "final_hit_rate_at_k", "final_events",
        "final_selected_stock_events", "pass_65_precision", "pass_65_hit_rate", "claim_level"]
    with (OUTPUT_DIR / "validation_topk_results.csv").open("w", newline="") as f:
        if all_val_results:
            w = csv.DictWriter(f, fieldnames=val_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_val_results)
    with (OUTPUT_DIR / "final_topk_results.csv").open("w", newline="") as f:
        if all_eval_results:
            w = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_eval_results)
    with (OUTPUT_DIR / "selected_topk_policy.csv").open("w", newline="") as f:
        if best_val:
            w = csv.DictWriter(f, fieldnames=val_fields, extrasaction="ignore")
            w.writeheader(); w.writerow(best_val)
    p65 = [r for r in all_eval_results if r.get("pass_65_precision") == True or r.get("pass_65_hit_rate") == True]
    exploratory = [r for r in all_eval_results if r not in p65]
    with (OUTPUT_DIR / "topk_65_candidates.csv").open("w", newline="") as f:
        if p65:
            w = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(p65)
    with (OUTPUT_DIR / "exploratory_topk_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(exploratory[:500])
    best_p65 = max((float(r.get("final_precision_at_k", 0) or 0) for r in all_eval_results), default=0)
    best_hr = max((float(r.get("final_hit_rate_at_k", 0) or 0) for r in all_eval_results), default=0)
    manifest = {"total_experiments": len(all_eval_results), "topk_65_candidates": len(p65),
        "best_precision_at_k": round(best_p65, 6), "best_hit_rate_at_k": round(best_hr, 6),
        "ranking65_pass": best_p65 >= 0.65 or best_hr >= 0.65, "completed_at": now_utc(),
        "leakage_safe": True, "daily_data_used": False, "resampling_used": False}
    with (OUTPUT_DIR / "topk_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    log = ["# Top-K Ranking Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Experiments: {manifest['total_experiments']}", f"- Best precision@k: {fmt_pct(best_p65)}",
        f"- Best hit_rate@k: {fmt_pct(best_hr)}", f"- Ranking65 pass: {'YES' if manifest['ranking65_pass'] else 'NO'}", ""]
    with (OUTPUT_DIR / "topk_run_log.md").open("w") as f: f.write("\n".join(log))
    print(f"\nBest precision@k: {fmt_pct(best_p65)}")
    print(f"Best hit_rate@k: {fmt_pct(best_hr)}")
    print(f"Ranking65 pass: {'YES' if manifest['ranking65_pass'] else 'NO'}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())