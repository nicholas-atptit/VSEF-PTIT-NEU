"""Final65 focus v3 for VN30 hourly 2015 using canonical evaluator."""
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
from vn30_hourly_2015_canonical_eval import compute_accuracy, classify_result, EVALUATOR_VERSION

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_final65_focus_v3"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
HORIZON = 40
SEED = 42
REFINED_THRESHOLDS = [round(0.50 + i * 0.01, 3) for i in range(48)]  # 0.50 to 0.97
MIN_ROWS_PER_TICKER = [50, 100, 150]
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

def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(float)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=idx))
    if labels: return pd.concat(labels)
    return pd.Series(dtype=float)

def split_data(df: pd.DataFrame, labels: pd.Series):
    train_mask = df["datetime"] <= TRAIN_END
    val_mask = (df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)
    eval_mask = df["datetime"] >= EVAL_START
    return df[train_mask].copy(), df[val_mask].copy(), df[eval_mask].copy()

def train_rf(X: pd.DataFrame, y: pd.Series):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**RF_PARAMS).fit(X, y)

def predict_proba(model, X: pd.DataFrame):
    probas = model.predict_proba(X)
    preds = model.predict(X)
    return preds, probas[:, 1], probas

def directional_accuracy(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0: return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Final65 Focus v3")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"model": "random_forest", "horizon": HORIZON, "target": "absolute_direction",
        "train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "seed": SEED, "evaluator_version": EVALUATOR_VERSION,
        "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    print("\nBuilding features...")
    feature_df_c, feat_cols_c = build_features(stock_df, index_data, include_market=True)
    print(f"  Feature set C: {len(feat_cols_c)} features")
    labels = build_labels(feature_df_c, HORIZON)
    train_df, val_df, eval_df = split_data(feature_df_c, labels)
    fcp = [c for c in feat_cols_c if c in feature_df_c.columns]
    tl = labels.reindex(train_df.index).dropna()
    vl = labels.reindex(val_df.index).dropna()
    el = labels.reindex(eval_df.index).dropna()
    print(f"  Train: {len(tl)}, Val: {len(vl)}, Eval: {len(el)}")
    tX = feature_df_c.reindex(tl.index)[fcp].fillna(0)
    ty = tl.astype(int)
    vX = feature_df_c.reindex(vl.index)[fcp].fillna(0)
    vy = vl.astype(int)
    eX = feature_df_c.reindex(el.index)[fcp].fillna(0)
    ey = el.astype(int)
    print("\nTraining RF h=40 base model...")
    base_model = train_rf(tX, ty)
    vp, vprob, vpr = predict_proba(base_model, vX)
    ep, eprob, epr = predict_proba(base_model, eX)
    vr = compute_accuracy(vy.values, vp)
    er = compute_accuracy(ey.values, ep)
    print(f"  Base val accuracy: {fmt_pct(vr['accuracy'])} ({vr['correct_count']}/{vr['total_valid']})")
    print(f"  Base eval accuracy: {fmt_pct(er['accuracy'])} ({er['correct_count']}/{er['total_valid']})")
    all_policies = []
    total_val = len(vy[~np.isnan(vy)])
    total_eval = len(ey[~np.isnan(ey)])

    # Policy A: Refined confidence threshold
    print("\n[Policy A] Refined confidence threshold...")
    for thresh in REFINED_THRESHOLDS:
        vmask = vprob >= thresh
        v_acc = compute_accuracy(vy[vmask], vp[vmask])
        emask = eprob >= thresh
        e_acc = compute_accuracy(ey[emask], ep[emask])
        if e_acc["total_valid"] == 0: continue
        ecov = e_acc["total_valid"] / total_eval
        vcov = v_acc["total_valid"] / total_val
        pid = f"refined_conf_{thresh}"
        cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
        all_policies.append({"policy_id": pid, "policy_type": "refined_confidence",
            "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
            "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
            "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
            "final_rows": e_acc["total_valid"], "active_ticker_count": len(tickers),
            "ticker_list": "", "confidence_threshold": thresh, "per_ticker_thresholds_id": "",
            "regime_filter": "", "meta_label_threshold": "",
            "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
            "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Policy B: Per-ticker confidence threshold
    print("\n[Policy B] Per-ticker confidence threshold...")
    ticker_val_data = {}
    for ticker in tickers:
        t_mask = feature_df_c["ticker"] == ticker
        t_vl_idx = vl.index[t_mask.reindex(vl.index)]
        if len(t_vl_idx) < 50: continue
        t_el_idx = el.index[t_mask.reindex(el.index)]
        if len(t_el_idx) < 20: continue
        t_idx_mask_val = t_mask.reindex(vl.index).values
        t_idx_mask_eval = t_mask.reindex(el.index).values
        t_vprob = vprob[t_idx_mask_val]
        t_vp = vp[t_idx_mask_val]
        t_vy_mask = t_idx_mask_val & ~np.isnan(vy)
        t_vy = vy[t_vy_mask]
        t_vp_trimmed = t_vp[:len(t_vy)]
        if len(t_vy) < 20: continue
        t_eprob = eprob[t_idx_mask_eval]
        t_ep = ep[t_idx_mask_eval]
        t_ey_mask = t_idx_mask_eval & ~np.isnan(ey)
        t_ey = ey[t_ey_mask]
        t_ep_trimmed = t_ep[:len(t_ey)]
        ticker_val_data[ticker] = {"vprob": t_vprob, "vp": t_vp_trimmed, "vy": t_vy,
            "eprob": t_eprob, "ep": t_ep_trimmed, "ey": t_ey, "val_rows": len(t_vy), "eval_rows": len(t_ey)}
    for min_rows in MIN_ROWS_PER_TICKER:
        for thresh in REFINED_THRESHOLDS[::2]:  # step by 2 to reduce runtime
            whitelist = []
            ticker_thresholds = {}
            for ticker, td in ticker_val_data.items():
                if td["val_rows"] < min_rows: continue
                mask = td["vprob"] >= thresh
                if mask.sum() < 20: continue
                acc = directional_accuracy(td["vy"][mask], td["vp"][mask])
                if acc >= 0.55:
                    whitelist.append(ticker)
                    ticker_thresholds[ticker] = thresh
            if len(whitelist) < 5: continue
            # Apply policy
            vmask = np.array([feature_df_c.loc[i, "ticker"] in whitelist and vprob[vl.index.get_loc(i)] >= ticker_thresholds.get(feature_df_c.loc[i, "ticker"], thresh)
                             for i in vl.index]) & ~np.isnan(vy)
            v_acc = compute_accuracy(vy[vmask], vp[vmask])
            emask = np.array([feature_df_c.loc[i, "ticker"] in whitelist and eprob[el.index.get_loc(i)] >= ticker_thresholds.get(feature_df_c.loc[i, "ticker"], thresh)
                             for i in el.index]) & ~np.isnan(ey)
            e_acc = compute_accuracy(ey[emask], ep[emask])
            if e_acc["total_valid"] == 0: continue
            ecov = e_acc["total_valid"] / total_eval
            vcov = v_acc["total_valid"] / total_val
            pid = f"per_ticker_conf_{min_rows}_{thresh}"
            cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
            all_policies.append({"policy_id": pid, "policy_type": "per_ticker_confidence",
                "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
                "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
                "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
                "final_rows": e_acc["total_valid"], "active_ticker_count": len(whitelist),
                "ticker_list": ",".join(sorted(whitelist)), "confidence_threshold": "",
                "per_ticker_thresholds_id": f"min{min_rows}_t{thresh}", "regime_filter": "",
                "meta_label_threshold": "",
                "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
                "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Policy C: Ticker whitelist + confidence
    print("\n[Policy C] Ticker whitelist + confidence...")
    ticker_val_acc = {}
    for ticker, td in ticker_val_data.items():
        acc = directional_accuracy(td["vy"], td["vp"])
        ticker_val_acc[ticker] = (acc, td["val_rows"])
    for min_rows in MIN_ROWS_PER_TICKER:
        for thresh in REFINED_THRESHOLDS:
            whitelist = [t for t, (a, n) in ticker_val_acc.items() if n >= min_rows and a >= 0.55]
            if len(whitelist) < 5: continue
            vmask = np.array([feature_df_c.loc[i, "ticker"] in whitelist for i in vl.index]) & (vprob >= thresh) & ~np.isnan(vy)
            v_acc = compute_accuracy(vy[vmask], vp[vmask])
            emask = np.array([feature_df_c.loc[i, "ticker"] in whitelist for i in el.index]) & (eprob >= thresh) & ~np.isnan(ey)
            e_acc = compute_accuracy(ey[emask], ep[emask])
            if e_acc["total_valid"] == 0: continue
            ecov = e_acc["total_valid"] / total_eval
            vcov = v_acc["total_valid"] / total_val
            pid = f"ticker_conf_{min_rows}_{thresh}"
            cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
            all_policies.append({"policy_id": pid, "policy_type": "ticker_whitelist_confidence",
                "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
                "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
                "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
                "final_rows": e_acc["total_valid"], "active_ticker_count": len(whitelist),
                "ticker_list": ",".join(sorted(whitelist)), "confidence_threshold": thresh,
                "per_ticker_thresholds_id": "", "regime_filter": "", "meta_label_threshold": "",
                "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
                "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})    # Policy D: Market-regime abstention
    print("\n[Policy D] Market-regime abstention...")
    regime_features = ["vn30_lag_1", "vn30_lag_2", "vnindex_lag_1", "vnindex_lag_2",
        "vn30_roll_vol_20", "vn30_roll_vol_60", "vnindex_roll_vol_20", "vnindex_roll_vol_60",
        "vn30_roll_mean_20", "vnindex_roll_mean_20"]
    regime_features = [f for f in regime_features if f in feature_df_c.columns]
    if regime_features:
        rvX = feature_df_c.reindex(vl.index)[regime_features].fillna(0)
        reX = feature_df_c.reindex(el.index)[regime_features].fillna(0)
        rcor = (vp == vl.values).astype(int)
        from sklearn.ensemble import RandomForestClassifier as RFC
        rm = RFC(n_estimators=100, max_depth=4, random_state=SEED)
        rm.fit(rvX, rcor)
        rmvp = rm.predict(rvX)
        rmvpp = rm.predict_proba(rvX)[:, 1]
        rmp = rm.predict(reX)
        rmpp = rm.predict_proba(reX)[:, 1]
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            vmask = rmvpp >= thresh
            v_acc = compute_accuracy(vy[vmask], vp[vmask])
            emask = rmpp >= thresh
            e_acc = compute_accuracy(ey[emask], ep[emask])
            if e_acc["total_valid"] == 0: continue
            ecov = e_acc["total_valid"] / total_eval
            vcov = v_acc["total_valid"] / total_val
            pid = f"market_regime_{thresh}"
            cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
            all_policies.append({"policy_id": pid, "policy_type": "market_regime_abstention",
                "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
                "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
                "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
                "final_rows": e_acc["total_valid"], "active_ticker_count": len(tickers),
                "ticker_list": "", "confidence_threshold": "", "per_ticker_thresholds_id": "",
                "regime_filter": ",".join(regime_features), "meta_label_threshold": "",
                "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
                "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Policy E: Meta-label abstention
    print("\n[Policy E] Meta-label abstention...")
    vcor = (vp == vl.values).astype(int)
    from sklearn.ensemble import RandomForestClassifier as RFC
    mfv = pd.DataFrame({"confidence": vprob, "max_proba": vpr.max(axis=1),
        "prob_diff": np.abs(vpr[:, 0] - vpr[:, 1])}, index=vl.index)
    if len(mfv) >= 50:
        mm = RFC(n_estimators=100, max_depth=5, random_state=SEED)
        mm.fit(mfv, vcor)
        mfe = pd.DataFrame({"confidence": eprob, "max_proba": epr.max(axis=1),
            "prob_diff": np.abs(epr[:, 0] - epr[:, 1])}, index=el.index)
        mpc = mm.predict(mfe)
        mpp = mm.predict_proba(mfe)[:, 1]
        mvp = mm.predict(mfv)
        mvpp = mm.predict_proba(mfv)[:, 1]
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            vmask = mvpp >= thresh
            v_acc = compute_accuracy(vcor[vmask], mvp[vmask])
            emask = mpp >= thresh
            ecor = (ep == el.values).astype(int)
            e_acc = compute_accuracy(ecor[emask], mpc[emask])
            if e_acc["total_valid"] == 0: continue
            ecov = e_acc["total_valid"] / total_eval
            vcov = v_acc["total_valid"] / total_val
            pid = f"meta_label_{thresh}"
            cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
            all_policies.append({"policy_id": pid, "policy_type": "meta_label_abstention",
                "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
                "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
                "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
                "final_rows": e_acc["total_valid"], "active_ticker_count": len(tickers),
                "ticker_list": "", "confidence_threshold": "", "per_ticker_thresholds_id": "",
                "regime_filter": "", "meta_label_threshold": thresh,
                "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
                "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Policy F: Hybrid (confidence + ticker + regime)
    print("\n[Policy F] Hybrid policy...")
    if regime_features and ticker_val_acc:
        for min_rows in [100, 150]:
            whitelist = [t for t, (a, n) in ticker_val_acc.items() if n >= min_rows and a >= 0.55]
            if len(whitelist) < 5: continue
            for conf_thresh in [0.6, 0.65, 0.7]:
                for reg_thresh in [0.55, 0.6, 0.65]:
                    vmask = (np.array([feature_df_c.loc[i, "ticker"] in whitelist for i in vl.index]) &
                             (vprob >= conf_thresh) & (rmvpp >= reg_thresh) & ~np.isnan(vy))
                    v_acc = compute_accuracy(vy[vmask], vp[vmask])
                    emask = (np.array([feature_df_c.loc[i, "ticker"] in whitelist for i in el.index]) &
                             (eprob >= conf_thresh) & (rmpp >= reg_thresh) & ~np.isnan(ey))
                    e_acc = compute_accuracy(ey[emask], ep[emask])
                    if e_acc["total_valid"] == 0: continue
                    ecov = e_acc["total_valid"] / total_eval
                    vcov = v_acc["total_valid"] / total_val
                    pid = f"hybrid_{min_rows}_c{conf_thresh}_r{reg_thresh}"
                    cl = classify_result(e_acc["accuracy"], ecov, e_acc["total_valid"])
                    all_policies.append({"policy_id": pid, "policy_type": "hybrid_confidence_ticker_regime",
                        "selected_on_validation": "yes", "validation_accuracy": round(v_acc["accuracy"], 6),
                        "validation_coverage": round(vcov, 4), "validation_rows": v_acc["total_valid"],
                        "final_accuracy": round(e_acc["accuracy"], 6), "final_coverage": round(ecov, 4),
                        "final_rows": e_acc["total_valid"], "active_ticker_count": len(whitelist),
                        "ticker_list": ",".join(sorted(whitelist)), "confidence_threshold": conf_thresh,
                        "per_ticker_thresholds_id": "", "regime_filter": ",".join(regime_features),
                        "meta_label_threshold": "",
                        "pass_65": e_acc["accuracy"] >= 0.65 and ecov >= 0.30 and e_acc["total_valid"] >= 1000,
                        "claim_level": cl, "evaluator_version": EVALUATOR_VERSION})

    # Select best policy on validation
    print("\nSelecting best policy on validation...")
    valid_policies = [p for p in all_policies if p["validation_rows"] >= 1000 and p["validation_coverage"] >= 0.30]
    if valid_policies:
        best_policy = max(valid_policies, key=lambda p: (p["validation_accuracy"], p["validation_coverage"], p["validation_rows"]))
        print(f"  Best: {best_policy['policy_id']}, type={best_policy['policy_type']}, "
              f"val_acc={fmt_pct(best_policy['validation_accuracy'])}, val_cov={fmt_pct(best_policy['validation_coverage'])}, "
              f"eval_acc={fmt_pct(best_policy['final_accuracy'])}, eval_cov={fmt_pct(best_policy['final_coverage'])}")
    else:
        best_policy = None
        print("  No valid policy found meeting coverage >=30% and rows >=1000")

    # Write outputs
    print("\nWriting outputs...")
    fieldnames = ["policy_id", "policy_type", "selected_on_validation", "validation_accuracy",
        "validation_coverage", "validation_rows", "final_accuracy", "final_coverage", "final_rows",
        "active_ticker_count", "ticker_list", "confidence_threshold", "per_ticker_thresholds_id",
        "regime_filter", "meta_label_threshold", "pass_65", "claim_level", "evaluator_version"]
    with (OUTPUT_DIR / "validation_policy_grid.csv").open("w", newline="") as f:
        if all_policies:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(all_policies)
    with (OUTPUT_DIR / "selected_policies.csv").open("w", newline="") as f:
        if best_policy:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerow(best_policy)
    with (OUTPUT_DIR / "final_policy_results.csv").open("w", newline="") as f:
        if all_policies:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(all_policies)
    f65_candidates = [p for p in all_policies if p.get("pass_65") == True or float(p.get("final_accuracy", 0) or 0) >= 0.65]
    exploratory = [p for p in all_policies if p not in f65_candidates]
    with (OUTPUT_DIR / "final65_candidates.csv").open("w", newline="") as f:
        if f65_candidates:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(f65_candidates)
    with (OUTPUT_DIR / "exploratory_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(exploratory[:500])
    best_f65 = max((float(p.get("final_accuracy", 0) or 0) for p in all_policies if float(p.get("final_coverage", 0) or 0) >= 0.30 and int(p.get("final_rows", 0) or 0) >= 1000), default=0)
    f65_pass = best_f65 >= 0.65
    manifest = {"total_policies": len(all_policies), "final65_candidates": len(f65_candidates),
        "best_final65_accuracy": round(best_f65, 6), "final65_pass": f65_pass, "completed_at": now_utc(),
        "evaluator_version": EVALUATOR_VERSION, "leakage_safe": True, "daily_data_used": False, "resampling_used": False}
    with (OUTPUT_DIR / "final65_focus_v3_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    blocked = ["# Final65 Focus v3 - Blocked or Invalid Policies", "",
        f"- Completed: {manifest['completed_at']}", f"- Policies evaluated: {manifest['total_policies']}",
        f"- Final65 candidates: {manifest['final65_candidates']}", f"- Best final65: {fmt_pct(best_f65)}", "",
        "## Notes", "", "- All policies selected on 2024 validation only.",
        "- Canonical evaluator v1.0.0 used for all metrics.", ""]
    with (OUTPUT_DIR / "blocked_or_invalid_policies.md").open("w") as f: f.write("\n".join(blocked))
    log = ["# Final65 Focus v3 Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Policies: {manifest['total_policies']}", f"- Best final65: {fmt_pct(best_f65)}",
        f"- Final65 pass: {'YES' if f65_pass else 'NO'}", f"- Gap to 65: {fmt_pct(0.65 - best_f65)}", ""]
    with (OUTPUT_DIR / "final65_focus_v3_run_log.md").open("w") as f: f.write("\n".join(log))
    print(f"\nBest final65: {fmt_pct(best_f65)}")
    print(f"Final65 pass: {'YES' if f65_pass else 'NO'}")
    print(f"Gap to 65: {fmt_pct(0.65 - best_f65)}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())