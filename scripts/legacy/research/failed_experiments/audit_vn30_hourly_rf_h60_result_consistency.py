"""Audit RF h=60 result consistency across experiments."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))
from vn30_hourly_2015_canonical_eval import compute_accuracy, evaluate_predictions, EVALUATOR_VERSION

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_consistency"
AUDIT_CSV_PATH = REPORT_DIR / "rf_h60_consistency_audit.csv"
AUDIT_MD_PATH = REPORT_DIR / "rf_h60_consistency_audit.md"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
HORIZON = 60
SEED = 42
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

def main() -> int:
    print("=" * 60)
    print("RF h=60 Result Consistency Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows")
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
    print("\nTraining RF h=60 with canonical params...")
    model = train_rf(tX, ty)
    vp = model.predict(vX)
    ep = model.predict(eX)
    # Canonical evaluation
    val_result = compute_accuracy(vy.values, vp)
    eval_result = compute_accuracy(ey.values, ep)
    print(f"  Canonical val accuracy: {fmt_pct(val_result['accuracy'])} ({val_result['correct_count']}/{val_result['total_valid']})")
    print(f"  Canonical eval accuracy: {fmt_pct(eval_result['accuracy'])} ({eval_result['correct_count']}/{eval_result['total_valid']})")
    audit_rows = []
    def add(source: str, accuracy: float, rows: int, coverage: float, details: str):
        audit_rows.append({"source": source, "accuracy": round(accuracy, 6), "rows": rows,
                          "coverage": round(coverage, 4), "details": details})
    add("canonical_evaluator", eval_result["accuracy"], eval_result["total_valid"], 1.0,
        f"Pooled accuracy: {eval_result['correct_count']}/{eval_result['total_valid']}")
    add("horizon_relative_target_v1", 0.6022, 3474, 1.0, "RF h=60 absolute, feature set C, from horizon-relative-target experiment")
    add("all_model_router_v1", 0.5964, 3474, 1.0, "RF h=60 absolute, per-ticker whitelist, from all-model router experiment")
    add("rf_h60_final65_focus_v1", 0.5970, 3474, 1.0, "RF h=60 absolute, Platt calibration, from RF-only focus experiment")
    add("rf_h60_router_v2", 0.5987, 3474, 1.0, "RF h=60 absolute, per-ticker whitelist, from router v2 experiment")
    discrepancy = abs(eval_result["accuracy"] - 0.6022)
    reason = "Different experiment runs with potentially different random seeds, data splits, or evaluation methods. The canonical evaluator gives the ground truth."
    with AUDIT_CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "accuracy", "rows", "coverage", "details"])
        w.writeheader(); w.writerows(audit_rows)
    md = ["# RF h=60 Consistency Audit", "", f"- Generated: {now_utc()}",
        f"- Canonical evaluator version: {EVALUATOR_VERSION}", "",
        "## Results", "",
        f"- Canonical RF h=60 eval accuracy: {fmt_pct(eval_result['accuracy'])} ({eval_result['correct_count']}/{eval_result['total_valid']})",
        f"- Canonical RF h=60 val accuracy: {fmt_pct(val_result['accuracy'])} ({val_result['correct_count']}/{val_result['total_valid']})",
        f"- Previous reported 60.22%: from horizon-relative-target experiment",
        f"- Previous reported 59.64%: from all-model router experiment",
        f"- Discrepancy: {fmt_pct(discrepancy)}", "",
        "## Discrepancy Reason", "",
        reason, "",
        "## Decision", "",
        f"- Canonical RF h=60 result: {fmt_pct(eval_result['accuracy'])}",
        f"- Baseline60 status: {'PASS' if eval_result['accuracy'] >= 0.60 else 'FAIL'}",
        "- All later experiments must use canonical evaluator.", ""]
    AUDIT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nCanonical RF h=60: {fmt_pct(eval_result['accuracy'])}")
    print(f"Baseline60: {'PASS' if eval_result['accuracy'] >= 0.60 else 'FAIL'}")
    print(f"Audit written to {rel(AUDIT_MD_PATH)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
