"""Audit accuracy drag for the current best daily target60 result."""
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

CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015_target60_v2"
EVAL_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
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

def build_features_cross(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extended + cross-sectional rank features."""
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        op = group["open"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 7, 10, 15, 20, 30, 60): prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20): prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
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
            gain = delta.clip(lower=0.0).rolling(period, min_periods=period//2).mean()
            loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=period//2).mean()
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
        for lag in (1, 2, 3, 5): prepared.loc[idx, f"volume_change_lag_{lag}"] = prepared.loc[idx, "volume_change_1"].shift(lag)
        vol_ma20 = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma20 - 1.0
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
    # Cross-sectional rank features
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        prepared.loc[idx, "cs_return_rank"] = prepared.loc[idx, "return_1"].rank(pct=True)
        prepared.loc[idx, "cs_volume_rank"] = prepared.loc[idx, "volume_shock_20"].rank(pct=True)
        prepared.loc[idx, "cs_momentum_rank"] = prepared.loc[idx, "momentum_20"].rank(pct=True)
    cs_cols = ["cs_return_rank", "cs_volume_rank", "cs_momentum_rank"]
    return prepared, feat_cols + cs_cols

def compute_future_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    returns = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        ret = (future_close - group["close"]) / group["close"]
        returns.append(pd.Series(ret.values, index=idx, name="future_return"))
    if returns: return pd.concat(returns)
    return pd.Series(dtype=float, name="future_return")

def main() -> int:
    print("=" * 60)
    print("VN30 Daily 2015 - Accuracy Drag Analysis")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    print(f"  {len(tickers)} tickers, {len(stock_df)} rows")

    print("\nBuilding features (daily_cross)...")
    fdf, fcols = build_features_cross(stock_df)
    print(f"  {len(fcols)} features")

    horizon = 40
    print(f"\nComputing future returns h={horizon}...")
    future_returns = compute_future_returns(fdf, horizon)
    labels = (future_returns > 0).astype(int)
    fcp = [c for c in fcols if c in fdf.columns]

    # Train on all pre-2025 data
    all_idx = future_returns.index[future_returns.notna()]
    t_idx = all_idx[(fdf.loc[all_idx, "datetime"] <= TRAIN_END) & future_returns.reindex(all_idx).notna()]
    e_idx = all_idx[(fdf.loc[all_idx, "datetime"] >= EVAL_START) & future_returns.reindex(all_idx).notna()]

    print(f"  Train: {len(t_idx)}, Eval: {len(e_idx)}")

    tX = fdf.reindex(t_idx)[fcp].fillna(0)
    ty = labels.reindex(t_idx)
    eX = fdf.reindex(e_idx)[fcp].fillna(0)
    ey = labels.reindex(e_idx)

    print("\nTraining LightGBM...")
    import lightgbm as lgb
    params = {"num_leaves": 20, "max_depth": 3, "learning_rate": 0.02, "n_estimators": 700,
        "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.6,
        "random_state": SEED, "verbose": -1}
    m = lgb.LGBMClassifier(**params).fit(tX, ty)
    ep = m.predict(eX)
    eprob = m.predict_proba(eX)[:, 1]

    # Build evaluation dataframe
    eval_df = pd.DataFrame({
        "datetime": fdf.reindex(e_idx)["datetime"].values,
        "ticker": fdf.reindex(e_idx)["ticker"].values,
        "y_true": ey.values,
        "y_pred": ep.astype(int),
        "y_prob": eprob,
        "is_correct": (ey.values == ep.astype(int)).astype(int),
    })
    eval_df["month"] = eval_df["datetime"].dt.month
    eval_df["quarter"] = eval_df["datetime"].dt.quarter
    eval_df["year"] = eval_df["datetime"].dt.year

    # Accuracy by ticker
    ticker_acc = eval_df.groupby("ticker", sort=True).agg(
        n_rows=("is_correct", "count"),
        n_correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        class_balance=("y_true", "mean"),
    ).reset_index()
    ticker_acc["class_balance"] = ticker_acc["class_balance"].round(4)
    ticker_acc["accuracy"] = ticker_acc["accuracy"].round(4)
    ticker_acc = ticker_acc.sort_values("accuracy")

    # Accuracy by time
    time_acc = eval_df.groupby(["year", "month"], sort=True).agg(
        n_rows=("is_correct", "count"),
        n_correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        class_balance=("y_true", "mean"),
    ).reset_index()
    time_acc["accuracy"] = time_acc["accuracy"].round(4)

    # Confusion matrix
    cm = pd.crosstab(eval_df["y_true"], eval_df["y_pred"], rownames=["actual"], colnames=["predicted"])

    # Worst/best tickers
    worst = ticker_acc.head(5)
    best = ticker_acc.tail(5)

    # Overall accuracy
    overall_acc = eval_df["is_correct"].mean()
    overall_class_balance = eval_df["y_true"].mean()

    # Write outputs
    ticker_acc.to_csv(OUTPUT_DIR / "daily_accuracy_drag_by_ticker.csv", index=False)
    time_acc.to_csv(OUTPUT_DIR / "daily_accuracy_drag_by_time.csv", index=False)

    # Report
    lines = [
        "# Daily Accuracy Drag Report", "",
        f"- Created at UTC: `{now_utc()}`.",
        f"- Model: LightGBM daily_cross h=40.",
        f"- Overall accuracy: {fmt_pct(overall_acc)}.",
        f"- Overall class balance (positive rate): {fmt_pct(overall_class_balance)}.",
        f"- Final rows: {len(eval_df)}.", "",
        "## Accuracy by Ticker (sorted ascending)", "",
        "| ticker | n_rows | n_correct | accuracy | class_balance |",
        "|---|---|---|---|---|",
    ]
    for _, r in ticker_acc.iterrows():
        lines.append(f"| {r['ticker']} | {int(r['n_rows'])} | {int(r['n_correct'])} | {fmt_pct(r['accuracy'])} | {r['class_balance']:.4f} |")

    lines.extend(["", "## Worst 5 Tickers (dragging accuracy)", ""])
    for _, r in worst.iterrows():
        lines.append(f"- **{r['ticker']}**: {fmt_pct(r['accuracy'])} ({int(r['n_rows'])} rows, class_balance={r['class_balance']:.4f})")

    lines.extend(["", "## Best 5 Tickers (supporting accuracy)", ""])
    for _, r in best.iterrows():
        lines.append(f"- **{r['ticker']}**: {fmt_pct(r['accuracy'])} ({int(r['n_rows'])} rows, class_balance={r['class_balance']:.4f})")

    lines.extend(["", "## Accuracy by Time", "",
        "| year | month | n_rows | n_correct | accuracy | class_balance |",
        "|---|---|---|---|---|---|"])
    for _, r in time_acc.iterrows():
        lines.append(f"| {int(r['year'])} | {int(r['month'])} | {int(r['n_rows'])} | {int(r['n_correct'])} | {fmt_pct(r['accuracy'])} | {r['class_balance']:.4f} |")

    lines.extend(["", "## Confusion Matrix", "",
        f"| | Predicted 0 | Predicted 1 |",
        "|---|---|---|",
        f"| Actual 0 | {int(cm.get(0, {}).get(0, 0))} | {int(cm.get(0, {}).get(1, 0))} |",
        f"| Actual 1 | {int(cm.get(1, {}).get(0, 0))} | {int(cm.get(1, {}).get(1, 0))} |"])

    # Class balance by horizon comparison (use previous results)
    lines.extend(["", "## Observations", ""])
    acc_std = ticker_acc["accuracy"].std()
    lines.append(f"- Accuracy std across tickers: {acc_std:.4f}")
    lines.append(f"- Worst ticker accuracy: {fmt_pct(worst.iloc[0]['accuracy'])} ({worst.iloc[0]['ticker']})")
    lines.append(f"- Best ticker accuracy: {fmt_pct(best.iloc[-1]['accuracy'])} ({best.iloc[-1]['ticker']})")
    lines.append(f"- Accuracy range: {fmt_pct(best.iloc[-1]['accuracy'] - worst.iloc[0]['accuracy'])}")

    # Check if errors cluster by ticker
    low_acc_tickers = ticker_acc[ticker_acc["accuracy"] < 0.50]
    lines.append(f"- Tickers below 50%: {len(low_acc_tickers)} ({', '.join(low_acc_tickers['ticker'].tolist()) if len(low_acc_tickers) > 0 else 'none'})")

    # Check time clustering
    low_time = time_acc[time_acc["accuracy"] < 0.50]
    if not low_time.empty:
        lines.append(f"- Periods below 50%: {len(low_time)}")
        for _, r in low_time.iterrows():
            lines.append(f"  - {int(r['year'])}-{int(r['month']):02d}: {fmt_pct(r['accuracy'])}")

    lines.extend(["", "## Conclusion", ""])
    lines.append(f"- Overall accuracy {fmt_pct(overall_acc)} is {fmt_pct(0.60 - overall_acc)} below the 60% target.")
    lines.append(f"- Accuracy varies across tickers (std={acc_std:.4f}).")
    if len(low_acc_tickers) > 0:
        lines.append(f"- {len(low_acc_tickers)} tickers are below 50% accuracy, contributing to the drag.")
    lines.append("- No hourly data used. Daily-only analysis.")

    (OUTPUT_DIR / "daily_accuracy_drag_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nOverall accuracy: {fmt_pct(overall_acc)}")
    print(f"Worst ticker: {worst.iloc[0]['ticker']} ({fmt_pct(worst.iloc[0]['accuracy'])})")
    print(f"Best ticker: {best.iloc[-1]['ticker']} ({fmt_pct(best.iloc[-1]['accuracy'])})")
    print(f"\nOutputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
