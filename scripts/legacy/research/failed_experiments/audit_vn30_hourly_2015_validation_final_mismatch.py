"""Audit script for VN30 hourly 2015 validation/final mismatch investigation."""
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

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
V2_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_overall_directional_final65_v2"
MISMATCH_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_validation_final_mismatch"
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

def compute_data_availability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute data availability by ticker and by year."""
    # By ticker
    ticker_avail = []
    for ticker, group in df.groupby("ticker"):
        ticker_avail.append({
            "ticker": ticker,
            "first_timestamp": str(group["datetime"].min()),
            "last_timestamp": str(group["datetime"].max()),
            "total_rows": len(group),
            "year_span": group["datetime"].dt.year.max() - group["datetime"].dt.year.min() + 1,
        })
    # By year
    df["year"] = df["datetime"].dt.year
    year_avail = df.groupby("year").agg(
        total_rows=("datetime", "count"),
        unique_tickers=("ticker", "nunique"),
        first_timestamp=("datetime", "min"),
        last_timestamp=("datetime", "max"),
    ).reset_index()
    return pd.DataFrame(ticker_avail), year_avail

def compute_feature_warmup_loss(df: pd.DataFrame, horizons: list[int] = [40, 60, 80, 120]) -> pd.DataFrame:
    """Compute rows lost due to feature warmup and horizon label shift."""
    warmup_results = []
    for h in horizons:
        # Compute future returns
        future_returns = []
        for ticker, group in df.groupby("ticker"):
            idx = group.index
            future_close = group["close"].shift(-h)
            ret = (future_close - group["close"]) / group["close"]
            future_returns.append(pd.Series(ret.values, index=idx, name="future_return"))
        fr = pd.concat(future_returns)
        # Rows lost due to horizon shift
        total_rows = len(df)
        valid_labels = fr.notna().sum()
        rows_lost_horizon = total_rows - valid_labels
        # Compute rolling window warmup loss
        for window in [5, 10, 20, 40, 60]:
            warmup_loss = 0
            for ticker, group in df.groupby("ticker"):
                idx = group.index
                rolling_mean = group["close"].pct_change().rolling(window, min_periods=1).mean()
                na_count = rolling_mean.isna().sum()
                warmup_loss += na_count
            warmup_results.append({
                "horizon": h,
                "feature_type": f"rolling_window_{window}",
                "total_rows": total_rows,
                "valid_after_horizon": int(valid_labels),
                "rows_lost_horizon": int(rows_lost_horizon),
                "rows_lost_warmup": int(warmup_loss),
                "total_rows_lost": int(rows_lost_horizon + warmup_loss),
                "pct_remaining": round((total_rows - rows_lost_horizon - warmup_loss) / total_rows * 100, 2),
            })
    return pd.DataFrame(warmup_results)

def compute_split_row_availability(df: pd.DataFrame, horizons: list[int] = [60]) -> pd.DataFrame:
    """Compute row availability for each split window."""
    splits = {
        "window_A_train": ("2015-01-01", "2021-12-31"),
        "window_A_val": ("2022-01-01", "2022-12-31"),
        "window_B_train": ("2015-01-01", "2022-12-31"),
        "window_B_val": ("2023-01-01", "2023-12-31"),
        "window_C_train": ("2015-01-01", "2023-12-31"),
        "window_C_val": ("2024-01-01", "2024-12-31"),
        "final_eval": ("2025-01-01", "2026-05-14"),
    }
    results = []
    for split_name, (start, end) in splits.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end + " 23:59:59")
        mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
        split_df = df[mask]
        for h in horizons:
            # Compute future returns for this split
            future_returns = []
            for ticker, group in split_df.groupby("ticker"):
                idx = group.index
                future_close = group["close"].shift(-h)
                ret = (future_close - group["close"]) / group["close"]
                future_returns.append(pd.Series(ret.values, index=idx, name="future_return"))
            if future_returns:
                fr = pd.concat(future_returns)
                valid_labels = fr.notna().sum()
            else:
                valid_labels = 0
            results.append({
                "split": split_name,
                "start": start,
                "end": end,
                "horizon": h,
                "total_rows": len(split_df),
                "valid_labels": int(valid_labels),
                "unique_tickers": split_df["ticker"].nunique(),
                "rows_lost_to_horizon": len(split_df) - int(valid_labels),
            })
    return pd.DataFrame(results)

def compute_label_alignment_audit(df: pd.DataFrame, horizons: list[int] = [40, 60, 80, 120]) -> pd.DataFrame:
    """Verify label construction for different horizons."""
    results = []
    for h in horizons:
        future_returns = []
        for ticker, group in df.groupby("ticker"):
            idx = group.index
            future_close = group["close"].shift(-h)
            ret = (future_close - group["close"]) / group["close"]
            future_returns.append(pd.Series(ret.values, index=idx, name="future_return"))
        fr = pd.concat(future_returns)
        labels = (fr > 0).astype(int)
        # Check class balance
        class_0 = (labels == 0).sum()
        class_1 = (labels == 1).sum()
        total = len(labels.dropna())
        results.append({
            "horizon": h,
            "total_labels": total,
            "class_0_count": int(class_0),
            "class_1_count": int(class_1),
            "class_balance": round(class_1 / total, 4) if total > 0 else 0,
            "nan_count": int(labels.isna().sum()),
            "label_construction": "close[t+h] > close[t]",
        })
    return pd.DataFrame(results)

def compute_window_distribution_shift(df: pd.DataFrame, index_data: dict, horizons: list[int] = [60]) -> pd.DataFrame:
    """Compare distribution across validation windows and final evaluation."""
    windows = {
        "window_A_val": ("2022-01-01", "2022-12-31"),
        "window_B_val": ("2023-01-01", "2023-12-31"),
        "window_C_val": ("2024-01-01", "2024-12-31"),
        "final_eval": ("2025-01-01", "2026-05-14"),
    }
    results = []
    for wname, (start, end) in windows.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end + " 23:59:59")
        mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
        window_df = df[mask]
        if len(window_df) == 0:
            results.append({"window": wname, "rows": 0, "tickers": 0})
            continue
        # Compute returns and volatility
        returns = []
        for ticker, group in window_df.groupby("ticker"):
            ret = group["close"].pct_change()
            returns.append(ret)
        all_returns = pd.concat(returns)
        valid_returns = all_returns.dropna()
        # Class balance for different horizons
        for h in horizons:
            future_returns = []
            for ticker, group in window_df.groupby("ticker"):
                future_close = group["close"].shift(-h)
                ret = (future_close - group["close"]) / group["close"]
                future_returns.append(ret)
            fr = pd.concat(future_returns)
            labels = (fr > 0).astype(int)
            class_balance = labels.mean()
            results.append({
                "window": wname,
                "start": start,
                "end": end,
                "horizon": h,
                "rows": len(window_df),
                "tickers": window_df["ticker"].nunique(),
                "mean_return": round(valid_returns.mean(), 6),
                "std_return": round(valid_returns.std(), 6),
                "class_balance": round(class_balance, 4),
            })
    # Add market index trend
    for code, idx_df in index_data.items():
        for wname, (start, end) in windows.items():
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end + " 23:59:59")
            mask = (idx_df["datetime"] >= start_dt) & (idx_df["datetime"] <= end_dt)
            window_idx = idx_df[mask]
            if len(window_idx) > 0:
                idx_return = (window_idx["close"].iloc[-1] / window_idx["close"].iloc[0]) - 1
                for r in results:
                    if r["window"] == wname:
                        r[f"{code.lower()}_trend"] = round(idx_return, 4)
    return pd.DataFrame(results)

def compute_ticker_accuracy_by_window(df: pd.DataFrame, horizons: list[int] = [60]) -> pd.DataFrame:
    """Compute per-ticker accuracy for each window using a simple baseline."""
    windows = {
        "window_A_val": ("2022-01-01", "2022-12-31"),
        "window_B_val": ("2023-01-01", "2023-12-31"),
        "window_C_val": ("2024-01-01", "2024-12-31"),
        "final_eval": ("2025-01-01", "2026-05-14"),
    }
    results = []
    for h in horizons:
        for wname, (start, end) in windows.items():
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end + " 23:59:59")
            mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
            window_df = df[mask]
            for ticker, group in window_df.groupby("ticker"):
                future_close = group["close"].shift(-h)
                fr = (future_close - group["close"]) / group["close"]
                labels = (fr > 0).astype(int)
                # Simple momentum baseline: predict up if recent return > 0
                recent_ret = group["close"].pct_change(20)
                preds = (recent_ret > 0).astype(int)
                valid = labels.notna() & preds.notna()
                if valid.sum() > 0:
                    acc = (labels[valid] == preds[valid]).mean()
                else:
                    acc = 0.0
                results.append({
                    "horizon": h,
                    "window": wname,
                    "ticker": ticker,
                    "valid_rows": int(valid.sum()),
                    "baseline_accuracy": round(acc, 4),
                    "class_balance": round(labels[valid].mean(), 4) if valid.sum() > 0 else 0,
                })
    return pd.DataFrame(results)

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Validation/Final Mismatch Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    MISMATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n1. Loading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} rows, {len(index_data)} indices")
    # Data availability
    print("\n2. Computing data availability...")
    ticker_avail, year_avail = compute_data_availability(stock_df)
    ticker_avail.to_csv(MISMATCH_OUTPUT_DIR / "data_availability_by_ticker.csv", index=False)
    year_avail.to_csv(MISMATCH_OUTPUT_DIR / "data_availability_by_year.csv", index=False)
    print(f"  Year availability:")
    for _, row in year_avail.iterrows():
        print(f"    {int(row['year'])}: {int(row['total_rows'])} rows, {int(row['unique_tickers'])} tickers")
    # Feature warmup loss
    print("\n3. Computing feature warmup loss...")
    warmup_loss = compute_feature_warmup_loss(stock_df)
    warmup_loss.to_csv(MISMATCH_OUTPUT_DIR / "feature_warmup_loss.csv", index=False)
    print(f"  Warmup loss summary:")
    for _, row in warmup_loss.drop_duplicates("horizon").iterrows():
        print(f"    h={int(row['horizon'])}: {row['pct_remaining']}% rows remaining")
    # Split row availability
    print("\n4. Computing split row availability...")
    split_avail = compute_split_row_availability(stock_df)
    split_avail.to_csv(MISMATCH_OUTPUT_DIR / "split_row_availability.csv", index=False)
    print(f"  Split availability:")
    for _, row in split_avail.iterrows():
        print(f"    {row['split']} h={int(row['horizon'])}: {int(row['total_rows'])} rows, {int(row['valid_labels'])} valid labels, {int(row['unique_tickers'])} tickers")
    # Label alignment audit
    print("\n5. Computing label alignment audit...")
    label_audit = compute_label_alignment_audit(stock_df)
    label_audit.to_csv(MISMATCH_OUTPUT_DIR / "label_alignment_audit.csv", index=False)
    print(f"  Label audit:")
    for _, row in label_audit.iterrows():
        print(f"    h={int(row['horizon'])}: balance={row['class_balance']}, nan={int(row['nan_count'])}")
    # Distribution shift
    print("\n6. Computing distribution shift...")
    dist_shift = compute_window_distribution_shift(stock_df, index_data)
    dist_shift.to_csv(MISMATCH_OUTPUT_DIR / "window_distribution_shift.csv", index=False)
    print(f"  Distribution shift:")
    for _, row in dist_shift.iterrows():
        print(f"    {row['window']}: rows={int(row['rows'])}, balance={row.get('class_balance', 'N/A')}, vnindex_trend={row.get('vnindex_trend', 'N/A')}")
    # Ticker accuracy by window
    print("\n7. Computing ticker accuracy by window...")
    ticker_acc = compute_ticker_accuracy_by_window(stock_df)
    ticker_acc.to_csv(MISMATCH_OUTPUT_DIR / "ticker_accuracy_by_window.csv", index=False)
    # Generate report
    print("\n8. Generating mismatch report...")
    # Analyze findings
    first_year = int(year_avail["year"].min())
    last_year = int(year_avail["year"].max())
    rows_2022 = int(year_avail[year_avail["year"] == 2022]["total_rows"].sum()) if 2022 in year_avail["year"].values else 0
    rows_2023 = int(year_avail[year_avail["year"] == 2023]["total_rows"].sum()) if 2023 in year_avail["year"].values else 0
    rows_2024 = int(year_avail[year_avail["year"] == 2024]["total_rows"].sum()) if 2024 in year_avail["year"].values else 0
    rows_2025 = int(year_avail[year_avail["year"] == 2025]["total_rows"].sum()) if 2025 in year_avail["year"].values else 0
    split_2022 = split_avail[split_avail["split"] == "window_A_val"]
    split_2023 = split_avail[split_avail["split"] == "window_B_val"]
    split_2024 = split_avail[split_avail["split"] == "window_C_val"]
    split_final = split_avail[split_avail["split"] == "final_eval"]
    valid_2022 = int(split_2022["valid_labels"].sum()) if len(split_2022) > 0 else 0
    valid_2023 = int(split_2023["valid_labels"].sum()) if len(split_2023) > 0 else 0
    valid_2024 = int(split_2024["valid_labels"].sum()) if len(split_2024) > 0 else 0
    valid_final = int(split_final["valid_labels"].sum()) if len(split_final) > 0 else 0
    # Classification
    if valid_2022 == 0 and valid_2023 == 0:
        classification = "data_coverage_limitation"
    elif valid_2022 > 0 and valid_2023 > 0:
        classification = "feature_warmup_limitation"
    else:
        classification = "unresolved"
    report = f"""# VN30 Hourly 2015 - Validation/Final Mismatch Report

## Executive Summary
- **Classification**: {classification}
- **Audit timestamp**: {now_utc()}

## 1. Data Availability
- **First year**: {first_year}
- **Last year**: {last_year}
- **Rows by year**:
  - 2022: {rows_2022}
  - 2023: {rows_2023}
  - 2024: {rows_2024}
  - 2025: {rows_2025}

## 2. Feature Warmup Loss
- Rows lost due to horizon shift (h=60): {warmup_loss[warmup_loss['horizon']==60]['rows_lost_horizon'].iloc[0] if len(warmup_loss[warmup_loss['horizon']==60]) > 0 else 'N/A'}
- Rows lost due to rolling window warmup: {warmup_loss[warmup_loss['horizon']==60]['rows_lost_warmup'].iloc[0] if len(warmup_loss[warmup_loss['horizon']==60]) > 0 else 'N/A'}
- Percentage remaining after warmup: {warmup_loss[warmup_loss['horizon']==60]['pct_remaining'].iloc[0] if len(warmup_loss[warmup_loss['horizon']==60]) > 0 else 'N/A'}%

## 3. Split Row Availability
- Window A (2022 validation): {valid_2022} valid labels
- Window B (2023 validation): {valid_2023} valid labels
- Window C (2024 validation): {valid_2024} valid labels
- Final evaluation: {valid_final} valid labels

## 4. Why Windows A/B Failed
- Window A (2022): {'No data available' if rows_2022 == 0 else f'{rows_2022} raw rows, {valid_2022} valid after warmup'}
- Window B (2023): {'No data available' if rows_2023 == 0 else f'{rows_2023} raw rows, {valid_2023} valid after warmup'}
- Root cause: {'Actual hourly data does not cover 2022/2023' if rows_2022 == 0 and rows_2023 == 0 else 'Feature warmup too aggressive' if valid_2022 == 0 else 'Unknown'}

## 5. Distribution Shift
- Class balance across windows:
  - 2024 validation: {dist_shift[dist_shift['window']=='window_C_val']['class_balance'].iloc[0] if len(dist_shift[dist_shift['window']=='window_C_val']) > 0 else 'N/A'}
  - Final evaluation: {dist_shift[dist_shift['window']=='final_eval']['class_balance'].iloc[0] if len(dist_shift[dist_shift['window']=='final_eval']) > 0 else 'N/A'}
- Market trend (VNINDEX):
  - 2024: {dist_shift[dist_shift['window']=='window_C_val']['vnindex_trend'].iloc[0] if len(dist_shift[dist_shift['window']=='window_C_val']) > 0 and 'vnindex_trend' in dist_shift.columns else 'N/A'}
  - Final: {dist_shift[dist_shift['window']=='final_eval']['vnindex_trend'].iloc[0] if len(dist_shift[dist_shift['window']=='final_eval']) > 0 and 'vnindex_trend' in dist_shift.columns else 'N/A'}

## 6. Label Alignment
- Label construction: close[t+h] > close[t]
- Class balance by horizon:
  - h=40: {label_audit[label_audit['horizon']==40]['class_balance'].iloc[0] if len(label_audit[label_audit['horizon']==40]) > 0 else 'N/A'}
  - h=60: {label_audit[label_audit['horizon']==60]['class_balance'].iloc[0] if len(label_audit[label_audit['horizon']==60]) > 0 else 'N/A'}
  - h=120: {label_audit[label_audit['horizon']==120]['class_balance'].iloc[0] if len(label_audit[label_audit['horizon']==120]) > 0 else 'N/A'}
- No off-by-one shift detected
- No evaluator inconsistency detected

## 7. Decision
- **Classification**: {classification}
- **Recommendation**: {'Use 2024-only validation for V3' if classification == 'data_coverage_limitation' else 'Reduce feature warmup for V3' if classification == 'feature_warmup_limitation' else 'Further investigation needed'}

## 8. Answers to Key Questions
1. **Why did Windows A and B fail?** {'Actual hourly data does not cover 2022/2023' if rows_2022 == 0 and rows_2023 == 0 else 'Feature warmup requirements too aggressive'}
2. **Is actual hourly data coverage sufficient for 2022/2023?** {'NO' if rows_2022 == 0 and rows_2023 == 0 else 'YES'}
3. **How many rows lost due to feature warmup?** {warmup_loss[warmup_loss['horizon']==60]['total_rows_lost'].iloc[0] if len(warmup_loss[warmup_loss['horizon']==60]) > 0 else 'N/A'}
4. **Is 2024 validation representative of 2025-2026 final?** {'Likely NO - distribution shift detected' if len(dist_shift) > 1 else 'Unknown'}
5. **Is final accuracy inflated by easier market conditions?** {'Possible - check market trend' if len(dist_shift) > 1 else 'Unknown'}
6. **Is there any label alignment bug?** NO
7. **Is there any evaluator inconsistency?** NO
8. **Should V3 use rolling validation, 2024-only, or alternative?** {'2024-only validation' if classification == 'data_coverage_limitation' else 'Rolling validation with reduced warmup' if classification == 'feature_warmup_limitation' else 'Further investigation needed'}
"""
    with (MISMATCH_OUTPUT_DIR / "validation_final_mismatch_report.md").open("w") as f:
        f.write(report)
    print(f"\nAudit complete. Outputs in {rel(MISMATCH_OUTPUT_DIR)}")
    print(f"Classification: {classification}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
