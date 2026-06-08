"""Audit the diagnostic Track A L2 Logistic h40 final65 row."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    LOCKED_RF_H60,
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    load_index_data,
    load_stock_data,
    markdown_table,
    pct,
    write_csv,
)
from scripts.research.vn30_hourly_track_a_regime_feature_v2 import FEATURE_SET_NAME, build_regime_feature_v2  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_diagnostic_65_row"
V2_RESULTS = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_regime_feature_v2" / "final_candidate_results.csv"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
HORIZON = 40
BASELINE_LOGISTIC_H40 = 0.6043200785468826
SELECTED_XGB_V2 = 0.5657830142366225


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    return {
        "train": labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index,
        "validation": labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index,
        "final": labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index,
    }


def accuracy(y_true: pd.Series, pred: np.ndarray | pd.Series) -> float:
    return float((y_true.astype(int).to_numpy() == np.asarray(pred).astype(int)).mean()) if len(y_true) else math.nan


def majority_prediction(train_y: pd.Series) -> int:
    return int(float(train_y.mean()) >= 0.5)


def summarize_group(frame: pd.DataFrame, group_cols: list[str], majority: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        acc = float(group["is_correct"].mean())
        majority_acc = float((group["target_direction"].astype(int) == majority).mean())
        row.update(
            {
                "slice_type": "+".join(group_cols),
                "rows": int(len(group)),
                "accuracy": acc,
                "majority_baseline_accuracy": majority_acc,
                "delta_vs_majority_baseline": acc - majority_acc,
                "target_up_rate": float(group["target_direction"].mean()),
                "prediction_up_rate": float(group["prediction"].mean()),
                "correct": int(group["is_correct"].sum()),
            }
        )
        rows.append(row)
    return rows


def classify_breadth(by_ticker: list[dict[str, Any]], month_rows: list[dict[str, Any]], quarter_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    ticker_count = len(by_ticker)
    tickers_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in by_ticker)
    tickers_above_65 = sum(float(row["accuracy"]) >= 0.65 for row in by_ticker)
    months_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in month_rows)
    quarters_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in quarter_rows)
    broad = tickers_above_60 >= max(15, math.ceil(ticker_count * 0.60)) and months_above_60 >= max(1, math.ceil(len(month_rows) * 0.50)) and quarters_above_60 >= max(1, math.ceil(len(quarter_rows) * 0.50))
    details = {
        "ticker_count": ticker_count,
        "tickers_above_60": tickers_above_60,
        "tickers_above_65": tickers_above_65,
        "month_count": len(month_rows),
        "months_above_60": months_above_60,
        "quarter_count": len(quarter_rows),
        "quarters_above_60": quarters_above_60,
    }
    return ("broad_based" if broad else "concentrated_or_mixed"), details


def v2_table_row() -> dict[str, Any]:
    if not V2_RESULTS.exists():
        return {}
    table = pd.read_csv(V2_RESULTS)
    row = table[
        (table["model"].astype(str) == "l2_logistic")
        & (table["horizon"].astype(int) == HORIZON)
        & (table["regime_method"].astype(str) == "global_v2")
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    features, feature_cols, feature_manifest = build_regime_feature_v2(load_stock_data(tickers), load_index_data())
    labels = add_absolute_labels(features, HORIZON)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(features.reindex(idx["train"])[feature_cols], train_y)
    val_pred = model.predict(features.reindex(idx["validation"])[feature_cols]).astype(int)
    final_pred = model.predict(features.reindex(idx["final"])[feature_cols]).astype(int)
    majority = majority_prediction(train_y)
    validation_accuracy = accuracy(val_y, val_pred)
    validation_baseline = accuracy(val_y, np.full(len(val_y), majority))
    final_accuracy = accuracy(final_y, final_pred)
    final_baseline = accuracy(final_y, np.full(len(final_y), majority))
    final = features.reindex(idx["final"])[["datetime", "ticker"]].copy()
    final["target_direction"] = final_y.astype(int)
    final["prediction"] = final_pred
    final["is_correct"] = (final["target_direction"] == final["prediction"]).astype(int)
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    by_ticker = summarize_group(final, ["ticker"], majority)
    by_month = summarize_group(final, ["month"], majority)
    by_quarter = summarize_group(final, ["quarter"], majority)
    for row in by_month:
        row["time_grain"] = "month"
    for row in by_quarter:
        row["time_grain"] = "quarter"
    broad_note, breadth_details = classify_breadth(by_ticker, by_month, by_quarter)
    worst_ticker = min(by_ticker, key=lambda row: float(row["accuracy"])) if by_ticker else {}
    worst_month = min(by_month, key=lambda row: float(row["accuracy"])) if by_month else {}
    worst_quarter = min(by_quarter, key=lambda row: float(row["accuracy"])) if by_quarter else {}
    v2_row = v2_table_row()
    global_rows = [
        {
            "model": "l2_logistic",
            "horizon": HORIZON,
            "feature_set": FEATURE_SET_NAME,
            "regime_method": "global_v2",
            "validation_accuracy": validation_accuracy,
            "validation_baseline_accuracy": validation_baseline,
            "validation_delta_vs_baseline": validation_accuracy - validation_baseline,
            "final_accuracy": final_accuracy,
            "final_baseline_accuracy": final_baseline,
            "final_delta_vs_baseline": final_accuracy - final_baseline,
            "final_rows": int(len(final_y)),
            "active_ticker_count": int(final["ticker"].nunique()),
            "delta_vs_logistic_h40_60_43": final_accuracy - BASELINE_LOGISTIC_H40,
            "delta_vs_rf_h60_60_31": final_accuracy - LOCKED_RF_H60,
            "delta_vs_selected_xgboost_v2_56_58": final_accuracy - SELECTED_XGB_V2,
            "target_up_rate": float(final["target_direction"].mean()),
            "prediction_up_rate": float(final["prediction"].mean()),
            "broad_based_or_concentrated": broad_note,
            "breadth_details": breadth_details,
            "worst_ticker": worst_ticker.get("ticker", ""),
            "worst_month": worst_month.get("month", ""),
            "worst_quarter": worst_quarter.get("quarter", ""),
            "v2_table_validation_accuracy": v2_row.get("validation_accuracy", ""),
            "v2_table_final_accuracy": v2_row.get("final_accuracy", ""),
            "selected_on_validation": False,
            "can_be_claimed": False,
            "feature_leakage_audit": feature_manifest,
        }
    ]
    write_csv(OUT_DIR / "diagnostic_65_global.csv", global_rows)
    write_csv(OUT_DIR / "diagnostic_65_by_ticker.csv", by_ticker)
    write_csv(OUT_DIR / "diagnostic_65_by_time.csv", by_month + by_quarter)
    worst_tickers = sorted(by_ticker, key=lambda row: float(row["accuracy"]))[:10]
    best_tickers = sorted(by_ticker, key=lambda row: float(row["accuracy"]), reverse=True)[:10]
    time_sorted = sorted(by_month + by_quarter, key=lambda row: float(row["accuracy"]))
    report = [
        "# Track A Diagnostic 65 Row Audit",
        "",
        "- Candidate: L2 Logistic h40 `regime_feature_v2` `global_v2`.",
        "- Status: diagnostic only; not selected on validation; not claimable.",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Validation baseline: {pct(validation_baseline)}.",
        f"- Validation delta: {pct(validation_accuracy - validation_baseline)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Final baseline: {pct(final_baseline)}.",
        f"- Final delta vs baseline: {pct(final_accuracy - final_baseline)}.",
        f"- Final rows: {len(final_y)}.",
        f"- Active ticker count: {final['ticker'].nunique()}.",
        f"- Delta vs Logistic h40 60.43: {pct(final_accuracy - BASELINE_LOGISTIC_H40)}.",
        f"- Delta vs selected XGBoost v2 56.58: {pct(final_accuracy - SELECTED_XGB_V2)}.",
        f"- Broad-based or concentrated: `{broad_note}`.",
        f"- Breadth details: `{breadth_details}`.",
        f"- Worst ticker: `{worst_ticker.get('ticker', '')}` at {pct(worst_ticker.get('accuracy', math.nan))}.",
        f"- Worst month: `{worst_month.get('month', '')}` at {pct(worst_month.get('accuracy', math.nan))}.",
        "",
        "## Worst Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], worst_tickers),
        "",
        "## Best Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], best_tickers),
        "",
        "## Worst Time Slices",
        "",
        markdown_table(["time_grain", "month", "quarter", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], time_sorted[:12]),
        "",
    ]
    (OUT_DIR / "diagnostic_65_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"diagnostic_65_audit_status=completed final_accuracy={final_accuracy:.6f} broad_based={broad_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
