"""Audit stability of the Track A Logistic Regression h40 baseline60 result."""

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
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    markdown_table,
    pct,
    write_csv,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_baseline60_stability"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
HORIZON = 40
PREVIOUS_TRACK_A_LOGISTIC_H40 = 0.6043200785468826


def prepare_predictions() -> tuple[pd.DataFrame, float, float]:
    tickers = active_stock_tickers()
    features, feature_cols = build_feature_set_c(load_stock_data(tickers), load_index_data())
    labels = add_absolute_labels(features, HORIZON)
    train_idx = features.index[features["datetime"].le(TRAIN_END)]
    val_idx = features.index[features["datetime"].between(VAL_START, VAL_END)]
    final_idx = features.index[features["datetime"].ge(EVAL_START)]
    train_y = labels.reindex(train_idx).dropna().astype(int)
    val_y = labels.reindex(val_idx).dropna().astype(int)
    final_y = labels.reindex(final_idx).dropna().astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )
    model.fit(features.reindex(train_y.index)[feature_cols], train_y)
    val_pred = model.predict(features.reindex(val_y.index)[feature_cols]).astype(int)
    final_pred = model.predict(features.reindex(final_y.index)[feature_cols]).astype(int)
    validation_accuracy = float((val_y.to_numpy() == val_pred).mean())
    final = features.reindex(final_y.index)[["datetime", "ticker"]].copy()
    final["target_direction"] = final_y.astype(int)
    final["prediction"] = final_pred
    final["is_correct"] = (final["target_direction"] == final["prediction"]).astype(int)
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    majority = int(train_y.mean() >= 0.5)
    final["majority_prediction"] = majority
    final["majority_correct"] = (final["target_direction"] == majority).astype(int)
    final_accuracy = float(final["is_correct"].mean())
    return final, validation_accuracy, final_accuracy


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        accuracy = float(group["is_correct"].mean())
        baseline = float(group["majority_correct"].mean())
        up_rate = float(group["target_direction"].mean())
        row.update(
            {
                "rows": int(len(group)),
                "accuracy": accuracy,
                "majority_baseline_accuracy": baseline,
                "delta_vs_majority_baseline": accuracy - baseline,
                "target_up_rate": up_rate,
                "prediction_up_rate": float(group["prediction"].mean()),
                "correct": int(group["is_correct"].sum()),
            }
        )
        rows.append(row)
    return rows


def concentration_note(by_ticker: list[dict[str, Any]], by_time: list[dict[str, Any]]) -> tuple[str, str, str]:
    ticker_acc = [float(row["accuracy"]) for row in by_ticker if int(row["rows"]) > 0]
    month_rows = [row for row in by_time if row.get("time_grain") == "month"]
    month_acc = [float(row["accuracy"]) for row in month_rows if int(row["rows"]) > 0]
    ticker_above_60 = sum(acc >= 0.60 for acc in ticker_acc)
    months_above_60 = sum(acc >= 0.60 for acc in month_acc)
    broad = ticker_above_60 >= 15 and months_above_60 >= max(1, len(month_acc) // 2)
    note = "broad_based" if broad else "concentrated_or_mixed"
    worst_ticker = min(by_ticker, key=lambda row: float(row["accuracy"])) if by_ticker else {}
    worst_month = min(month_rows, key=lambda row: float(row["accuracy"])) if month_rows else {}
    return note, str(worst_ticker.get("ticker", "")), str(worst_month.get("month", ""))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final, validation_accuracy, final_accuracy = prepare_predictions()
    final_majority = float(final["majority_correct"].mean())
    by_ticker = summarize_group(final, ["ticker"])
    month_rows = summarize_group(final, ["month"])
    for row in month_rows:
        row["time_grain"] = "month"
    quarter_rows = summarize_group(final, ["quarter"])
    for row in quarter_rows:
        row["time_grain"] = "quarter"
    by_time = month_rows + quarter_rows
    broad_note, worst_ticker, worst_month = concentration_note(by_ticker, by_time)
    global_rows = [
        {
            "model": "logistic_regression",
            "horizon": HORIZON,
            "feature_set": "feature_set_C_closest",
            "validation_accuracy": validation_accuracy,
            "final_accuracy": final_accuracy,
            "final_rows": int(len(final)),
            "final_coverage": 1.0,
            "final_majority_baseline_accuracy": final_majority,
            "delta_vs_majority_baseline": final_accuracy - final_majority,
            "delta_vs_rf_h60_60_31": final_accuracy - LOCKED_RF_H60,
            "previous_track_a_logistic_h40": PREVIOUS_TRACK_A_LOGISTIC_H40,
            "target_up_rate": float(final["target_direction"].mean()),
            "prediction_up_rate": float(final["prediction"].mean()),
            "broad_based_or_concentrated": broad_note,
            "worst_ticker": worst_ticker,
            "worst_month": worst_month,
            "claim_level": "exploratory_baseline60",
        }
    ]
    write_csv(OUT_DIR / "track_a_baseline60_global.csv", global_rows)
    write_csv(OUT_DIR / "track_a_baseline60_by_ticker.csv", by_ticker)
    write_csv(OUT_DIR / "track_a_baseline60_by_time.csv", by_time)
    ticker_sorted = sorted(by_ticker, key=lambda row: float(row["accuracy"]))
    time_sorted = sorted(by_time, key=lambda row: float(row["accuracy"]))
    report = [
        "# Track A Baseline60 Stability Audit",
        "",
        f"- Model: Logistic Regression h={HORIZON}.",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Final majority baseline: {pct(final_majority)}.",
        f"- Delta vs majority baseline: {pct(final_accuracy - final_majority)}.",
        f"- Delta vs RF h60 60.31: {pct(final_accuracy - LOCKED_RF_H60)}.",
        f"- Broad-based or concentrated: `{broad_note}`.",
        f"- Worst ticker drag: `{worst_ticker}`.",
        f"- Worst month drag: `{worst_month}`.",
        "",
        "## Worst Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "target_up_rate", "prediction_up_rate"], ticker_sorted[:10]),
        "",
        "## Worst Time Slices",
        "",
        markdown_table(["time_grain", "month", "quarter", "rows", "accuracy", "target_up_rate"], time_sorted[:12]),
        "",
    ]
    (OUT_DIR / "track_a_baseline60_stability_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"track_a_baseline60_stability_status=completed final_accuracy={final_accuracy:.6f} broad_based={broad_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
