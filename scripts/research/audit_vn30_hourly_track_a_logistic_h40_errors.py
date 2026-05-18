"""Audit Track A Logistic Regression h40 baseline errors by ticker/time/regime."""

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
from scripts.research.vn30_hourly_track_a_regime_feature_v2 import build_regime_feature_v2  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")

OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_regime_feature_v2"
CANONICAL_RESULTS = REPO_ROOT / "reports" / "generated" / "vn30_hourly_canonical_expanded_model_comparison" / "model_comparison_summary.csv"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
HORIZON = 40
BASELINE_LOGISTIC_H40 = 0.6043200785468826


def majority_accuracy(train_y: pd.Series, y: pd.Series) -> float:
    majority = int(float(train_y.mean()) >= 0.5)
    return float((y.astype(int).to_numpy() == majority).mean())


def summarize(frame: pd.DataFrame, group_cols: list[str], train_y: pd.Series) -> list[dict[str, Any]]:
    majority = int(float(train_y.mean()) >= 0.5)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        accuracy = float(group["is_correct"].mean())
        baseline = float((group["target_direction"].astype(int) == majority).mean())
        row.update(
            {
                "slice_type": "+".join(group_cols),
                "rows": int(len(group)),
                "accuracy": accuracy,
                "majority_baseline_accuracy": baseline,
                "delta_vs_majority_baseline": accuracy - baseline,
                "target_up_rate": float(group["target_direction"].mean()),
                "prediction_up_rate": float(group["prediction"].mean()),
                "correct": int(group["is_correct"].sum()),
            }
        )
        rows.append(row)
    return rows


def canonical_context() -> list[dict[str, Any]]:
    if not CANONICAL_RESULTS.exists():
        return []
    table = pd.read_csv(CANONICAL_RESULTS)
    h40 = table[(table["horizon"].astype(int) == HORIZON) & (table["model"].astype(str) != "logistic_regression")]
    context_rows = []
    for _, row in h40.iterrows():
        context_rows.append(
            {
                "slice_type": "complex_model_context",
                "model": row["model"],
                "horizon": int(row["horizon"]),
                "feature_set": row["feature_set"],
                "validation_accuracy": float(row["validation_accuracy"]),
                "final_accuracy": float(row["final_accuracy"]),
                "logistic_final_delta": BASELINE_LOGISTIC_H40 - float(row["final_accuracy"]),
                "where_logistic_beats_complex_models": bool(BASELINE_LOGISTIC_H40 > float(row["final_accuracy"])),
            }
        )
    return context_rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    features, feature_cols = build_feature_set_c(stock_df, index_data)
    regime_features, _regime_cols, _manifest = build_regime_feature_v2(stock_df, index_data)
    labels = add_absolute_labels(features, HORIZON)
    train_idx = labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index
    val_idx = labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index
    final_idx = labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index
    train_y = labels.reindex(train_idx).astype(int)
    val_y = labels.reindex(val_idx).astype(int)
    final_y = labels.reindex(final_idx).astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )
    model.fit(features.reindex(train_y.index)[feature_cols], train_y)
    val_pred = model.predict(features.reindex(val_y.index)[feature_cols]).astype(int)
    final_pred = model.predict(features.reindex(final_y.index)[feature_cols]).astype(int)
    final = features.reindex(final_y.index)[["datetime", "ticker"]].copy()
    final["target_direction"] = final_y.astype(int)
    final["prediction"] = final_pred
    final["is_correct"] = (final["target_direction"] == final["prediction"]).astype(int)
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    regime_slice = regime_features.reindex(final_y.index)[["market_regime_v2"]].copy()
    if "vnindex_vol_regime_v2" in regime_features.columns:
        regime_slice["vnindex_vol_regime_v2"] = regime_features.reindex(final_y.index)["vnindex_vol_regime_v2"]
    else:
        regime_slice["vnindex_vol_regime_v2"] = np.nan
    final = final.join(regime_slice)
    final["market_regime_v2"] = final["market_regime_v2"].fillna("unknown")
    final["vnindex_vol_regime_v2"] = final["vnindex_vol_regime_v2"].fillna(-1).astype(int)
    validation_accuracy = float((val_y.to_numpy() == val_pred).mean())
    final_accuracy = float(final["is_correct"].mean())
    slice_rows: list[dict[str, Any]] = []
    for group_cols in [["ticker"], ["month"], ["quarter"], ["market_regime_v2"], ["vnindex_vol_regime_v2"], ["ticker", "market_regime_v2"]]:
        slice_rows.extend(summarize(final, group_cols, train_y))
    context_rows = canonical_context()
    worst_ticker = min([row for row in slice_rows if row["slice_type"] == "ticker"], key=lambda row: float(row["accuracy"]))
    worst_month = min([row for row in slice_rows if row["slice_type"] == "month"], key=lambda row: float(row["accuracy"]))
    worst_quarter = min([row for row in slice_rows if row["slice_type"] == "quarter"], key=lambda row: float(row["accuracy"]))
    global_rows = [
        {
            "slice_type": "global",
            "model": "logistic_regression",
            "horizon": HORIZON,
            "feature_set": "feature_set_C_closest",
            "validation_accuracy": validation_accuracy,
            "final_accuracy": final_accuracy,
            "final_rows": int(len(final)),
            "active_ticker_count": int(final["ticker"].nunique()),
            "majority_baseline_accuracy": majority_accuracy(train_y, final_y),
            "delta_vs_majority_baseline": final_accuracy - majority_accuracy(train_y, final_y),
            "worst_ticker_drag": worst_ticker.get("ticker", ""),
            "worst_month_drag": worst_month.get("month", ""),
            "worst_quarter_drag": worst_quarter.get("quarter", ""),
            "where_logistic_fails": "ticker/month/regime slices with negative delta vs majority or low absolute accuracy",
            "where_logistic_beats_complex_models": "canonical h40 final accuracy exceeds RF/ExtraTrees/CART/XGBoost/LightGBM h40",
        }
    ]
    output_rows = global_rows + slice_rows + context_rows
    write_csv(OUT_DIR / "logistic_h40_error_audit.csv", output_rows)
    worst_slices = sorted(slice_rows, key=lambda row: float(row["accuracy"]))[:15]
    report = [
        "# Track A Logistic h40 Error Audit",
        "",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Final rows: {len(final)}.",
        f"- Active ticker count: {final['ticker'].nunique()}.",
        f"- Majority baseline: {pct(majority_accuracy(train_y, final_y))}.",
        f"- Worst ticker drag: `{worst_ticker.get('ticker', '')}` at {pct(worst_ticker.get('accuracy', math.nan))}.",
        f"- Worst month drag: `{worst_month.get('month', '')}` at {pct(worst_month.get('accuracy', math.nan))}.",
        f"- Worst quarter drag: `{worst_quarter.get('quarter', '')}` at {pct(worst_quarter.get('accuracy', math.nan))}.",
        "",
        "## Where Logistic Beats Complex Models",
        "",
        markdown_table(["model", "horizon", "final_accuracy", "logistic_final_delta"], context_rows),
        "",
        "## Worst Slices",
        "",
        markdown_table(["slice_type", "ticker", "month", "quarter", "market_regime_v2", "vnindex_vol_regime_v2", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], worst_slices),
        "",
        "## Where Logistic Fails",
        "",
        "Logistic h40 failures are concentrated in the lowest-accuracy ticker, month, quarter, and regime slices listed above, especially slices where the model delta vs the majority baseline is negative.",
        "",
    ]
    (OUT_DIR / "logistic_h40_error_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"logistic_h40_error_audit_status=completed final_accuracy={final_accuracy:.6f} output_dir={OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
