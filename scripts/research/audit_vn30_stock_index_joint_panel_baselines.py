"""Audit deterministic baselines for the joint stock + index panel."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_stock_index_joint_panel_features import (
    BASELINE_STOCK_RF_H60_REFERENCE,
    FEATURE_SETS,
    HORIZONS,
    REPORT_DIR,
    add_time_splits,
    baseline_prediction,
    build_model_dataset,
    markdown_table,
    pct,
    split_metrics,
    write_csv,
)


BASELINES = ("majority_class", "always_up", "previous_direction", "moving_average_signal")


def summarize_baseline(horizon: int, method: str) -> dict[str, Any]:
    data, _features = build_model_dataset(horizon, "own_instrument_features")
    data = add_time_splits(data)
    data["prediction"] = baseline_prediction(data, method)
    metrics = split_metrics(data, "final")
    return {
        "baseline": method,
        "horizon": horizon,
        "combined_accuracy": metrics["final_combined_accuracy"],
        "stock_only_accuracy": metrics["final_stock_only_accuracy"],
        "index_only_accuracy": metrics["final_index_only_accuracy"],
        "combined_rows": metrics["final_rows_combined"],
        "stock_only_rows": metrics["final_rows_stock_only"],
        "index_only_rows": metrics["final_rows_index_only"],
        "stock_instrument_count": metrics["final_stock_instrument_count"],
        "index_instrument_count": metrics["final_index_instrument_count"],
        "total_instrument_count": metrics["final_total_instrument_count"],
        "reference_only": False,
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in BASELINES:
            rows.append(summarize_baseline(horizon, method))
    rows.append(
        {
            "baseline": "existing_stock_hourly_rf_h60_reference",
            "horizon": 60,
            "combined_accuracy": math.nan,
            "stock_only_accuracy": BASELINE_STOCK_RF_H60_REFERENCE,
            "index_only_accuracy": math.nan,
            "combined_rows": "",
            "stock_only_rows": "",
            "index_only_rows": "",
            "stock_instrument_count": 30,
            "index_instrument_count": 0,
            "total_instrument_count": 30,
            "reference_only": True,
        }
    )
    write_csv(REPORT_DIR / "joint_baseline_summary.csv", rows)
    comparable = [row for row in rows if not row["reference_only"]]
    best = max(comparable, key=lambda row: row["combined_accuracy"] if pd.notna(row["combined_accuracy"]) else -1)
    content = [
        "# Joint Panel Baseline Report",
        "",
        "Baselines are deterministic and do not train models.",
        "",
        f"- Best deterministic combined baseline: `{best['baseline']}` h={best['horizon']} at {pct(best['combined_accuracy'])}.",
        f"- Stock-only RF h=60 historical reference: {pct(BASELINE_STOCK_RF_H60_REFERENCE)}; reference only, not a replacement for joint-panel scoring.",
        "- Existing index benchmark best results are reference-only and cannot replace combined 36-instrument scoring.",
        "",
        "## Baseline Summary",
        "",
        markdown_table(
            [
                "baseline",
                "horizon",
                "combined_accuracy",
                "stock_only_accuracy",
                "index_only_accuracy",
                "combined_rows",
                "stock_instrument_count",
                "index_instrument_count",
                "reference_only",
            ],
            rows,
            max_rows=80,
        ),
        "",
    ]
    (REPORT_DIR / "joint_baseline_report.md").write_text("\n".join(content), encoding="utf-8")
    print(f"best_baseline={best['baseline']} h={best['horizon']} combined={best['combined_accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
