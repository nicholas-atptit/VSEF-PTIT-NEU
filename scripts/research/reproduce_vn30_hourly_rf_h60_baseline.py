"""Reproduce/check the locked VN30 hourly RF h60 baseline in the current pipeline."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import read_joint_panel_universe, write_csv  # noqa: E402

HISTORICAL_CONSISTENCY_CSV = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_consistency" / "rf_h60_consistency_audit.csv"
HISTORICAL_LOCK_MD = REPO_ROOT / "reports" / "VN30_HOURLY_2015_BASELINE60_RESULT_LOCK.md"
HISTORICAL_RUN_CONFIG = REPO_ROOT / "outputs" / "vn30_hourly_2015_horizon_relative_target_experiments" / "run_config.json"
CURRENT_RESULTS_CSV = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening" / "final_candidate_results.csv"
CURRENT_RUN_CONFIG = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening" / "run_config.json"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_rf_h60_reproduction"
SUMMARY_CSV = OUT_DIR / "rf_h60_reproduction_summary.csv"
REPORT_MD = OUT_DIR / "rf_h60_reproduction_report.md"

HISTORICAL_ACCURACY_FALLBACK = 0.603051
HISTORICAL_ROWS_FALLBACK = 3474
HISTORICAL_FEATURE_SET = "feature_set_C_stock_lagged_plus_market_index_context"
CURRENT_CLOSEST_FEATURE_SET = "stock_lagged_rolling_plus_index_context"

SUMMARY_COLUMNS = [
    "check_item",
    "historical_value",
    "current_value",
    "matched",
    "detail",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_historical_reference() -> dict[str, Any]:
    if HISTORICAL_CONSISTENCY_CSV.exists():
        with HISTORICAL_CONSISTENCY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        canonical = next((row for row in rows if row.get("source") == "canonical_evaluator"), None)
        if canonical:
            return {
                "source": "canonical_evaluator",
                "accuracy": float(canonical.get("accuracy") or HISTORICAL_ACCURACY_FALLBACK),
                "rows": int(float(canonical.get("rows") or HISTORICAL_ROWS_FALLBACK)),
                "coverage": float(canonical.get("coverage") or 1.0),
                "details": canonical.get("details", ""),
            }
    return {
        "source": "fallback_locked_reference",
        "accuracy": HISTORICAL_ACCURACY_FALLBACK,
        "rows": HISTORICAL_ROWS_FALLBACK,
        "coverage": 1.0,
        "details": "Fallback from locked baseline context.",
    }


def current_rf_h60_rows() -> pd.DataFrame:
    if not CURRENT_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Current expanded-model results missing: {rel(CURRENT_RESULTS_CSV)}")
    frame = pd.read_csv(CURRENT_RESULTS_CSV)
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    return frame[(frame["model_name"].eq("random_forest")) & (frame["horizon"].eq(60))].copy()


def closest_current_row(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {}
    closest = rows[rows["feature_set"].eq(CURRENT_CLOSEST_FEATURE_SET)]
    if closest.empty:
        closest = rows.sort_values("feature_set").head(1)
    return closest.iloc[0].to_dict()


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def build_summary_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    historical = read_historical_reference()
    historical_config = read_json(HISTORICAL_RUN_CONFIG)
    current_config = read_json(CURRENT_RUN_CONFIG)
    stocks, _indices = read_joint_panel_universe()
    current_rows = current_rf_h60_rows()
    current = closest_current_row(current_rows)

    current_accuracy = float(current.get("final_accuracy", math.nan)) if current else math.nan
    current_rows_count = int(float(current.get("final_rows", 0) or 0)) if current else 0
    current_feature = str(current.get("feature_set", "")) if current else ""
    accuracy_gap = current_accuracy - float(historical["accuracy"]) if math.isfinite(current_accuracy) else math.nan
    row_gap = current_rows_count - int(historical["rows"])

    historical_final_window = "2025-01-01 to 2026-05-14; calendar final window from historical lock/canonical artifacts"
    current_final_window = "per-ticker final 20% chronological split over current available cache; not calendar 2025-01-01 to 2026-05-14"
    if current_config:
        current_final_window = "per-ticker final 20% chronological split; run config has no fixed calendar final window"
    historical_target = "absolute direction: future_close > current_close"
    current_target = "absolute direction: future_close > current_close"
    historical_split = f"train_end={historical_config.get('train_end', '2023-12-31 23:59:59')}; val=2024; eval_start={historical_config.get('eval_start', '2025-01-01 00:00:00')}"
    current_split = "60/20/20 per-ticker chronological split inside expanded-model script"

    same_universe = len(stocks) == 30
    same_horizon = int(float(current.get("horizon", -1) or -1)) == 60 if current else False
    same_target = historical_target == current_target
    same_rows = current_rows_count == int(historical["rows"])
    same_final_window = False
    closest_feature = current_feature == CURRENT_CLOSEST_FEATURE_SET
    same_feature = False
    reproduced = (
        same_universe
        and same_horizon
        and same_target
        and same_rows
        and same_final_window
        and same_feature
        and math.isfinite(current_accuracy)
        and abs(accuracy_gap) <= 0.001
    )
    comparable = same_universe and same_horizon and same_target and closest_feature

    rows = [
        {
            "check_item": "historical_rf_h60_accuracy",
            "historical_value": historical["accuracy"],
            "current_value": current_accuracy,
            "matched": bool_text(math.isfinite(current_accuracy) and abs(accuracy_gap) <= 0.001),
            "detail": f"gap={accuracy_gap:.6f}" if math.isfinite(accuracy_gap) else "current accuracy missing",
        },
        {
            "check_item": "universe_30_of_30",
            "historical_value": "30/30 active VN30 January 2025 stocks",
            "current_value": f"{len(stocks)}/30 active VN30 January 2025 stocks",
            "matched": bool_text(same_universe),
            "detail": ",".join(stocks),
        },
        {
            "check_item": "horizon",
            "historical_value": "h=60",
            "current_value": f"h={int(float(current.get('horizon', -1) or -1)) if current else ''}",
            "matched": bool_text(same_horizon),
            "detail": "same horizon required for reproduction",
        },
        {
            "check_item": "final_window",
            "historical_value": historical_final_window,
            "current_value": current_final_window,
            "matched": bool_text(same_final_window),
            "detail": "historical uses fixed calendar final window; expanded pipeline uses per-ticker 20% final split",
        },
        {
            "check_item": "target_definition",
            "historical_value": historical_target,
            "current_value": current_target,
            "matched": bool_text(same_target),
            "detail": "both are absolute direction labels",
        },
        {
            "check_item": "row_count",
            "historical_value": historical["rows"],
            "current_value": current_rows_count,
            "matched": bool_text(same_rows),
            "detail": f"current_minus_historical={row_gap}",
        },
        {
            "check_item": "feature_set",
            "historical_value": HISTORICAL_FEATURE_SET,
            "current_value": current_feature,
            "matched": bool_text(same_feature),
            "detail": "current closest available feature set includes stock lagged/rolling plus lagged index context, but is not the historical feature set C implementation",
        },
        {
            "check_item": "split_design",
            "historical_value": historical_split,
            "current_value": current_split,
            "matched": "no",
            "detail": "split design differs",
        },
        {
            "check_item": "reproduced_inside_current_pipeline",
            "historical_value": "required: same universe/horizon/window/target/rows/features and materially equal accuracy",
            "current_value": bool_text(reproduced),
            "matched": bool_text(reproduced),
            "detail": "not reproduced because final window, row count, split design, feature implementation, and accuracy differ",
        },
        {
            "check_item": "expanded_failure_comparable_to_locked_baseline",
            "historical_value": "apples-to-apples required",
            "current_value": "partially comparable only" if comparable else "not comparable",
            "matched": "no",
            "detail": "same universe/horizon/target, but different row count, split/window, and feature implementation",
        },
    ]
    context = {
        "historical_accuracy": float(historical["accuracy"]),
        "historical_rows": int(historical["rows"]),
        "current_accuracy": current_accuracy,
        "current_rows": current_rows_count,
        "current_feature_set": current_feature,
        "accuracy_gap": accuracy_gap,
        "row_gap": row_gap,
        "same_universe": same_universe,
        "same_horizon": same_horizon,
        "same_target": same_target,
        "same_rows": same_rows,
        "same_final_window": same_final_window,
        "same_feature": same_feature,
        "reproduced": reproduced,
        "comparable": comparable,
        "historical_final_window": historical_final_window,
        "current_final_window": current_final_window,
        "historical_target": historical_target,
        "current_target": current_target,
    }
    return rows, context


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = SUMMARY_COLUMNS
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    historical_accuracy = context["historical_accuracy"]
    current_accuracy = context["current_accuracy"]
    lines = [
        "# VN30 Hourly RF h60 Baseline Reproduction Report",
        "",
        f"- Reproduced inside current expanded-model pipeline: {bool_text(bool(context['reproduced']))}.",
        f"- Historical canonical RF h60 accuracy: {pct(historical_accuracy)}.",
        f"- Current expanded-pipeline RF h60 closest accuracy: {pct(current_accuracy)}.",
        f"- Accuracy gap: {context['accuracy_gap']:.6f}.",
        f"- Historical rows: {context['historical_rows']}.",
        f"- Current rows: {context['current_rows']}.",
        f"- Row count difference: {context['row_gap']}.",
        f"- Historical final window: {context['historical_final_window']}.",
        f"- Current final window: {context['current_final_window']}.",
        f"- Historical target: {context['historical_target']}.",
        f"- Current target: {context['current_target']}.",
        f"- Current closest feature set: `{context['current_feature_set']}`.",
        "",
        "## Decision",
        "",
        "- The 60.31% canonical RF h60 result remains valid as the locked historical baseline in its own historical/canonical setup.",
        "- The result is not reproduced inside the current expanded-model pipeline.",
        "- The expanded model-pool failure is only partially comparable: universe, horizon, and target align, but row count, final-window/split design, and feature implementation differ.",
        "",
        "## Check Matrix",
        "",
        markdown_table(rows),
        "",
        "No data fetch, new broad tuning, paper/DOCX generation, or trading/profitability claim was made.",
        "",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows, context = build_summary_rows()
    write_csv(SUMMARY_CSV, rows, SUMMARY_COLUMNS)
    write_report(rows, context)
    print(
        "rf_h60_reproduction_status="
        f"reproduced_{bool_text(bool(context['reproduced']))} "
        f"historical={context['historical_accuracy']:.6f} "
        f"current={context['current_accuracy']:.6f} "
        f"rows={context['historical_rows']}->{context['current_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
