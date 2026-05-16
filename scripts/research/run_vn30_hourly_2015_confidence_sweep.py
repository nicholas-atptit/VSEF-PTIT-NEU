"""Post-benchmark confidence-threshold diagnostics for VN30 hourly 2015."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_PATH = REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark" / "hourly" / "predicted_vs_actual.csv"
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "confidence"
SUMMARY_PATH = OUTPUT_DIR / "vn30_confidence_sweep_summary.csv"
REPORT_PATH = OUTPUT_DIR / "vn30_confidence_sweep_report.md"

THRESHOLDS = [round(0.50 + index * 0.025, 3) for index in range(17)]
COVERAGE_FLOORS = [0.50, 0.40, 0.30]
SUMMARY_COLUMNS = [
    "model",
    "horizon",
    "threshold",
    "coverage_floor",
    "total_rows",
    "evaluated_rows",
    "coverage_ratio",
    "filtered_accuracy",
    "passed_60pct",
    "coverage_ok",
    "diagnostic_only",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark predictions: {rel(path)}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"model", "horizon", "frequency", "confidence", "is_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    frame = frame[frame["frequency"].astype(str).str.lower().eq("hourly")].copy()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["is_correct"] = pd.to_numeric(frame["is_correct"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame = frame[
        frame["confidence"].between(0.0, 1.0)
        & frame["is_correct"].isin([0, 1])
        & frame["horizon"].notna()
    ].copy()
    frame["horizon"] = frame["horizon"].astype(int)
    return frame


def build_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for (model, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        total_rows = int(len(group))
        for threshold in THRESHOLDS:
            selected = group[group["confidence"] >= threshold].copy()
            evaluated_rows = int(len(selected))
            coverage_ratio = evaluated_rows / total_rows if total_rows else 0.0
            filtered_accuracy = float(selected["is_correct"].mean()) if evaluated_rows else float("nan")
            passed_60pct = bool(evaluated_rows > 0 and filtered_accuracy >= 0.60)
            for coverage_floor in COVERAGE_FLOORS:
                rows.append(
                    {
                        "model": str(model),
                        "horizon": int(horizon),
                        "threshold": float(threshold),
                        "coverage_floor": float(coverage_floor),
                        "total_rows": total_rows,
                        "evaluated_rows": evaluated_rows,
                        "coverage_ratio": coverage_ratio,
                        "filtered_accuracy": filtered_accuracy,
                        "passed_60pct": passed_60pct,
                        "coverage_ok": bool(coverage_ratio >= coverage_floor),
                        "diagnostic_only": True,
                    }
                )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def best_by_floor(sweep: pd.DataFrame) -> list[dict[str, Any]]:
    if sweep.empty:
        return []
    rows: list[dict[str, Any]] = []
    for floor in COVERAGE_FLOORS:
        candidates = sweep[
            (pd.to_numeric(sweep["coverage_floor"], errors="coerce") == floor)
            & sweep["coverage_ok"].astype(bool)
            & (pd.to_numeric(sweep["evaluated_rows"], errors="coerce") > 0)
        ].copy()
        if candidates.empty:
            rows.append(
                {
                    "coverage_floor": fmt_pct(floor),
                    "model": "",
                    "horizon": "",
                    "threshold": "",
                    "evaluated_rows": 0,
                    "coverage_ratio": "",
                    "filtered_accuracy": "",
                    "passed_60pct": "no",
                }
            )
            continue
        candidates = candidates.sort_values(["filtered_accuracy", "coverage_ratio", "evaluated_rows"], ascending=[False, False, False])
        best = candidates.iloc[0]
        rows.append(
            {
                "coverage_floor": fmt_pct(floor),
                "model": best["model"],
                "horizon": int(best["horizon"]),
                "threshold": f"{float(best['threshold']):.3f}",
                "evaluated_rows": int(best["evaluated_rows"]),
                "coverage_ratio": fmt_pct(best["coverage_ratio"]),
                "filtered_accuracy": fmt_pct(best["filtered_accuracy"]),
                "passed_60pct": "yes" if bool(best["passed_60pct"]) else "no",
            }
        )
    return rows


def write_report(sweep: pd.DataFrame) -> None:
    rows = best_by_floor(sweep)
    passed_rows = int((sweep["passed_60pct"].astype(bool) & sweep["coverage_ok"].astype(bool)).sum()) if not sweep.empty else 0
    content = [
        "# VN30 Hourly 2015 Confidence Sweep Diagnostics",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(PREDICTIONS_PATH)}`.",
        "- Frequency: hourly only.",
        "- Threshold grid: 0.500 to 0.900 in 0.025 increments.",
        "- Coverage floors: 50%, 40%, 30%.",
        "- This is a post-hoc diagnostic. It does not create a new global benchmark pass.",
        "",
        "## Best Diagnostic Slice By Coverage Floor",
        "",
        markdown_table(
            [
                "coverage_floor",
                "model",
                "horizon",
                "threshold",
                "evaluated_rows",
                "coverage_ratio",
                "filtered_accuracy",
                "passed_60pct",
            ],
            rows,
        ),
        "",
        "## Boundary",
        "",
        f"- Sweep rows generated: {len(sweep)}.",
        f"- Rows with filtered accuracy at or above 60% and coverage floor satisfied: {passed_rows}.",
        "- Confidence slices are conditional diagnostics only and must be reported with coverage and post-hoc limitations.",
        "- No trading-readiness, profitability, or stable 60%+ method claim is made.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly 2015 confidence-threshold diagnostics.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PREDICTIONS_PATH, OUTPUT_DIR, SUMMARY_PATH, REPORT_PATH
    PREDICTIONS_PATH = args.predictions
    OUTPUT_DIR = args.output_dir
    SUMMARY_PATH = OUTPUT_DIR / "vn30_confidence_sweep_summary.csv"
    REPORT_PATH = OUTPUT_DIR / "vn30_confidence_sweep_report.md"

    predictions = load_predictions(PREDICTIONS_PATH)
    sweep = build_sweep(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(SUMMARY_PATH, index=False)
    write_report(sweep)
    print(f"VN30 confidence sweep complete: rows={len(sweep)} report={rel(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
