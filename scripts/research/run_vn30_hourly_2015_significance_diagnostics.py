"""Post-benchmark directional significance diagnostics for VN30 hourly 2015."""

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
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "significance"
SUMMARY_PATH = OUTPUT_DIR / "vn30_significance_summary.csv"
REPORT_PATH = OUTPUT_DIR / "vn30_significance_report.md"

SUMMARY_COLUMNS = [
    "model",
    "horizon",
    "observations",
    "correct",
    "accuracy",
    "p_value",
    "statistically_above_50pct",
    "bonferroni_alpha",
    "statistically_above_50pct_bonferroni",
    "multiple_testing_limitation",
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


def fmt_p(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6g}"


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
    required = {"model", "horizon", "frequency", "is_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    frame = frame[frame["frequency"].astype(str).str.lower().eq("hourly")].copy()
    frame["is_correct"] = pd.to_numeric(frame["is_correct"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame = frame[frame["is_correct"].isin([0, 1]) & frame["horizon"].notna()].copy()
    frame["horizon"] = frame["horizon"].astype(int)
    return frame


def binomial_p_value_greater(successes: int, observations: int, null_accuracy: float = 0.50) -> float:
    if observations <= 0:
        return float("nan")
    try:
        from scipy.stats import binomtest

        return float(binomtest(successes, observations, p=null_accuracy, alternative="greater").pvalue)
    except Exception:
        p_hat = successes / observations
        variance = null_accuracy * (1.0 - null_accuracy) / observations
        if variance <= 0.0:
            return float("nan")
        z_score = (p_hat - null_accuracy) / math.sqrt(variance)
        return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    grouped = predictions.groupby(["model", "horizon"], sort=True)
    bonferroni_alpha = 0.05 / max(int(grouped.ngroups), 1)
    rows: list[dict[str, Any]] = []
    for (model, horizon), group in grouped:
        observations = int(len(group))
        correct = int(group["is_correct"].sum())
        accuracy = correct / observations if observations else float("nan")
        p_value = binomial_p_value_greater(correct, observations, 0.50)
        rows.append(
            {
                "model": str(model),
                "horizon": int(horizon),
                "observations": observations,
                "correct": correct,
                "accuracy": accuracy,
                "p_value": p_value,
                "statistically_above_50pct": bool(math.isfinite(p_value) and p_value < 0.05),
                "bonferroni_alpha": bonferroni_alpha,
                "statistically_above_50pct_bonferroni": bool(math.isfinite(p_value) and p_value < bonferroni_alpha),
                "multiple_testing_limitation": "yes",
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(["p_value", "accuracy"], ascending=[True, False]).reset_index(drop=True)


def report_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "horizon": int(row.horizon),
                "observations": int(row.observations),
                "accuracy": fmt_pct(row.accuracy),
                "p_value": fmt_p(row.p_value),
                "statistically_above_50pct": "yes" if bool(row.statistically_above_50pct) else "no",
                "bonferroni_significant": "yes" if bool(row.statistically_above_50pct_bonferroni) else "no",
            }
        )
    return rows


def write_report(summary: pd.DataFrame) -> None:
    unadjusted = int(summary["statistically_above_50pct"].astype(bool).sum()) if not summary.empty else 0
    bonferroni = int(summary["statistically_above_50pct_bonferroni"].astype(bool).sum()) if not summary.empty else 0
    content = [
        "# VN30 Hourly 2015 Significance Diagnostics",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(PREDICTIONS_PATH)}`.",
        "- Null hypothesis: directional accuracy is 50%.",
        "- Test: one-sided binomial/sign test by model and horizon.",
        "- Multiple-testing limitation: 16 model/horizon rows are tested, so unadjusted p-values are diagnostic only.",
        "",
        "## Results",
        "",
        markdown_table(
            [
                "model",
                "horizon",
                "observations",
                "accuracy",
                "p_value",
                "statistically_above_50pct",
                "bonferroni_significant",
            ],
            report_rows(summary),
        ),
        "",
        "## Boundary",
        "",
        f"- Unadjusted rows statistically above 50% at alpha=0.05: {unadjusted}.",
        f"- Bonferroni-adjusted rows statistically above 50%: {bonferroni}.",
        "- Statistical evidence against a 50% null is not a trading-readiness or profitability claim.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly 2015 significance diagnostics.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PREDICTIONS_PATH, OUTPUT_DIR, SUMMARY_PATH, REPORT_PATH
    PREDICTIONS_PATH = args.predictions
    OUTPUT_DIR = args.output_dir
    SUMMARY_PATH = OUTPUT_DIR / "vn30_significance_summary.csv"
    REPORT_PATH = OUTPUT_DIR / "vn30_significance_report.md"

    predictions = load_predictions(PREDICTIONS_PATH)
    summary = build_summary(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(summary)
    print(f"VN30 significance diagnostics complete: rows={len(summary)} report={rel(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
