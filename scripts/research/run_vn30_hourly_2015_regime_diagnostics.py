"""Post-benchmark regime diagnostics for VN30 hourly 2015 predictions."""

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
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "regime"
SUMMARY_PATH = OUTPUT_DIR / "vn30_regime_diagnostics_summary.csv"
REPORT_PATH = OUTPUT_DIR / "vn30_regime_diagnostics_report.md"

SUMMARY_COLUMNS = [
    "regime_dimension",
    "regime",
    "model",
    "horizon",
    "observations",
    "ticker_count",
    "accuracy",
    "passed_60pct",
    "label_source",
    "diagnostic_status",
    "limitation",
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


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int = 30) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows[:max_rows]:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark predictions: {rel(path)}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"model", "horizon", "frequency", "ticker", "is_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    frame = frame[frame["frequency"].astype(str).str.lower().eq("hourly")].copy()
    frame["is_correct"] = pd.to_numeric(frame["is_correct"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame = frame[frame["is_correct"].isin([0, 1]) & frame["horizon"].notna()].copy()
    frame["horizon"] = frame["horizon"].astype(int)
    return frame


def usable_label_columns(predictions: pd.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("return_regime", "regime"),
        ("volatility_regime", "volatility_regime"),
    ]
    usable: list[tuple[str, str]] = []
    for dimension, column in candidates:
        if column not in predictions.columns:
            continue
        values = predictions[column].fillna("").astype(str).str.strip().str.lower()
        known = values[~values.isin(["", "unknown", "nan", "none"])]
        if not known.empty:
            usable.append((dimension, column))
    return usable


def limitation_frame(message: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "regime_dimension": "",
                "regime": "",
                "model": "",
                "horizon": "",
                "observations": 0,
                "ticker_count": 0,
                "accuracy": "",
                "passed_60pct": False,
                "label_source": "not_available",
                "diagnostic_status": "limitation_only",
                "limitation": message,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    label_columns = usable_label_columns(predictions)
    if predictions.empty:
        return limitation_frame("Benchmark prediction file is empty.")
    if not label_columns:
        return limitation_frame("No non-empty existing benchmark regime labels are available; no labels were fabricated.")

    rows: list[dict[str, Any]] = []
    for dimension, column in label_columns:
        working = predictions.copy()
        working["regime_value"] = working[column].fillna("").astype(str).str.strip().str.lower()
        working = working[~working["regime_value"].isin(["", "unknown", "nan", "none"])].copy()
        if working.empty:
            continue
        for (regime, model, horizon), group in working.groupby(["regime_value", "model", "horizon"], sort=True):
            observations = int(len(group))
            accuracy = float(group["is_correct"].mean()) if observations else float("nan")
            rows.append(
                {
                    "regime_dimension": dimension,
                    "regime": str(regime),
                    "model": str(model),
                    "horizon": int(horizon),
                    "observations": observations,
                    "ticker_count": int(group["ticker"].astype(str).str.upper().nunique()),
                    "accuracy": accuracy,
                    "passed_60pct": bool(observations > 0 and accuracy >= 0.60),
                    "label_source": "existing_benchmark_prediction_labels",
                    "diagnostic_status": "computed",
                    "limitation": "Existing labels are benchmark-internal regime proxies; no new external ex-ante regime labels were created.",
                }
            )
    if not rows:
        return limitation_frame("Existing regime columns were present but no usable label values were found.")
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def top_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty or "diagnostic_status" not in summary.columns:
        return []
    computed = summary[summary["diagnostic_status"].astype(str).eq("computed")].copy()
    if computed.empty:
        return []
    computed["accuracy"] = pd.to_numeric(computed["accuracy"], errors="coerce")
    computed = computed.sort_values(["accuracy", "observations"], ascending=[False, False]).head(20)
    rows: list[dict[str, Any]] = []
    for row in computed.itertuples(index=False):
        rows.append(
            {
                "regime_dimension": row.regime_dimension,
                "regime": row.regime,
                "model": row.model,
                "horizon": int(row.horizon),
                "observations": int(row.observations),
                "accuracy": fmt_pct(row.accuracy),
                "passed_60pct": "yes" if bool(row.passed_60pct) else "no",
            }
        )
    return rows


def write_report(summary: pd.DataFrame) -> None:
    computed = summary[summary["diagnostic_status"].astype(str).eq("computed")].copy() if not summary.empty else pd.DataFrame()
    content = [
        "# VN30 Hourly 2015 Regime Diagnostics",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(PREDICTIONS_PATH)}`.",
        "- Frequency: hourly only.",
        "- Regime source: existing labels already emitted by the base benchmark predictions.",
        "- No new regime labels were created or fabricated.",
        "",
        "## Top Regime Diagnostic Rows",
        "",
        markdown_table(
            ["regime_dimension", "regime", "model", "horizon", "observations", "accuracy", "passed_60pct"],
            top_rows(summary),
        ),
        "",
        "## Boundary",
        "",
    ]
    if computed.empty:
        limitations = sorted(summary["limitation"].dropna().astype(str).unique().tolist()) if "limitation" in summary.columns else []
        content.append("Only a limitation report is available.")
        for limitation in limitations:
            content.append(f"- {limitation}")
    else:
        pass_rows = int(computed["passed_60pct"].astype(bool).sum())
        content.append(f"- Regime diagnostic rows generated: {len(computed)}.")
        content.append(f"- Rows at or above 60% accuracy: {pass_rows}.")
        content.append("- These are benchmark-internal regime slices, not an independently validated market-regime system.")
    content.extend(
        [
            "- No trading-readiness or profitability claim is made.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly 2015 regime diagnostics.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PREDICTIONS_PATH, OUTPUT_DIR, SUMMARY_PATH, REPORT_PATH
    PREDICTIONS_PATH = args.predictions
    OUTPUT_DIR = args.output_dir
    SUMMARY_PATH = OUTPUT_DIR / "vn30_regime_diagnostics_summary.csv"
    REPORT_PATH = OUTPUT_DIR / "vn30_regime_diagnostics_report.md"

    predictions = load_predictions(PREDICTIONS_PATH)
    summary = build_summary(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(summary)
    print(f"VN30 regime diagnostics complete: rows={len(summary)} report={rel(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
