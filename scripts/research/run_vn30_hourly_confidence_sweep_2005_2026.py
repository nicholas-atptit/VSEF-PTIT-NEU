"""Run a confidence-threshold sweep from official frozen VN30 hourly predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    REPORT_ROOT,
    format_cell,
    load_hourly_predictions,
    markdown_table,
    rel,
    save_placeholder_figure,
)


DEFAULT_OUTPUT_DIR = REPORT_ROOT / "confidence"
THRESHOLDS = [round(value / 100, 2) for value in range(50, 91)]
COVERAGE_FLOORS = (0.50, 0.40, 0.30, 0.20)
SUMMARY_COLUMNS = [
    "frequency",
    "model",
    "horizon",
    "threshold",
    "total_rows",
    "evaluated_rows",
    "coverage_ratio",
    "filtered_accuracy",
    "passed_60pct",
    "coverage_floor_50pct_ok",
    "coverage_floor_40pct_ok",
    "coverage_floor_30pct_ok",
    "coverage_floor_20pct_ok",
    "selected_at_50pct_floor",
    "selected_at_40pct_floor",
    "selected_at_30pct_floor",
    "selected_at_20pct_floor",
    "ticker_count",
    "top_ticker",
    "top1_prediction_share",
    "top3_prediction_share",
    "ticker_concentration_warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly confidence sweep diagnostics.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def candidate_label(row: pd.Series | dict[str, Any]) -> str:
    return f"hourly {row['model']} h={int(row['horizon'])} threshold {float(row['threshold']):.2f}"


def concentration(data: pd.DataFrame, model: str, horizon: int, threshold: float) -> dict[str, Any]:
    selected = data[(data["model"] == model) & (data["horizon"] == horizon) & (data["confidence"] >= threshold)]
    if selected.empty:
        return {
            "ticker_count": 0,
            "top_ticker": "",
            "top1_prediction_share": None,
            "top3_prediction_share": None,
            "ticker_concentration_warning": "missing",
        }
    counts = selected.groupby("ticker").size().sort_values(ascending=False)
    total = float(counts.sum())
    top1 = float(counts.iloc[0] / total)
    top3 = float(counts.head(3).sum() / total)
    warning = "yes" if int(counts.shape[0]) < 10 or top1 >= 0.35 or top3 >= 0.60 else "no"
    return {
        "ticker_count": int(counts.shape[0]),
        "top_ticker": str(counts.index[0]),
        "top1_prediction_share": top1,
        "top3_prediction_share": top3,
        "ticker_concentration_warning": warning,
    }


def build_sweep(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    required = {"model", "horizon", "confidence", "is_correct", "ticker"}
    if required.difference(data.columns):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    working = data.dropna(subset=["model", "horizon", "confidence", "is_correct", "ticker"]).copy()
    working = working[working["frequency"].astype(str).str.lower().eq("hourly")]
    if working.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (model, horizon), group in working.groupby(["model", "horizon"], sort=True):
        total_rows = int(len(group))
        for threshold in THRESHOLDS:
            selected = group[group["confidence"] >= threshold]
            evaluated_rows = int(len(selected))
            coverage = evaluated_rows / total_rows if total_rows else 0.0
            accuracy = float(selected["is_correct"].mean()) if evaluated_rows else None
            row: dict[str, Any] = {
                "frequency": "hourly",
                "model": str(model),
                "horizon": int(horizon),
                "threshold": threshold,
                "total_rows": total_rows,
                "evaluated_rows": evaluated_rows,
                "coverage_ratio": coverage,
                "filtered_accuracy": accuracy,
                "passed_60pct": bool(accuracy is not None and accuracy >= 0.60),
            }
            for floor in COVERAGE_FLOORS:
                row[f"coverage_floor_{int(floor * 100)}pct_ok"] = bool(coverage >= floor)
                row[f"selected_at_{int(floor * 100)}pct_floor"] = False
            row.update(concentration(working, str(model), int(horizon), threshold))
            rows.append(row)
    sweep = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if sweep.empty:
        return sweep
    for floor in COVERAGE_FLOORS:
        flag = f"selected_at_{int(floor * 100)}pct_floor"
        eligible = sweep[(sweep["coverage_ratio"] >= floor) & sweep["filtered_accuracy"].notna()].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values(
            ["filtered_accuracy", "coverage_ratio", "evaluated_rows", "threshold"],
            ascending=[False, False, False, True],
        )
        sweep.loc[eligible.index[0], flag] = True
    return sweep


def best_rows(sweep: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in COVERAGE_FLOORS:
        flag = f"selected_at_{int(floor * 100)}pct_floor"
        selected = sweep[sweep[flag] == True] if not sweep.empty and flag in sweep.columns else pd.DataFrame()  # noqa: E712
        if selected.empty:
            rows.append(
                {
                    "coverage_floor": f">= {int(floor * 100)}%",
                    "candidate": "missing",
                    "evaluated_rows": "",
                    "coverage_ratio": "",
                    "filtered_accuracy": "",
                    "passed_60pct": "",
                    "ticker_concentration_warning": "",
                }
            )
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "coverage_floor": f">= {int(floor * 100)}%",
                "candidate": candidate_label(row),
                "evaluated_rows": int(row["evaluated_rows"]),
                "coverage_ratio": float(row["coverage_ratio"]),
                "filtered_accuracy": float(row["filtered_accuracy"]),
                "passed_60pct": bool(row["passed_60pct"]),
                "ticker_count": int(row["ticker_count"]),
                "top_ticker": row["top_ticker"],
                "top1_prediction_share": row["top1_prediction_share"],
                "top3_prediction_share": row["top3_prediction_share"],
                "ticker_concentration_warning": row["ticker_concentration_warning"],
            }
        )
    return rows


def write_report(path: Path, artifact_dir: Path, sweep: pd.DataFrame) -> None:
    best = best_rows(sweep)
    content = [
        "# VN30 Hourly Confidence Sweep Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Threshold grid: 0.50 to 0.90.",
        "- Coverage floors: 50%, 40%, 30%, 20%.",
        "",
        "## Best Candidate by Coverage Floor",
        "",
        markdown_table(
            [
                "coverage_floor",
                "candidate",
                "evaluated_rows",
                "coverage_ratio",
                "filtered_accuracy",
                "passed_60pct",
                "ticker_count",
                "top_ticker",
                "top1_prediction_share",
                "top3_prediction_share",
                "ticker_concentration_warning",
            ],
            best,
        ),
        "",
        "## Boundary",
        "",
    ]
    if sweep.empty:
        content.append("No confidence-sweep claims are available because official hourly VN30 predictions are missing or empty.")
    else:
        warnings = int((sweep["ticker_concentration_warning"].astype(str) == "yes").sum())
        content.append(f"- Sweep rows evaluated: {len(sweep)}.")
        content.append(f"- Rows with ticker concentration warnings: {warnings}.")
        content.append("- A passing filtered slice is diagnostic only and does not establish live trading readiness.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, sweep: pd.DataFrame) -> str:
    if sweep.empty:
        return save_placeholder_figure(
            path,
            "VN30 hourly confidence sweep",
            "No official hourly predictions were available because the full VN30 coverage gate failed.",
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        selected_groups = (
            sweep[
                sweep[[f"selected_at_{int(floor * 100)}pct_floor" for floor in COVERAGE_FLOORS]].any(axis=1)
            ][["model", "horizon"]]
            .drop_duplicates()
            .head(6)
        )
        if selected_groups.empty:
            selected_groups = sweep[["model", "horizon"]].drop_duplicates().head(6)
        for _, group_row in selected_groups.iterrows():
            group = sweep[
                (sweep["model"] == group_row["model"])
                & (pd.to_numeric(sweep["horizon"], errors="coerce") == int(group_row["horizon"]))
            ].sort_values("threshold")
            label = f"{group_row['model']} h={int(group_row['horizon'])}"
            axes[0].plot(group["threshold"], group["filtered_accuracy"], marker="o", markersize=3, linewidth=1, label=label)
            axes[1].plot(group["threshold"], group["coverage_ratio"], marker="o", markersize=3, linewidth=1, label=label)
        axes[0].axhline(0.60, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("VN30 hourly confidence threshold sweep")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Coverage")
        axes[1].set_xlabel("Confidence threshold")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(path, "VN30 hourly confidence sweep", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    data = load_hourly_predictions(args.artifact_dir)
    sweep = build_sweep(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_hourly_confidence_sweep_summary.csv"
    report_path = args.output_dir / "vn30_hourly_confidence_sweep_report.md"
    figure_path = args.output_dir / "vn30_hourly_confidence_threshold_coverage_accuracy.png"
    sweep.to_csv(summary_path, index=False)
    write_report(report_path, args.artifact_dir, sweep)
    figure_status = write_plot(figure_path, sweep)
    print(
        "VN30 hourly confidence sweep complete: "
        f"rows={len(sweep)} report={rel(report_path)} figure={rel(figure_path)} figure_status={format_cell(figure_status)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
