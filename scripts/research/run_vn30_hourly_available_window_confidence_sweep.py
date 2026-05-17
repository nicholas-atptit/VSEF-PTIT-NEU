"""Run confidence-threshold diagnostics for VN30 hourly available-window predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_vn30_hourly_confidence_sweep_2005_2026 import (  # noqa: E402
    COVERAGE_FLOORS,
    SUMMARY_COLUMNS,
    best_rows,
    build_sweep,
)
from scripts.research.vn30_hourly_available_window_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    CONFIDENCE_DIR,
    load_available_window_predictions,
    markdown_table,
    rel,
    save_placeholder_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly available-window confidence sweep.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=CONFIDENCE_DIR)
    return parser.parse_args()


def write_report(path: Path, artifact_dir: Path, sweep: pd.DataFrame) -> None:
    best = best_rows(sweep)
    content = [
        "# VN30 Hourly Available-Window Confidence Sweep Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Study: VN30 hourly available-window.",
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
        content.append("No confidence-sweep claims are available because available-window hourly predictions are missing or empty.")
    else:
        warnings = int((sweep["ticker_concentration_warning"].astype(str) == "yes").sum())
        content.append(f"- Sweep rows evaluated: {len(sweep)}.")
        content.append(f"- Coverage floors evaluated: {', '.join(str(int(floor * 100)) + '%' for floor in COVERAGE_FLOORS)}.")
        content.append(f"- Rows with ticker concentration warnings: {warnings}.")
        content.append("- A filtered slice is diagnostic only and does not establish trading readiness.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, sweep: pd.DataFrame) -> str:
    if sweep.empty:
        return save_placeholder_figure(
            path,
            "VN30 hourly available-window confidence sweep",
            "No available-window hourly predictions were available.",
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flags = [f"selected_at_{int(floor * 100)}pct_floor" for floor in COVERAGE_FLOORS]
        selected_groups = sweep[sweep[flags].any(axis=1)][["model", "horizon"]].drop_duplicates().head(6)
        if selected_groups.empty:
            selected_groups = sweep[["model", "horizon"]].drop_duplicates().head(6)
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
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
        axes[0].set_title("VN30 hourly available-window confidence sweep")
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
        return save_placeholder_figure(path, "VN30 hourly available-window confidence sweep", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    data = load_available_window_predictions(args.artifact_dir)
    sweep = build_sweep(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_available_window_confidence_sweep_summary.csv"
    report_path = args.output_dir / "vn30_available_window_confidence_sweep_report.md"
    figure_path = args.output_dir / "vn30_available_window_confidence_threshold_coverage_accuracy.png"
    if sweep.empty:
        sweep = pd.DataFrame(columns=SUMMARY_COLUMNS)
    sweep.to_csv(summary_path, index=False)
    write_report(report_path, args.artifact_dir, sweep)
    figure_status = write_plot(figure_path, sweep)
    print(
        "VN30 hourly available-window confidence sweep complete: "
        f"rows={len(sweep)} report={rel(report_path)} figure={rel(figure_path)} status={figure_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
