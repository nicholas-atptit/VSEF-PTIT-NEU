"""Validate listing-aware VN30 hourly predictions with ex-ante regime labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_vn30_hourly_exante_regime_validation_2005_2026 import (  # noqa: E402
    MIN_PRIOR_OBS,
    REGIME_WINDOW,
    SUMMARY_COLUMNS,
    aggregate_rows,
    build_summary,
)
from scripts.research.vn30_hourly_common import load_hourly_predictions, markdown_table, rel, save_placeholder_figure  # noqa: E402
from scripts.research.vn30_hourly_listing_aware_common import BENCHMARK_OUTPUT_DIR, REPORT_ROOT  # noqa: E402


DEFAULT_OUTPUT_DIR = REPORT_ROOT / "regime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 listing-aware ex-ante regime validation.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_report(path: Path, artifact_dir: Path, summary: pd.DataFrame) -> None:
    aggregate = aggregate_rows(summary)
    content = [
        "# VN30 Hourly Listing-Aware Ex-Ante Regime Validation Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Ex-ante proxy rule: labels use shifted prior actual returns only.",
        f"- Rolling window: {REGIME_WINDOW}; minimum prior observations: {MIN_PRIOR_OBS}.",
        "",
        "## Aggregate Regime Diagnostics",
        "",
        markdown_table(
            [
                "regime_source",
                "model",
                "horizon",
                "regime",
                "observation_count",
                "accuracy",
                "passed_60pct",
                "passed_63pct",
            ],
            aggregate,
            max_rows=40,
        )
        if aggregate
        else "No regime diagnostics are available.",
        "",
        "## Boundary",
        "",
    ]
    if summary.empty:
        content.append("No ex-ante regime claims are available because listing-aware hourly predictions are missing or empty.")
    else:
        content.append(f"- Per-ticker regime rows: {len(summary)}.")
        content.append("- Ex-ante proxy labels avoid current/future target leakage but remain diagnostics.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, summary: pd.DataFrame) -> str:
    if summary.empty:
        return save_placeholder_figure(path, "VN30 listing-aware ex-ante regime accuracy", "No hourly predictions were available.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        aggregate = pd.DataFrame(aggregate_rows(summary))
        aggregate = aggregate[aggregate["regime_source"].astype(str).eq("exante_proxy")].copy()
        if aggregate.empty:
            return save_placeholder_figure(path, "VN30 listing-aware ex-ante regime accuracy", "No ex-ante aggregate rows were available.")
        aggregate = aggregate.sort_values(["accuracy", "observation_count"], ascending=[False, False]).head(12)
        labels = [f"{row.model} h={int(row.horizon)} {row.regime}" for row in aggregate.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, aggregate["accuracy"], color="#356f8c")
        ax.axvline(0.60, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.63, color="#7a3b2e", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("VN30 listing-aware ex-ante regime accuracy")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(path, "VN30 listing-aware ex-ante regime accuracy", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    summary = build_summary(load_hourly_predictions(args.artifact_dir))
    if summary.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_listing_aware_exante_regime_accuracy_summary.csv"
    report_path = args.output_dir / "vn30_listing_aware_exante_regime_validation_report.md"
    figure_path = args.output_dir / "vn30_listing_aware_regime_accuracy.png"
    summary.to_csv(summary_path, index=False)
    write_report(report_path, args.artifact_dir, summary)
    status = write_plot(figure_path, summary)
    print(f"VN30 listing-aware ex-ante regime validation complete: rows={len(summary)} report={rel(report_path)} figure_status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
