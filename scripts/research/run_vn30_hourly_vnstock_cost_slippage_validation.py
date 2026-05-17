"""Run proxy cost/slippage diagnostics for vnstock-fetched VN30 hourly benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_vn30_hourly_cost_slippage_validation_2005_2026 import (  # noqa: E402
    BASELINES,
    EQUITY_COLUMNS,
    SUMMARY_COLUMNS,
    TRADE_COLUMNS,
    run_diagnostics,
)
from scripts.research.vn30_hourly_common import load_hourly_predictions, markdown_table, rel, save_placeholder_figure  # noqa: E402
from scripts.research.vn30_hourly_vnstock_common import BENCHMARK_OUTPUT_DIR, FULL_REPORT_ROOT  # noqa: E402


DEFAULT_OUTPUT_DIR = FULL_REPORT_ROOT / "cost_slippage"
DEFAULT_CONFIDENCE_PATH = FULL_REPORT_ROOT / "confidence" / "vn30_hourly_vnstock_confidence_sweep_summary.csv"
DEFAULT_REGIME_PATH = FULL_REPORT_ROOT / "regime" / "vn30_hourly_vnstock_exante_regime_accuracy_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly vnstock cost/slippage diagnostics.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--confidence-summary", type=Path, default=DEFAULT_CONFIDENCE_PATH)
    parser.add_argument("--regime-summary", type=Path, default=DEFAULT_REGIME_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_report(path: Path, artifact_dir: Path, summary: pd.DataFrame) -> None:
    top_rows = [] if summary.empty else summary.sort_values(["net_return", "row_count"], ascending=[False, False]).head(20).to_dict("records")
    content = [
        "# VN30 Hourly vnstock Cost/Slippage Proxy Validation Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Data source: fetched vnstock/vnstock_data normalized cache.",
        "- Cost grid: transaction cost bps 5, 10, 15, 20; slippage bps 5, 10, 15, 20.",
        f"- Baselines: {', '.join(item for item in BASELINES if item != 'model_signal')}.",
        "",
        "## Top Proxy Rows",
        "",
        markdown_table(
            [
                "slice_name",
                "baseline",
                "transaction_cost_bps",
                "slippage_bps",
                "row_count",
                "gross_return",
                "net_return",
                "turnover",
                "max_drawdown",
                "profit_factor",
                "win_rate",
                "trade_count",
                "exposure",
                "benchmark_comparison",
            ],
            top_rows,
        )
        if top_rows
        else "No cost/slippage proxy diagnostics are available.",
        "",
        "## Boundary",
        "",
        "This remains proxy diagnostics. It does not establish live trading readiness because real order book depth, fills, liquidity filters, and execution policy are not implemented.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, equity_curve: pd.DataFrame) -> str:
    if equity_curve.empty:
        return save_placeholder_figure(path, "VN30 hourly vnstock proxy equity curve", "No selected candidates were available for diagnostics.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        working = equity_curve.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        groups = working[["slice_name", "baseline"]].drop_duplicates().head(8)
        fig, ax = plt.subplots(figsize=(9, 5))
        for row in groups.itertuples(index=False):
            group = working[(working["slice_name"] == row.slice_name) & (working["baseline"] == row.baseline)].sort_values("timestamp")
            ax.plot(group["timestamp"], group["net_equity"], linewidth=1, label=f"{row.slice_name} {row.baseline}")
        ax.set_title("VN30 hourly vnstock cost/slippage proxy equity curve")
        ax.set_ylabel("Net equity")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(path, "VN30 hourly vnstock proxy equity curve", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    data = load_hourly_predictions(args.artifact_dir)
    summary, trade_list, equity_curve = run_diagnostics(data, args.confidence_summary, args.regime_summary)
    if summary.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    if trade_list.empty:
        trade_list = pd.DataFrame(columns=TRADE_COLUMNS)
    if equity_curve.empty:
        equity_curve = pd.DataFrame(columns=EQUITY_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_hourly_vnstock_cost_slippage_summary.csv"
    report_path = args.output_dir / "vn30_hourly_vnstock_cost_slippage_validation_report.md"
    trade_path = args.output_dir / "vn30_hourly_vnstock_trade_list.csv"
    equity_path = args.output_dir / "vn30_hourly_vnstock_equity_curve.csv"
    figure_path = args.output_dir / "vn30_hourly_vnstock_equity_curve.png"
    summary.to_csv(summary_path, index=False)
    trade_list.to_csv(trade_path, index=False)
    equity_curve.to_csv(equity_path, index=False)
    write_report(report_path, args.artifact_dir, summary)
    figure_status = write_plot(figure_path, equity_curve)
    print(
        "VN30 hourly vnstock cost/slippage diagnostics complete: "
        f"rows={len(summary)} report={rel(report_path)} figure={rel(figure_path)} status={figure_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
