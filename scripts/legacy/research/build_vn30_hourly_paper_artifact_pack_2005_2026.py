"""Build VN30 hourly NCKH paper tables, figures, and notes from generated artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    EVAL_END_TEXT,
    EVAL_START_TEXT,
    REPORT_ROOT,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START_TEXT,
    VN30_TICKERS,
    as_bool,
    markdown_table,
    read_csv_rows,
    read_json,
    rel,
    save_placeholder_figure,
    write_csv,
)


DEFAULT_REPORT_DIR = REPORT_ROOT
TABLE_DIR_NAME = "paper_tables"
FIGURE_DIR_NAME = "paper_figures"
NOTE_DIR_NAME = "paper_notes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN30 hourly paper artifact pack.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_table(table_dir: Path, key: str, title: str, rows: list[dict[str, Any]], headers: list[str], note: str) -> None:
    if not rows:
        rows = [{"status": "missing", "note": note}]
        headers = ["status", "note"]
    csv_path = table_dir / f"{key}.csv"
    md_path = table_dir / f"{key}.md"
    write_csv(csv_path, rows, fieldnames=headers)
    content = [
        f"# {title}",
        "",
        markdown_table(headers, rows),
        "",
        "## Note",
        "",
        note,
        "",
    ]
    md_path.write_text("\n".join(content), encoding="utf-8")


def table1_rows(report_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(report_dir / "audit" / "vn30_hourly_coverage_audit.csv")
    return [
        {
            "ticker": row.get("ticker", ""),
            "index": "VN30",
            "frequency": "hourly",
            "first_available_hourly_timestamp": row.get("first_available_hourly_timestamp", ""),
            "last_available_hourly_timestamp": row.get("last_available_hourly_timestamp", ""),
            "hourly_rows": row.get("hourly_rows", ""),
            "benchmark_usable": row.get("benchmark_usable", ""),
            "missing_reason": row.get("missing_reason", ""),
        }
        for row in rows
    ]


def table2_rows(run_config: dict[str, Any]) -> list[dict[str, Any]]:
    models = run_config.get("models") or ["xgboost", "lightgbm", "random_forest", "stacking"]
    baselines = [
        "buy_and_hold",
        "flat_no_trade",
        "always_up",
        "moving_average_signal",
        "previous_direction_signal",
    ]
    rows = [{"type": "model", "name": model, "frequency": "hourly", "target": "direction"} for model in models]
    rows.extend({"type": "baseline", "name": baseline, "frequency": "hourly", "target": "direction"} for baseline in baselines)
    return rows


def table3_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    if not summary:
        return []
    return [
        {
            "frequency": "hourly",
            "status": summary.get("status", ""),
            "benchmark_run": summary.get("benchmark_run", ""),
            "n_predictions": summary.get("n_predictions", ""),
            "overall_accuracy": summary.get("overall_accuracy", ""),
            "passed_60pct": summary.get("passed", ""),
            "evaluated_tickers": ",".join(summary.get("evaluated_tickers", [])),
            "benchmark_usable_ticker_count": summary.get("benchmark_usable_ticker_count", ""),
        }
    ]


def top_csv_rows(path: Path, sort_col: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    if sort_col and sort_col in frame.columns:
        frame[sort_col] = pd.to_numeric(frame[sort_col], errors="coerce")
        frame = frame.sort_values(sort_col, ascending=False)
    return frame.head(limit).fillna("").to_dict("records")


def confidence_rows(report_dir: Path) -> list[dict[str, Any]]:
    path = report_dir / "confidence" / "vn30_hourly_confidence_sweep_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    flags = [column for column in frame.columns if column.startswith("selected_at_")]
    selected = frame[frame[flags].any(axis=1)] if flags else frame.head(0)
    if selected.empty:
        selected = frame.sort_values(["filtered_accuracy", "coverage_ratio"], ascending=[False, False]).head(10)
    return selected.fillna("").to_dict("records")


def limitation_rows(audit_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    usable = sum(1 for row in audit_rows if as_bool(row.get("benchmark_usable")))
    return [
        {
            "area": "Hourly data coverage",
            "status": "blocking" if usable < 30 else "passed",
            "evidence": f"{usable}/30 VN30 tickers benchmark-usable",
            "claim_boundary": "No full VN30 evidence claim unless 30/30 are usable.",
        },
        {
            "area": "Period design",
            "status": "fixed",
            "evidence": f"{TRAIN_START_TEXT} to {EVAL_END_TEXT}",
            "claim_boundary": "The requested period is not silently shortened.",
        },
        {
            "area": "Frequency",
            "status": "fixed",
            "evidence": "hourly only",
            "claim_boundary": "Daily data and daily-to-hourly resampling are excluded.",
        },
        {
            "area": "Leakage",
            "status": "declared",
            "evidence": manifest.get("training_label_cutoff_rule", "target_timestamp <= train_cutoff"),
            "claim_boundary": "Evaluation labels after 2025-01-01 do not enter training labels.",
        },
        {
            "area": "Trading readiness",
            "status": "proxy_only",
            "evidence": "Cost/slippage diagnostics lack real fills and liquidity filters.",
            "claim_boundary": "No live trading readiness claim.",
        },
    ]


def copy_or_placeholder(src: Path, dst: Path, title: str, message: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.stat().st_size > 0:
        shutil.copyfile(src, dst)
        return "copied"
    return save_placeholder_figure(dst, title, message)


def plot_accuracy(src: Path, dst: Path) -> str:
    if not src.exists():
        return save_placeholder_figure(dst, "VN30 hourly accuracy by model/horizon", "Benchmark accuracy rows are missing.")
    frame = pd.read_csv(src)
    if frame.empty:
        return save_placeholder_figure(dst, "VN30 hourly accuracy by model/horizon", "Benchmark accuracy rows are empty.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grouped = frame.groupby(["model", "horizon"], as_index=False).apply(
            lambda group: pd.Series(
                {
                    "n_obs": pd.to_numeric(group["n_obs"], errors="coerce").fillna(0).sum(),
                    "accuracy": (
                        pd.to_numeric(group["accuracy"], errors="coerce")
                        .mul(pd.to_numeric(group["n_obs"], errors="coerce").fillna(0))
                        .sum()
                    )
                    / max(pd.to_numeric(group["n_obs"], errors="coerce").fillna(0).sum(), 1),
                }
            )
        )
        grouped = grouped.sort_values("accuracy", ascending=False).head(16)
        labels = [f"{row.model} h={int(row.horizon)}" for row in grouped.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, grouped["accuracy"], color="#356f8c")
        ax.axvline(0.60, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("VN30 hourly accuracy by model/horizon")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(dst, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(dst, "VN30 hourly accuracy by model/horizon", f"Plot rendering failed: {exc}")


def build_figures(report_dir: Path, artifact_dir: Path, figure_dir: Path) -> list[dict[str, str]]:
    figures = [
        {
            "figure": "figure1_research_pipeline.png",
            "status": save_placeholder_figure(
                figure_dir / "figure1_research_pipeline.png",
                "VN30 hourly research pipeline",
                "Universe freeze -> hourly coverage audit -> benchmark gate -> diagnostics -> paper artifacts.",
            ),
        },
        {
            "figure": "figure2_hourly_walk_forward_validation_design.png",
            "status": save_placeholder_figure(
                figure_dir / "figure2_hourly_walk_forward_validation_design.png",
                "Hourly walk-forward design",
                f"Train labels: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}; evaluation: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
            ),
        },
        {
            "figure": "figure3_hourly_accuracy_by_model_horizon.png",
            "status": plot_accuracy(
                artifact_dir / "hourly" / "accuracy_summary.csv",
                figure_dir / "figure3_hourly_accuracy_by_model_horizon.png",
            ),
        },
        {
            "figure": "figure4_hourly_confidence_threshold_vs_coverage_accuracy.png",
            "status": copy_or_placeholder(
                report_dir / "confidence" / "vn30_hourly_confidence_threshold_coverage_accuracy.png",
                figure_dir / "figure4_hourly_confidence_threshold_vs_coverage_accuracy.png",
                "VN30 hourly confidence threshold vs coverage/accuracy",
                "Confidence sweep figure is unavailable.",
            ),
        },
        {
            "figure": "figure5_hourly_regime_specific_accuracy.png",
            "status": copy_or_placeholder(
                report_dir / "regime" / "vn30_hourly_regime_accuracy.png",
                figure_dir / "figure5_hourly_regime_specific_accuracy.png",
                "VN30 hourly regime-specific accuracy",
                "Regime accuracy figure is unavailable.",
            ),
        },
        {
            "figure": "figure6_hourly_exante_regime_accuracy.png",
            "status": copy_or_placeholder(
                report_dir / "regime" / "vn30_hourly_regime_accuracy.png",
                figure_dir / "figure6_hourly_exante_regime_accuracy.png",
                "VN30 hourly ex-ante regime accuracy",
                "Ex-ante regime figure is unavailable.",
            ),
        },
        {
            "figure": "figure7_hourly_cost_slippage_proxy_equity_curve.png",
            "status": copy_or_placeholder(
                report_dir / "cost_slippage" / "vn30_hourly_equity_curve.png",
                figure_dir / "figure7_hourly_cost_slippage_proxy_equity_curve.png",
                "VN30 hourly cost/slippage proxy equity curve",
                "Cost/slippage equity curve is unavailable.",
            ),
        },
    ]
    return figures


def write_notes(note_dir: Path, report_dir: Path, artifact_dir: Path, figure_status: list[dict[str, str]]) -> None:
    manifest = read_json(artifact_dir / "manifest.json")
    full_usable = bool(manifest.get("all_30_benchmark_usable"))
    status_lines = [
        "# VN30 Hourly Paper Artifact Status",
        "",
        f"- Artifact directory: `{rel(artifact_dir)}`.",
        f"- Report directory: `{rel(report_dir)}`.",
        f"- All 30 benchmark-usable: {str(full_usable).lower()}.",
        f"- Training labels: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation labels: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Daily data used: false.",
        "- Daily-to-hourly resampling used: false.",
        "- VN100 seven-ticker evidence reused: false.",
        "",
        "## Figure Status",
        "",
        markdown_table(["figure", "status"], figure_status),
        "",
    ]
    (note_dir / "paper_artifact_status.md").write_text("\n".join(status_lines), encoding="utf-8")
    boundary = [
        "# VN30 Hourly Claim Boundary Notes",
        "",
        "The VN30 hourly rerun did not achieve full 30-ticker benchmark usability under the requested 2005-2026 hourly design."
        if not full_usable
        else "The study evaluates all 30 frozen VN30 constituents using hourly data under the selected research design.",
        "",
        "No final paper should be written unless the full benchmark gate passes 30/30 ticker usability.",
        "No daily data, daily-to-hourly resampling, or prior VN100 seven-ticker evidence is used in this VN30 artifact pack.",
        "",
    ]
    (note_dir / "claim_boundary_notes.md").write_text("\n".join(boundary), encoding="utf-8")
    if not full_usable:
        notice = [
            "# Final Paper Not Written",
            "",
            "The requested final VN30 hourly paper was not generated because the benchmark gate did not pass.",
            "This follows the requirement to stop before final claims when all 30 VN30 tickers are not benchmark-usable.",
            "",
        ]
        (note_dir / "final_paper_not_written.md").write_text("\n".join(notice), encoding="utf-8")


def main() -> int:
    args = parse_args()
    table_dir = args.report_dir / TABLE_DIR_NAME
    figure_dir = args.report_dir / FIGURE_DIR_NAME
    note_dir = args.report_dir / NOTE_DIR_NAME
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)

    run_config = read_json(args.artifact_dir / "run_config.json")
    manifest = read_json(args.artifact_dir / "manifest.json")
    audit_rows = read_csv_rows(args.report_dir / "audit" / "vn30_hourly_coverage_audit.csv")

    table_specs = [
        (
            "table1_vn30_universe_hourly_data_scope",
            "Table 1: VN30 universe and hourly data scope",
            table1_rows(args.report_dir),
            [
                "ticker",
                "index",
                "frequency",
                "first_available_hourly_timestamp",
                "last_available_hourly_timestamp",
                "hourly_rows",
                "benchmark_usable",
                "missing_reason",
            ],
            "Frozen VN30 ticker scope and actual local hourly data coverage.",
        ),
        (
            "table2_model_and_baseline_list",
            "Table 2: Model and baseline list",
            table2_rows(run_config),
            ["type", "name", "frequency", "target"],
            "Model and baseline list configured for hourly directional classification.",
        ),
        (
            "table3_hourly_global_benchmark_results",
            "Table 3: Hourly global benchmark results",
            table3_rows(args.artifact_dir),
            [
                "frequency",
                "status",
                "benchmark_run",
                "n_predictions",
                "overall_accuracy",
                "passed_60pct",
                "evaluated_tickers",
                "benchmark_usable_ticker_count",
            ],
            "Global hourly benchmark summary. Empty or stopped status means no empirical claim is supported.",
        ),
        (
            "table4_baseline_delta_summary",
            "Table 4: Baseline delta summary",
            top_csv_rows(args.artifact_dir / "hourly" / "baseline_delta_summary.csv", sort_col="accuracy_delta"),
            [
                "frequency",
                "model",
                "horizon",
                "baseline",
                "model_accuracy",
                "baseline_accuracy",
                "accuracy_delta",
                "model_n_obs",
                "baseline_n_obs",
                "model_better_than_baseline",
            ],
            "Hourly model-versus-baseline directional-accuracy deltas.",
        ),
        (
            "table5_confidence_filtered_diagnostics",
            "Table 5: Confidence-filtered diagnostics",
            confidence_rows(args.report_dir),
            [
                "frequency",
                "model",
                "horizon",
                "threshold",
                "evaluated_rows",
                "coverage_ratio",
                "filtered_accuracy",
                "passed_60pct",
                "ticker_count",
                "ticker_concentration_warning",
            ],
            "Confidence-filtered diagnostics from official hourly predictions.",
        ),
        (
            "table6_regime_specific_diagnostics",
            "Table 6: Regime-specific diagnostics",
            top_csv_rows(args.artifact_dir / "hourly" / "regime_accuracy_summary.csv", sort_col="accuracy"),
            ["frequency", "regime", "model", "horizon", "n_obs", "accuracy", "passed_60pct", "reliable"],
            "Post-hoc regime diagnostics from hourly benchmark artifacts, if available.",
        ),
        (
            "table7_exante_regime_diagnostics",
            "Table 7: Ex-ante regime diagnostics",
            top_csv_rows(args.report_dir / "regime" / "vn30_hourly_exante_regime_accuracy_summary.csv", sort_col="accuracy"),
            [
                "regime_source",
                "frequency",
                "model",
                "horizon",
                "regime",
                "ticker",
                "observation_count",
                "accuracy",
                "reliability",
                "passed_60pct",
                "passed_63pct",
            ],
            "Ex-ante proxy regime diagnostics using prior information only.",
        ),
        (
            "table8_statistical_significance_summary",
            "Table 8: Statistical significance summary",
            top_csv_rows(args.artifact_dir / "hourly" / "significance_summary.csv", sort_col="accuracy"),
            [
                "frequency",
                "model",
                "horizon",
                "n_obs",
                "accuracy",
                "null_accuracy",
                "binomial_p_value",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "significant_at_5pct",
                "significant_at_10pct",
            ],
            "Hourly statistical significance diagnostics, if benchmark predictions exist.",
        ),
        (
            "table9_cost_slippage_proxy_diagnostics",
            "Table 9: Cost/slippage proxy diagnostics",
            top_csv_rows(args.report_dir / "cost_slippage" / "vn30_hourly_cost_slippage_summary.csv", sort_col="net_return"),
            [
                "slice_name",
                "candidate_source",
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
                "average_trade_return",
                "exposure",
                "benchmark_comparison",
                "status",
            ],
            "Proxy cost/slippage diagnostics only; not live-trading evidence.",
        ),
        (
            "table10_limitation_and_robustness_matrix",
            "Table 10: Limitation and robustness matrix",
            limitation_rows(audit_rows, manifest),
            ["area", "status", "evidence", "claim_boundary"],
            "Claim boundaries for the VN30 hourly rerun.",
        ),
    ]
    for key, title, rows, headers, note in table_specs:
        write_table(table_dir, key, title, rows, headers, note)

    figure_status = build_figures(args.report_dir, args.artifact_dir, figure_dir)
    write_notes(note_dir, args.report_dir, args.artifact_dir, figure_status)
    print(
        "VN30 hourly paper artifact pack complete: "
        f"tables={rel(table_dir)} figures={rel(figure_dir)} notes={rel(note_dir)} tickers={len(VN30_TICKERS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
