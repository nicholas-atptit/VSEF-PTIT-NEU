"""Build paper tables, figures, notes, and paper draft for vnstock VN30 hourly track."""

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
    EVAL_END_TEXT,
    EVAL_START_TEXT,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START_TEXT,
    VN30_TICKERS,
    markdown_table,
    read_json,
    rel,
    save_placeholder_figure,
    write_csv,
)
from scripts.research.vn30_hourly_vnstock_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    DOCX_NOTES_PATH,
    FULL_REPORT_ROOT,
    MISSING_EVIDENCE_PATH,
    PAPER_PATH,
    build_docx_notes,
    read_validation_rows,
    validation_gate_passed,
    write_missing_evidence_report,
)


PAPER_TABLE_DIR = FULL_REPORT_ROOT / "paper_tables"
PAPER_FIGURE_DIR = FULL_REPORT_ROOT / "paper_figures"
PAPER_NOTE_DIR = FULL_REPORT_ROOT / "paper_notes"
CONFIDENCE_DIR = FULL_REPORT_ROOT / "confidence"
REGIME_DIR = FULL_REPORT_ROOT / "regime"
COST_SLIPPAGE_DIR = FULL_REPORT_ROOT / "cost_slippage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN30 hourly vnstock paper artifact pack.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    return parser.parse_args()


def benchmark_outputs_exist(artifact_dir: Path) -> bool:
    predictions = artifact_dir / "hourly" / "predicted_vs_actual.csv"
    summary = artifact_dir / "hourly" / "benchmark_summary.json"
    return predictions.exists() and predictions.stat().st_size > 0 and summary.exists()


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


def write_table(table_dir: Path, key: str, title: str, rows: list[dict[str, Any]], headers: list[str], note: str) -> None:
    if not rows:
        rows = [{"status": "missing", "note": note}]
        headers = ["status", "note"]
    write_csv(table_dir / f"{key}.csv", rows, fieldnames=headers)
    (table_dir / f"{key}.md").write_text(
        "\n".join([f"# {title}", "", markdown_table(headers, rows, max_rows=80), "", "## Note", "", note, ""]),
        encoding="utf-8",
    )


def copy_or_placeholder(src: Path, dst: Path, title: str, message: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.stat().st_size > 0:
        shutil.copyfile(src, dst)
        return "copied"
    return save_placeholder_figure(dst, title, message)


def plot_accuracy(src: Path, dst: Path) -> str:
    if not src.exists():
        return save_placeholder_figure(dst, "VN30 hourly vnstock accuracy by model/horizon", "Benchmark accuracy rows are missing.")
    frame = pd.read_csv(src)
    if frame.empty:
        return save_placeholder_figure(dst, "VN30 hourly vnstock accuracy by model/horizon", "Benchmark accuracy rows are empty.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        working = frame.copy()
        working["n_obs"] = pd.to_numeric(working["n_obs"], errors="coerce").fillna(0)
        working["accuracy"] = pd.to_numeric(working["accuracy"], errors="coerce")
        rows = []
        for (model, horizon), group in working.groupby(["model", "horizon"], sort=True):
            n_obs = int(group["n_obs"].sum())
            accuracy = float((group["accuracy"] * group["n_obs"]).sum() / n_obs) if n_obs else 0.0
            rows.append({"model": model, "horizon": int(horizon), "n_obs": n_obs, "accuracy": accuracy})
        grouped = pd.DataFrame(rows).sort_values("accuracy", ascending=False).head(16)
        labels = [f"{row.model} h={int(row.horizon)}" for row in grouped.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, grouped["accuracy"], color="#356f8c")
        ax.axvline(0.60, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("VN30 hourly vnstock accuracy by model/horizon")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(dst, "VN30 hourly vnstock accuracy by model/horizon", f"Plot rendering failed: {exc}")


def write_figures(artifact_dir: Path) -> None:
    save_placeholder_figure(
        PAPER_FIGURE_DIR / "figure_1_research_pipeline.png",
        "VN30 hourly vnstock research pipeline",
        "vnstock fetch -> validation gate -> full VN30 hourly benchmark -> diagnostics -> paper artifacts.",
    )
    save_placeholder_figure(
        PAPER_FIGURE_DIR / "figure_2_walk_forward_design.png",
        "VN30 hourly vnstock walk-forward design",
        f"Train/history {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}; evaluation {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
    )
    plot_accuracy(artifact_dir / "hourly" / "accuracy_summary.csv", PAPER_FIGURE_DIR / "figure_3_hourly_accuracy_by_model_horizon.png")
    copy_or_placeholder(
        CONFIDENCE_DIR / "vn30_hourly_vnstock_confidence_threshold_coverage_accuracy.png",
        PAPER_FIGURE_DIR / "figure_4_confidence_threshold_coverage_accuracy.png",
        "VN30 hourly vnstock confidence threshold vs coverage/accuracy",
        "Confidence sweep figure is missing.",
    )
    copy_or_placeholder(
        artifact_dir / "hourly" / "regime_accuracy_summary.png",
        PAPER_FIGURE_DIR / "figure_5_regime_specific_accuracy.png",
        "VN30 hourly vnstock regime-specific accuracy",
        "Regime summary figure is not available from the benchmark artifact.",
    )
    copy_or_placeholder(
        REGIME_DIR / "vn30_hourly_vnstock_regime_accuracy.png",
        PAPER_FIGURE_DIR / "figure_6_exante_regime_accuracy.png",
        "VN30 hourly vnstock ex-ante regime accuracy",
        "Ex-ante regime figure is missing.",
    )
    copy_or_placeholder(
        COST_SLIPPAGE_DIR / "vn30_hourly_vnstock_equity_curve.png",
        PAPER_FIGURE_DIR / "figure_7_cost_slippage_equity_curve.png",
        "VN30 hourly vnstock cost/slippage equity curve",
        "Cost/slippage equity curve is missing.",
    )


def write_tables(artifact_dir: Path, validation_rows: list[dict[str, Any]]) -> None:
    run_config = read_json(artifact_dir / "run_config.json")
    manifest = read_json(artifact_dir / "manifest.json")
    benchmark_summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    selected_rows = [
        {
            "ticker": ticker,
            "status": "selected",
            "reason": "frozen_vn30_full_design",
        }
        for ticker in VN30_TICKERS
    ]
    model_rows = [{"type": "model", "name": model, "frequency": "hourly", "target": "direction"} for model in run_config.get("models", [])]
    model_rows.extend(
        {"type": "baseline", "name": baseline, "frequency": "hourly", "target": "direction"}
        for baseline in ["buy-and-hold", "flat/no-trade", "always-up", "moving-average signal", "previous-direction signal"]
    )
    write_table(PAPER_TABLE_DIR, "table_1_vn30_hourly_data_scope", "Table 1: VN30 hourly fetched data scope", validation_rows, [
        "symbol",
        "asset_type",
        "gate_required",
        "row_count",
        "first_datetime",
        "last_datetime",
        "benchmark_usable",
        "failure_reason",
    ], "Scope is the vnstock/vnstock_data fetched normalized cache.")
    write_table(PAPER_TABLE_DIR, "table_2_selected_and_excluded_tickers", "Table 2: Selected and excluded tickers", selected_rows, [
        "ticker",
        "status",
        "reason",
    ], "All 30 frozen VN30 tickers are selected only when validation and benchmark gates pass.")
    write_table(PAPER_TABLE_DIR, "table_3_model_and_baseline_list", "Table 3: Model and baseline list", model_rows, [
        "type",
        "name",
        "frequency",
        "target",
    ], "Models and baselines are read from the benchmark run config.")
    write_table(PAPER_TABLE_DIR, "table_4_hourly_global_benchmark_results", "Table 4: Hourly global benchmark results", [benchmark_summary] if benchmark_summary else [], [
        "status",
        "benchmark_run",
        "n_predictions",
        "overall_accuracy",
        "passed",
    ], "Global benchmark summary.")
    write_table(PAPER_TABLE_DIR, "table_5_baseline_delta_summary", "Table 5: Baseline delta summary", top_csv_rows(artifact_dir / "hourly" / "baseline_delta_summary.csv"), [
        "frequency",
        "model",
        "horizon",
        "baseline",
        "model_accuracy",
        "baseline_accuracy",
        "accuracy_delta",
    ], "Baseline delta rows.")
    write_table(PAPER_TABLE_DIR, "table_6_confidence_filtered_diagnostics", "Table 6: Confidence-filtered diagnostics", top_csv_rows(CONFIDENCE_DIR / "vn30_hourly_vnstock_confidence_sweep_summary.csv", "filtered_accuracy"), [
        "model",
        "horizon",
        "threshold",
        "coverage_ratio",
        "filtered_accuracy",
        "passed_60pct",
    ], "Confidence filtered diagnostics.")
    write_table(PAPER_TABLE_DIR, "table_7_regime_diagnostics", "Table 7: Regime diagnostics", top_csv_rows(artifact_dir / "hourly" / "regime_accuracy_summary.csv", "accuracy"), [
        "frequency",
        "model",
        "horizon",
        "regime",
        "n_obs",
        "accuracy",
    ], "Post-hoc regime diagnostics from benchmark artifacts.")
    write_table(PAPER_TABLE_DIR, "table_8_exante_regime_diagnostics", "Table 8: Ex-ante regime diagnostics", top_csv_rows(REGIME_DIR / "vn30_hourly_vnstock_exante_regime_accuracy_summary.csv", "accuracy"), [
        "regime_source",
        "model",
        "horizon",
        "regime",
        "ticker",
        "observation_count",
        "accuracy",
    ], "Ex-ante regime diagnostics.")
    write_table(PAPER_TABLE_DIR, "table_9_statistical_significance", "Table 9: Statistical significance", top_csv_rows(artifact_dir / "hourly" / "significance_summary.csv"), [
        "frequency",
        "model",
        "horizon",
        "n_obs",
        "accuracy",
        "p_value",
    ], "Statistical significance diagnostics.")
    write_table(PAPER_TABLE_DIR, "table_10_cost_slippage_proxy", "Table 10: Cost/slippage proxy diagnostics", top_csv_rows(COST_SLIPPAGE_DIR / "vn30_hourly_vnstock_cost_slippage_summary.csv", "net_return"), [
        "slice_name",
        "baseline",
        "transaction_cost_bps",
        "slippage_bps",
        "row_count",
        "net_return",
        "trade_count",
    ], "Cost/slippage proxy diagnostics.")
    limitations = [
        {"area": "VN30 coverage", "status": "full_30_if_gate_passed", "claim_boundary": "All 30 frozen VN30 stocks only if validation passed."},
        {"area": "Frequency", "status": "hourly_only", "claim_boundary": "No daily evidence or resampling claims."},
        {"area": "Data source", "status": "vnstock_fetched_cache", "claim_boundary": "Provider limitations must be disclosed."},
        {"area": "Index context", "status": manifest.get("vnindex_benchmark_usable", ""), "claim_boundary": "VNINDEX is market context, not stock target labels."},
        {"area": "Trading readiness", "status": "not_claimed", "claim_boundary": "Cost/slippage diagnostics are proxies only."},
    ]
    write_table(PAPER_TABLE_DIR, "table_11_limitation_claim_boundary", "Table 11: Limitation and claim boundary matrix", limitations, [
        "area",
        "status",
        "claim_boundary",
    ], "Claim boundaries for the vnstock full-history track.")


def write_paper(artifact_dir: Path, validation_rows: list[dict[str, Any]]) -> None:
    summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    manifest = read_json(artifact_dir / "manifest.json")
    content = [
        "# NCKH Full Paper Draft: VN30 Hourly vnstock 2005-2026",
        "",
        "## Abstract",
        "",
        "This paper evaluates all 30 frozen VN30 constituents using hourly data fetched from vnstock/vnstock_data. "
        f"The training/history period is {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}, and the evaluation/comparison period is {EVAL_START_TEXT} to {EVAL_END_TEXT}. "
        "The study uses hourly evidence only, excludes old VN100 evidence, and does not use daily or resampled data.",
        "",
        "## Methods",
        "",
        "- Universe: all 30 frozen VN30 tickers.",
        "- Target: stock direction/classification.",
        "- Leakage rule: target_timestamp <= train_cutoff.",
        "- VNINDEX is included as validated market context if supported by the existing feature pipeline; it is not a stock target label.",
        "- VN30INDEX and VNXALL are optional context indices; exact-code support status is reported in the manifest.",
        "",
        "## Results",
        "",
        f"- Benchmark status: {summary.get('status', '')}.",
        f"- Predictions: {summary.get('n_predictions', '')}.",
        f"- Overall accuracy: {summary.get('overall_accuracy', '')}.",
        f"- VNINDEX benchmark-usable: {manifest.get('vnindex_benchmark_usable', '')}.",
        f"- VN30INDEX supported: {manifest.get('vn30index_supported', '')}.",
        f"- VNXALL supported: {manifest.get('vnxall_supported', '')}.",
        "",
        "## Tables and Figures",
        "",
        f"- Tables: `{rel(PAPER_TABLE_DIR)}`.",
        f"- Figures: `{rel(PAPER_FIGURE_DIR)}`.",
        "",
        "## Claim Boundaries",
        "",
        "- The paper does not claim trading readiness; cost/slippage outputs are proxy diagnostics.",
        "- The paper does not use daily evidence or old VN100 evidence.",
        "- No hourly or index data is fabricated.",
        "",
        "## Validation Rows",
        "",
        markdown_table(
            ["symbol", "asset_type", "row_count", "first_datetime", "last_datetime", "benchmark_usable", "failure_reason"],
            validation_rows,
            max_rows=80,
        ),
        "",
    ]
    PAPER_PATH.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validation_rows = read_validation_rows()
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_NOTE_DIR.mkdir(parents=True, exist_ok=True)
    if not benchmark_outputs_exist(args.artifact_dir) or not validation_gate_passed(validation_rows):
        write_missing_evidence_report(MISSING_EVIDENCE_PATH, validation_rows, source_script=Path(__file__).name)
        build_docx_notes(paper_exists=False, validation_rows=validation_rows)
        (PAPER_NOTE_DIR / "vn30_hourly_vnstock_paper_not_written.md").write_text(
            "The full VN30 hourly vnstock paper was not written because benchmark outputs or validation gate evidence are missing.\n",
            encoding="utf-8",
        )
        print(f"VN30 hourly vnstock paper artifact pack stopped: missing evidence report={rel(MISSING_EVIDENCE_PATH)}")
        return 2

    write_tables(args.artifact_dir, validation_rows)
    write_figures(args.artifact_dir)
    write_paper(args.artifact_dir, validation_rows)
    build_docx_notes(paper_exists=True, validation_rows=validation_rows)
    (PAPER_NOTE_DIR / "vn30_hourly_vnstock_docx_notes_path.txt").write_text(f"{rel(DOCX_NOTES_PATH)}\n", encoding="utf-8")
    print(f"VN30 hourly vnstock paper artifact pack complete: paper={rel(PAPER_PATH)} tables={rel(PAPER_TABLE_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
