"""Build paper artifacts for the VN30 hourly listing-aware benchmark."""

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
    EVAL_START_TEXT,
    TRAIN_CUTOFF_TEXT,
    VN30_TICKERS,
    markdown_table,
    read_json,
    rel,
    save_placeholder_figure,
    write_csv,
)
from scripts.research.vn30_hourly_listing_aware_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    DOCX_NOTES_PATH,
    MISSING_EVIDENCE_PATH,
    PAPER_PATH,
    REPORT_ROOT,
    REQUESTED_EVAL_END_TEXT,
    compute_actual_eval_end,
    read_validation_rows,
    validation_gate_passed,
    write_docx_notes,
    write_missing_evidence_report,
)


PAPER_TABLE_DIR = REPORT_ROOT / "paper_tables"
PAPER_FIGURE_DIR = REPORT_ROOT / "paper_figures"
PAPER_NOTE_DIR = REPORT_ROOT / "paper_notes"
CONFIDENCE_DIR = REPORT_ROOT / "confidence"
REGIME_DIR = REPORT_ROOT / "regime"
COST_SLIPPAGE_DIR = REPORT_ROOT / "cost_slippage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN30 hourly listing-aware paper artifact pack.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    return parser.parse_args()


def benchmark_outputs_exist(artifact_dir: Path) -> bool:
    return (artifact_dir / "hourly" / "predicted_vs_actual.csv").exists() and (artifact_dir / "hourly" / "benchmark_summary.json").exists()


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
        return save_placeholder_figure(dst, "VN30 listing-aware accuracy by model/horizon", "Benchmark accuracy rows are missing.")
    frame = pd.read_csv(src)
    if frame.empty:
        return save_placeholder_figure(dst, "VN30 listing-aware accuracy by model/horizon", "Benchmark accuracy rows are empty.")
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
        ax.set_title("VN30 listing-aware accuracy by model/horizon")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(dst, "VN30 listing-aware accuracy by model/horizon", f"Plot rendering failed: {exc}")


def write_tables(artifact_dir: Path, validation_rows: list[dict[str, Any]]) -> None:
    run_config = read_json(artifact_dir / "run_config.json")
    manifest = read_json(artifact_dir / "manifest.json")
    benchmark_summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    selected_rows = [{"ticker": ticker, "status": "selected", "reason": "frozen_vn30_listing_aware_design"} for ticker in VN30_TICKERS]
    model_rows = [{"type": "model", "name": model, "frequency": "hourly", "target": "direction"} for model in run_config.get("models", [])]
    model_rows.extend(
        {"type": "baseline", "name": baseline, "frequency": "hourly", "target": "direction"}
        for baseline in ["buy-and-hold", "flat/no-trade", "always-up", "moving-average signal", "previous-direction signal"]
    )
    write_table(PAPER_TABLE_DIR, "table_1_listing_aware_data_scope", "Table 1: VN30 hourly listing-aware data scope", validation_rows, [
        "symbol",
        "asset_type",
        "listing_date_used",
        "ticker_training_start",
        "row_count",
        "first_datetime",
        "last_datetime",
        "benchmark_usable",
        "missing_reason",
    ], "Each ticker starts from first trading date or first provider-available hourly timestamp, whichever is later.")
    write_table(PAPER_TABLE_DIR, "table_2_selected_and_excluded_tickers", "Table 2: Selected and excluded tickers", selected_rows, [
        "ticker",
        "status",
        "reason",
    ], "No subset fallback is allowed in the listing-aware full VN30 design.")
    write_table(PAPER_TABLE_DIR, "table_3_model_and_baseline_list", "Table 3: Model and baseline list", model_rows, [
        "type",
        "name",
        "frequency",
        "target",
    ], "Models and baselines are read from the benchmark config.")
    write_table(PAPER_TABLE_DIR, "table_4_hourly_global_benchmark_results", "Table 4: Hourly global benchmark results", [benchmark_summary] if benchmark_summary else [], [
        "status",
        "benchmark_run",
        "n_predictions",
        "overall_accuracy",
        "passed",
        "actual_eval_end",
    ], "Global benchmark summary.")
    write_table(PAPER_TABLE_DIR, "table_5_baseline_delta_summary", "Table 5: Baseline delta summary", top_csv_rows(artifact_dir / "hourly" / "baseline_delta_summary.csv"), [
        "frequency",
        "model",
        "horizon",
        "baseline",
        "model_accuracy",
        "baseline_accuracy",
        "accuracy_delta",
    ], "Baseline deltas.")
    write_table(PAPER_TABLE_DIR, "table_6_confidence_filtered_diagnostics", "Table 6: Confidence-filtered diagnostics", top_csv_rows(CONFIDENCE_DIR / "vn30_listing_aware_confidence_sweep_summary.csv", "filtered_accuracy"), [
        "model",
        "horizon",
        "threshold",
        "coverage_ratio",
        "filtered_accuracy",
        "passed_60pct",
    ], "Confidence diagnostics.")
    write_table(PAPER_TABLE_DIR, "table_7_regime_diagnostics", "Table 7: Regime diagnostics", top_csv_rows(artifact_dir / "hourly" / "regime_accuracy_summary.csv", "accuracy"), [
        "frequency",
        "model",
        "horizon",
        "regime",
        "n_obs",
        "accuracy",
    ], "Post-hoc regime diagnostics.")
    write_table(PAPER_TABLE_DIR, "table_8_exante_regime_diagnostics", "Table 8: Ex-ante regime diagnostics", top_csv_rows(REGIME_DIR / "vn30_listing_aware_exante_regime_accuracy_summary.csv", "accuracy"), [
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
    write_table(PAPER_TABLE_DIR, "table_10_cost_slippage_proxy", "Table 10: Cost/slippage proxy diagnostics", top_csv_rows(COST_SLIPPAGE_DIR / "vn30_listing_aware_cost_slippage_summary.csv", "net_return"), [
        "slice_name",
        "baseline",
        "transaction_cost_bps",
        "slippage_bps",
        "row_count",
        "net_return",
        "trade_count",
    ], "Cost/slippage proxy diagnostics.")
    limitations = [
        {"area": "Listing-aware start", "status": "applied", "claim_boundary": "No ticker is forced before first trading or first provider hourly timestamp."},
        {"area": "VN30 coverage", "status": manifest.get("all_30_stocks_usable", ""), "claim_boundary": "No subset fallback."},
        {"area": "Frequency", "status": "hourly_only", "claim_boundary": "No daily evidence or resampling."},
        {"area": "Index context", "status": manifest.get("vnindex_benchmark_usable", ""), "claim_boundary": "VNINDEX is context, not a stock target label."},
        {"area": "Trading readiness", "status": "not_claimed", "claim_boundary": "Cost/slippage diagnostics are proxies only."},
    ]
    write_table(PAPER_TABLE_DIR, "table_11_limitation_claim_boundary", "Table 11: Limitation and claim boundary matrix", limitations, [
        "area",
        "status",
        "claim_boundary",
    ], "Claim boundaries.")


def write_figures(artifact_dir: Path) -> None:
    save_placeholder_figure(
        PAPER_FIGURE_DIR / "figure_1_research_pipeline.png",
        "VN30 listing-aware research pipeline",
        "vnstock fetch -> listing-aware validation gate -> benchmark -> diagnostics -> paper artifacts.",
    )
    save_placeholder_figure(
        PAPER_FIGURE_DIR / "figure_2_listing_aware_walk_forward_design.png",
        "VN30 listing-aware walk-forward design",
        f"Training labels end {TRAIN_CUTOFF_TEXT}; evaluation starts {EVAL_START_TEXT}; actual end is provider-derived.",
    )
    plot_accuracy(artifact_dir / "hourly" / "accuracy_summary.csv", PAPER_FIGURE_DIR / "figure_3_hourly_accuracy_by_model_horizon.png")
    copy_or_placeholder(
        CONFIDENCE_DIR / "vn30_listing_aware_confidence_threshold_coverage_accuracy.png",
        PAPER_FIGURE_DIR / "figure_4_confidence_threshold_coverage_accuracy.png",
        "VN30 listing-aware confidence threshold vs coverage/accuracy",
        "Confidence sweep figure missing.",
    )
    copy_or_placeholder(
        artifact_dir / "hourly" / "regime_accuracy_summary.png",
        PAPER_FIGURE_DIR / "figure_5_regime_specific_accuracy.png",
        "VN30 listing-aware regime-specific accuracy",
        "Regime summary figure not available.",
    )
    copy_or_placeholder(
        REGIME_DIR / "vn30_listing_aware_regime_accuracy.png",
        PAPER_FIGURE_DIR / "figure_6_exante_regime_accuracy.png",
        "VN30 listing-aware ex-ante regime accuracy",
        "Ex-ante regime figure missing.",
    )
    copy_or_placeholder(
        COST_SLIPPAGE_DIR / "vn30_listing_aware_equity_curve.png",
        PAPER_FIGURE_DIR / "figure_7_cost_slippage_equity_curve.png",
        "VN30 listing-aware cost/slippage equity curve",
        "Cost/slippage equity curve missing.",
    )


def write_paper(artifact_dir: Path, validation_rows: list[dict[str, Any]]) -> None:
    summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    manifest = read_json(artifact_dir / "manifest.json")
    actual_eval_end = compute_actual_eval_end(validation_rows)
    content = [
        "# NCKH Full Paper Draft: VN30 Hourly Listing-Aware Benchmark",
        "",
        "## Abstract",
        "",
        "This is a VN30 hourly listing-aware historical benchmark using all 30 frozen VN30 tickers. "
        "Each ticker starts from its first trading date or first provider-available hourly timestamp, whichever is later. "
        "The benchmark does not force pre-listing data, uses hourly data only, and excludes old VN100 evidence and daily-to-hourly resampling.",
        "",
        "## Methods",
        "",
        "- Universe: all 30 frozen VN30 tickers.",
        "- Frequency: hourly only.",
        "- Per-ticker start rule: max(first trading/listing date, first provider-available hourly timestamp).",
        f"- Training labels end at: {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation starts at: {EVAL_START_TEXT}.",
        f"- Evaluation ends at actual_eval_end: {actual_eval_end}.",
        "- Target: stock direction/classification.",
        "- Leakage rule: target_timestamp <= train_cutoff for training labels.",
        "- VNINDEX is used as market context if fetched and validated; VN30INDEX/VNXALL support status is reported.",
        "",
        "## Results",
        "",
        f"- Benchmark status: {summary.get('status', '')}.",
        f"- Predictions: {summary.get('n_predictions', '')}.",
        f"- Overall accuracy: {summary.get('overall_accuracy', '')}.",
        f"- VNINDEX usable: {manifest.get('vnindex_benchmark_usable', '')}.",
        f"- VN30INDEX support: {manifest.get('vn30index_supported', '')}.",
        f"- VNXALL support: {manifest.get('vnxall_supported', '')}.",
        "",
        "## Tables and Figures",
        "",
        f"- Tables: `{rel(PAPER_TABLE_DIR)}`.",
        f"- Figures: `{rel(PAPER_FIGURE_DIR)}`.",
        "",
        "## Claim Boundaries",
        "",
        "- No trading-readiness claim is made without execution-ready evidence.",
        "- No daily data, no daily-to-hourly resampling, no old VN100 evidence, and no fabricated bars are used.",
        f"- Requested evaluation end was {REQUESTED_EVAL_END_TEXT}; actual_eval_end is provider-derived.",
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
        write_docx_notes(paper_exists=False, validation_rows=validation_rows)
        (PAPER_NOTE_DIR / "vn30_hourly_listing_aware_paper_not_written.md").write_text(
            "The listing-aware VN30 hourly paper was not written because benchmark outputs or validation gate evidence are missing.\n",
            encoding="utf-8",
        )
        print(f"VN30 listing-aware paper artifact pack stopped: missing evidence report={rel(MISSING_EVIDENCE_PATH)}")
        return 2
    write_tables(args.artifact_dir, validation_rows)
    write_figures(args.artifact_dir)
    write_paper(args.artifact_dir, validation_rows)
    write_docx_notes(paper_exists=True, validation_rows=validation_rows)
    (PAPER_NOTE_DIR / "vn30_hourly_listing_aware_docx_notes_path.txt").write_text(f"{rel(DOCX_NOTES_PATH)}\n", encoding="utf-8")
    print(f"VN30 listing-aware paper artifact pack complete: paper={rel(PAPER_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
