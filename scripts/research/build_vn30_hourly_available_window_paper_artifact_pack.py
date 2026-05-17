"""Build tables, figures, notes, and paper draft for VN30 hourly available-window research."""

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

from scripts.research.vn30_hourly_available_window_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    CONFIDENCE_DIR,
    COST_SLIPPAGE_DIR,
    DESIGN_DECISION_JSON,
    PAPER_FIGURE_DIR,
    PAPER_NOTE_DIR,
    PAPER_TABLE_DIR,
    REGIME_DIR,
    REPORT_ROOT,
    excluded_tickers,
    final_paper_can_proceed,
    load_design_decision,
    markdown_table,
    read_csv_rows,
    rel,
    save_placeholder_figure,
    selected_tickers,
    write_csv,
    write_docx_build_notes,
)
from scripts.research.vn30_hourly_common import read_json  # noqa: E402


PAPER_PATH = REPO_ROOT / "reports" / "NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_AVAILABLE_WINDOW_V1_WITH_FIGURES.md"
DOCX_NOTES_PATH = REPO_ROOT / "reports" / "NCKH_VN30_HOURLY_AVAILABLE_WINDOW_DOCX_BUILD_NOTES.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN30 hourly available-window paper artifact pack.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--design-json", type=Path, default=DESIGN_DECISION_JSON)
    return parser.parse_args()


def write_table(table_dir: Path, key: str, title: str, rows: list[dict[str, Any]], headers: list[str], note: str) -> None:
    if not rows:
        rows = [{"status": "missing", "note": note}]
        headers = ["status", "note"]
    csv_path = table_dir / f"{key}.csv"
    md_path = table_dir / f"{key}.md"
    write_csv(csv_path, rows, fieldnames=headers)
    md_path.write_text(
        "\n".join([f"# {title}", "", markdown_table(headers, rows), "", "## Note", "", note, ""]),
        encoding="utf-8",
    )


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


def table1_rows(report_dir: Path, decision: dict[str, Any]) -> list[dict[str, Any]]:
    audit_rows = read_csv_rows(report_dir / "audit" / "vn30_hourly_available_window_audit.csv")
    selected = set(selected_tickers(decision))
    return [
        {
            "ticker": row.get("ticker", ""),
            "frequency": "hourly",
            "selected": str(row.get("ticker", "") in selected).lower(),
            "first_available_hourly_timestamp": row.get("first_available_hourly_timestamp", ""),
            "last_available_hourly_timestamp": row.get("last_available_hourly_timestamp", ""),
            "hourly_rows": row.get("hourly_rows", ""),
        }
        for row in audit_rows
    ]


def table2_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "ticker": ticker,
            "status": "selected",
            "reason": "included_in_available_window_design",
        }
        for ticker in selected_tickers(decision)
    ]
    rows.extend(
        {
            "ticker": ticker,
            "status": "excluded",
            "reason": decision.get("exclusion_reasons", {}).get(ticker, "excluded_by_available_window_design"),
        }
        for ticker in excluded_tickers(decision)
    )
    return rows


def table3_rows(run_config: dict[str, Any]) -> list[dict[str, Any]]:
    models = run_config.get("models") or ["xgboost", "lightgbm", "random_forest", "stacking"]
    baselines = ["buy_and_hold", "flat_no_trade", "always_up", "moving_average_signal", "previous_direction_signal"]
    rows = [{"type": "model", "name": model, "frequency": "hourly", "target": "direction"} for model in models]
    rows.extend({"type": "baseline", "name": baseline, "frequency": "hourly", "target": "direction"} for baseline in baselines)
    return rows


def table4_rows(artifact_dir: Path, decision: dict[str, Any]) -> list[dict[str, Any]]:
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
            "selected_ticker_count": len(selected_tickers(decision)),
            "full_vn30_representativeness": decision.get("full_vn30_representativeness", False),
        }
    ]


def confidence_rows() -> list[dict[str, Any]]:
    path = CONFIDENCE_DIR / "vn30_available_window_confidence_sweep_summary.csv"
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


def limitation_rows(decision: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected_count = len(selected_tickers(decision))
    return [
        {
            "area": "Full-history requirement",
            "status": "not_satisfied",
            "evidence": "External hourly 2005-2026 data is still required.",
            "claim_boundary": "Do not claim 2005-2026 full-history VN30 evidence.",
        },
        {
            "area": "Constituent coverage",
            "status": "subset" if selected_count < 30 else "full_30_available_window",
            "evidence": f"{selected_count}/30 tickers selected.",
            "claim_boundary": "No full VN30 representativeness claim unless selected tickers are 30/30.",
        },
        {
            "area": "Frequency",
            "status": "hourly_only",
            "evidence": "Local hourly files only.",
            "claim_boundary": "Daily data and daily-to-hourly resampling are excluded.",
        },
        {
            "area": "Leakage",
            "status": "declared",
            "evidence": manifest.get("training_label_cutoff_rule", "target_timestamp <= train_cutoff"),
            "claim_boundary": "Training labels are bounded by the selected train cutoff.",
        },
        {
            "area": "Trading readiness",
            "status": "not_claimed",
            "evidence": "Cost/slippage outputs are proxy diagnostics.",
            "claim_boundary": "No trading readiness claim without execution-ready evidence.",
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
        return save_placeholder_figure(dst, "VN30 hourly available-window accuracy by model/horizon", "Benchmark accuracy rows are missing.")
    frame = pd.read_csv(src)
    if frame.empty:
        return save_placeholder_figure(dst, "VN30 hourly available-window accuracy by model/horizon", "Benchmark accuracy rows are empty.")
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
        ax.set_title("VN30 hourly available-window accuracy by model/horizon")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(dst, "VN30 hourly available-window accuracy by model/horizon", f"Plot rendering failed: {exc}")


def plot_regime_accuracy(src: Path, dst: Path) -> str:
    if not src.exists():
        return save_placeholder_figure(dst, "VN30 hourly available-window regime-specific accuracy", "Regime accuracy rows are missing.")
    frame = pd.read_csv(src)
    if frame.empty:
        return save_placeholder_figure(dst, "VN30 hourly available-window regime-specific accuracy", "Regime accuracy rows are empty.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        working = frame.copy()
        working["n_obs"] = pd.to_numeric(working["n_obs"], errors="coerce").fillna(0)
        working["accuracy"] = pd.to_numeric(working["accuracy"], errors="coerce")
        rows = []
        for (model, horizon, regime), group in working.groupby(["model", "horizon", "regime"], sort=True):
            n_obs = int(group["n_obs"].sum())
            accuracy = float((group["accuracy"] * group["n_obs"]).sum() / n_obs) if n_obs else 0.0
            rows.append({"model": model, "horizon": int(horizon), "regime": regime, "n_obs": n_obs, "accuracy": accuracy})
        grouped = pd.DataFrame(rows).sort_values(["accuracy", "n_obs"], ascending=[False, False]).head(14)
        labels = [f"{row.model} h={int(row.horizon)} {row.regime}" for row in grouped.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, grouped["accuracy"], color="#4d755f")
        ax.axvline(0.60, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("VN30 hourly available-window regime-specific accuracy")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(dst, "VN30 hourly available-window regime-specific accuracy", f"Plot rendering failed: {exc}")


def build_figures(artifact_dir: Path, figure_dir: Path, decision: dict[str, Any]) -> list[dict[str, str]]:
    figures = [
        {
            "figure": "figure1_research_pipeline.png",
            "status": save_placeholder_figure(
                figure_dir / "figure1_research_pipeline.png",
                "VN30 hourly available-window research pipeline",
                "Local hourly audit -> design gate -> benchmark -> diagnostics -> paper artifacts.",
            ),
        },
        {
            "figure": "figure2_available_window_walk_forward_design.png",
            "status": save_placeholder_figure(
                figure_dir / "figure2_available_window_walk_forward_design.png",
                "Available-window walk-forward design",
                f"Train: {decision.get('training_start')} to {decision.get('training_cutoff')}; eval: {decision.get('evaluation_start')} to {decision.get('evaluation_end')}.",
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
            "figure": "figure4_confidence_threshold_vs_coverage_accuracy.png",
            "status": copy_or_placeholder(
                CONFIDENCE_DIR / "vn30_available_window_confidence_threshold_coverage_accuracy.png",
                figure_dir / "figure4_confidence_threshold_vs_coverage_accuracy.png",
                "VN30 hourly available-window confidence threshold vs coverage/accuracy",
                "Confidence sweep figure is unavailable.",
            ),
        },
        {
            "figure": "figure5_regime_specific_accuracy.png",
            "status": plot_regime_accuracy(
                artifact_dir / "hourly" / "regime_accuracy_summary.csv",
                figure_dir / "figure5_regime_specific_accuracy.png",
            ),
        },
        {
            "figure": "figure6_exante_regime_accuracy.png",
            "status": copy_or_placeholder(
                REGIME_DIR / "vn30_available_window_regime_accuracy.png",
                figure_dir / "figure6_exante_regime_accuracy.png",
                "VN30 hourly available-window ex-ante regime accuracy",
                "Ex-ante regime figure is unavailable.",
            ),
        },
        {
            "figure": "figure7_cost_slippage_equity_curve.png",
            "status": copy_or_placeholder(
                COST_SLIPPAGE_DIR / "vn30_available_window_equity_curve.png",
                figure_dir / "figure7_cost_slippage_equity_curve.png",
                "VN30 hourly available-window cost/slippage equity curve",
                "Cost/slippage equity curve is unavailable.",
            ),
        },
    ]
    return figures


def write_notes(note_dir: Path, artifact_dir: Path, decision: dict[str, Any], figure_status: list[dict[str, str]]) -> None:
    note_dir.mkdir(parents=True, exist_ok=True)
    status = [
        "# VN30 Hourly Available-Window Paper Artifact Status",
        "",
        f"- Artifact directory: `{rel(artifact_dir)}`.",
        f"- Report directory: `{rel(REPORT_ROOT)}`.",
        f"- Selected tickers: {len(selected_tickers(decision))}/30.",
        f"- Training: {decision.get('training_start', '')} to {decision.get('training_cutoff', '')}.",
        f"- Evaluation: {decision.get('evaluation_start', '')} to {decision.get('evaluation_end', '')}.",
        f"- Full VN30 representativeness: {str(bool(decision.get('full_vn30_representativeness'))).lower()}.",
        "- Daily data used: false.",
        "- VN100 evidence reused: false.",
        "",
        "## Figure Status",
        "",
        markdown_table(["figure", "status"], figure_status),
        "",
    ]
    (note_dir / "paper_artifact_status.md").write_text("\n".join(status), encoding="utf-8")
    boundary = [
        "# VN30 Hourly Available-Window Claim Boundary Notes",
        "",
        decision.get("claim_boundary", ""),
        "",
        "This artifact pack does not satisfy the 2005-2026 full-history requirement. External hourly data remains required for that design.",
        "",
    ]
    (note_dir / "claim_boundary_notes.md").write_text("\n".join(boundary), encoding="utf-8")


def write_paper_if_allowed(artifact_dir: Path, decision: dict[str, Any]) -> None:
    if not final_paper_can_proceed(decision):
        return
    summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    selected = selected_tickers(decision)
    excluded = excluded_tickers(decision)
    subset_sentence = (
        "The study is an hourly available-window VN30 subset analysis rather than a full-constituent VN30 historical evaluation."
        if len(selected) < 30
        else "The study uses all 30 frozen VN30 tickers within the available hourly window."
    )
    content = [
        "# NCKH Full Paper Draft: VN30 Hourly Available-Window V1 With Figures",
        "",
        "## Abstract",
        "",
        f"This paper reports a VN30 hourly available-window forecasting study using real local hourly data only. {subset_sentence} "
        "The design does not claim to satisfy the 2005-2026 full-history requirement, does not use daily evidence, and does not reuse old VN100 evidence.",
        "",
        "## 1. Scope and Claim Boundary",
        "",
        "- Study label: VN30 hourly available-window.",
        f"- Selected tickers: {len(selected)}/30.",
        f"- Selected ticker list: {', '.join(selected)}.",
        f"- Excluded ticker list: {', '.join(excluded) if excluded else 'None'}.",
        f"- Full VN30 representativeness: {str(bool(decision.get('full_vn30_representativeness'))).lower()}.",
        "- Trading readiness: not claimed.",
        "- Full-history 2005-2026 VN30 evidence: not claimed.",
        "",
        "## 2. Methods",
        "",
        f"{subset_sentence}",
        "",
        "The benchmark uses hourly OHLCV rows from the local hourly cache. The target is directional classification. "
        "Training labels obey the leakage boundary `target_timestamp <= train_cutoff`.",
        "",
        f"- Actual training period: {decision.get('training_start')} to {decision.get('training_cutoff')}.",
        f"- Actual evaluation period: {decision.get('evaluation_start')} to {decision.get('evaluation_end')}.",
        "- Frequency: hourly only.",
        "",
        "## 3. Data",
        "",
        "The audit found that a full 2005-2026 hourly VN30 design is not supported by local data. The available-window design was selected from the local hourly coverage audit.",
        "",
        "See `reports/generated/vn30_hourly_available_window/audit/vn30_hourly_available_window_audit.md` for per-ticker coverage.",
        "",
        "## 4. Benchmark Results",
        "",
        f"- Prediction rows: {summary.get('n_predictions', '')}.",
        f"- Overall accuracy: {summary.get('overall_accuracy', '')}.",
        f"- Benchmark status: {summary.get('status', '')}.",
        f"- Models executed: {', '.join(summary.get('executed_models', [])) if summary.get('executed_models') else 'None reported'}.",
        "",
        "Detailed benchmark tables are in `reports/generated/vn30_hourly_available_window/paper_tables/`.",
        "",
        "## 5. Diagnostics",
        "",
        "- Confidence-filtered diagnostics are included as diagnostic evidence only.",
        "- Ex-ante regime labels use prior information only.",
        "- Cost/slippage outputs are proxy diagnostics and do not establish live execution readiness.",
        "",
        "## 6. Limitations",
        "",
        "- External hourly data is still required for the original 2005-2026 full design.",
        "- The selected hourly window is short relative to a full historical study.",
        "- VN30 representativeness is not claimed unless the selected design includes all 30 tickers.",
        "- The paper does not use old VN100 evidence.",
        "- The paper does not use daily evidence.",
        "- The paper does not fabricate data.",
        "",
        "## Figures",
        "",
        "![Figure 1: Research pipeline](generated/vn30_hourly_available_window/paper_figures/figure1_research_pipeline.png)",
        "",
        "![Figure 2: Available-window walk-forward design](generated/vn30_hourly_available_window/paper_figures/figure2_available_window_walk_forward_design.png)",
        "",
        "![Figure 3: Hourly accuracy by model/horizon](generated/vn30_hourly_available_window/paper_figures/figure3_hourly_accuracy_by_model_horizon.png)",
        "",
        "![Figure 4: Confidence threshold vs coverage/accuracy](generated/vn30_hourly_available_window/paper_figures/figure4_confidence_threshold_vs_coverage_accuracy.png)",
        "",
        "![Figure 5: Regime-specific accuracy](generated/vn30_hourly_available_window/paper_figures/figure5_regime_specific_accuracy.png)",
        "",
        "![Figure 6: Ex-ante regime accuracy](generated/vn30_hourly_available_window/paper_figures/figure6_exante_regime_accuracy.png)",
        "",
        "![Figure 7: Cost/slippage equity curve](generated/vn30_hourly_available_window/paper_figures/figure7_cost_slippage_equity_curve.png)",
        "",
    ]
    PAPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PATH.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    decision = load_design_decision(args.design_json)
    table_dir = PAPER_TABLE_DIR
    figure_dir = PAPER_FIGURE_DIR
    note_dir = PAPER_NOTE_DIR
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)

    run_config = read_json(args.artifact_dir / "run_config.json")
    manifest = read_json(args.artifact_dir / "manifest.json")

    table_specs = [
        (
            "table1_available_window_vn30_hourly_data_scope",
            "Table 1: Available-window VN30 hourly data scope",
            table1_rows(args.report_dir, decision),
            ["ticker", "frequency", "selected", "first_available_hourly_timestamp", "last_available_hourly_timestamp", "hourly_rows"],
            "Frozen VN30 local hourly data scope and selected available-window coverage.",
        ),
        (
            "table2_selected_and_excluded_tickers",
            "Table 2: Selected and excluded tickers",
            table2_rows(decision),
            ["ticker", "status", "reason"],
            "Selected tickers are used in the available-window benchmark; excluded tickers start too late for the selected split.",
        ),
        (
            "table3_model_and_baseline_list",
            "Table 3: Model and baseline list",
            table3_rows(run_config),
            ["type", "name", "frequency", "target"],
            "Model and baseline list for hourly directional classification.",
        ),
        (
            "table4_hourly_global_benchmark_results",
            "Table 4: Hourly global benchmark results",
            table4_rows(args.artifact_dir, decision),
            ["frequency", "status", "benchmark_run", "n_predictions", "overall_accuracy", "passed_60pct", "selected_ticker_count", "full_vn30_representativeness"],
            "Global available-window hourly benchmark results.",
        ),
        (
            "table5_baseline_delta_summary",
            "Table 5: Baseline delta summary",
            top_csv_rows(args.artifact_dir / "hourly" / "baseline_delta_summary.csv", sort_col="accuracy_delta"),
            ["frequency", "model", "horizon", "baseline", "model_accuracy", "baseline_accuracy", "accuracy_delta", "model_n_obs", "baseline_n_obs", "model_better_than_baseline"],
            "Hourly model-versus-baseline directional-accuracy deltas.",
        ),
        (
            "table6_confidence_filtered_diagnostics",
            "Table 6: Confidence-filtered diagnostics",
            confidence_rows(),
            ["frequency", "model", "horizon", "threshold", "evaluated_rows", "coverage_ratio", "filtered_accuracy", "passed_60pct", "ticker_count", "ticker_concentration_warning"],
            "Confidence-filtered diagnostics from available-window hourly predictions.",
        ),
        (
            "table7_regime_diagnostics",
            "Table 7: Regime diagnostics",
            top_csv_rows(args.artifact_dir / "hourly" / "regime_accuracy_summary.csv", sort_col="accuracy"),
            ["frequency", "regime", "model", "horizon", "n_obs", "accuracy", "passed_60pct", "reliable"],
            "Post-hoc regime diagnostics from benchmark artifacts.",
        ),
        (
            "table8_exante_regime_diagnostics",
            "Table 8: Ex-ante regime diagnostics",
            top_csv_rows(REGIME_DIR / "vn30_available_window_exante_regime_accuracy_summary.csv", sort_col="accuracy"),
            ["regime_source", "frequency", "model", "horizon", "regime", "ticker", "observation_count", "accuracy", "reliability", "passed_60pct", "passed_63pct"],
            "Ex-ante proxy regime diagnostics using prior information only.",
        ),
        (
            "table9_statistical_significance",
            "Table 9: Statistical significance",
            top_csv_rows(args.artifact_dir / "hourly" / "significance_summary.csv", sort_col="accuracy"),
            ["frequency", "model", "horizon", "n_obs", "accuracy", "null_accuracy", "binomial_p_value", "bootstrap_ci_low", "bootstrap_ci_high", "significant_at_5pct", "significant_at_10pct"],
            "Statistical significance diagnostics for available-window hourly predictions.",
        ),
        (
            "table10_cost_slippage_proxy_diagnostics",
            "Table 10: Cost/slippage proxy diagnostics",
            top_csv_rows(COST_SLIPPAGE_DIR / "vn30_available_window_cost_slippage_summary.csv", sort_col="net_return"),
            ["slice_name", "candidate_source", "baseline", "transaction_cost_bps", "slippage_bps", "row_count", "gross_return", "net_return", "turnover", "max_drawdown", "profit_factor", "win_rate", "trade_count", "average_trade_return", "exposure", "benchmark_comparison", "status"],
            "Proxy cost/slippage diagnostics only; not live-trading evidence.",
        ),
        (
            "table11_limitation_and_claim_boundary_matrix",
            "Table 11: Limitation and claim boundary matrix",
            limitation_rows(decision, manifest),
            ["area", "status", "evidence", "claim_boundary"],
            "Claim boundaries for the available-window VN30 hourly study.",
        ),
    ]
    for key, title, rows, headers, note in table_specs:
        write_table(table_dir, key, title, rows, headers, note)

    figure_status = build_figures(args.artifact_dir, figure_dir, decision)
    write_notes(note_dir, args.artifact_dir, decision, figure_status)
    write_paper_if_allowed(args.artifact_dir, decision)
    write_docx_build_notes(DOCX_NOTES_PATH, decision)
    print(
        "VN30 hourly available-window paper artifact pack complete: "
        f"tables={rel(table_dir)} figures={rel(figure_dir)} notes={rel(note_dir)} paper={str(PAPER_PATH.exists()).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
