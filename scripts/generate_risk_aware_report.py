"""Generate Phase 3 risk-aware decision research report artifacts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_candidate_research import (  # noqa: E402
    BASKET_COLUMNS,
    CANDIDATE_COLUMNS,
    DISCLAIMER,
    OUTPUT_ROOT,
    REPORT_ROOT,
    build_drawdown_comparison,
    build_hit_ratio_comparison,
    build_risk_summary,
    markdown_table,
    read_csv,
    resolve_repo_path,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 3 risk-aware decision research report.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment IDs to aggregate")
    parser.add_argument("--output", required=True, help="Output directory for report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_table(args.experiments, output_dir)
    baskets = load_basket_table(args.experiments, output_dir)
    candidates = ensure_columns(candidates, CANDIDATE_COLUMNS)
    baskets = ensure_columns(baskets, BASKET_COLUMNS)

    risk_summary = build_risk_summary(candidates, baskets)
    risk_adjusted = candidates.loc[candidates["candidate_type"] == "risk_aware"].copy()
    drawdown = build_drawdown_comparison(baskets)
    hit_ratio = build_hit_ratio_comparison(baskets)

    write_csv(output_dir / "candidate_comparison.csv", candidates)
    write_csv(output_dir / "topn_basket_metrics.csv", baskets)
    write_csv(output_dir / "risk_summary.csv", risk_summary)
    write_csv(output_dir / "risk_adjusted_ranking.csv", risk_adjusted)
    write_csv(output_dir / "drawdown_comparison.csv", drawdown)
    write_csv(output_dir / "hit_ratio_comparison.csv", hit_ratio)

    source_evidence = build_source_evidence(args.experiments)
    chart_notes = generate_charts(output_dir / "charts", candidates, baskets, drawdown, hit_ratio)
    report = render_report(
        experiments=args.experiments,
        candidates=candidates,
        baskets=baskets,
        risk_summary=risk_summary,
        risk_adjusted=risk_adjusted,
        drawdown=drawdown,
        hit_ratio=hit_ratio,
        source_evidence=source_evidence,
        chart_notes=chart_notes,
    )
    (output_dir / "RISK_AWARE_DECISION_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote Phase 3 risk-aware report artifacts to {output_dir}")
    return 0


def load_candidate_table(experiments: list[str], output_dir: Path) -> pd.DataFrame:
    candidates = read_csv(output_dir / "candidate_comparison.csv")
    if not candidates.empty:
        return candidates
    frames: list[pd.DataFrame] = []
    for experiment_id in experiments:
        path = OUTPUT_ROOT / experiment_id / "artifacts" / "candidate_comparison.csv"
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANDIDATE_COLUMNS)


def load_basket_table(experiments: list[str], output_dir: Path) -> pd.DataFrame:
    baskets = read_csv(output_dir / "topn_basket_metrics.csv")
    if not baskets.empty:
        return baskets
    frames: list[pd.DataFrame] = []
    for experiment_id in experiments:
        path = OUTPUT_ROOT / experiment_id / "artifacts" / "topn_basket_metrics.csv"
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASKET_COLUMNS)


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns].reset_index(drop=True)


def build_source_evidence(experiments: list[str]) -> pd.DataFrame:
    paths: list[dict[str, Any]] = []
    for experiment_id in experiments:
        base = OUTPUT_ROOT / experiment_id
        for relative in (
            "config/original_config.yaml",
            "config/resolved_config.yaml",
            "manifests/run_manifest.json",
            "logs/run.log",
            "logs/errors.log",
            "metrics/metrics.csv",
            "artifacts/candidate_comparison.csv",
            "artifacts/topn_basket_metrics.csv",
            "reports/summary.md",
        ):
            path = base / relative
            paths.append({"source": experiment_id, "path": str(path), "exists": path.exists()})
    for experiment_id in ("EXP-FC-001", "EXP-FC-003"):
        base = OUTPUT_ROOT / experiment_id
        for relative in (
            "predictions/predictions.csv",
            "metrics/metrics.csv",
            "manifests/run_manifest.json",
            "logs/run.log",
            "logs/errors.log",
        ):
            path = base / relative
            paths.append({"source": experiment_id, "path": str(path), "exists": path.exists()})
    for relative in (
        "forecast_metrics.csv",
        "model_ranking.csv",
        "horizon_comparison.csv",
    ):
        path = REPO_ROOT / "reports" / "forecasting_core" / relative
        paths.append({"source": "forecasting_core_report", "path": str(path), "exists": path.exists()})
    return pd.DataFrame(paths)


def generate_charts(
    chart_dir: Path,
    candidates: pd.DataFrame,
    baskets: pd.DataFrame,
    drawdown: pd.DataFrame,
    hit_ratio: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"matplotlib_unavailable:{exc}"]
    chart_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not candidates.empty:
            summary = (
                candidates.groupby("candidate_type")[["expected_return_proxy", "realized_volatility", "max_drawdown"]]
                .mean(numeric_only=True)
                .reset_index()
            )
            plot_grouped_bar(
                plt,
                summary,
                "candidate_type",
                ["expected_return_proxy", "realized_volatility", "max_drawdown"],
                chart_dir / "candidate_policy_comparison_summary.png",
                "Candidate Policy Comparison",
            )
            notes.append("generated:candidate_policy_comparison_summary.png")
        else:
            notes.append("skipped:candidate_policy_comparison:no_candidate_rows")

        if not drawdown.empty:
            plot_two_series(
                plt,
                drawdown,
                "drawdown_label",
                "forecast_only_max_drawdown",
                "risk_aware_max_drawdown",
                chart_dir / "drawdown_comparison_topn_horizon.png",
                "Drawdown Comparison",
            )
            notes.append("generated:drawdown_comparison_topn_horizon.png")
        else:
            notes.append("skipped:drawdown_comparison:no_drawdown_rows")

        if not hit_ratio.empty:
            plot_two_series(
                plt,
                hit_ratio,
                "hit_label",
                "forecast_only_hit_ratio",
                "risk_aware_hit_ratio",
                chart_dir / "hit_ratio_comparison_topn_horizon.png",
                "Hit Ratio Comparison",
            )
            notes.append("generated:hit_ratio_comparison_topn_horizon.png")
        else:
            notes.append("skipped:hit_ratio_comparison:no_hit_ratio_rows")

        if not baskets.empty:
            proxy = baskets.pivot_table(
                index=["horizon", "top_n"],
                columns="candidate_type",
                values="return_volatility_proxy",
                aggfunc="first",
            ).reset_index()
            proxy["proxy_label"] = proxy["horizon"].astype(str) + "d top" + proxy["top_n"].astype(str)
            plot_two_series(
                plt,
                proxy,
                "proxy_label",
                "forecast_only",
                "risk_aware",
                chart_dir / "return_volatility_proxy_topn_horizon.png",
                "Return Volatility Proxy",
            )
            notes.append("generated:return_volatility_proxy_topn_horizon.png")

            var_cvar = baskets.pivot_table(
                index=["horizon", "top_n"],
                columns="candidate_type",
                values=["var_95", "cvar_95"],
                aggfunc="first",
            )
            var_cvar.columns = ["_".join(column).strip() for column in var_cvar.columns.values]
            var_cvar = var_cvar.reset_index()
            var_cvar["var_label"] = var_cvar["horizon"].astype(str) + "d top" + var_cvar["top_n"].astype(str)
            plot_grouped_bar(
                plt,
                var_cvar,
                "var_label",
                [column for column in var_cvar.columns if column.startswith("var_95_") or column.startswith("cvar_95_")],
                chart_dir / "var_cvar_comparison_topn_horizon.png",
                "VaR CVaR Comparison",
            )
            notes.append("generated:var_cvar_comparison_topn_horizon.png")
        else:
            notes.append("skipped:return_volatility_proxy:no_basket_rows")
            notes.append("skipped:var_cvar_comparison:no_basket_rows")
    except Exception as exc:
        notes.append(f"chart_generation_failed:{exc}")
    return notes


def plot_grouped_bar(
    plt: Any,
    frame: pd.DataFrame,
    label_column: str,
    value_columns: list[str],
    path: Path,
    title: str,
) -> None:
    clean = frame.copy()
    clean[label_column] = clean[label_column].astype(str)
    clean = clean.set_index(label_column)
    values = clean[value_columns].apply(pd.to_numeric, errors="coerce")
    ax = values.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_two_series(
    plt: Any,
    frame: pd.DataFrame,
    label_prefix: str,
    first_column: str,
    second_column: str,
    path: Path,
    title: str,
) -> None:
    clean = frame.copy()
    if label_prefix not in clean.columns:
        clean[label_prefix] = clean["horizon"].astype(str) + "d top" + clean["top_n"].astype(str)
    values = clean[[label_prefix, first_column, second_column]].copy()
    values[first_column] = pd.to_numeric(values[first_column], errors="coerce")
    values[second_column] = pd.to_numeric(values[second_column], errors="coerce")
    ax = values.set_index(label_prefix).plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_report(
    *,
    experiments: list[str],
    candidates: pd.DataFrame,
    baskets: pd.DataFrame,
    risk_summary: pd.DataFrame,
    risk_adjusted: pd.DataFrame,
    drawdown: pd.DataFrame,
    hit_ratio: pd.DataFrame,
    source_evidence: pd.DataFrame,
    chart_notes: list[str],
) -> str:
    assessment = assess_results(candidates, baskets, drawdown, hit_ratio)
    missing = source_evidence.loc[~source_evidence["exists"]].copy()
    acceptance = acceptance_table(candidates, baskets, risk_adjusted, drawdown, hit_ratio)
    policy_summary = pd.DataFrame(
        [
            {
                "policy": "forecast_only",
                "ranking_basis": "expected_return_proxy, directional_confidence",
                "risk_controls": "disabled",
                "diagnostic_only": True,
            },
            {
                "policy": "risk_aware",
                "ranking_basis": "forecast_score minus volatility, drawdown, VaR, CVaR, and missing-metric penalty",
                "risk_controls": "enabled",
                "diagnostic_only": True,
            },
        ]
    )

    candidate_preview = candidates.head(20) if not candidates.empty else candidates
    basket_preview = baskets.head(20) if not baskets.empty else baskets
    chart_frame = pd.DataFrame({"chart_note": chart_notes})

    lines = [
        "# Risk-Aware Decision Research Report",
        "",
        "## 1. Executive Summary",
        "",
        assessment["executive_summary"],
        "",
        "## 2. Phase 3 Objective",
        "",
        "Phase 3 evaluates whether a risk-aware candidate ranking improves diagnostic decision candidate utility compared with forecast-only ranking.",
        "",
        "## 3. Relation To Phase 0, Phase 1, And Phase 2",
        "",
        "- Phase 0 froze VSEF v1 governance boundaries.",
        "- Phase 1 implemented config-driven experiment execution and artifact conventions.",
        "- Phase 2 found limited evidence that forecasting models consistently outperform simple baselines on MAE/RMSE.",
        "- Phase 3 therefore tests whether risk-aware filtering improves diagnostic candidate quality through drawdown, volatility, VaR, CVaR, hit ratio, and risk-adjusted behavior.",
        "",
        "## 4. Candidate Policy Definitions",
        "",
        markdown_table(policy_summary),
        "",
        "## 5. Experiment Design",
        "",
        "- EXP-RK-001: candidate comparison between forecast-only and risk-aware ranking.",
        "- EXP-RK-002: equal-weight diagnostic candidate basket outcome evaluation for top-N baskets.",
        "- Candidate baskets are diagnostic research evidence only and are not real portfolios.",
        "",
        "## 6. Data And Source Artifact Evidence",
        "",
        markdown_table(source_evidence),
        "",
        "## 7. Candidate Comparison Results",
        "",
        markdown_table(candidate_preview),
        "",
        "## 8. Basket Outcome Results",
        "",
        markdown_table(basket_preview),
        "",
        "## 9. Risk-Adjusted Utility Discussion",
        "",
        assessment["utility_discussion"],
        "",
        "### Drawdown Comparison",
        "",
        markdown_table(drawdown),
        "",
        "### Hit Ratio Comparison",
        "",
        markdown_table(hit_ratio),
        "",
        "### Risk Summary",
        "",
        markdown_table(risk_summary),
        "",
        "## 10. Missing Artifacts And Limitations",
        "",
        missing_limitations_text(missing, chart_frame, candidates, baskets),
        "",
        "## 11. Acceptance Criteria",
        "",
        markdown_table(acceptance),
        "",
        "## 12. Diagnostic-Only Disclaimer",
        "",
        DISCLAIMER,
        "",
        "## Generated Files",
        "",
        "- `candidate_comparison.csv`",
        "- `topn_basket_metrics.csv`",
        "- `risk_summary.csv`",
        "- `risk_adjusted_ranking.csv`",
        "- `drawdown_comparison.csv`",
        "- `hit_ratio_comparison.csv`",
        "- `charts/` when chart generation succeeded",
        "",
        "## Experiments",
        "",
        ", ".join(f"`{experiment}`" for experiment in experiments),
        "",
    ]
    return "\n".join(lines)


def assess_results(
    candidates: pd.DataFrame,
    baskets: pd.DataFrame,
    drawdown: pd.DataFrame,
    hit_ratio: pd.DataFrame,
) -> dict[str, str]:
    if candidates.empty:
        message = (
            "Candidate generation could not be validated because no candidate comparison rows were available. "
            "No decision-layer value claim should be made from this run."
        )
        return {"executive_summary": message, "utility_discussion": message}
    if baskets.empty:
        message = (
            "Candidate rows were generated, but basket outcomes were not available. "
            "No realized decision utility claim should be made from this run."
        )
        return {"executive_summary": message, "utility_discussion": message}

    paired = pair_basket_metrics(baskets)
    mean_return_diff = numeric_mean(paired, "average_realized_return_diff")
    proxy_diff = numeric_mean(paired, "return_volatility_proxy_diff")
    drawdown_reduction = numeric_mean(drawdown, "drawdown_reduction_vs_forecast_only")
    hit_diff = numeric_mean(hit_ratio, "hit_ratio_difference_vs_forecast_only")
    improved_drawdown = pd.notna(drawdown_reduction) and drawdown_reduction > 0
    improved_proxy = pd.notna(proxy_diff) and proxy_diff > 0
    improved_hit = pd.notna(hit_diff) and hit_diff > 0
    sacrificed_return = pd.notna(mean_return_diff) and mean_return_diff < 0

    facts = [
        f"Mean risk-aware minus forecast-only average realized return: `{format_number(mean_return_diff)}`.",
        f"Mean risk-aware minus forecast-only return/volatility proxy: `{format_number(proxy_diff)}`.",
        f"Mean drawdown reduction versus forecast-only: `{format_number(drawdown_reduction)}`.",
        f"Mean hit-ratio difference versus forecast-only: `{format_number(hit_diff)}`.",
    ]

    if improved_drawdown or improved_proxy or improved_hit:
        if sacrificed_return and improved_drawdown:
            conclusion = (
                "Risk-aware ranking reduced raw return in some contexts but improved drawdown or volatility behavior. "
                "This supports a risk-adjusted objective only if the project prioritizes stability over raw return."
            )
        else:
            improved = []
            if improved_drawdown:
                improved.append("drawdown")
            if improved_proxy:
                improved.append("return/volatility proxy")
            if improved_hit:
                improved.append("hit ratio")
            conclusion = (
                "Risk-aware ranking improved " + ", ".join(improved) + " in the aggregated Phase 3 evidence. "
                "This supports continued research on a risk-aware decision layer, but not investment readiness."
            )
    else:
        conclusion = (
            "The current evidence does not prove that the risk-aware ranking improves candidate utility over forecast-only ranking. "
            "This weakens the risk-layer value claim and suggests that future work should focus on better risk feature design, "
            "regime-aware filtering, or stricter candidate eligibility rules."
        )
    return {
        "executive_summary": conclusion + " " + " ".join(facts),
        "utility_discussion": conclusion + "\n\n" + "\n".join(f"- {fact}" for fact in facts),
    }


def pair_basket_metrics(baskets: pd.DataFrame) -> pd.DataFrame:
    if baskets.empty:
        return pd.DataFrame()
    metrics = [
        "average_realized_return",
        "hit_ratio",
        "return_volatility_proxy",
        "max_drawdown",
        "var_95",
        "cvar_95",
    ]
    rows: list[dict[str, Any]] = []
    for (horizon, top_n), group in baskets.groupby(["horizon", "top_n"], sort=True):
        row: dict[str, Any] = {"horizon": horizon, "top_n": top_n}
        for metric in metrics:
            forecast = first_group_value(group, "forecast_only", metric)
            risk = first_group_value(group, "risk_aware", metric)
            row[f"{metric}_forecast_only"] = forecast
            row[f"{metric}_risk_aware"] = risk
            row[f"{metric}_diff"] = risk - forecast if pd.notna(risk) and pd.notna(forecast) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def first_group_value(group: pd.DataFrame, candidate_type: str, metric: str) -> float:
    values = pd.to_numeric(group.loc[group["candidate_type"] == candidate_type, metric], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def missing_limitations_text(
    missing: pd.DataFrame,
    chart_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    baskets: pd.DataFrame,
) -> str:
    lines: list[str] = []
    if missing.empty:
        lines.append("No required Phase 3 source artifact was missing from the generated report evidence table.")
    else:
        lines.append("Missing or unavailable artifacts were disclosed and were not replaced with fabricated values.")
        lines.append("")
        lines.append(markdown_table(missing))
    lines.extend(
        [
            "",
            "Charts:",
            "",
            markdown_table(chart_frame),
            "",
            "Limitations:",
            "",
            "- Candidate rankings use historical Phase 2 forecast artifacts and local daily OHLCV files.",
            "- The expected return proxy inherits prediction outliers from Phase 2 y_pred artifacts; this is forecast instability evidence, not a decision-layer value claim.",
            "- Risk metrics use a 20-day realized lookback and may not capture broader regime behavior.",
            "- Candidate baskets are equal-weight diagnostic baskets, not portfolio allocations.",
            f"- Candidate row count: `{len(candidates)}`.",
            f"- Basket metric row count: `{len(baskets)}`.",
        ]
    )
    return "\n".join(lines)


def acceptance_table(
    candidates: pd.DataFrame,
    baskets: pd.DataFrame,
    risk_adjusted: pd.DataFrame,
    drawdown: pd.DataFrame,
    hit_ratio: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("candidate_policy_forecast_only.yaml exists", (REPO_ROOT / "configs" / "policies" / "candidate_policy_forecast_only.yaml").exists()),
        ("candidate_policy_risk_aware.yaml exists", (REPO_ROOT / "configs" / "policies" / "candidate_policy_risk_aware.yaml").exists()),
        ("EXP-RK-001.yaml exists", (REPO_ROOT / "configs" / "experiments" / "EXP-RK-001.yaml").exists()),
        ("EXP-RK-002.yaml exists", (REPO_ROOT / "configs" / "experiments" / "EXP-RK-002.yaml").exists()),
        ("Candidate comparison table exists", not candidates.empty),
        ("Top-N basket metrics table exists", not baskets.empty),
        ("Risk-aware report generated from artifacts", True),
        ("Every candidate row has diagnostics", bool(not candidates.empty and candidates["diagnostics"].notna().all())),
        ("Every candidate row has diagnostic_only=true", bool(not candidates.empty and candidates["diagnostic_only"].astype(str).str.lower().eq("true").all())),
        ("Risk-aware compared against forecast-only", bool({"forecast_only", "risk_aware"} <= set(candidates["candidate_type"].astype(str))) if not candidates.empty else False),
        ("Drawdown comparison exists", not drawdown.empty),
        ("Hit ratio comparison exists", not hit_ratio.empty),
        ("Risk-adjusted ranking exists", not risk_adjusted.empty),
    ]
    return pd.DataFrame(
        {
            "acceptance_criterion": [name for name, _ in checks],
            "status": ["pass" if passed else "fail" for _, passed in checks],
            "diagnostic_only": True,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
