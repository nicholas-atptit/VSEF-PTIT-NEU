from __future__ import annotations

import json

import pandas as pd

from src.ml.benchmark.acceptance import evaluate_benchmark_acceptance
from src.ml.benchmark.system_benchmark import SystemBenchmarkRunner


def test_strong_evidence_produces_accepted() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.01,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is True
    assert result.status == "accepted"
    assert result.to_dict()["accepted"] is True


def test_mixed_evidence_produces_inconclusive() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.20,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "inconclusive"
    assert any(reason.startswith("dm_p_value_not_significant") for reason in result.decision_reasons)


def test_missing_evidence_produces_exploratory_only() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=None,
        dm_p_value=None,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "exploratory_only"
    assert "missing_evidence:bootstrap_ci,dm_p_value" in result.warnings


def test_negative_or_weak_effect_produces_rejected() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.0,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.01,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "rejected"
    assert any(reason.startswith("effect_size_not_positive") for reason in result.decision_reasons)


def test_unstable_bootstrap_ci_is_not_accepted() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(-0.25, 0.30),
        dm_p_value=0.01,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "inconclusive"
    assert any(reason.startswith("bootstrap_ci_lower_below_tolerance") for reason in result.decision_reasons)


def test_weak_dm_evidence_is_not_accepted() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.08,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "inconclusive"
    assert any(reason.startswith("dm_p_value_not_significant") for reason in result.decision_reasons)


def test_cost_or_turnover_penalty_destroys_promotion() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.01,
        turnover_delta=1.50,
        cost_adjusted_delta=-0.02,
        sample_size=120,
        comparison_count=3,
    )

    assert result.accepted is False
    assert result.status == "rejected"
    assert any(reason.startswith("cost_adjusted_delta_below_threshold") for reason in result.decision_reasons)
    assert any(reason.startswith("turnover_penalty_exceeds_threshold") for reason in result.decision_reasons)


def test_multiple_comparison_warning_appears_when_count_is_high() -> None:
    result = evaluate_benchmark_acceptance(
        prediction_metric_delta=0.04,
        economic_metric_delta=0.25,
        bootstrap_ci=(0.01, 0.08),
        dm_p_value=0.01,
        turnover_delta=0.0,
        cost_adjusted_delta=0.12,
        sample_size=120,
        comparison_count=25,
    )

    assert result.accepted is False
    assert result.status == "inconclusive"
    assert "many_unadjusted_comparisons:25>20" in result.warnings


def test_benchmark_report_includes_acceptance_status(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            _benchmark_detail_row(
                benchmark_mode="legacy_forecast_only",
                sharpe=0.10,
                cagr=0.02,
                turnover=2.0,
                directional_accuracy=0.52,
            ),
            _benchmark_detail_row(
                benchmark_mode="full_system",
                sharpe=0.25,
                cagr=0.05,
                turnover=2.1,
                directional_accuracy=0.56,
            ),
        ]
    )
    summary = SystemBenchmarkRunner._summary(detail)
    runner = SystemBenchmarkRunner(model_root=tmp_path / "models")

    runner._write_outputs(
        detail_df=detail,
        summary_df=summary,
        detail_path=tmp_path / "reports" / "benchmark.csv",
        summary_path=tmp_path / "reports" / "benchmark_summary.csv",
        json_path=tmp_path / "reports" / "benchmark.json",
        markdown_path=tmp_path / "reports" / "system_benchmark.md",
    )

    markdown = (tmp_path / "reports" / "system_benchmark.md").read_text(encoding="utf-8")
    assert "## Acceptance Governance" in markdown
    assert "exploratory_only" in markdown
    assert "Leaderboard position alone is not benchmark promotion" in markdown

    payload = json.loads((tmp_path / "reports" / "benchmark.json").read_text(encoding="utf-8"))
    assert {"accepted", "status", "effect_size", "bootstrap_ci", "dm_p_value", "warnings"} <= set(
        payload["summary_rows"][0]
    )
    full_system_row = next(row for row in payload["summary_rows"] if row["benchmark_mode"] == "full_system")
    assert full_system_row["accepted"] is False
    assert full_system_row["status"] == "exploratory_only"


def _benchmark_detail_row(
    *,
    benchmark_mode: str,
    sharpe: float,
    cagr: float,
    turnover: float,
    directional_accuracy: float,
) -> dict[str, object]:
    return {
        "benchmark_mode": benchmark_mode,
        "ticker": "TEST",
        "horizon": "short",
        "algorithm": "cart",
        "mode_description": benchmark_mode,
        "train_rows": 80,
        "val_rows": 20,
        "test_rows": 60,
        "cumulative_return": cagr,
        "cagr": cagr,
        "volatility": 0.10,
        "sharpe": sharpe,
        "sortino": sharpe,
        "calmar": sharpe,
        "max_drawdown": -0.05,
        "avg_drawdown": -0.02,
        "tail_loss": -0.01,
        "turnover": turnover,
        "exposure": 0.50,
        "trade_count": 4,
        "rmse": 0.10,
        "mae": 0.08,
        "directional_accuracy": directional_accuracy,
    }
