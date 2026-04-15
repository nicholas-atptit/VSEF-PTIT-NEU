from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.combined_signal import CombinedSignalAnalysisRunner, CombinedSignalConfig


def _sample_joined_frame() -> pd.DataFrame:
    rows = [
        ["2026-01-05", "AAA", "3d", "cart", 0.03, 0.02, 1, 1, 0.66],
        ["2026-01-05", "BBB", "3d", "cart", 0.01, 0.01, 1, 1, 0.58],
        ["2026-01-05", "CCC", "3d", "cart", -0.02, -0.01, 0, 0, 0.49],
        ["2026-01-06", "AAA", "3d", "cart", 0.04, 0.03, 1, 1, 0.70],
        ["2026-01-06", "BBB", "3d", "cart", -0.01, 0.015, 0, 1, 0.62],
        ["2026-01-06", "CCC", "3d", "cart", -0.03, -0.02, 0, 0, 0.45],
        ["2026-01-05", "AAA", "5d", "cart", 0.05, 0.025, 1, 1, 0.68],
        ["2026-01-05", "BBB", "5d", "cart", 0.00, 0.005, 0, 1, 0.56],
        ["2026-01-05", "CCC", "5d", "cart", -0.03, -0.015, 0, 0, 0.44],
        ["2026-01-06", "AAA", "5d", "cart", 0.06, 0.035, 1, 1, 0.72],
        ["2026-01-06", "BBB", "5d", "cart", -0.02, 0.012, 0, 1, 0.61],
        ["2026-01-06", "CCC", "5d", "cart", -0.04, -0.03, 0, 0, 0.43],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "ticker",
            "horizon",
            "model_name",
            "actual_return",
            "predicted_return",
            "actual_profit_label",
            "predicted_profit_label",
            "predicted_profit_probability",
        ],
    )


def _write_joined_frame(tmp_path: Path, frame: pd.DataFrame) -> Path:
    dual_task_dir = tmp_path / "dual_task" / "summary"
    dual_task_dir.mkdir(parents=True, exist_ok=True)
    path = dual_task_dir / "joined_regression_classification_evaluation.csv"
    frame.to_csv(path, index=False)
    return path


def test_prepare_horizon_table_combines_scores_and_labels(tmp_path) -> None:
    frame = _sample_joined_frame()
    _write_joined_frame(tmp_path, frame)
    runner = CombinedSignalAnalysisRunner(
        CombinedSignalConfig(
            dual_task_dir=str(tmp_path / "dual_task"),
            output_dir=str(tmp_path / "combined"),
            horizons=["3d"],
        )
    )

    prepared = runner._prepare_horizon_table(frame[frame["horizon"] == "3d"].copy())
    strong = prepared[(prepared["ticker"] == "AAA") & (prepared["date"] == "2026-01-06")].iloc[0]
    moderate = prepared[(prepared["ticker"] == "BBB") & (prepared["date"] == "2026-01-05")].iloc[0]
    reject = prepared[(prepared["ticker"] == "CCC") & (prepared["date"] == "2026-01-06")].iloc[0]

    assert strong["normalized_predicted_return"] == 1.0
    assert strong["combined_score"] == 0.85
    assert bool(strong["gated_valid_signal"]) is True
    assert strong["combined_signal_label"] == "strong_positive"
    assert moderate["combined_signal_label"] == "moderate_positive"
    assert reject["combined_signal_label"] == "reject"


def test_build_ranking_summary_produces_topk_comparison(tmp_path) -> None:
    frame = _sample_joined_frame()
    _write_joined_frame(tmp_path, frame)
    runner = CombinedSignalAnalysisRunner(
        CombinedSignalConfig(
            dual_task_dir=str(tmp_path / "dual_task"),
            output_dir=str(tmp_path / "combined"),
            horizons=["3d"],
            top_k_values=[1, 3],
        )
    )

    prepared = runner._prepare_horizon_table(frame[frame["horizon"] == "3d"].copy())
    ranking = runner._build_ranking_summary(prepared)
    top1_combined = ranking[
        (ranking["ranking_method"] == "combined_weighted_linear") & (ranking["top_k"] == 1)
    ].iloc[0]
    top1_return = ranking[
        (ranking["ranking_method"] == "predicted_return") & (ranking["top_k"] == 1)
    ].iloc[0]

    assert set(ranking["ranking_method"]) >= {
        "predicted_return",
        "predicted_profit_probability",
        "combined_weighted_linear",
        "combined_rank_based",
        "combined_weighted_linear_gated",
    }
    assert top1_combined["average_actual_return"] == 0.035
    assert top1_combined["profit_rate"] == 1.0
    assert top1_return["average_actual_return"] == 0.035
    assert top1_return["precision_at_top_k"] == 1.0


def test_build_calibration_summary_buckets_profit_probabilities(tmp_path) -> None:
    frame = _sample_joined_frame()
    _write_joined_frame(tmp_path, frame)
    runner = CombinedSignalAnalysisRunner(
        CombinedSignalConfig(
            dual_task_dir=str(tmp_path / "dual_task"),
            output_dir=str(tmp_path / "combined"),
            horizons=["3d"],
        )
    )

    prepared = runner._prepare_horizon_table(frame[frame["horizon"] == "3d"].copy())
    calibration = runner._build_calibration_summary(prepared)
    high_bucket = calibration[calibration["probability_bucket"] == "0.65+"].iloc[0]

    assert high_bucket["observations"] == 2
    assert high_bucket["realized_profit_rate"] == 1.0
    assert round(high_bucket["avg_predicted_probability"], 2) == 0.68


def test_combined_signal_runner_writes_artifacts_and_preserves_alignment(tmp_path) -> None:
    frame = _sample_joined_frame()
    _write_joined_frame(tmp_path, frame)
    runner = CombinedSignalAnalysisRunner(
        CombinedSignalConfig(
            dual_task_dir=str(tmp_path / "dual_task"),
            output_dir=str(tmp_path / "combined_signal"),
            horizons=["3d", "5d"],
        )
    )
    result = runner.run()

    combined_path = Path(result["horizons"]["3d"]["paths"]["combined_signal_table"])
    ranking_path = Path(result["horizons"]["3d"]["paths"]["combined_ranking_summary"])
    calibration_path = Path(result["horizons"]["3d"]["paths"]["probability_calibration_summary"])
    summary_path = Path(result["summary_paths"]["overall_combined_signal_summary"])
    cross_path = Path(result["summary_paths"]["cross_horizon_combined_ranking"])
    assert combined_path.exists()
    assert ranking_path.exists()
    assert calibration_path.exists()
    assert summary_path.exists()
    assert cross_path.exists()

    combined_frame = pd.read_csv(combined_path)
    original_3d = frame[frame["horizon"] == "3d"].copy()
    assert combined_frame[["date", "ticker", "horizon", "model_name"]].equals(
        original_3d[["date", "ticker", "horizon", "model_name"]].reset_index(drop=True)
    )
    assert {
        "normalized_predicted_return",
        "combined_score",
        "combined_signal_label",
        "actual_return",
        "actual_profit_label",
    } <= set(combined_frame.columns)

    run_config = json.loads((combined_path.parent / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["analysis_only"] is True
    assert run_config["live_execution_enabled"] is False
