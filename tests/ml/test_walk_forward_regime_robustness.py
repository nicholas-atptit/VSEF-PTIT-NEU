from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.walk_forward_regime_robustness import (
    WalkForwardRegimeRobustnessConfig,
    WalkForwardRegimeRobustnessRunner,
)


def _fake_fold_payload(tmp_root: Path, fold: dict[str, str], *, reg_model: str, cls_model: str, combined_model: str, combined_method: str) -> dict:
    fold_dir = tmp_root / fold["fold_id"]
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_config.json").write_text(json.dumps({"fold_id": fold["fold_id"]}), encoding="utf-8")

    fold_summary = pd.DataFrame(
        [
            {
                "fold_id": fold["fold_id"],
                "fold_number": fold["fold_number"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "eval_start": fold["eval_start"],
                "eval_end": fold["eval_end"],
                "regime": "bull",
                "best_regression_model": reg_model,
                "best_regression_horizon": "3d",
                "best_classification_model": cls_model,
                "best_classification_horizon": "5d",
                "best_combined_method": combined_method,
                "best_combined_model": combined_model,
                "best_horizon": "5d",
                "key_takeaway": "synthetic",
            },
            {
                "fold_id": fold["fold_id"],
                "fold_number": fold["fold_number"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "eval_start": fold["eval_start"],
                "eval_end": fold["eval_end"],
                "regime": "bear",
                "best_regression_model": "xgboost",
                "best_regression_horizon": "3d",
                "best_classification_model": "cart",
                "best_classification_horizon": "20d",
                "best_combined_method": "combined_weighted_linear_gated",
                "best_combined_model": "xgboost",
                "best_horizon": "3d",
                "key_takeaway": "synthetic",
            },
        ]
    )
    model_ranking = pd.DataFrame(
        [
            {
                "fold_id": fold["fold_id"],
                "horizon": "3d",
                "regime": "bull",
                "model_name": reg_model,
                "rmse": 0.05,
                "f1": 0.45,
                "ranking_method": combined_method,
                "profit_rate": 0.6,
                "average_actual_return": 0.02,
                "rank_regression_rmse": 1.0,
                "rank_classification_f1": 2.0,
                "rank_combined_profit_rate": 1.0,
                "overall_rank": 1.33,
            },
            {
                "fold_id": fold["fold_id"],
                "horizon": "5d",
                "regime": "bull",
                "model_name": cls_model,
                "rmse": 0.07,
                "f1": 0.60,
                "ranking_method": combined_method,
                "profit_rate": 0.62,
                "average_actual_return": 0.03,
                "rank_regression_rmse": 2.0,
                "rank_classification_f1": 1.0,
                "rank_combined_profit_rate": 1.0,
                "overall_rank": 1.33,
            },
        ]
    )
    combined_method_ranking = pd.DataFrame(
        [
            {
                "fold_id": fold["fold_id"],
                "horizon": "5d",
                "regime": "bull",
                "model_name": combined_model,
                "ranking_method": combined_method,
                "top_k": 3,
                "average_actual_return": 0.03,
                "profit_rate": 0.62,
                "rank_in_regime": 1,
            },
            {
                "fold_id": fold["fold_id"],
                "horizon": "3d",
                "regime": "bear",
                "model_name": "xgboost",
                "ranking_method": "combined_weighted_linear_gated",
                "top_k": 3,
                "average_actual_return": 0.04,
                "profit_rate": 0.7,
                "rank_in_regime": 1,
            },
        ]
    )
    joined_sample = pd.DataFrame(
        [
            {
                "date": fold["eval_start"],
                "ticker": "AAA",
                "horizon": "3d",
                "model_name": reg_model,
                "prediction_date": fold["train_end"],
                "benchmark_date": fold["train_end"],
                "regime": "bull",
            }
        ]
    )
    pd.DataFrame([{
        **fold,
        "benchmark_source_used": "market_proxy_csv",
        "rows_analyzed": 12,
        "bull_count": 6,
        "sideway_count": 3,
        "bear_count": 3,
        "available_algorithms": "cart,xgboost",
        "skipped_algorithms": "[]",
    }]).to_csv(fold_dir / "fold_summary.csv", index=False)
    fold_summary.to_csv(fold_dir / "regime_summary.csv", index=False)
    model_ranking.to_csv(fold_dir / "model_ranking.csv", index=False)
    combined_method_ranking.to_csv(fold_dir / "combined_method_ranking.csv", index=False)
    joined_sample.to_csv(fold_dir / "joined_evaluation_sample.csv", index=False)
    return {
        "status": "completed",
        "fold": fold,
        "fold_dir": fold_dir,
        "fold_overview_row": {
            **fold,
            "status": "completed",
            "attempts_used": 1,
            "benchmark_source_used": "market_proxy_csv",
            "rows_analyzed": 12,
            "bull_count": 6,
            "sideway_count": 3,
            "bear_count": 3,
            "available_algorithms": "cart,xgboost",
            "skipped_algorithms": "[]",
        },
        "fold_summary": fold_summary,
        "model_ranking": model_ranking,
        "combined_method_ranking": combined_method_ranking,
        "joined_sample": joined_sample,
    }


def test_generate_folds_preserves_time_order_and_no_leakage() -> None:
    runner = WalkForwardRegimeRobustnessRunner(
        WalkForwardRegimeRobustnessConfig(
            train_start="2020-01-01",
            first_eval_start="2023-01-01",
            last_eval_end="2023-05-31",
            eval_window_days=60,
            step_size_days=30,
            max_folds=3,
        )
    )
    folds = runner._generate_folds()

    assert len(folds) == 3
    assert folds[0]["train_end"] == "2022-12-31"
    assert folds[0]["eval_start"] == "2023-01-01"
    assert folds[1]["eval_start"] == "2023-01-31"
    for fold in folds:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["eval_start"])


def test_stability_summary_builders_compute_win_rates() -> None:
    runner = WalkForwardRegimeRobustnessRunner(WalkForwardRegimeRobustnessConfig())
    fold_summaries = pd.DataFrame(
        [
            {"fold_id": "fold_001", "regime": "bull", "best_regression_model": "xgboost", "best_regression_horizon": "3d", "best_classification_model": "cart", "best_classification_horizon": "20d", "best_combined_method": "combined_weighted_linear_gated", "best_horizon": "5d"},
            {"fold_id": "fold_002", "regime": "bull", "best_regression_model": "xgboost", "best_regression_horizon": "3d", "best_classification_model": "xgboost", "best_classification_horizon": "5d", "best_combined_method": "combined_rank_based", "best_horizon": "3d"},
        ]
    )
    model_rankings = pd.DataFrame(
        [
            {"fold_id": "fold_001", "regime": "bull", "model_name": "xgboost", "horizon": "3d", "rank_regression_rmse": 1.0, "rank_classification_f1": 3.0, "overall_rank": 2.0},
            {"fold_id": "fold_002", "regime": "bull", "model_name": "xgboost", "horizon": "3d", "rank_regression_rmse": 1.0, "rank_classification_f1": 2.0, "overall_rank": 1.5},
            {"fold_id": "fold_001", "regime": "bull", "model_name": "cart", "horizon": "20d", "rank_regression_rmse": 2.0, "rank_classification_f1": 1.0, "overall_rank": 1.5},
        ]
    )
    combined_rankings = pd.DataFrame(
        [
            {"fold_id": "fold_001", "regime": "bull", "ranking_method": "combined_weighted_linear_gated", "model_name": "xgboost", "horizon": "5d", "profit_rate": 0.6, "average_actual_return": 0.02, "rank_in_regime": 1},
            {"fold_id": "fold_002", "regime": "bull", "ranking_method": "combined_rank_based", "model_name": "xgboost", "horizon": "3d", "profit_rate": 0.55, "average_actual_return": 0.01, "rank_in_regime": 1},
        ]
    )

    model_stability = runner._build_model_stability_summary(fold_summaries, model_rankings)
    regime_stability = runner._build_regime_stability_summary(fold_summaries)
    combined_stability = runner._build_combined_method_stability_summary(combined_rankings)

    regression_row = model_stability[(model_stability["component"] == "regression_model") & (model_stability["model_name"] == "xgboost")].iloc[0]
    assert regression_row["fold_win_count"] == 2
    assert regression_row["win_rate"] == 1.0
    bull_combined = regime_stability[(regime_stability["regime"] == "bull") & (regime_stability["component"] == "combined_method")].iloc[0]
    assert bull_combined["unique_choices"] == 2
    assert set(combined_stability["ranking_method"]) == {"combined_rank_based", "combined_weighted_linear_gated"}
    assert combined_stability["win_rate"].le(1.0).all()


def test_runner_writes_fold_and_global_artifacts(monkeypatch, tmp_path) -> None:
    runner = WalkForwardRegimeRobustnessRunner(
        WalkForwardRegimeRobustnessConfig(
            output_dir=str(tmp_path / "artifacts" / "wf"),
            last_eval_end="2023-04-30",
            max_folds=2,
            eval_window_days=30,
            step_size_days=30,
        )
    )

    def _fake_run_fold(fold):
        if fold["fold_id"] == "fold_001":
            return _fake_fold_payload(Path(runner.output_dir), fold, reg_model="xgboost", cls_model="cart", combined_model="xgboost", combined_method="combined_weighted_linear_gated")
        return _fake_fold_payload(Path(runner.output_dir), fold, reg_model="lightgbm", cls_model="xgboost", combined_model="lightgbm", combined_method="combined_rank_based")

    monkeypatch.setattr(runner, "_run_fold", _fake_run_fold)
    result = runner.run()

    assert Path(result["summary_paths"]["fold_overview"]).exists()
    assert Path(result["summary_paths"]["model_stability_summary"]).exists()
    assert Path(result["summary_paths"]["overall_robustness_report"]).exists()
    fold_one = Path(runner.output_dir) / "fold_001"
    assert (fold_one / "fold_config.json").exists()
    assert (fold_one / "fold_summary.csv").exists()
    assert (fold_one / "regime_summary.csv").exists()
    assert (fold_one / "model_ranking.csv").exists()
    assert (fold_one / "combined_method_ranking.csv").exists()
    assert (fold_one / "joined_evaluation_sample.csv").exists()

    report = pd.read_csv(result["summary_paths"]["overall_robustness_report"])
    assert {"component", "best_overall_choice", "stability_level", "supporting_evidence", "caution_note"} <= set(report.columns)


def test_runner_continues_after_fold_failure_and_records_status(monkeypatch, tmp_path) -> None:
    runner = WalkForwardRegimeRobustnessRunner(
        WalkForwardRegimeRobustnessConfig(
            output_dir=str(tmp_path / "artifacts" / "wf_failure"),
            last_eval_end="2023-04-30",
            max_folds=2,
            eval_window_days=30,
            step_size_days=30,
            fold_retry_count=1,
            continue_on_fold_error=True,
        )
    )
    attempts = {"fold_001": 0, "fold_002": 0}

    def _fake_run_fold_once(fold):
        attempts[fold["fold_id"]] += 1
        if fold["fold_id"] == "fold_001":
            raise ValueError("No vnstock OHLCV data returned for HPG")
        return _fake_fold_payload(
            Path(runner.output_dir),
            fold,
            reg_model="lightgbm",
            cls_model="xgboost",
            combined_model="lightgbm",
            combined_method="combined_rank_based",
        )

    monkeypatch.setattr(runner, "_run_fold_once", _fake_run_fold_once)
    result = runner.run()

    overview = result["fold_overview"]
    failed_row = overview.loc[overview["fold_id"] == "fold_001"].iloc[0]
    completed_row = overview.loc[overview["fold_id"] == "fold_002"].iloc[0]

    assert failed_row["status"] == "failed"
    assert failed_row["attempts_used"] == 2
    assert "No vnstock OHLCV data returned" in failed_row["error_message"]
    assert completed_row["status"] == "completed"
    assert result["completed_folds"] == 1
    assert result["failed_folds"] == 1
    assert Path(result["summary_paths"]["fold_overview"]).exists()
    assert not result["model_stability_summary"].empty
