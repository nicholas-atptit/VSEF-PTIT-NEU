from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.meta_selector import MetaSelectorConfig, RegimeConditionedMetaSelectorRunner


def _write_fold_artifacts(
    root: Path,
    *,
    fold_id: str,
    fold_number: int,
    eval_start: str,
    regime_rows: list[dict],
    model_rows: list[dict],
    combined_rows: list[dict],
) -> None:
    fold_dir = root / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "fold_number": fold_number,
                "train_start": "2020-01-01",
                "train_end": "2022-12-31",
                "eval_start": eval_start,
                "eval_end": eval_start,
                "training_window_mode": "expanding",
                "benchmark_source_used": "vnstock_index",
                "rows_analyzed": len(regime_rows),
                "bull_count": sum(1 for row in regime_rows if row["regime"] == "bull"),
                "sideway_count": sum(1 for row in regime_rows if row["regime"] == "sideway"),
                "bear_count": sum(1 for row in regime_rows if row["regime"] == "bear"),
                "available_algorithms": "cart,xgboost,lightgbm",
                "skipped_algorithms": "[]",
                "status": "completed",
                "attempts_used": 0,
            }
        ]
    ).to_csv(fold_dir / "fold_summary.csv", index=False)

    with (fold_dir / "fold_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold_id": fold_id, "fold_number": fold_number}, handle)

    pd.DataFrame(model_rows).to_csv(fold_dir / "model_ranking.csv", index=False)
    pd.DataFrame(combined_rows).to_csv(fold_dir / "combined_method_ranking.csv", index=False)

    for horizon in {"3d", "5d"}:
        horizon_dir = fold_dir / "regime_aware" / horizon
        horizon_dir.mkdir(parents=True, exist_ok=True)
        horizon_rows = [row for row in regime_rows if row["horizon"] == horizon]
        pd.DataFrame(horizon_rows).to_csv(horizon_dir / "regime_labeled_signal_table.csv", index=False)


def _base_regime_row(
    *,
    date: str,
    prediction_date: str,
    ticker: str,
    regime: str,
    model_name: str,
    horizon: str,
    actual_return: float,
    predicted_return: float,
    probability: float,
    combined_score: float,
    rank_score: float,
) -> dict:
    return {
        "date": date,
        "prediction_date": prediction_date,
        "target_date": date,
        "ticker": ticker,
        "model_name": model_name,
        "horizon": horizon,
        "horizon_days": 3 if horizon == "3d" else 5,
        "actual_return": actual_return,
        "predicted_return": predicted_return,
        "actual_profit_label": int(actual_return > 0),
        "predicted_profit_probability": probability,
        "combined_score": combined_score,
        "rank_based_joint_score": rank_score,
        "regime": regime,
    }


def test_selector_uses_only_prior_fold_history_and_fallback(tmp_path) -> None:
    root = tmp_path / "artifacts" / "walk_forward"
    fold1_rows = [
        _base_regime_row(
            date="2023-01-03",
            prediction_date="2022-12-29",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="3d",
            actual_return=0.03,
            predicted_return=0.02,
            probability=0.80,
            combined_score=0.75,
            rank_score=0.70,
        ),
        _base_regime_row(
            date="2023-01-03",
            prediction_date="2022-12-29",
            ticker="AAA",
            regime="bull",
            model_name="lightgbm",
            horizon="5d",
            actual_return=0.01,
            predicted_return=0.01,
            probability=0.60,
            combined_score=0.55,
            rank_score=0.50,
        ),
    ]
    _write_fold_artifacts(
        root,
        fold_id="fold_001",
        fold_number=1,
        eval_start="2023-01-01",
        regime_rows=fold1_rows,
        model_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.70,
                "positive_class_precision": 0.75,
                "rank_regression_rmse": 1.0,
            },
            {
                "fold_id": "fold_001",
                "horizon": "5d",
                "regime": "bull",
                "model_name": "lightgbm",
                "directional_accuracy": 0.55,
                "positive_class_precision": 0.60,
                "rank_regression_rmse": 2.0,
            },
        ],
        combined_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "combined_weighted_linear_gated",
                "return_threshold": 0.01,
                "probability_threshold": 0.55,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.030,
                "profit_rate": 0.80,
            },
            {
                "fold_id": "fold_001",
                "horizon": "5d",
                "regime": "bull",
                "model_name": "lightgbm",
                "ranking_method": "combined_weighted_linear",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.010,
                "profit_rate": 0.60,
            },
        ],
    )

    fold2_rows = [
        _base_regime_row(
            date="2023-06-30",
            prediction_date="2023-06-27",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="3d",
            actual_return=0.02,
            predicted_return=0.015,
            probability=0.82,
            combined_score=0.72,
            rank_score=0.68,
        ),
        _base_regime_row(
            date="2023-06-30",
            prediction_date="2023-06-27",
            ticker="AAA",
            regime="bull",
            model_name="lightgbm",
            horizon="5d",
            actual_return=0.01,
            predicted_return=0.012,
            probability=0.58,
            combined_score=0.50,
            rank_score=0.45,
        ),
        _base_regime_row(
            date="2023-06-30",
            prediction_date="2023-06-27",
            ticker="BBB",
            regime="bear",
            model_name="xgboost",
            horizon="3d",
            actual_return=-0.01,
            predicted_return=-0.005,
            probability=0.40,
            combined_score=0.30,
            rank_score=0.25,
        ),
    ]
    _write_fold_artifacts(
        root,
        fold_id="fold_002",
        fold_number=2,
        eval_start="2023-06-30",
        regime_rows=fold2_rows,
        model_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.68,
                "positive_class_precision": 0.72,
                "rank_regression_rmse": 1.0,
            }
        ],
        combined_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "combined_weighted_linear_gated",
                "return_threshold": 0.01,
                "probability_threshold": 0.55,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.020,
                "profit_rate": 0.75,
            }
        ],
    )

    fold3_rows = [
        _base_regime_row(
            date="2023-12-27",
            prediction_date="2023-12-22",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="3d",
            actual_return=0.025,
            predicted_return=0.018,
            probability=0.78,
            combined_score=0.74,
            rank_score=0.71,
        ),
        _base_regime_row(
            date="2023-12-27",
            prediction_date="2023-12-22",
            ticker="BBB",
            regime="bear",
            model_name="xgboost",
            horizon="3d",
            actual_return=-0.02,
            predicted_return=-0.01,
            probability=0.35,
            combined_score=0.22,
            rank_score=0.20,
        ),
    ]
    _write_fold_artifacts(
        root,
        fold_id="fold_003",
        fold_number=3,
        eval_start="2023-12-27",
        regime_rows=fold3_rows,
        model_rows=[
            {
                "fold_id": "fold_003",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.70,
                "positive_class_precision": 0.74,
                "rank_regression_rmse": 1.0,
            }
        ],
        combined_rows=[
            {
                "fold_id": "fold_003",
                "horizon": "3d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "combined_weighted_linear_gated",
                "return_threshold": 0.01,
                "probability_threshold": 0.55,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.025,
                "profit_rate": 0.78,
            }
        ],
    )

    runner = RegimeConditionedMetaSelectorRunner(
        MetaSelectorConfig(
            walk_forward_dir=str(root),
            output_dir=str(tmp_path / "artifacts" / "meta_selector"),
            minimum_prior_folds_per_regime=2,
            minimum_samples_per_regime=30,
        )
    )
    result = runner.run()

    fold2_selected = pd.read_csv(Path(runner.output_dir) / "fold_002" / "selected_candidates.csv")
    bull_row = fold2_selected.loc[fold2_selected["regime"] == "bull"].iloc[0]
    bear_row = fold2_selected.loc[fold2_selected["regime"] == "bear"].iloc[0]

    assert bull_row["selected_model_name"] == "xgboost"
    assert bull_row["selected_horizon"] == "3d"
    assert bear_row["fallback_used"]
    assert "fallback_global used because bear had insufficient history" in bear_row["selection_reason"]

    overview = result["meta_selector_overview"]
    assert {"fold_id", "selector_mode", "status"} <= set(overview.columns)


def test_utility_and_global_summary_generation(tmp_path) -> None:
    root = tmp_path / "artifacts" / "walk_forward"
    base_rows = [
        _base_regime_row(
            date="2023-01-03",
            prediction_date="2022-12-29",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="3d",
            actual_return=0.03,
            predicted_return=0.02,
            probability=0.80,
            combined_score=0.75,
            rank_score=0.70,
        )
    ]
    for index in range(1, 4):
        fold_id = f"fold_{index:03d}"
        _write_fold_artifacts(
            root,
            fold_id=fold_id,
            fold_number=index,
            eval_start=f"2023-0{index}-01",
            regime_rows=base_rows,
            model_rows=[
                {
                    "fold_id": fold_id,
                    "horizon": "3d",
                    "regime": "bull",
                    "model_name": "xgboost",
                    "directional_accuracy": 0.70,
                    "positive_class_precision": 0.75,
                    "rank_regression_rmse": 1.0,
                }
            ],
            combined_rows=[
                {
                    "fold_id": fold_id,
                    "horizon": "3d",
                    "regime": "bull",
                    "model_name": "xgboost",
                    "ranking_method": "combined_weighted_linear_gated",
                    "return_threshold": 0.01,
                    "probability_threshold": 0.55,
                    "top_k": 3,
                    "observations": 40,
                    "average_actual_return": 0.03,
                    "profit_rate": 0.80,
                }
            ],
        )

    runner = RegimeConditionedMetaSelectorRunner(
        MetaSelectorConfig(
            walk_forward_dir=str(root),
            output_dir=str(tmp_path / "artifacts" / "meta_selector"),
            minimum_prior_folds_per_regime=1,
            minimum_samples_per_regime=1,
        )
    )
    result = runner.run()

    stability = result["selector_stability_summary"]
    assert not stability.empty
    assert "candidate_label" in stability.columns

    baseline_summary = result["selector_vs_baselines_summary"]
    assert "fixed_best_global_setup" in set(baseline_summary["entity_name"])
    assert "naive_global_baseline" in set(baseline_summary["entity_name"])

    report = result["overall_meta_selector_report"]
    assert {"component", "best_overall_choice", "stability_level"} <= set(report.columns)
    for path in result["summary_paths"].values():
        assert Path(path).exists()
