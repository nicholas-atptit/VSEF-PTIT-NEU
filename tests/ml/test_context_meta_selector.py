from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.context_meta_selector import (
    ContextConditionedMetaSelectorRunner,
    ContextMetaSelectorConfig,
)
from src.ml.backtest.meta_selector import MetaSelectorConfig, RegimeConditionedMetaSelectorRunner


def _regime_row(
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
    market_return: float,
    market_volatility: float,
) -> dict:
    normalized_predicted_return = 0.9 if predicted_return > 0.03 else 0.4
    return {
        "date": date,
        "prediction_date": prediction_date,
        "target_date": date,
        "ticker": ticker,
        "model_name": model_name,
        "horizon": horizon,
        "horizon_days": 20 if horizon == "20d" else 3,
        "actual_return": actual_return,
        "predicted_return": predicted_return,
        "actual_profit_label": int(actual_return > 0),
        "predicted_profit_label": int(probability >= 0.55),
        "predicted_profit_probability": probability,
        "combined_score": 0.5 * normalized_predicted_return + 0.5 * probability,
        "rank_based_joint_score": 0.5 * normalized_predicted_return + 0.25,
        "normalized_predicted_return": normalized_predicted_return,
        "market_return_lookback": market_return,
        "market_volatility_lookback": market_volatility,
        "regime": regime,
    }


def _write_walk_forward_fold(
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
                "available_algorithms": "xgboost,lightgbm",
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

    for horizon in {"3d", "20d"}:
        horizon_dir = fold_dir / "regime_aware" / horizon
        horizon_dir.mkdir(parents=True, exist_ok=True)
        horizon_rows = [row for row in regime_rows if row["horizon"] == horizon]
        pd.DataFrame(horizon_rows).to_csv(horizon_dir / "regime_labeled_signal_table.csv", index=False)


def _run_meta_selector(walk_forward_dir: Path, output_dir: Path) -> None:
    runner = RegimeConditionedMetaSelectorRunner(
        MetaSelectorConfig(
            walk_forward_dir=str(walk_forward_dir),
            output_dir=str(output_dir),
            minimum_prior_folds_per_regime=1,
            minimum_samples_per_regime=1,
        )
    )
    runner.run()


def test_context_selector_uses_prior_folds_only(tmp_path) -> None:
    walk_forward_dir = tmp_path / "artifacts" / "walk_forward"
    meta_selector_dir = tmp_path / "artifacts" / "meta_selector"

    fold1_rows = [
        _regime_row(
            date="2023-01-25",
            prediction_date="2023-01-05",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="20d",
            actual_return=0.06,
            predicted_return=0.05,
            probability=0.82,
            market_return=0.04,
            market_volatility=0.02,
        ),
        _regime_row(
            date="2023-01-25",
            prediction_date="2023-01-05",
            ticker="AAA",
            regime="bull",
            model_name="lightgbm",
            horizon="20d",
            actual_return=0.01,
            predicted_return=0.01,
            probability=0.55,
            market_return=0.04,
            market_volatility=0.02,
        ),
    ]
    _write_walk_forward_fold(
        walk_forward_dir,
        fold_id="fold_001",
        fold_number=1,
        eval_start="2023-01-01",
        regime_rows=fold1_rows,
        model_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.8,
                "positive_class_precision": 0.85,
                "rank_regression_rmse": 1.0,
            },
            {
                "fold_id": "fold_001",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "directional_accuracy": 0.5,
                "positive_class_precision": 0.55,
                "rank_regression_rmse": 2.0,
            },
        ],
        combined_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.06,
                "profit_rate": 0.85,
            },
            {
                "fold_id": "fold_001",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.01,
                "profit_rate": 0.55,
            },
        ],
    )

    fold2_rows = [
        _regime_row(
            date="2023-06-30",
            prediction_date="2023-06-05",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="20d",
            actual_return=0.04,
            predicted_return=0.04,
            probability=0.80,
            market_return=0.03,
            market_volatility=0.02,
        ),
        _regime_row(
            date="2023-06-30",
            prediction_date="2023-06-05",
            ticker="AAA",
            regime="bull",
            model_name="lightgbm",
            horizon="20d",
            actual_return=0.00,
            predicted_return=0.01,
            probability=0.54,
            market_return=0.03,
            market_volatility=0.02,
        ),
    ]
    _write_walk_forward_fold(
        walk_forward_dir,
        fold_id="fold_002",
        fold_number=2,
        eval_start="2023-06-01",
        regime_rows=fold2_rows,
        model_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.75,
                "positive_class_precision": 0.80,
                "rank_regression_rmse": 1.0,
            },
            {
                "fold_id": "fold_002",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "directional_accuracy": 0.45,
                "positive_class_precision": 0.52,
                "rank_regression_rmse": 2.0,
            },
        ],
        combined_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.04,
                "profit_rate": 0.80,
            },
            {
                "fold_id": "fold_002",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.00,
                "profit_rate": 0.52,
            },
        ],
    )

    fold3_rows = [
        _regime_row(
            date="2023-12-29",
            prediction_date="2023-12-05",
            ticker="AAA",
            regime="bull",
            model_name="xgboost",
            horizon="20d",
            actual_return=0.01,
            predicted_return=0.02,
            probability=0.55,
            market_return=0.02,
            market_volatility=0.02,
        ),
        _regime_row(
            date="2023-12-29",
            prediction_date="2023-12-05",
            ticker="AAA",
            regime="bull",
            model_name="lightgbm",
            horizon="20d",
            actual_return=0.07,
            predicted_return=0.05,
            probability=0.85,
            market_return=0.02,
            market_volatility=0.02,
        ),
    ]
    _write_walk_forward_fold(
        walk_forward_dir,
        fold_id="fold_003",
        fold_number=3,
        eval_start="2023-12-01",
        regime_rows=fold3_rows,
        model_rows=[
            {
                "fold_id": "fold_003",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "directional_accuracy": 0.45,
                "positive_class_precision": 0.55,
                "rank_regression_rmse": 2.0,
            },
            {
                "fold_id": "fold_003",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "directional_accuracy": 0.85,
                "positive_class_precision": 0.90,
                "rank_regression_rmse": 1.0,
            },
        ],
        combined_rows=[
            {
                "fold_id": "fold_003",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "xgboost",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.01,
                "profit_rate": 0.55,
            },
            {
                "fold_id": "fold_003",
                "horizon": "20d",
                "regime": "bull",
                "model_name": "lightgbm",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 40,
                "average_actual_return": 0.07,
                "profit_rate": 0.90,
            },
        ],
    )

    _run_meta_selector(walk_forward_dir, meta_selector_dir)

    runner = ContextConditionedMetaSelectorRunner(
        ContextMetaSelectorConfig(
            walk_forward_dir=str(walk_forward_dir),
            meta_selector_dir=str(meta_selector_dir),
            audit_output_dir=str(tmp_path / "artifacts" / "meta_selector_audit"),
            output_dir=str(tmp_path / "artifacts" / "context_meta_selector"),
            minimum_prior_samples_for_context_match=1,
            minimum_prior_folds=1,
            knn_neighbors=5,
        )
    )
    runner.run()

    fold2_selected = pd.read_csv(Path(runner.output_dir) / "fold_002" / "selected_candidates.csv")
    assert not fold2_selected.empty
    assert set(fold2_selected["selected_model_name"]) == {"xgboost"}


def test_context_selector_audit_and_fallback_outputs(tmp_path) -> None:
    walk_forward_dir = tmp_path / "artifacts" / "walk_forward"
    meta_selector_dir = tmp_path / "artifacts" / "meta_selector"

    fold1_rows = [
        _regime_row(
            date="2023-03-15",
            prediction_date="2023-03-10",
            ticker="AAA",
            regime="bear",
            model_name="xgboost",
            horizon="3d",
            actual_return=-0.01,
            predicted_return=-0.01,
            probability=0.40,
            market_return=-0.05,
            market_volatility=0.03,
        ),
        _regime_row(
            date="2023-03-15",
            prediction_date="2023-03-10",
            ticker="AAA",
            regime="bear",
            model_name="lightgbm",
            horizon="3d",
            actual_return=-0.02,
            predicted_return=-0.02,
            probability=0.35,
            market_return=-0.05,
            market_volatility=0.03,
        ),
    ]
    _write_walk_forward_fold(
        walk_forward_dir,
        fold_id="fold_001",
        fold_number=1,
        eval_start="2023-03-10",
        regime_rows=fold1_rows,
        model_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "3d",
                "regime": "bear",
                "model_name": "xgboost",
                "directional_accuracy": 0.6,
                "positive_class_precision": 0.4,
                "rank_regression_rmse": 1.0,
            }
        ],
        combined_rows=[
            {
                "fold_id": "fold_001",
                "horizon": "3d",
                "regime": "bear",
                "model_name": "xgboost",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 10,
                "average_actual_return": -0.01,
                "profit_rate": 0.20,
            }
        ],
    )
    fold2_rows = [
        _regime_row(
            date="2023-04-15",
            prediction_date="2023-04-10",
            ticker="AAA",
            regime="bear",
            model_name="xgboost",
            horizon="3d",
            actual_return=-0.01,
            predicted_return=-0.01,
            probability=0.40,
            market_return=-0.04,
            market_volatility=0.04,
        ),
    ]
    _write_walk_forward_fold(
        walk_forward_dir,
        fold_id="fold_002",
        fold_number=2,
        eval_start="2023-04-10",
        regime_rows=fold2_rows,
        model_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "3d",
                "regime": "bear",
                "model_name": "xgboost",
                "directional_accuracy": 0.6,
                "positive_class_precision": 0.4,
                "rank_regression_rmse": 1.0,
            }
        ],
        combined_rows=[
            {
                "fold_id": "fold_002",
                "horizon": "3d",
                "regime": "bear",
                "model_name": "xgboost",
                "ranking_method": "predicted_return",
                "return_threshold": None,
                "probability_threshold": None,
                "top_k": 3,
                "observations": 10,
                "average_actual_return": -0.01,
                "profit_rate": 0.20,
            }
        ],
    )

    _run_meta_selector(walk_forward_dir, meta_selector_dir)

    runner = ContextConditionedMetaSelectorRunner(
        ContextMetaSelectorConfig(
            walk_forward_dir=str(walk_forward_dir),
            meta_selector_dir=str(meta_selector_dir),
            audit_output_dir=str(tmp_path / "artifacts" / "meta_selector_audit"),
            output_dir=str(tmp_path / "artifacts" / "context_meta_selector"),
            minimum_prior_samples_for_context_match=100,
            minimum_prior_folds=2,
            knn_neighbors=5,
        )
    )
    result = runner.run()

    audit_dir = Path(runner.audit_output_dir)
    assert (audit_dir / "benchmark_audit_report.csv").exists()
    assert (audit_dir / "baseline_definition_check.csv").exists()

    fold2_selected = pd.read_csv(Path(runner.output_dir) / "fold_002" / "selected_candidates.csv")
    assert not fold2_selected.empty
    assert bool(fold2_selected["fallback_used"].all())

    summary = result["context_selector_vs_baselines_summary"]
    assert not summary.empty
    assert "fixed_best_global_setup" in set(summary["entity_name"])
    assert Path(result["summary_paths"]["overall_context_selector_report"]).exists()
