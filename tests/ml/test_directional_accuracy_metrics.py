import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.run_vn100_hybrid_frequency_accuracy_benchmark as benchmark_runner
from scripts.run_vn100_hybrid_frequency_accuracy_benchmark import (
    _benchmark_usability_metrics,
    _binomial_p_value_greater,
    _bootstrap_accuracy_ci,
    _training_label_mask,
    build_confidence_filter_summary,
    build_confidence_threshold_sweep_summary,
    build_strategy_selection_summary,
    run_frequency_benchmark,
)
from src.ml.metrics import compute_directional_accuracy_from_returns


def test_directional_accuracy_basic_correct_incorrect_sign_matching():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, -0.20, 0.30, -0.40],
        predicted_return=[0.05, -0.10, -0.20, 0.10],
    )

    assert result["n_obs"] == 4
    assert result["accuracy"] == 0.5


def test_directional_accuracy_ignores_nan_values():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, np.nan, -0.30, 0.20],
        predicted_return=[0.01, 0.20, np.nan, -0.10],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 0.5


def test_directional_accuracy_ignores_infinite_values():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, np.inf, -0.30, -np.inf],
        predicted_return=[0.01, 0.20, -0.10, -0.10],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 1.0


def test_directional_accuracy_ignores_zero_actual_returns():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.0, 0.10, -0.10, 0.0],
        predicted_return=[0.50, 0.20, 0.20, -0.50],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 0.5


def test_directional_accuracy_empty_input_returns_nan_accuracy():
    result = compute_directional_accuracy_from_returns(actual_return=[], predicted_return=[])

    assert result["n_obs"] == 0
    assert math.isnan(result["accuracy"])


def test_benchmark_significance_helpers_return_bounded_values():
    p_value = _binomial_p_value_greater(successes=8, n_obs=10, null_accuracy=0.50)
    ci_low, ci_high = _bootstrap_accuracy_ci(pd.Series([1, 1, 1, 0, 1, 0]), samples=200, seed=7)

    assert p_value is None or 0.0 <= p_value <= 1.0
    assert ci_low is not None and ci_high is not None
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_confidence_filter_summary_reports_coverage_and_filtered_accuracy():
    predictions = pd.DataFrame(
        [
            {"frequency": "daily", "model": "xgboost", "horizon": 1, "is_correct": 1, "filtered_out": False},
            {"frequency": "daily", "model": "xgboost", "horizon": 1, "is_correct": 0, "filtered_out": True},
            {"frequency": "daily", "model": "xgboost", "horizon": 1, "is_correct": 1, "filtered_out": False},
        ]
    )

    summary = build_confidence_filter_summary(
        predictions,
        enabled=True,
        confidence_threshold=0.55,
        min_coverage_after_filter=0.30,
        threshold=0.60,
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["total_rows"] == 3
    assert row["evaluated_rows"] == 2
    assert bool(row["coverage_ok"]) is True
    assert row["filtered_accuracy"] == 1.0


def test_confidence_threshold_sweep_selects_best_covered_candidate():
    predictions = pd.DataFrame(
        [
            {
                "frequency": "hourly",
                "model": "stacking",
                "horizon": 1,
                "target_mode": "classification",
                "confidence": 0.51,
                "is_correct": 0,
            },
            {
                "frequency": "hourly",
                "model": "stacking",
                "horizon": 1,
                "target_mode": "classification",
                "confidence": 0.56,
                "is_correct": 1,
            },
            {
                "frequency": "hourly",
                "model": "stacking",
                "horizon": 1,
                "target_mode": "classification",
                "confidence": 0.66,
                "is_correct": 1,
            },
            {
                "frequency": "hourly",
                "model": "stacking",
                "horizon": 1,
                "target_mode": "classification",
                "confidence": 0.72,
                "is_correct": 0,
            },
        ]
    )

    summary = build_confidence_threshold_sweep_summary(
        predictions,
        enabled=True,
        thresholds=[0.50, 0.55, 0.65, 0.70],
        min_sweep_coverage=0.50,
        global_threshold=0.60,
    )

    selected = summary[summary["selected_candidate"].astype(bool)]
    assert len(selected) == 1
    row = selected.iloc[0]
    assert row["threshold"] == 0.55
    assert row["evaluated_rows"] == 3
    assert row["coverage_ratio"] == 0.75
    assert row["filtered_accuracy"] == 2 / 3
    assert bool(row["passed_60pct"]) is True


def _train_cutoff_test_config(**overrides):
    values = {
        "train_cutoff": "2024-12-31",
        "eval_start": "2025-01-01",
        "eval_end": "2025-01-05",
        "coverage_end_tolerance_days": 0,
        "min_pre_eval_rows_daily": 2,
        "min_pre_eval_rows_hourly": 2,
        "min_eval_rows_daily": 2,
        "min_eval_rows_hourly": 2,
        "target_mode": "classification",
        "enable_regime_evaluation": False,
        "enable_confidence_filter": True,
        "confidence_threshold": 0.55,
        "no_trade_band": 0.0,
        "enable_confidence_threshold_sweep": False,
        "enable_horizon_tuning": False,
        "tuning_models": [],
        "tuning_trials": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_benchmark_usability_counts_eval_rows_after_train_cutoff():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-12-30 10:00",
                    "2024-12-31 10:00",
                    "2025-01-02 10:00",
                    "2025-01-03 10:00",
                ]
            ),
            "ticker": ["AAA"] * 4,
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.5, 10.6, 10.7, 10.8],
            "low": [9.8, 9.9, 10.0, 10.1],
            "close": [10.2, 10.3, 10.4, 10.5],
            "volume": [1000, 1001, 1002, 1003],
        }
    )
    config = _train_cutoff_test_config(eval_end="2025-01-03")

    usability = _benchmark_usability_metrics(frame, "hourly", config)

    assert usability["benchmark_usable"] is True
    assert usability["pre_eval_rows"] == 2
    assert usability["eval_rows"] == 2


def test_training_label_mask_excludes_2025_targets_with_train_cutoff():
    labeled = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-12-29", "2024-12-30", "2024-12-31"]),
            "target_timestamp": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"]),
            "actual_return": [0.01, -0.01, 0.02],
        }
    )
    config = _train_cutoff_test_config()

    mask = _training_label_mask(
        labeled,
        initial_train_start="2024-01-01",
        forecast_chunk_start=pd.Timestamp("2025-02-01"),
        config=config,
    )

    assert mask.tolist() == [True, True, False]


def test_train_cutoff_allows_2025_eval_predictions_without_2025_training_labels(monkeypatch):
    dates = pd.bdate_range("2024-06-03", "2025-01-20")
    close = np.linspace(10.0, 15.0, len(dates))
    raw = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0,
        }
    )
    captured_train_targets = []

    def fake_fit_predict_classifier(model_name, train_df, predict_df, feature_columns, seed, params=None):
        captured_train_targets.append(pd.to_datetime(train_df["target_timestamp"]).max())
        return predict_df["actual_direction"].astype(int).to_numpy(), np.full(len(predict_df), 0.80)

    monkeypatch.setattr(benchmark_runner, "fit_predict_classifier", fake_fit_predict_classifier)
    config = _train_cutoff_test_config(eval_end="2025-01-10")

    predictions, _, summary, _, errors, _, _ = run_frequency_benchmark(
        raw_df=raw,
        frequency="daily",
        horizons=[1],
        models=["xgboost"],
        initial_train_start="2024-06-03",
        initial_train_end="2024-12-31",
        eval_start="2025-01-02",
        eval_end="2025-01-10",
        threshold=0.60,
        provider="synthetic",
        universe="TEST",
        retrain_frequency="never",
        seed=42,
        min_history_days=None,
        min_obs_per_group=1,
        max_daily_gap_days=30,
        config=config,
    )

    assert errors.empty
    assert not predictions.empty
    assert pd.to_datetime(predictions["timestamp"]).min() >= pd.Timestamp("2025-01-02")
    assert captured_train_targets
    assert max(captured_train_targets) <= pd.Timestamp("2024-12-31")
    assert summary["train_cutoff"] == "2024-12-31"
    assert summary["training_label_cutoff_rule"] == "target_timestamp <= train_cutoff"


def test_strategy_selection_summary_includes_regime_strategy_level_pass_63():
    rows = []
    for idx in range(400):
        rows.append(
            {
                "frequency": "daily",
                "model": "xgboost",
                "horizon": 20,
                "target_mode": "classification",
                "regime": "bear",
                "volatility_regime": "low_volatility",
                "is_correct": 1 if idx < 260 else 0,
            }
        )
    for idx in range(600):
        rows.append(
            {
                "frequency": "daily",
                "model": "xgboost",
                "horizon": 20,
                "target_mode": "classification",
                "regime": "sideways",
                "volatility_regime": "low_volatility",
                "is_correct": 0,
            }
        )
    predictions = pd.DataFrame(rows)

    summary = build_strategy_selection_summary(
        predictions,
        pd.DataFrame(),
        threshold=0.60,
    )

    bear = summary[
        (summary["frequency"] == "daily")
        & (summary["model"] == "xgboost")
        & (summary["horizon"] == 20)
        & (summary["regime"] == "bear")
    ].iloc[0]
    assert bear["candidate_type"] == "regime"
    assert bear["n_obs"] == 400
    assert bear["total_eligible_rows"] == 1000
    assert bear["coverage_ratio"] == 0.4
    assert bear["accuracy"] == 0.65
    assert bool(bear["pass_63"]) is True
    assert bear["pass_level"] == "regime_strategy_level"
    assert bool(bear["selected_candidate"]) is True
