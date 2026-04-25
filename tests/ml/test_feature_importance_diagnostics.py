from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.backtest.feature_importance_diagnostics import (
    FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS,
    FEATURE_IMPORTANCE_STABILITY_COLUMNS,
    LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS,
    compare_linear_and_importance_diagnostics,
    empty_feature_importance_diagnostics_frame,
    empty_feature_importance_stability_summary_frame,
    empty_linear_vs_importance_comparison_frame,
    extract_feature_importance_rows,
    importance_stability_level,
    summarize_feature_importance_stability,
)


def _fold_context(fold_id: str = "fold_001") -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "step_size": 1,
        "forecast_sequence_index": 0,
        "ticker": "AAA",
        "prediction_date": "2024-02-01",
        "model": "xgboost",
        "horizon": "short_5d",
        "task": "return",
        "train_start": "2024-01-01",
        "train_end": "2024-01-31",
        "eval_start": "2024-02-01",
        "eval_end": "2024-02-08",
    }


class _SupportedWrappedModel:
    def __init__(self, importances: list[float]) -> None:
        self.model = type("_InnerModel", (), {"feature_importances_": np.asarray(importances, dtype=float)})()


class _SupportedEstimatorModel:
    def __init__(self, importances: list[float]) -> None:
        self.estimator = type("_Estimator", (), {"feature_importances_": np.asarray(importances, dtype=float)})()


class _UnsupportedModel:
    pass


def test_extract_feature_importance_rows_normalizes_and_ranks_supported_model() -> None:
    rows = extract_feature_importance_rows(
        model_name="xgboost",
        model=_SupportedWrappedModel([2.0, 1.0, 0.0]),
        feature_columns=["f_one", "f_two", "f_three"],
        fold_context=_fold_context(),
    )

    assert list(rows.columns) == FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS
    assert rows["importance_normalized"].sum() == 1.0
    assert rows.loc[rows["feature"] == "f_one", "importance_rank"].iloc[0] == 1
    assert rows.loc[rows["feature"] == "f_three", "importance_normalized"].iloc[0] == 0.0


def test_extract_feature_importance_rows_supports_estimator_attribute() -> None:
    rows = extract_feature_importance_rows(
        model_name="random_forest",
        model=_SupportedEstimatorModel([0.25, 0.75]),
        feature_columns=["f_one", "f_two"],
        fold_context=_fold_context(),
    )

    assert set(rows["model"]) == {"random_forest"}
    assert rows.loc[rows["feature"] == "f_two", "importance_rank"].iloc[0] == 1


def test_unsupported_or_mismatched_models_return_empty_rows_without_error() -> None:
    unsupported = extract_feature_importance_rows(
        model_name="linear",
        model=_UnsupportedModel(),
        feature_columns=["f_one"],
        fold_context=_fold_context(),
    )
    mismatched = extract_feature_importance_rows(
        model_name="cart",
        model=_SupportedWrappedModel([0.5, 0.5]),
        feature_columns=["f_one"],
        fold_context=_fold_context(),
    )

    assert unsupported.empty
    assert mismatched.empty


def test_feature_importance_stability_summary_counts_top_rank_ratios_and_levels() -> None:
    rows = pd.DataFrame(
        [
            {**_fold_context("fold_001"), "feature": "f_high", "importance": 0.4, "importance_rank": 1, "importance_normalized": 0.4},
            {**_fold_context("fold_002"), "feature": "f_high", "importance": 0.3, "importance_rank": 2, "importance_normalized": 0.3},
            {**_fold_context("fold_003"), "feature": "f_high", "importance": 0.2, "importance_rank": 3, "importance_normalized": 0.2},
            {**_fold_context("fold_001"), "feature": "f_medium", "importance": 0.1, "importance_rank": 8, "importance_normalized": 0.1},
            {**_fold_context("fold_002"), "feature": "f_medium", "importance": 0.0, "importance_rank": 12, "importance_normalized": 0.0},
            {**_fold_context("fold_003"), "feature": "f_medium", "importance": 0.1, "importance_rank": 10, "importance_normalized": 0.1},
            {**_fold_context("fold_001"), "feature": "f_low", "importance": 0.1, "importance_rank": 11, "importance_normalized": 0.1},
            {**_fold_context("fold_002"), "feature": "f_low", "importance": 0.1, "importance_rank": 12, "importance_normalized": 0.1},
        ]
    )

    summary = summarize_feature_importance_stability(rows)
    high = summary[summary["feature"] == "f_high"].iloc[0]
    medium = summary[summary["feature"] == "f_medium"].iloc[0]
    low = summary[summary["feature"] == "f_low"].iloc[0]

    assert list(summary.columns) == FEATURE_IMPORTANCE_STABILITY_COLUMNS
    assert high["top_10_ratio"] == 1.0
    assert high["importance_stability_level"] == "high"
    assert medium["top_10_count"] == 2
    assert medium["top_10_ratio"] == 2 / 3
    assert medium["importance_stability_level"] == "medium"
    assert low["fold_count"] == 2
    assert low["importance_stability_level"] == "low"


def test_importance_stability_level_rule_requires_three_folds() -> None:
    assert importance_stability_level(1.0, 2) == "low"
    assert importance_stability_level(0.8, 3) == "high"
    assert importance_stability_level(0.5, 3) == "medium"
    assert importance_stability_level(0.49, 3) == "low"


def test_linear_vs_importance_comparison_assigns_alignment_labels() -> None:
    linear_summary = pd.DataFrame(
        [
            {"model": "linear", "horizon": "short_5d", "task": "return", "feature": "f_aligned", "mean_abs_coefficient": 0.2, "sign_consistency_ratio": 1.0, "stability_level": "high"},
            {"model": "ridge", "horizon": "short_5d", "task": "return", "feature": "f_linear", "mean_abs_coefficient": 0.1, "sign_consistency_ratio": 0.7, "stability_level": "medium"},
            {"model": "lasso", "horizon": "short_5d", "task": "return", "feature": "f_unstable", "mean_abs_coefficient": 0.01, "sign_consistency_ratio": 0.2, "stability_level": "low"},
        ]
    )
    importance_summary = pd.DataFrame(
        [
            {"model": "xgboost", "horizon": "short_5d", "task": "return", "feature": "f_aligned", "mean_importance_normalized": 0.3, "top_10_ratio": 1.0, "importance_stability_level": "high"},
            {"model": "cart", "horizon": "short_5d", "task": "return", "feature": "f_importance", "mean_importance_normalized": 0.2, "top_10_ratio": 0.6, "importance_stability_level": "medium"},
        ]
    )

    comparison = compare_linear_and_importance_diagnostics(
        linear_summary=linear_summary,
        importance_summary=importance_summary,
    )
    labels = dict(zip(comparison["feature"], comparison["alignment_label"]))

    assert list(comparison.columns) == LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS
    assert labels["f_aligned"] == "aligned_stable"
    assert labels["f_linear"] == "linear_only"
    assert labels["f_importance"] == "importance_only"
    assert labels["f_unstable"] == "unstable_or_missing"


def test_empty_output_frames_expose_required_csv_schema() -> None:
    assert list(empty_feature_importance_diagnostics_frame().columns) == FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS
    assert list(empty_feature_importance_stability_summary_frame().columns) == FEATURE_IMPORTANCE_STABILITY_COLUMNS
    assert list(empty_linear_vs_importance_comparison_frame().columns) == LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS
