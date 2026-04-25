from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.backtest.linear_fold_diagnostics import (
    COEFFICIENT_DIAGNOSTIC_COLUMNS,
    COEFFICIENT_STABILITY_COLUMNS,
    coefficient_rows_from_metadata,
    coefficient_stability_level,
    empty_coefficient_diagnostics_frame,
    empty_coefficient_stability_summary_frame,
    fit_linear_fold_diagnostics,
    summarize_coefficient_stability,
)


def _fold_context(fold_id: str = "fold_001") -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "step_size": 1,
        "forecast_sequence_index": 0,
        "ticker": "AAA",
        "prediction_date": "2024-02-01",
        "horizon": "short_5d",
        "horizon_days": 5,
        "task": "return",
        "train_start": "2024-01-01",
        "train_end": "2024-01-31",
        "eval_start": "2024-02-01",
        "eval_end": "2024-02-08",
    }


def test_stability_summary_calculates_sign_consistency_and_levels() -> None:
    rows = pd.DataFrame(
        [
            {**_fold_context("fold_001"), "model": "linear", "feature": "f_high", "coefficient": 0.10, "coefficient_sign": "positive", "coefficient_magnitude": 0.10, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_002"), "model": "linear", "feature": "f_high", "coefficient": 0.20, "coefficient_sign": "positive", "coefficient_magnitude": 0.20, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_003"), "model": "linear", "feature": "f_high", "coefficient": 0.15, "coefficient_sign": "positive", "coefficient_magnitude": 0.15, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_001"), "model": "linear", "feature": "f_medium", "coefficient": -0.10, "coefficient_sign": "negative", "coefficient_magnitude": 0.10, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_002"), "model": "linear", "feature": "f_medium", "coefficient": -0.20, "coefficient_sign": "negative", "coefficient_magnitude": 0.20, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_003"), "model": "linear", "feature": "f_medium", "coefficient": 0.05, "coefficient_sign": "positive", "coefficient_magnitude": 0.05, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_001"), "model": "linear", "feature": "f_low", "coefficient": 0.00, "coefficient_sign": "zero", "coefficient_magnitude": 0.00, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
            {**_fold_context("fold_002"), "model": "linear", "feature": "f_low", "coefficient": 0.10, "coefficient_sign": "positive", "coefficient_magnitude": 0.10, "intercept": 0.0, "nonzero_coefficient_count": 2, "feature_count": 2},
        ]
    )

    summary = summarize_coefficient_stability(rows)
    high = summary[summary["feature"] == "f_high"].iloc[0]
    medium = summary[summary["feature"] == "f_medium"].iloc[0]
    low = summary[summary["feature"] == "f_low"].iloc[0]

    assert high["fold_count"] == 3
    assert high["sign_consistency_ratio"] == 1.0
    assert high["stability_level"] == "high"
    assert medium["sign_negative_count"] == 2
    assert medium["sign_consistency_ratio"] == 2 / 3
    assert medium["stability_level"] == "medium"
    assert low["fold_count"] == 2
    assert low["stability_level"] == "low"


def test_stability_level_rule_requires_three_folds() -> None:
    assert coefficient_stability_level(1.0, 2) == "low"
    assert coefficient_stability_level(0.8, 3) == "high"
    assert coefficient_stability_level(0.6, 3) == "medium"
    assert coefficient_stability_level(0.5, 3) == "low"


def test_linear_diagnostic_fit_uses_metadata_and_skips_non_linear_names() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    x1 = np.linspace(-1.0, 1.0, len(dates))
    x2 = np.linspace(1.0, -1.0, len(dates))
    train_frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "feature_one": x1,
            "feature_two": x2,
            "target_return_short_5d": 0.03 * x1 - 0.01 * x2 + 0.001,
        }
    )

    diagnostics = fit_linear_fold_diagnostics(
        train_frame=train_frame,
        feature_columns=["feature_one", "feature_two"],
        target_column="target_return_short_5d",
        fold_context=_fold_context(),
        model_names=["linear", "cart"],
    )

    assert set(COEFFICIENT_DIAGNOSTIC_COLUMNS) == set(diagnostics.columns)
    assert set(diagnostics["model"]) == {"linear"}
    assert diagnostics["feature_count"].eq(2).all()
    assert set(diagnostics["coefficient_sign"]) <= {"positive", "negative", "zero"}


def test_non_linear_metadata_is_ignored_without_error() -> None:
    rows = coefficient_rows_from_metadata(
        model_name="cart",
        metadata={"model_name": "cart"},
        fold_context=_fold_context(),
    )

    assert rows == []


def test_empty_output_frames_expose_required_csv_schema() -> None:
    assert list(empty_coefficient_diagnostics_frame().columns) == COEFFICIENT_DIAGNOSTIC_COLUMNS
    assert list(empty_coefficient_stability_summary_frame().columns) == COEFFICIENT_STABILITY_COLUMNS
