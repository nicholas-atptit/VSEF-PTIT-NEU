from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.backtest.context_coverage_diagnostics import (
    CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS,
    CONTEXT_COVERAGE_SUMMARY_COLUMNS,
    build_context_coverage_rows,
    coverage_warning_level,
    summarize_context_coverage,
)
from src.ml.data_loader import apply_context_features
from src.ml.feature_engineering import FeatureEngineer


def _fold_context(**overrides) -> dict[str, object]:
    context = {
        "ticker": "AAA",
        "fold_id": "fold_001",
        "step_size": 1,
        "forecast_sequence_index": 0,
        "prediction_date": "2024-01-05",
        "horizon": "short_5d",
        "train_start": "2024-01-02",
        "train_end": "2024-01-05",
        "eval_start": "2024-01-05",
        "eval_end": "2024-01-12",
    }
    context.update(overrides)
    return context


def test_context_coverage_rows_compute_counts_and_rates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "breadth_context_available": [True, True, False, False],
            "breadth_context_missing": [False, False, True, True],
            "foreign_flow_context_available": [True, False, True, False],
            "foreign_flow_context_missing": [False, True, False, True],
        }
    )

    rows = build_context_coverage_rows(feature_frame=frame, fold_context=_fold_context())
    row = rows.iloc[0]

    assert list(rows.columns) == CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS
    assert row["row_count"] == 4
    assert row["breadth_available_count"] == 2
    assert row["breadth_missing_count"] == 2
    assert row["breadth_available_rate"] == 0.5
    assert row["breadth_missing_rate"] == 0.5
    assert row["foreign_flow_available_count"] == 2
    assert row["foreign_flow_missing_count"] == 2
    assert row["foreign_flow_missing_rate"] == 0.5
    assert row["coverage_warning_level"] == "weak_coverage"


def test_coverage_warning_level_rule() -> None:
    assert coverage_warning_level(0.0, 0.05) == "ok"
    assert coverage_warning_level(0.051, 0.0) == "review"
    assert coverage_warning_level(0.0, 0.25) == "review"
    assert coverage_warning_level(0.251, 0.0) == "weak_coverage"
    assert coverage_warning_level(np.nan, np.nan) == "metadata_unavailable"


def test_context_coverage_summary_aggregates_fold_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                **_fold_context(fold_id="fold_001"),
                "row_count": 100,
                "breadth_missing_rate": 0.0,
                "foreign_flow_missing_rate": 0.02,
                "coverage_warning_level": "ok",
            },
            {
                **_fold_context(fold_id="fold_002"),
                "row_count": 100,
                "breadth_missing_rate": 0.10,
                "foreign_flow_missing_rate": 0.02,
                "coverage_warning_level": "review",
            },
            {
                **_fold_context(fold_id="fold_003"),
                "row_count": 100,
                "breadth_missing_rate": 0.30,
                "foreign_flow_missing_rate": 0.02,
                "coverage_warning_level": "weak_coverage",
            },
        ]
    )

    summary = summarize_context_coverage(rows)
    row = summary.iloc[0]

    assert list(summary.columns) == CONTEXT_COVERAGE_SUMMARY_COLUMNS
    assert row["fold_count"] == 3
    assert row["mean_breadth_missing_rate"] == 0.13333333333333333
    assert row["max_breadth_missing_rate"] == 0.30
    assert row["weak_coverage_fold_count"] == 1
    assert row["review_fold_count"] == 1
    assert row["overall_coverage_warning_level"] == "weak_coverage"


def test_missing_metadata_columns_do_not_crash_coverage_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close_return_5d": [0.01, -0.02],
        }
    )

    rows = build_context_coverage_rows(feature_frame=frame, fold_context=_fold_context())
    row = rows.iloc[0]

    assert row["row_count"] == 2
    assert pd.isna(row["breadth_missing_rate"])
    assert pd.isna(row["foreign_flow_missing_rate"])
    assert row["coverage_warning_level"] == "metadata_unavailable"
    assert row["coverage_metadata_status"] == "missing_breadth_and_foreign_flow"


def test_context_metadata_columns_remain_excluded_from_model_features() -> None:
    dates = pd.bdate_range("2024-01-02", periods=70)
    close = pd.Series(np.linspace(100.0, 110.0, len(dates)))
    df = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, len(dates)),
        }
    )
    breadth = pd.DataFrame({"date": dates, "market_breadth": 0.0})
    foreign_flow = pd.DataFrame({"ticker": "AAA", "date": dates, "foreign_net_value": 0.0})
    contextual = apply_context_features(df, "AAA", breadth_df=breadth, foreign_flow_df=foreign_flow)
    feature_frame = FeatureEngineer().build_feature_frame(contextual, build_mode="fast_core_mode")

    feature_columns = FeatureEngineer().get_feature_columns(feature_frame)

    assert "breadth_context_available" in feature_frame.columns
    assert "foreign_flow_context_available" in feature_frame.columns
    assert "breadth_context_available" not in feature_columns
    assert "foreign_flow_context_available" not in feature_columns
