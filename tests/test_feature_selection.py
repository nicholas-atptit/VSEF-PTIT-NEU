from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.features import (
    WalkForwardFeatureSelectionConfig,
    build_feature_panel,
    detect_forward_fill_boundary_leakage,
    detect_future_join_leakage,
    load_feature_registry,
    run_walk_forward_feature_selection,
    validate_walk_forward_folds,
)


def _make_raw_panel(num_days: int = 520) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    tickers = ["AAA", "BBB", "CCC"]
    for idx, ticker in enumerate(tickers):
        frame = generate_mock_data(ticker=ticker, num_days=num_days, seed=42 + idx)
        base_return = pd.Series(frame["close"]).pct_change().fillna(0.0)
        frame["ticker"] = ticker
        frame["m_ret"] = base_return.rolling(3, min_periods=1).mean().fillna(0.0) * 0.9
        frame["s_ret"] = base_return.rolling(5, min_periods=1).mean().fillna(0.0) * 0.7
        frame["market_breadth"] = np.linspace(-0.15, 0.20, len(frame))
        frame["advancing_share"] = np.linspace(0.42, 0.62, len(frame))
        frame["advance_decline_ratio"] = np.linspace(0.8, 1.3, len(frame))
        frame["sector_dispersion"] = np.linspace(0.01, 0.03, len(frame))
        frame["pct_above_ma20"] = np.linspace(0.45, 0.60, len(frame))
        frame["pct_above_ma50"] = np.linspace(0.40, 0.55, len(frame))
        frame["new_highs_252"] = np.linspace(4, 20, len(frame))
        frame["new_lows_252"] = np.linspace(15, 2, len(frame))
        frame["new_high_low_spread"] = np.linspace(-0.20, 0.25, len(frame))
        frame["up_volume"] = frame["volume"] * np.linspace(0.55, 0.75, len(frame))
        frame["down_volume"] = frame["volume"] * np.linspace(0.45, 0.25, len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_validate_walk_forward_folds_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        validate_walk_forward_folds(
            pd.DataFrame(
                {
                    "fold_id": ["fold_001"],
                    "train_start": ["2024-01-01"],
                    "train_end": ["2024-06-01"],
                    "validation_start": ["2024-05-15"],
                    "validation_end": ["2024-06-30"],
                }
            )
        )


def test_run_walk_forward_feature_selection_emits_task_sets() -> None:
    panel = build_feature_panel(_make_raw_panel())
    config = WalkForwardFeatureSelectionConfig(
        train_window_days=220,
        validation_window_days=45,
        step_days=35,
        max_folds=2,
        min_train_rows=260,
        min_validation_rows=110,
        top_k=10,
        permutation_top_n=12,
        regression_max_features=8,
        classification_max_features=8,
        regime_max_features=6,
        risk_max_features=6,
    )

    results = run_walk_forward_feature_selection(
        panel,
        registry=load_feature_registry(),
        config=config,
    )

    assert results["validation"]["selection_scope"] == "walk_forward_train_only"
    assert results["validation"]["duplicate_key_count"] == 0
    assert not results["folds"].empty
    assert not results["feature_summary"].empty
    assert set(results["final_task_feature_sets"]) == {
        "regression_forecasting",
        "directional_classification",
        "regime_detection",
        "risk_layer",
    }
    assert results["final_task_feature_sets"]["regression_forecasting"]
    assert results["final_task_feature_sets"]["directional_classification"]


def test_feature_selection_is_prefix_invariant_for_first_fold() -> None:
    panel = build_feature_panel(_make_raw_panel())
    config = WalkForwardFeatureSelectionConfig(
        train_window_days=220,
        validation_window_days=40,
        step_days=30,
        max_folds=1,
        min_train_rows=260,
        min_validation_rows=100,
        top_k=8,
        permutation_top_n=10,
        regression_max_features=6,
        classification_max_features=6,
        regime_max_features=5,
        risk_max_features=5,
    )

    full_results = run_walk_forward_feature_selection(
        panel,
        registry=load_feature_registry(),
        config=config,
    )
    first_fold = full_results["folds"].iloc[0]
    prefix = panel[panel["date"] <= pd.Timestamp(first_fold["validation_end"])].copy()
    prefix_results = run_walk_forward_feature_selection(
        prefix,
        registry=load_feature_registry(),
        config=config,
    )

    full_top = (
        full_results["fold_scores"]
        .query("task_name == 'regression_forecasting' and fold_id == 'fold_001'")
        .sort_values("combined_score", ascending=False)
        .head(5)["feature_name"]
        .tolist()
    )
    prefix_top = (
        prefix_results["fold_scores"]
        .query("task_name == 'regression_forecasting' and fold_id == 'fold_001'")
        .sort_values("combined_score", ascending=False)
        .head(5)["feature_name"]
        .tolist()
    )

    assert full_top == prefix_top


def test_detect_future_join_leakage() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "source_date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        }
    )
    assert detect_future_join_leakage(frame) == 1


def test_detect_forward_fill_boundary_leakage() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "source_ticker": ["AAA", "AAA"],
        }
    )
    assert detect_forward_fill_boundary_leakage(frame) == 1
