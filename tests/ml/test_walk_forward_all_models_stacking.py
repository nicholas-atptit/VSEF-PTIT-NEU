from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.backtest.walk_forward_all_models_stacking import (
    FINAL_STACKING_MODEL_NAME,
    WalkForwardAllModelsStackingConfig,
    WalkForwardAllModelsStackingRunner,
)
from src.ml.backtest.linear_fold_diagnostics import (
    COEFFICIENT_DIAGNOSTIC_COLUMNS,
    COEFFICIENT_STABILITY_COLUMNS,
)
from src.ml.backtest.feature_importance_diagnostics import (
    FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS,
    FEATURE_IMPORTANCE_STABILITY_COLUMNS,
    LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS,
)
from src.ml.backtest.feature_governance_review import FEATURE_GOVERNANCE_REVIEW_COLUMNS
from src.ml.backtest.context_coverage_diagnostics import (
    CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS,
    CONTEXT_COVERAGE_SUMMARY_COLUMNS,
)
from src.ml.data_loader import generate_mock_data


def _mock_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="B")
    df = generate_mock_data(ticker=ticker, num_days=len(dates), seed=100 + len(ticker))
    df["date"] = dates
    df["ticker"] = ticker
    return df


def test_walk_forward_all_models_runner_writes_required_outputs(tmp_path, monkeypatch) -> None:
    histories = {
        "AAA": _mock_history("AAA", "2017-06-01", "2021-01-29"),
        "BBB": _mock_history("BBB", "2017-06-01", "2021-01-29"),
    }
    benchmark = _mock_history("VNINDEX", "2017-06-01", "2021-01-29")

    class _FakeAdapter:
        def __init__(self, symbol_list=None) -> None:
            self.symbol_list = symbol_list or []

        def get_ohlcv(self, symbol: str, start_date: str, end_date: str, interval: str = "1D") -> pd.DataFrame:
            df = histories[symbol.upper()].copy()
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

        def get_index_ohlcv(self, symbol: str, start_date: str, end_date: str, interval: str = "1D") -> pd.DataFrame:
            df = benchmark.copy()
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _FakeAdapter,
    )

    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA", "BBB"],
        history_start="2018-01-01",
        history_end="2021-01-29",
        initial_train_start="2018-01-01",
        initial_train_end="2020-12-31",
        forecast_start="2021-01-01",
        forecast_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "walk_forward_all_models"),
        horizons=["short_5d"],
        step_sizes=[1, 2],
        algorithms=["cart"],
        max_depth=3,
        meta_min_samples=1,
    )
    result = WalkForwardAllModelsStackingRunner(config).run()

    assert Path(result["csv_dir"]).exists()
    assert Path(result["charts_dir"]).exists()
    assert Path(result["report_path"]).exists()

    predictions = pd.read_csv(Path(config.output_dir) / "csv" / "predictions_detailed.csv")
    stacking = pd.read_csv(Path(config.output_dir) / "csv" / "stacking_predictions_detailed.csv")
    coverage = pd.read_csv(Path(config.output_dir) / "csv" / "forecast_coverage_summary.csv")
    coefficient_diagnostics = pd.read_csv(Path(config.output_dir) / "csv" / "linear_coefficient_diagnostics.csv")
    coefficient_summary = pd.read_csv(Path(config.output_dir) / "csv" / "linear_coefficient_stability_summary.csv")
    importance_diagnostics = pd.read_csv(Path(config.output_dir) / "csv" / "feature_importance_diagnostics.csv")
    importance_summary = pd.read_csv(Path(config.output_dir) / "csv" / "feature_importance_stability_summary.csv")
    linear_vs_importance = pd.read_csv(Path(config.output_dir) / "csv" / "linear_vs_importance_feature_comparison.csv")
    governance_review = pd.read_csv(Path(config.output_dir) / "csv" / "feature_governance_review.csv")
    context_coverage = pd.read_csv(Path(config.output_dir) / "csv" / "context_coverage_diagnostics.csv")
    context_coverage_summary = pd.read_csv(Path(config.output_dir) / "csv" / "context_coverage_summary.csv")

    assert {
        "ticker",
        "prediction_date",
        "horizon",
        "model_name",
        "predicted_return",
        "predicted_direction",
        "actual_return",
        "actual_direction",
        "actual_realized_forward_return",
        "actual_realized_direction",
        "absolute_error",
        "squared_error",
        "direction_correct",
        "evaluation_eligible",
    } <= set(predictions.columns)
    assert {
        "ticker",
        "prediction_date",
        "horizon",
        "model_name",
        "final_predicted_return",
        "final_predicted_direction",
        "actual_return",
        "actual_direction",
        "actual_realized_forward_return",
        "actual_realized_direction",
        "absolute_error",
        "squared_error",
        "direction_correct",
        "evaluation_eligible",
    } <= set(stacking.columns)
    assert set(predictions["step_size"]) == {1, 2}
    assert set(stacking["model_name"]) == {FINAL_STACKING_MODEL_NAME}
    assert (pd.to_datetime(predictions["prediction_date"]) == pd.to_datetime(predictions["feature_date"])).all()
    assert (predictions["evaluation_eligible"] == False).any()
    assert (stacking["evaluation_eligible"] == False).any()
    assert not coverage.empty
    assert set(COEFFICIENT_DIAGNOSTIC_COLUMNS) <= set(coefficient_diagnostics.columns)
    assert set(COEFFICIENT_STABILITY_COLUMNS) <= set(coefficient_summary.columns)
    assert set(coefficient_diagnostics["model"]) == {"linear", "ridge", "lasso"}
    assert coefficient_summary["stability_level"].isin(["high", "medium", "low"]).all()
    assert set(FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS) <= set(importance_diagnostics.columns)
    assert set(FEATURE_IMPORTANCE_STABILITY_COLUMNS) <= set(importance_summary.columns)
    assert set(LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS) <= set(linear_vs_importance.columns)
    assert set(importance_diagnostics["model"]) == {"cart"}
    assert importance_diagnostics["importance_normalized"].between(0.0, 1.0).all()
    assert importance_summary["importance_stability_level"].isin(["high", "medium", "low"]).all()
    assert linear_vs_importance["alignment_label"].isin(
        ["aligned_stable", "linear_only", "importance_only", "unstable_or_missing"]
    ).all()
    assert set(FEATURE_GOVERNANCE_REVIEW_COLUMNS) <= set(governance_review.columns)
    assert not governance_review.empty
    assert governance_review["governance_category"].isin(
        ["safe_trailing", "requires_review", "alias_or_redundant", "potential_leakage", "target_derived", "unknown"]
    ).all()
    assert governance_review["recommended_action"].isin(
        ["keep", "keep_but_document", "review_timing", "review_redundancy", "exclude_until_verified"]
    ).all()
    assert set(CONTEXT_COVERAGE_DIAGNOSTIC_COLUMNS) <= set(context_coverage.columns)
    assert set(CONTEXT_COVERAGE_SUMMARY_COLUMNS) <= set(context_coverage_summary.columns)
    assert not context_coverage.empty
    assert context_coverage["coverage_warning_level"].isin(
        ["ok", "review", "weak_coverage", "metadata_unavailable"]
    ).all()
    assert context_coverage_summary["overall_coverage_warning_level"].isin(
        ["ok", "review", "weak_coverage", "metadata_unavailable"]
    ).all()


def test_stacking_layer_switches_from_fallback_to_prequential_model(tmp_path, monkeypatch) -> None:
    class _FakeAdapter:
        def __init__(self, symbol_list=None) -> None:
            self.symbol_list = symbol_list or []

    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _FakeAdapter,
    )

    config = WalkForwardAllModelsStackingConfig(
        output_dir=str(tmp_path / "artifacts" / "walk_forward_all_models"),
        meta_min_samples=1,
    )
    runner = WalkForwardAllModelsStackingRunner(config)

    prediction_dates = pd.to_datetime(
        ["2025-01-02", "2025-01-06", "2025-01-10", "2025-01-15", "2025-01-20"]
    )
    target_dates = pd.to_datetime(
        ["2025-01-03", "2025-01-07", "2025-01-13", "2025-01-16", "2025-01-27"]
    )
    rows = []
    for prediction_date, target_date, actual_return, cart_pred, xgb_pred in zip(
        prediction_dates,
        target_dates,
        [0.01, 0.02, 0.015, 0.03, 0.025],
        [0.011, 0.019, 0.014, 0.029, 0.026],
        [0.012, 0.021, 0.016, 0.031, 0.024],
    ):
        for model_name, predicted_return in [("cart", cart_pred), ("xgboost", xgb_pred)]:
            rows.append(
                {
                    "step_size": 1,
                    "ticker": "AAA",
                    "prediction_date": prediction_date,
                    "feature_date": prediction_date,
                    "target_date": target_date,
                    "horizon": "short_5d",
                    "horizon_days": 5,
                    "model_name": model_name,
                    "predicted_return": predicted_return,
                    "predicted_direction": int(predicted_return > 0.0),
                    "actual_return": actual_return,
                    "actual_direction": int(actual_return > 0.0),
                    "absolute_error": abs(predicted_return - actual_return),
                    "squared_error": (predicted_return - actual_return) ** 2,
                    "direction_correct": 1,
                    "evaluation_eligible": True,
                }
            )
    base_df = pd.DataFrame(rows)

    stacking = runner._build_stacking_predictions(base_df)
    first_row = stacking.sort_values("prediction_date").iloc[0]
    last_row = stacking.sort_values("prediction_date").iloc[-1]

    assert first_row["stacking_method"] == "mean_fallback"
    assert first_row["training_meta_rows"] == 0
    assert last_row["stacking_method"] == "prequential_ridge_on_oos_base_predictions"
    assert last_row["training_meta_rows"] == 4
