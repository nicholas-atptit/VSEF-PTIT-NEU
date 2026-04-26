from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.contracts import FORECAST_REQUIRED_COLUMNS
from src.forecast.ml.lasso import LassoForecastModel
from src.forecast.ml.linear import LinearForecastModel
from src.forecast.ml.ridge import RidgeForecastModel
from src.forecast.statistical.naive import NaiveForecastModel


def _forecast_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="B"),
            "ticker": ["AAA"] * 30,
            "target_forward_return": np.linspace(-0.02, 0.03, 30),
            "daily_return": np.linspace(-0.01, 0.01, 30),
            "close_return_1d": np.linspace(-0.01, 0.01, 30),
            "feature_one": rng.normal(size=30),
            "feature_two": rng.normal(size=30),
            "window_id": ["window_001"] * 30,
        }
    )


def test_linear_forecast_model_returns_shared_contract() -> None:
    frame = _forecast_frame()
    train = frame.iloc[:20].copy()
    test = frame.iloc[20:].copy()

    model = LinearForecastModel()
    predictions = model.fit(
        train_df=train,
        features=["feature_one", "feature_two"],
        target="target_forward_return",
        horizon=5,
        config={"seed": 42},
    ).predict(test)

    assert set(FORECAST_REQUIRED_COLUMNS) <= set(predictions.columns)
    assert predictions["model_name"].eq("linear").all()
    assert predictions["horizon"].eq(5).all()


def test_linear_family_metadata_exposes_coefficient_diagnostics() -> None:
    frame = _forecast_frame()
    train = frame.iloc[:24].copy()
    features = ["feature_one", "feature_two"]

    models = [
        LinearForecastModel(),
        RidgeForecastModel(alpha=1.0),
        LassoForecastModel(alpha=0.0001, max_iter=10_000),
    ]

    for model in models:
        model.fit(
            train_df=train,
            features=features,
            target="target_forward_return",
            horizon=5,
            config={"seed": 42},
        )
        diagnostics = model.get_metadata()["coefficient_diagnostics"]

        assert diagnostics["available"] is True
        assert diagnostics["selected_feature_names"] == features
        assert diagnostics["coefficient_count"] == len(features)
        assert diagnostics["fold_level_coefficient_stability"]["available"] is False
        assert isinstance(diagnostics["intercept"], float)
        assert [row["feature"] for row in diagnostics["coefficients"]] == features
        assert {row["sign"] for row in diagnostics["coefficients"]} <= {"negative", "zero", "positive"}
        assert all(row["magnitude"] == abs(row["coefficient"]) for row in diagnostics["coefficients"])


def test_naive_forecast_model_does_not_require_feature_columns() -> None:
    frame = _forecast_frame()
    train = frame.iloc[:20].copy()
    test = frame.iloc[20:].copy()

    model = NaiveForecastModel()
    predictions = model.fit(
        train_df=train,
        features=[],
        target="target_forward_return",
        horizon=3,
        config={},
    ).predict(test)

    assert set(FORECAST_REQUIRED_COLUMNS) <= set(predictions.columns)
    assert predictions["model_name"].eq("naive").all()
