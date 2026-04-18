"""Linear regression forecast model."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from src.forecast.base import SklearnForecastModel


class LinearForecastModel(SklearnForecastModel):
    """Plain linear regression over the prepared feature set."""

    model_name = "linear"
    estimator_cls = LinearRegression

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        positive: bool = False,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(fit_intercept=fit_intercept, positive=positive, target_type=target_type)
