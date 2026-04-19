"""Lasso regression forecast model."""

from __future__ import annotations

from sklearn.linear_model import Lasso

from src.forecast.base import SklearnForecastModel


class LassoForecastModel(SklearnForecastModel):
    """L1-regularized regression over the prepared feature set."""

    model_name = "lasso"
    estimator_cls = Lasso

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        fit_intercept: bool = True,
        max_iter: int = 5000,
        random_state: int = 42,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=random_state,
            target_type=target_type,
        )
