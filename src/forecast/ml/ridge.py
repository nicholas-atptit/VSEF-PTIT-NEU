"""Ridge regression forecast model."""

from __future__ import annotations

from sklearn.linear_model import Ridge

from src.forecast.base import SklearnForecastModel


class RidgeForecastModel(SklearnForecastModel):
    """Ridge regression over the prepared feature set."""

    model_name = "ridge"
    estimator_cls = Ridge

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: int | None = None,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=random_state,
            target_type=target_type,
        )
