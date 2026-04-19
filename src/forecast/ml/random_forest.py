"""Random forest forecast model."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.forecast.base import SklearnForecastModel


class RandomForestForecastModel(SklearnForecastModel):
    """Random-forest regressor over the prepared feature set."""

    model_name = "random_forest"
    estimator_cls = RandomForestRegressor

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = 6,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            target_type=target_type,
        )
