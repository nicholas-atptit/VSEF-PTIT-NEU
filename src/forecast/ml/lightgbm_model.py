"""LightGBM forecast model."""

from __future__ import annotations

from typing import Any

from src.forecast.base import SklearnForecastModel


class LightGBMForecastModel(SklearnForecastModel):
    """LightGBM regressor over the prepared feature set."""

    model_name = "lightgbm"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_leaves: int = 31,
        random_state: int = 42,
        n_jobs: int = 1,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            target_type=target_type,
        )

    def _build_estimator(self) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "LightGBMForecastModel requires lightgbm. Install lightgbm to use this baseline."
            ) from exc
        return LGBMRegressor(
            **self.estimator_params,
            verbosity=-1,
        )
