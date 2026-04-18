"""XGBoost forecast model."""

from __future__ import annotations

from typing import Any

from src.forecast.base import SklearnForecastModel


class XGBoostForecastModel(SklearnForecastModel):
    """Gradient-boosted tree regressor over the prepared feature set."""

    model_name = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
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
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            target_type=target_type,
        )

    def _build_estimator(self) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "XGBoostForecastModel requires xgboost. Install xgboost to use this baseline."
            ) from exc
        return XGBRegressor(
            **self.estimator_params,
            objective="reg:squarederror",
            verbosity=0,
        )
