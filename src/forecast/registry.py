"""Minimal forecast model registry for the Phase 1 stack."""

from __future__ import annotations

from collections.abc import Callable

from src.forecast.base import ForecastModel


def _naive_cls():
    from src.forecast.statistical.naive import NaiveForecastModel

    return NaiveForecastModel


def _moving_average_cls():
    from src.forecast.statistical.moving_average import MovingAverageForecastModel

    return MovingAverageForecastModel


def _sarimax_cls():
    from src.forecast.statistical.sarimax import SARIMAXForecastModel

    return SARIMAXForecastModel


def _ets_cls():
    from src.forecast.statistical.ets import ETSForecastModel

    return ETSForecastModel


def _linear_cls():
    from src.forecast.ml.linear import LinearForecastModel

    return LinearForecastModel


def _ridge_cls():
    from src.forecast.ml.ridge import RidgeForecastModel

    return RidgeForecastModel


def _lasso_cls():
    from src.forecast.ml.lasso import LassoForecastModel

    return LassoForecastModel


def _random_forest_cls():
    from src.forecast.ml.random_forest import RandomForestForecastModel

    return RandomForestForecastModel


def _xgboost_cls():
    from src.forecast.ml.xgboost_model import XGBoostForecastModel

    return XGBoostForecastModel


def _lightgbm_cls():
    from src.forecast.ml.lightgbm_model import LightGBMForecastModel

    return LightGBMForecastModel


FORECAST_MODEL_REGISTRY: dict[str, Callable[[], type[ForecastModel]]] = {
    "naive": _naive_cls,
    "moving_average": _moving_average_cls,
    "sarimax": _sarimax_cls,
    "ets": _ets_cls,
    "linear": _linear_cls,
    "ridge": _ridge_cls,
    "lasso": _lasso_cls,
    "random_forest": _random_forest_cls,
    "xgboost": _xgboost_cls,
    "lightgbm": _lightgbm_cls,
}


def create_forecast_model(name: str, **kwargs) -> ForecastModel:
    key = str(name).strip().lower()
    if key not in FORECAST_MODEL_REGISTRY:
        raise ValueError(f"Unsupported forecast model '{name}'. Available: {sorted(FORECAST_MODEL_REGISTRY)}")
    return FORECAST_MODEL_REGISTRY[key]()( **kwargs)


def supported_forecast_models() -> list[str]:
    return sorted(FORECAST_MODEL_REGISTRY)
