"""Governed forecast model registry for quant-core execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.model_governance import ModelGovernanceEntry, filter_governance_entries, governance_table
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


@dataclass(frozen=True)
class ForecastModelRegistration:
    """Forecast-model registration with governance metadata and lazy loader."""

    governance: ModelGovernanceEntry
    builder: Any
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def create(self, **kwargs: Any) -> ForecastModel:
        payload = dict(self.default_kwargs)
        payload.update(kwargs)
        return self.builder()(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.governance.to_dict(),
            "registration_kind": "forecast_model",
        }


def _registration(
    model_name: str,
    *,
    family: str,
    builder: Any,
    status: str,
    research_priority: int,
    role: str,
    enabled_for_full_forecast: bool = True,
    enabled_for_research_core: bool = False,
    enabled_for_decision_core: bool = False,
    baseline_only: bool = False,
    comparator_only: bool = False,
    parked: bool = False,
    supports_return: bool = True,
    supports_direction: bool = True,
    supports_policy_eval: bool = True,
    notes: str = "",
    default_kwargs: dict[str, Any] | None = None,
) -> ForecastModelRegistration:
    return ForecastModelRegistration(
        governance=ModelGovernanceEntry(
            model_name=model_name,
            family=family,
            status=status,
            research_priority=int(research_priority),
            role=role,  # type: ignore[arg-type]
            enabled_for_full_forecast=bool(enabled_for_full_forecast),
            enabled_for_research_core=bool(enabled_for_research_core),
            enabled_for_decision_core=bool(enabled_for_decision_core),
            baseline_only=bool(baseline_only),
            comparator_only=bool(comparator_only),
            parked=bool(parked),
            supports_return=bool(supports_return),
            supports_direction=bool(supports_direction),
            supports_policy_eval=bool(supports_policy_eval),
            notes=notes,
        ),
        builder=builder,
        default_kwargs=dict(default_kwargs or {}),
    )


FORECAST_MODEL_SPECS: dict[str, ForecastModelRegistration] = {
    "lightgbm": _registration(
        "lightgbm",
        family="boosting",
        builder=_lightgbm_cls,
        status="active",
        research_priority=10,
        role="primary_research",
        enabled_for_research_core=True,
        enabled_for_decision_core=True,
        notes="Current primary research model in the narrow edge lane.",
    ),
    "xgboost": _registration(
        "xgboost",
        family="boosting",
        builder=_xgboost_cls,
        status="active",
        research_priority=20,
        role="primary_research",
        enabled_for_research_core=True,
        enabled_for_decision_core=True,
        notes="Current primary research model in the narrow edge lane.",
    ),
    "random_forest": _registration(
        "random_forest",
        family="tree",
        builder=_random_forest_cls,
        status="active",
        research_priority=30,
        role="primary_research",
        enabled_for_research_core=True,
        enabled_for_decision_core=True,
        notes="Tree-based primary research model kept in the decision-authorized subset.",
    ),
    "ets": _registration(
        "ets",
        family="statistical",
        builder=_ets_cls,
        status="active",
        research_priority=40,
        role="comparator",
        enabled_for_research_core=True,
        comparator_only=True,
        notes="Statistical comparator retained for research-lane benchmarking.",
    ),
    "sarimax": _registration(
        "sarimax",
        family="statistical",
        builder=_sarimax_cls,
        status="active",
        research_priority=50,
        role="comparator",
        enabled_for_research_core=True,
        comparator_only=True,
        notes="Time-series comparator retained for research-lane benchmarking.",
    ),
    "naive": _registration(
        "naive",
        family="baseline",
        builder=_naive_cls,
        status="baseline",
        research_priority=60,
        role="baseline_only",
        baseline_only=True,
        notes="Baseline checkpoint; stays in full-forecast and baseline-only modes.",
    ),
    "moving_average": _registration(
        "moving_average",
        family="baseline",
        builder=_moving_average_cls,
        status="baseline",
        research_priority=70,
        role="baseline_only",
        baseline_only=True,
        notes="Baseline checkpoint; stays in full-forecast and baseline-only modes.",
    ),
    "linear": _registration(
        "linear",
        family="linear",
        builder=_linear_cls,
        status="shadow",
        research_priority=80,
        role="shadow_only",
        notes="Preserved in the full zoo for analysis continuity but not in the active research core.",
    ),
    "ridge": _registration(
        "ridge",
        family="linear",
        builder=_ridge_cls,
        status="shadow",
        research_priority=90,
        role="shadow_only",
        notes="Preserved in the full zoo for analysis continuity but not in the active research core.",
    ),
    "lasso": _registration(
        "lasso",
        family="linear",
        builder=_lasso_cls,
        status="shadow",
        research_priority=100,
        role="shadow_only",
        notes="Preserved in the full zoo for analysis continuity but not in the active research core.",
    ),
}

FORECAST_MODEL_REGISTRY: dict[str, Any] = {
    name: registration.builder for name, registration in FORECAST_MODEL_SPECS.items()
}


def get_forecast_model_registration(name: str) -> ForecastModelRegistration:
    key = str(name).strip().lower()
    if key not in FORECAST_MODEL_SPECS:
        raise ValueError(f"Unsupported forecast model '{name}'. Available: {sorted(FORECAST_MODEL_SPECS)}")
    return FORECAST_MODEL_SPECS[key]


def forecast_model_governance_table(
    *,
    run_mode: str | None = None,
    roles: list[str] | None = None,
    model_names: list[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = True,
) -> pd.DataFrame:
    entries = [registration.governance for registration in FORECAST_MODEL_SPECS.values()]
    return governance_table(
        entries,
        run_mode=run_mode,
        roles=roles,
        model_names=model_names,
        target_type=target_type,
        require_policy_eval=require_policy_eval,
        include_parked=include_parked,
    )


def resolve_forecast_model_registrations(
    *,
    run_mode: str | None = "full_forecast",
    roles: list[str] | None = None,
    model_names: list[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = False,
) -> list[ForecastModelRegistration]:
    selected_entries = filter_governance_entries(
        [registration.governance for registration in FORECAST_MODEL_SPECS.values()],
        run_mode=run_mode,
        roles=roles,
        model_names=model_names,
        target_type=target_type,
        require_policy_eval=require_policy_eval,
        include_parked=include_parked,
    )
    return [FORECAST_MODEL_SPECS[entry.model_name] for entry in selected_entries]


def create_forecast_model(name: str, **kwargs: Any) -> ForecastModel:
    return get_forecast_model_registration(name).create(**kwargs)


def create_forecast_models(
    *,
    run_mode: str | None = "full_forecast",
    roles: list[str] | None = None,
    model_names: list[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = False,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[ForecastModel]:
    kwargs_by_model = {str(key).lower(): dict(value) for key, value in dict(model_kwargs or {}).items()}
    models: list[ForecastModel] = []
    for registration in resolve_forecast_model_registrations(
        run_mode=run_mode,
        roles=roles,
        model_names=model_names,
        target_type=target_type,
        require_policy_eval=require_policy_eval,
        include_parked=include_parked,
    ):
        models.append(registration.create(**kwargs_by_model.get(registration.governance.model_name, {})))
    return models


def supported_forecast_models(
    *,
    run_mode: str | None = "full_forecast",
    roles: list[str] | None = None,
    model_names: list[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = False,
) -> list[str]:
    return [
        registration.governance.model_name
        for registration in resolve_forecast_model_registrations(
            run_mode=run_mode,
            roles=roles,
            model_names=model_names,
            target_type=target_type,
            require_policy_eval=require_policy_eval,
            include_parked=include_parked,
        )
    ]
