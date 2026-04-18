"""Helpers for the forecast-layer rehabilitation benchmark matrix."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.evaluation.hardening import DEFAULT_TICKER_GROUPS
from src.evaluation.targets import ForecastTargetSpec, build_target_spec
from src.forecast.registry import create_forecast_model
from src.ml.feature_engineering import FeatureEngineer
from src.ml.features.registry import final_task_feature_sets


FEATURE_FAMILY_ORDER = [
    "technical_core",
    "technical_plus_market",
    "technical_plus_market_plus_sector",
    "short_lag",
    "long_lag",
    "reduced_compact",
    "current_full",
]

FEATURE_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "technical_core": "Active price, return, technical-indicator, and liquidity features without market/sector context.",
    "technical_plus_market": "technical_core plus non-sector market breadth and macro context.",
    "technical_plus_market_plus_sector": "technical_plus_market plus sector-relative context features.",
    "short_lag": "Short-memory contextual set with lags up to 3 days and without 60/200-day memory blocks.",
    "long_lag": "Contextual set with both short lags and longer 60/200-day memory blocks.",
    "reduced_compact": "Compact union of the current regression and directional registry-selected baselines.",
    "current_full": "Current task-specific working baseline resolved from the governed feature registry.",
}

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "group_names": ["small_banks"],
        "horizons": [1, 5],
        "target_names": ["forward_return", "forward_log_return", "direction_binary"],
        "feature_families": ["current_full", "reduced_compact", "technical_plus_market_plus_sector"],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 1,
        },
    },
    "medium": {
        "group_names": ["small_banks", "mixed_large_cap"],
        "horizons": [1, 5, 10],
        "target_names": ["forward_return", "direction_binary"],
        "feature_families": FEATURE_FAMILY_ORDER,
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "rehab_full": {
        "group_names": ["small_banks", "mixed_large_cap", "vn100_subset"],
        "horizons": [1, 5, 10],
        "target_names": ["forward_return", "forward_log_return", "direction_binary"],
        "feature_families": FEATURE_FAMILY_ORDER,
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
}

CURRENT_REGRESSION_FEATURES = tuple(final_task_feature_sets()["regression_forecasting"])
CURRENT_DIRECTION_FEATURES = tuple(final_task_feature_sets()["directional_classification"])
REDUCED_COMPACT_FEATURES = tuple(dict.fromkeys([*CURRENT_REGRESSION_FEATURES, *CURRENT_DIRECTION_FEATURES]))

_LAG_PATTERN = re.compile(r"_lag_(\d+)$")


def forecast_rehab_policy_baseline() -> dict[str, Any]:
    """Return the fixed Phase 2.6 downstream execution baseline."""

    return {
        "policy_variant": "regime_threshold_adaptive_drawdown",
        "policy_label": "Phase 2.6 default candidate",
        "threshold": 0.010,
        "risk_budget": 0.02,
        "max_position_size": 1.0,
        "allow_short": False,
        "sizing_profile": "adaptive_current",
        "threshold_policy": "regime_aware",
        "sizing_mode": "adaptive",
        "use_risk_context": True,
        "use_regime_context": True,
        "use_volatility_sizing": True,
        "use_drawdown_control": True,
        "use_regime_sizing": True,
        "fixed_position_size": None,
        "min_position_size": 0.0,
        "volatility_target_scale": 1.0,
        "drawdown_haircut_strength": 1.0,
        "regime_multiplier_strength": 1.0,
        "transaction_fee_bps": 15.0,
        "slippage_bps": 20.0,
        "source_phase": "phase26_calibration",
    }


def build_forecast_rehab_matrix_config(
    preset: str = "medium",
    *,
    ticker_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the bounded forecast rehab matrix."""

    normalized_preset = str(preset or "medium").strip().lower()
    if normalized_preset not in PRESET_CONFIGS:
        raise ValueError(f"Unsupported forecast rehab preset '{preset}'")
    groups = dict(DEFAULT_TICKER_GROUPS)
    groups.update({str(name): [str(ticker).upper() for ticker in tickers] for name, tickers in dict(ticker_groups or {}).items()})
    preset_config = PRESET_CONFIGS[normalized_preset]
    selected_groups = []
    for group_name in preset_config["group_names"]:
        if group_name not in groups:
            raise ValueError(f"Ticker group '{group_name}' is not defined")
        selected_groups.append(
            {
                "group_name": str(group_name),
                "tickers": list(groups[group_name]),
            }
        )
    return {
        "preset": normalized_preset,
        "ticker_groups": selected_groups,
        "horizons": [int(value) for value in preset_config["horizons"]],
        "target_names": list(preset_config["target_names"]),
        "feature_families": list(preset_config["feature_families"]),
        "feature_family_descriptions": dict(FEATURE_FAMILY_DESCRIPTIONS),
        "evaluation_config": dict(preset_config["evaluation_config"]),
        "policy_baseline": forecast_rehab_policy_baseline(),
    }


def build_forecast_rehab_core_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation_config = dict(matrix_config.get("evaluation_config", {}))
    for group in matrix_config.get("ticker_groups", []):
        for horizon in matrix_config.get("horizons", []):
            for target_name in matrix_config.get("target_names", []):
                target_spec = build_target_spec(target_name)
                for feature_family in matrix_config.get("feature_families", []):
                    rows.append(
                        {
                            "core_run_id": f"{group['group_name']}_h{int(horizon):02d}_{target_name}_{feature_family}",
                            "preset": str(matrix_config.get("preset", "medium")),
                            "group_name": str(group["group_name"]),
                            "tickers": list(group["tickers"]),
                            "ticker_count": len(group["tickers"]),
                            "horizon": int(horizon),
                            "target_name": target_spec.name,
                            "target_type": target_spec.target_type,
                            "target_column": target_spec.target_column,
                            "target_family": target_spec.target_family,
                            "target_tradable": bool(target_spec.tradable_output),
                            "feature_family": str(feature_family),
                            **evaluation_config,
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "core_run_id",
                "preset",
                "group_name",
                "tickers",
                "ticker_count",
                "horizon",
                "target_name",
                "target_type",
                "target_column",
                "target_family",
                "target_tradable",
                "feature_family",
                "train_size",
                "test_size",
                "step_size",
                "gap_size",
                "max_windows",
            ]
        )
    return pd.DataFrame(rows).sort_values(["group_name", "horizon", "target_name", "feature_family"]).reset_index(drop=True)


def _is_sector_context(feature_name: str) -> bool:
    value = str(feature_name)
    return (
        value.startswith(("s_", "sector_", "rel_to_sector"))
        or "sector_" in value
        or value == "sector_member_count"
    )


def _is_lag_feature(feature_name: str) -> bool:
    return _LAG_PATTERN.search(str(feature_name)) is not None


def _lag_depth(feature_name: str) -> int | None:
    match = _LAG_PATTERN.search(str(feature_name))
    if match is None:
        return None
    return int(match.group(1))


def _is_long_memory_feature(feature_name: str) -> bool:
    value = str(feature_name)
    if _lag_depth(value) in {5}:
        return True
    long_markers = (
        "_60",
        "_60d",
        "sma_200",
        "close_to_sma_200",
        "ema_50",
        "rolling_max_60",
        "rolling_min_60",
        "dist_ma_60",
        "turnover_ma_60",
        "rolling_volatility_60",
        "market_return_60d",
        "sector_return_60d",
        "relative_strength_market_60",
        "relative_strength_sector_60",
        "rolling_beta_market_60",
        "rolling_beta_sector_60",
        "rolling_corr_market_60",
        "rolling_corr_sector_60",
    )
    return any(marker in value for marker in long_markers)


def build_feature_inventory_table(frame: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    inventory = engineer.build_feature_inventory(frame)
    inventory["feature_name"] = inventory["feature_name"].astype(str)
    inventory = inventory[~inventory["feature_name"].isin({"daily_return", "current_log_return_1d", "current_direction_1d"})].copy()
    inventory["is_current_regression_baseline"] = inventory["feature_name"].isin(CURRENT_REGRESSION_FEATURES)
    inventory["is_current_direction_baseline"] = inventory["feature_name"].isin(CURRENT_DIRECTION_FEATURES)
    inventory["is_reduced_compact"] = inventory["feature_name"].isin(REDUCED_COMPACT_FEATURES)
    return inventory.sort_values("feature_name").reset_index(drop=True)


def _ordered_feature_list(frame: pd.DataFrame, selected: set[str]) -> list[str]:
    return [column for column in frame.columns if column in selected]


def build_feature_family_columns(
    frame: pd.DataFrame,
    *,
    family_name: str,
    target_name: str,
) -> list[str]:
    """Resolve one explicit feature family against the available columns."""

    inventory = build_feature_inventory_table(frame)
    active = inventory[inventory["status"] == "active"].copy()

    technical_like = active[active["category"].isin(["price_volume_core", "technical_indicator", "flow_microstructure"])]
    technical_core = technical_like[
        ~technical_like["feature_name"].map(_is_lag_feature)
        & ~technical_like["feature_name"].map(_is_long_memory_feature)
    ]

    market_context = active[active["category"] == "market_context"].copy()
    market_only = market_context[~market_context["feature_name"].map(_is_sector_context)]
    sector_only = market_context[market_context["feature_name"].map(_is_sector_context)]
    macro_context = active[active["category"] == "macro_cross_asset"]

    contextual_base = active[
        active["category"].isin(["price_volume_core", "technical_indicator", "flow_microstructure", "market_context", "macro_cross_asset"])
    ].copy()
    contextual_no_lags = contextual_base[~contextual_base["feature_name"].map(_is_lag_feature)]
    contextual_no_lags_or_long = contextual_no_lags[~contextual_no_lags["feature_name"].map(_is_long_memory_feature)]
    lag_features = active[active["feature_name"].map(_is_lag_feature)]
    short_lags = lag_features[lag_features["feature_name"].map(lambda value: (_lag_depth(value) or 0) <= 3)]

    family_key = str(family_name).strip().lower()
    if family_key == "technical_core":
        selected = set(technical_core["feature_name"])
    elif family_key == "technical_plus_market":
        selected = set(technical_core["feature_name"]) | set(market_only["feature_name"]) | set(macro_context["feature_name"])
    elif family_key == "technical_plus_market_plus_sector":
        selected = (
            set(technical_core["feature_name"])
            | set(market_only["feature_name"])
            | set(macro_context["feature_name"])
            | set(sector_only["feature_name"])
        )
    elif family_key == "short_lag":
        selected = set(contextual_no_lags_or_long["feature_name"]) | set(short_lags["feature_name"])
    elif family_key == "long_lag":
        selected = set(contextual_no_lags["feature_name"]) | set(lag_features["feature_name"])
    elif family_key == "reduced_compact":
        selected = set(REDUCED_COMPACT_FEATURES)
    elif family_key == "current_full":
        if str(target_name).strip().lower() == "direction_binary":
            selected = set(CURRENT_DIRECTION_FEATURES)
        else:
            selected = set(CURRENT_REGRESSION_FEATURES)
    else:
        raise ValueError(f"Unsupported feature family '{family_name}'. Available: {FEATURE_FAMILY_ORDER}")

    resolved = _ordered_feature_list(frame, selected)
    if not resolved:
        raise ValueError(f"Feature family '{family_name}' resolved to no available columns")
    return resolved


def create_rehab_forecast_model(
    model_name: str,
    *,
    target_spec: ForecastTargetSpec,
) -> Any:
    """Create one forecast model aligned to the selected target framing."""

    kwargs: dict[str, Any] = {"target_type": target_spec.target_type}
    if model_name in {"naive", "moving_average"}:
        if target_spec.name == "forward_log_return":
            kwargs.update(
                {
                    "value_column": "current_log_return_1d",
                    "fallback_column": "current_log_return_1d",
                }
            )
        elif target_spec.name == "direction_binary":
            kwargs.update(
                {
                    "value_column": "current_direction_1d",
                    "fallback_column": "current_direction_1d",
                }
            )
        else:
            kwargs.update(
                {
                    "value_column": "daily_return",
                    "fallback_column": "close_return_1d",
                }
            )
    return create_forecast_model(model_name, **kwargs)
