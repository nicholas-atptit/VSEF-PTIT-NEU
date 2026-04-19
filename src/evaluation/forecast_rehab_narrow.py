"""Helpers for the narrowed post-F1 forecast rehabilitation matrix."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.forecast_rehab import (
    build_feature_inventory_table,
    create_rehab_forecast_model,
    forecast_rehab_policy_baseline,
)
from src.evaluation.hardening import DEFAULT_TICKER_GROUPS
from src.evaluation.targets import ForecastTargetSpec, build_target_spec


PRIMARY_GROUPS = {"small_banks": list(DEFAULT_TICKER_GROUPS["small_banks"])}
PRIMARY_HORIZONS = [5, 10]
PRIMARY_TARGETS = ["forward_return", "direction_binary"]
PRIMARY_MODELS = ["lightgbm", "random_forest", "xgboost"]
COMPARATOR_MODELS = ["ets", "sarimax"]
BASELINE_ONLY_MODELS = ["naive", "moving_average"]
DEMOTED_MODELS = ["linear", "ridge", "lasso"]

NARROW_FEATURE_FAMILY_ORDER = [
    "tech_core_v1",
    "compact_v1",
    "compact_plus_longlag_v1",
    "compact_plus_longlag_v2",
]

NARROW_FEATURE_FAMILY_METADATA: dict[str, dict[str, Any]] = {
    "tech_core_v1": {
        "source_family": "technical_core",
        "rationale": "Compressed technical and flow core that keeps medium-horizon trend, liquidity, and volatility state without broad context expansion.",
        "features": (
            "log_return",
            "momentum_5",
            "momentum_20",
            "dist_ma_20",
            "turnover_ratio_20",
            "volume_spike_zscore_20",
            "amihud_20",
            "rsi_14",
            "macd_signal",
            "bb_width",
            "adx_14",
            "close_to_close_return_1d",
            "close_return_5d",
            "close_return_10d",
            "close_return_20d",
            "open_close_spread_pct",
            "overnight_return_1d",
            "high_low_range_pct",
            "atr_14",
            "atr_proxy_10",
            "rolling_volatility_10",
            "rolling_volatility_20",
            "foreign_net_value_ratio",
            "foreign_participation_20",
            "foreign_flow_intensity_zscore_20",
        ),
    },
    "compact_v1": {
        "source_family": "reduced_compact",
        "rationale": "Reduced compact baseline after removing raw breadth-volume counts and redundant long-memory aliases.",
        "features": (
            "m_ret_5d",
            "m_ret_20d",
            "declining_share",
            "pct_above_ma20",
            "pct_above_ma50",
            "range_20",
            "rolling_min_5",
            "dist_ma_20",
            "dist_ma_60",
            "ema_50",
            "close_to_sma_200",
            "turnover_ma_60",
            "macd_signal",
            "bb_width",
            "close_return_10d",
            "rolling_volatility_60",
            "market_return_60d",
            "breadth_thrust_10",
            "new_high_low_spread_5",
            "up_down_volume_ratio_5",
        ),
    },
    "compact_plus_longlag_v1": {
        "source_family": "reduced_compact+long_lag",
        "rationale": "Compact baseline plus 60-day trend and market-memory overlays aimed at horizon-10 persistence.",
        "features": (
            "m_ret_5d",
            "m_ret_20d",
            "declining_share",
            "pct_above_ma20",
            "pct_above_ma50",
            "range_20",
            "rolling_min_5",
            "dist_ma_20",
            "dist_ma_60",
            "ema_50",
            "close_to_sma_200",
            "turnover_ma_60",
            "macd_signal",
            "bb_width",
            "close_return_10d",
            "rolling_volatility_60",
            "market_return_60d",
            "breadth_thrust_10",
            "new_high_low_spread_5",
            "up_down_volume_ratio_5",
            "roc_60",
            "rolling_min_60",
            "rolling_max_60",
            "turnover_ratio_60",
            "volume_ratio_60",
            "relative_strength_market_60",
            "rolling_beta_market_60",
            "rolling_corr_market_60",
        ),
    },
    "compact_plus_longlag_v2": {
        "source_family": "reduced_compact+long_lag",
        "rationale": "Compact baseline plus lagged return and momentum persistence features to test whether raw state carry matters more than broad context.",
        "features": (
            "m_ret_5d",
            "m_ret_20d",
            "declining_share",
            "pct_above_ma20",
            "pct_above_ma50",
            "range_20",
            "rolling_min_5",
            "dist_ma_20",
            "dist_ma_60",
            "ema_50",
            "close_to_sma_200",
            "turnover_ma_60",
            "macd_signal",
            "bb_width",
            "close_return_10d",
            "rolling_volatility_60",
            "market_return_60d",
            "breadth_thrust_10",
            "new_high_low_spread_5",
            "up_down_volume_ratio_5",
            "close_lag_3",
            "close_lag_5",
            "pct_return_lag_3",
            "pct_return_lag_5",
            "log_return_lag_3",
            "log_return_lag_5",
            "rsi_14_lag_5",
            "volume_lag_5",
        ),
    },
}

NARROW_COST_MODES: dict[str, dict[str, float]] = {
    "baseline": {
        "transaction_fee_bps": 15.0,
        "slippage_bps": 20.0,
    },
    "elevated": {
        "transaction_fee_bps": 30.0,
        "slippage_bps": 35.0,
    },
}

NARROW_PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "group_names": ["small_banks"],
        "horizons": [5],
        "target_names": ["forward_return", "direction_binary"],
        "feature_families": ["compact_v1", "compact_plus_longlag_v1"],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "medium": {
        "group_names": ["small_banks"],
        "horizons": [5, 10],
        "target_names": PRIMARY_TARGETS,
        "feature_families": NARROW_FEATURE_FAMILY_ORDER,
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 4,
        },
    },
    "narrow_full": {
        "group_names": ["small_banks"],
        "horizons": [5, 10],
        "target_names": PRIMARY_TARGETS,
        "feature_families": NARROW_FEATURE_FAMILY_ORDER,
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 6,
        },
    },
}


def build_narrow_scope_table() -> pd.DataFrame:
    rows = [
        {
            "dimension": "ticker_group",
            "in_scope": "small_banks",
            "comparator_only": "",
            "baseline_only": "",
            "out_or_deemphasized": "mixed_large_cap, vn100_subset",
            "reason": "F1 showed the edge cluster is concentrated in small banks and collapses when generalized too early.",
        },
        {
            "dimension": "horizons",
            "in_scope": "5, 10",
            "comparator_only": "",
            "baseline_only": "",
            "out_or_deemphasized": "1",
            "reason": "F1 showed horizon 10 is least bad and horizon 5 remains materially better than horizon 1.",
        },
        {
            "dimension": "feature_families",
            "in_scope": ", ".join(NARROW_FEATURE_FAMILY_ORDER),
            "comparator_only": "",
            "baseline_only": "",
            "out_or_deemphasized": "technical_plus_market, technical_plus_market_plus_sector, current_full, short_lag",
            "reason": "F1 favored technical_core, reduced_compact, and selected long-memory overlays over broad context expansion.",
        },
        {
            "dimension": "model_families",
            "in_scope": ", ".join(PRIMARY_MODELS),
            "comparator_only": ", ".join(COMPARATOR_MODELS),
            "baseline_only": ", ".join(BASELINE_ONLY_MODELS),
            "out_or_deemphasized": ", ".join(DEMOTED_MODELS),
            "reason": "Tree models led the best-slice cluster, ETS stayed useful as a benchmark, SARIMAX remained usable, and the linear family no longer justified primary scope.",
        },
        {
            "dimension": "target_framing",
            "in_scope": ", ".join(PRIMARY_TARGETS),
            "comparator_only": "",
            "baseline_only": "",
            "out_or_deemphasized": "forward_log_return, future_realized_volatility",
            "reason": "F1 left forward_return versus direction_binary unresolved, while volatility was audit-only and log-return did not justify extra surface area.",
        },
    ]
    return pd.DataFrame(rows)


def build_narrow_matrix_config(
    preset: str = "medium",
    *,
    include_baselines: bool = True,
) -> dict[str, Any]:
    key = str(preset or "medium").strip().lower()
    if key not in NARROW_PRESET_CONFIGS:
        raise ValueError(f"Unsupported narrow forecast rehab preset '{preset}'")

    preset_config = NARROW_PRESET_CONFIGS[key]
    group_payload = [
        {
            "group_name": group_name,
            "tickers": list(PRIMARY_GROUPS[group_name]),
        }
        for group_name in preset_config["group_names"]
    ]
    model_sequence = [*PRIMARY_MODELS, *COMPARATOR_MODELS]
    if include_baselines:
        model_sequence.extend(BASELINE_ONLY_MODELS)

    policy_baseline = forecast_rehab_policy_baseline()
    return {
        "preset": key,
        "ticker_groups": group_payload,
        "horizons": [int(value) for value in preset_config["horizons"]],
        "target_names": list(preset_config["target_names"]),
        "feature_families": list(preset_config["feature_families"]),
        "feature_family_metadata": {name: dict(payload) for name, payload in NARROW_FEATURE_FAMILY_METADATA.items()},
        "primary_models": list(PRIMARY_MODELS),
        "comparator_models": list(COMPARATOR_MODELS),
        "baseline_only_models": list(BASELINE_ONLY_MODELS if include_baselines else []),
        "demoted_models": list(DEMOTED_MODELS),
        "models": model_sequence,
        "cost_modes": {name: dict(payload) for name, payload in NARROW_COST_MODES.items()},
        "evaluation_config": dict(preset_config["evaluation_config"]),
        "policy_baseline": policy_baseline,
    }


def build_narrow_core_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values(["group_name", "horizon", "target_name", "feature_family"]).reset_index(drop=True)


def build_narrow_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    inventory = build_feature_inventory_table(frame)
    rows: list[dict[str, Any]] = []
    for family_name in NARROW_FEATURE_FAMILY_ORDER:
        payload = dict(NARROW_FEATURE_FAMILY_METADATA[family_name])
        requested = list(payload["features"])
        resolved = [column for column in requested if column in frame.columns]
        missing = [column for column in requested if column not in frame.columns]
        categories = sorted(
            set(
                inventory.loc[inventory["feature_name"].isin(resolved), "category"].astype(str).tolist()
            )
        )
        rows.append(
            {
                "feature_family": family_name,
                "source_family": str(payload["source_family"]),
                "feature_count": len(resolved),
                "requested_feature_count": len(requested),
                "resolved_feature_count": len(resolved),
                "missing_feature_count": len(missing),
                "feature_categories": ",".join(categories),
                "rationale": str(payload["rationale"]),
                "features": ",".join(resolved),
                "missing_features": ",".join(missing),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_family").reset_index(drop=True)


def resolve_narrow_feature_family_columns(
    frame: pd.DataFrame,
    *,
    family_name: str,
) -> list[str]:
    key = str(family_name).strip().lower()
    if key not in NARROW_FEATURE_FAMILY_METADATA:
        raise ValueError(f"Unsupported narrow feature family '{family_name}'")
    requested = list(NARROW_FEATURE_FAMILY_METADATA[key]["features"])
    resolved = [column for column in requested if column in frame.columns]
    if not resolved:
        raise ValueError(f"Narrow feature family '{family_name}' resolved to no available columns")
    return resolved


def build_narrow_policy_configuration(policy_baseline: dict[str, Any], *, cost_mode: str) -> dict[str, Any]:
    if cost_mode not in NARROW_COST_MODES:
        raise ValueError(f"Unsupported cost mode '{cost_mode}'")
    payload = dict(policy_baseline)
    payload["cost_mode"] = str(cost_mode)
    payload.update(NARROW_COST_MODES[cost_mode])
    return payload


def create_narrow_forecast_model(model_name: str, *, target_spec: ForecastTargetSpec) -> Any:
    return create_rehab_forecast_model(model_name, target_spec=target_spec)

