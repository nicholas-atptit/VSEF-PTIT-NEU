"""Phase 2.6 calibration matrix helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.hardening import DEFAULT_TICKER_GROUPS


CANONICAL_POLICY_LIBRARY: dict[str, dict[str, Any]] = {
    "forecast_only_current": {
        "policy_variant": "forecast_only_current",
        "policy_label": "Forecast-only current path",
        "policy_family": "forecast_only",
        "strategy_variant": "forecast_only",
        "threshold_policy": "fixed",
        "sizing_mode": "adaptive",
        "use_risk_context": False,
        "use_regime_context": False,
        "use_volatility_sizing": False,
        "use_drawdown_control": False,
        "use_regime_sizing": False,
        "sizing_profile_names": ["adaptive_current"],
        "ablation_labels": ["A"],
    },
    "fixed_threshold_fixed_fraction": {
        "policy_variant": "fixed_threshold_fixed_fraction",
        "policy_label": "Fixed threshold + fixed fraction",
        "policy_family": "simple_baseline",
        "strategy_variant": "forecast_only",
        "threshold_policy": "fixed",
        "sizing_mode": "fixed_fraction",
        "use_risk_context": False,
        "use_regime_context": False,
        "use_volatility_sizing": False,
        "use_drawdown_control": False,
        "use_regime_sizing": False,
        "sizing_profile_names": ["fixed_fraction_full"],
        "ablation_labels": ["B"],
    },
    "fixed_threshold_adaptive": {
        "policy_variant": "fixed_threshold_adaptive",
        "policy_label": "Fixed threshold + adaptive volatility sizing",
        "policy_family": "risk_component",
        "strategy_variant": "forecast_plus_risk",
        "threshold_policy": "fixed",
        "sizing_mode": "adaptive",
        "use_risk_context": True,
        "use_regime_context": False,
        "use_volatility_sizing": True,
        "use_drawdown_control": False,
        "use_regime_sizing": False,
        "sizing_profile_names": ["adaptive_current"],
        "ablation_labels": ["C"],
    },
    "regime_threshold_fixed_fraction": {
        "policy_variant": "regime_threshold_fixed_fraction",
        "policy_label": "Regime threshold + fixed fraction",
        "policy_family": "regime_component",
        "strategy_variant": "forecast_plus_regime",
        "threshold_policy": "regime_aware",
        "sizing_mode": "fixed_fraction",
        "use_risk_context": False,
        "use_regime_context": True,
        "use_volatility_sizing": False,
        "use_drawdown_control": False,
        "use_regime_sizing": False,
        "sizing_profile_names": ["fixed_fraction_full"],
        "ablation_labels": ["D", "H"],
    },
    "fixed_threshold_adaptive_drawdown": {
        "policy_variant": "fixed_threshold_adaptive_drawdown",
        "policy_label": "Fixed threshold + drawdown-only adaptive sizing",
        "policy_family": "risk_component",
        "strategy_variant": "forecast_plus_risk",
        "threshold_policy": "fixed",
        "sizing_mode": "adaptive",
        "use_risk_context": True,
        "use_regime_context": False,
        "use_volatility_sizing": False,
        "use_drawdown_control": True,
        "use_regime_sizing": False,
        "sizing_profile_names": ["adaptive_current"],
        "ablation_labels": ["E"],
    },
    "risk_only_no_regime": {
        "policy_variant": "risk_only_no_regime",
        "policy_label": "Risk-only adaptive stack",
        "policy_family": "risk_stack",
        "strategy_variant": "forecast_plus_risk",
        "threshold_policy": "fixed",
        "sizing_mode": "adaptive",
        "use_risk_context": True,
        "use_regime_context": False,
        "use_volatility_sizing": True,
        "use_drawdown_control": True,
        "use_regime_sizing": False,
        "sizing_profile_names": [
            "adaptive_current",
            "adaptive_capped_floor",
            "adaptive_lighter_vol",
            "adaptive_lighter_drawdown",
        ],
        "ablation_labels": ["G"],
    },
    "regime_threshold_adaptive_drawdown": {
        "policy_variant": "regime_threshold_adaptive_drawdown",
        "policy_label": "Regime threshold + risk stack",
        "policy_family": "regime_risk_stack",
        "strategy_variant": "forecast_plus_risk_and_regime",
        "threshold_policy": "regime_aware",
        "sizing_mode": "adaptive",
        "use_risk_context": True,
        "use_regime_context": True,
        "use_volatility_sizing": True,
        "use_drawdown_control": True,
        "use_regime_sizing": True,
        "sizing_profile_names": [
            "adaptive_current",
            "adaptive_capped_floor",
            "adaptive_lighter_vol",
            "adaptive_lighter_drawdown",
        ],
        "ablation_labels": ["F"],
    },
}

SIZING_PROFILE_LIBRARY: dict[str, dict[str, Any]] = {
    "fixed_fraction_full": {
        "sizing_profile": "fixed_fraction_full",
        "sizing_label": "Fixed fraction 1.0",
        "fixed_position_size": 1.0,
        "min_position_size": 0.0,
        "max_position_size": 1.0,
        "volatility_target_scale": 1.0,
        "drawdown_haircut_strength": 1.0,
        "regime_multiplier_strength": 1.0,
    },
    "adaptive_current": {
        "sizing_profile": "adaptive_current",
        "sizing_label": "Adaptive current",
        "fixed_position_size": None,
        "min_position_size": 0.0,
        "max_position_size": 1.0,
        "volatility_target_scale": 1.0,
        "drawdown_haircut_strength": 1.0,
        "regime_multiplier_strength": 1.0,
    },
    "adaptive_capped_floor": {
        "sizing_profile": "adaptive_capped_floor",
        "sizing_label": "Adaptive capped/floored",
        "fixed_position_size": None,
        "min_position_size": 0.20,
        "max_position_size": 0.75,
        "volatility_target_scale": 1.0,
        "drawdown_haircut_strength": 1.0,
        "regime_multiplier_strength": 1.0,
    },
    "adaptive_lighter_vol": {
        "sizing_profile": "adaptive_lighter_vol",
        "sizing_label": "Adaptive lighter vol penalty",
        "fixed_position_size": None,
        "min_position_size": 0.0,
        "max_position_size": 1.0,
        "volatility_target_scale": 1.5,
        "drawdown_haircut_strength": 1.0,
        "regime_multiplier_strength": 1.0,
    },
    "adaptive_lighter_drawdown": {
        "sizing_profile": "adaptive_lighter_drawdown",
        "sizing_label": "Adaptive lighter drawdown haircut",
        "fixed_position_size": None,
        "min_position_size": 0.0,
        "max_position_size": 1.0,
        "volatility_target_scale": 1.0,
        "drawdown_haircut_strength": 0.5,
        "regime_multiplier_strength": 1.0,
    },
}

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "group_names": ["small_banks", "mixed_large_cap"],
        "horizons": [1, 5],
        "thresholds": [0.001, 0.005, 0.010],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
        ],
        "policy_names": [
            "forecast_only_current",
            "fixed_threshold_fixed_fraction",
            "risk_only_no_regime",
            "regime_threshold_adaptive_drawdown",
        ],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "medium": {
        "group_names": ["small_banks", "mixed_large_cap", "vn100_subset"],
        "horizons": [1, 5, 10],
        "thresholds": [0.001, 0.003, 0.005, 0.007, 0.010],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
        ],
        "policy_names": list(CANONICAL_POLICY_LIBRARY.keys()),
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "calibration_full": {
        "group_names": ["small_banks", "mixed_large_cap", "vn100_subset"],
        "horizons": [1, 5, 10],
        "thresholds": [0.001, 0.003, 0.005, 0.007, 0.010],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
            {"cost_mode": "conservative", "transaction_fee_bps": 45.0, "slippage_bps": 45.0},
        ],
        "policy_names": list(CANONICAL_POLICY_LIBRARY.keys()),
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 3,
        },
    },
}


def _format_token(value: float) -> str:
    formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "p")


def build_phase26_matrix_config(
    preset: str = "medium",
    *,
    ticker_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the bounded Phase 2.6 calibration matrix."""

    normalized_preset = str(preset or "medium").strip().lower()
    if normalized_preset not in PRESET_CONFIGS:
        raise ValueError(f"Unsupported Phase 2.6 preset '{preset}'")
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
    policies = [dict(CANONICAL_POLICY_LIBRARY[name]) for name in preset_config["policy_names"]]
    ablation_labels = sorted({label for policy in policies for label in policy.get("ablation_labels", [])})
    sizing_profiles = {
        name: dict(SIZING_PROFILE_LIBRARY[name])
        for policy in policies
        for name in policy.get("sizing_profile_names", [])
    }
    return {
        "preset": normalized_preset,
        "ticker_groups": selected_groups,
        "horizons": [int(value) for value in preset_config["horizons"]],
        "thresholds": [float(value) for value in preset_config["thresholds"]],
        "threshold_mode": "symmetric",
        "cost_modes": [dict(item) for item in preset_config["cost_modes"]],
        "policies": policies,
        "sizing_profiles": sizing_profiles,
        "ablation_labels": ablation_labels,
        "evaluation_config": dict(preset_config["evaluation_config"]),
    }


def build_phase26_core_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    """Return one row per forecast/risk/regime core evaluation scenario."""

    rows: list[dict[str, Any]] = []
    evaluation_config = dict(matrix_config.get("evaluation_config", {}))
    for group in matrix_config.get("ticker_groups", []):
        for horizon in matrix_config.get("horizons", []):
            group_name = str(group["group_name"])
            rows.append(
                {
                    "core_run_id": f"{group_name}_h{int(horizon):02d}",
                    "preset": str(matrix_config.get("preset", "medium")),
                    "group_name": group_name,
                    "tickers": list(group["tickers"]),
                    "ticker_count": len(group["tickers"]),
                    "horizon": int(horizon),
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
                "train_size",
                "test_size",
                "step_size",
                "gap_size",
                "max_windows",
            ]
        )
    return pd.DataFrame(rows).sort_values(["group_name", "horizon"]).reset_index(drop=True)


def build_phase26_sweep_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    """Expand the Phase 2.6 calibration matrix into explicit policy runs."""

    core_frame = build_phase26_core_frame(matrix_config)
    rows: list[dict[str, Any]] = []
    for core_row in core_frame.itertuples(index=False):
        for threshold in matrix_config.get("thresholds", []):
            for cost_mode in matrix_config.get("cost_modes", []):
                for policy in matrix_config.get("policies", []):
                    for sizing_profile_name in policy.get("sizing_profile_names", []):
                        sizing_profile = dict(matrix_config.get("sizing_profiles", {}).get(sizing_profile_name, {}))
                        threshold_token = _format_token(float(threshold))
                        run_id = "_".join(
                            [
                                str(core_row.core_run_id),
                                f"thr_{threshold_token}",
                                str(cost_mode["cost_mode"]),
                                str(policy["policy_variant"]),
                                str(sizing_profile_name),
                            ]
                        )
                        rows.append(
                            {
                                **core_row._asdict(),
                                "run_id": run_id,
                                "threshold": float(threshold),
                                "threshold_label": threshold_token,
                                "threshold_mode": str(matrix_config.get("threshold_mode", "symmetric")),
                                "cost_mode": str(cost_mode["cost_mode"]),
                                "transaction_fee_bps": float(cost_mode["transaction_fee_bps"]),
                                "slippage_bps": float(cost_mode["slippage_bps"]),
                                **policy,
                                **sizing_profile,
                            }
                        )
    if not rows:
        return pd.DataFrame(columns=["run_id", "core_run_id", "group_name", "horizon", "threshold", "cost_mode", "policy_variant"])
    return (
        pd.DataFrame(rows)
        .sort_values(["group_name", "horizon", "threshold", "cost_mode", "policy_variant", "sizing_profile"])
        .reset_index(drop=True)
    )
