"""Phase 2.5 benchmark matrix helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_STRATEGY_VARIANTS = (
    "forecast_only",
    "forecast_plus_risk",
    "forecast_plus_risk_and_regime",
)

DEFAULT_TICKER_GROUPS: dict[str, list[str]] = {
    "small_banks": ["ACB", "BID", "CTG", "MBB"],
    "mixed_large_cap": ["FPT", "HPG", "MWG", "VNM"],
    "vn100_subset": ["GAS", "REE", "SSI", "TCB", "VCB", "VHM"],
}

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "group_names": ["small_banks", "mixed_large_cap"],
        "horizons": [1, 5],
        "thresholds": [0.005, 0.010],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
        ],
        "sizing_modes": [
            {"sizing_mode": "adaptive", "fixed_position_size": None},
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
        "thresholds": [0.003, 0.007],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
        ],
        "sizing_modes": [
            {"sizing_mode": "adaptive", "fixed_position_size": None},
            {"sizing_mode": "fixed_fraction", "fixed_position_size": 1.0},
        ],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "full": {
        "group_names": ["small_banks", "mixed_large_cap", "vn100_subset"],
        "horizons": [1, 5, 10],
        "thresholds": [0.0025, 0.0050, 0.0075, 0.0100],
        "cost_modes": [
            {"cost_mode": "baseline", "transaction_fee_bps": 15.0, "slippage_bps": 20.0},
            {"cost_mode": "elevated", "transaction_fee_bps": 30.0, "slippage_bps": 35.0},
            {"cost_mode": "conservative", "transaction_fee_bps": 45.0, "slippage_bps": 45.0},
        ],
        "sizing_modes": [
            {"sizing_mode": "adaptive", "fixed_position_size": None},
            {"sizing_mode": "fixed_fraction", "fixed_position_size": 1.0},
        ],
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


def build_phase25_matrix_config(
    preset: str = "medium",
    *,
    ticker_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build a bounded hardening matrix configuration."""

    normalized_preset = str(preset or "medium").strip().lower()
    if normalized_preset not in PRESET_CONFIGS:
        raise ValueError(f"Unsupported Phase 2.5 preset '{preset}'")
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
        "horizons": list(preset_config["horizons"]),
        "thresholds": [float(value) for value in preset_config["thresholds"]],
        "cost_modes": [dict(item) for item in preset_config["cost_modes"]],
        "sizing_modes": [dict(item) for item in preset_config["sizing_modes"]],
        "strategy_variants": list(DEFAULT_STRATEGY_VARIANTS),
        "evaluation_config": dict(preset_config["evaluation_config"]),
    }


def build_phase25_core_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    """Return one row per forecast/risk/regime evaluation scenario."""

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


def build_phase25_sweep_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    """Expand the bounded hardening matrix into policy sweep rows."""

    core_frame = build_phase25_core_frame(matrix_config)
    rows: list[dict[str, Any]] = []
    for core_row in core_frame.itertuples(index=False):
        for threshold in matrix_config.get("thresholds", []):
            for cost_mode in matrix_config.get("cost_modes", []):
                for sizing_mode in matrix_config.get("sizing_modes", []):
                    threshold_token = _format_token(float(threshold))
                    sizing_name = str(sizing_mode["sizing_mode"])
                    run_id = "_".join(
                        [
                            str(core_row.core_run_id),
                            f"thr_{threshold_token}",
                            str(cost_mode["cost_mode"]),
                            sizing_name,
                        ]
                    )
                    rows.append(
                        {
                            **core_row._asdict(),
                            "run_id": run_id,
                            "threshold": float(threshold),
                            "threshold_label": threshold_token,
                            "cost_mode": str(cost_mode["cost_mode"]),
                            "transaction_fee_bps": float(cost_mode["transaction_fee_bps"]),
                            "slippage_bps": float(cost_mode["slippage_bps"]),
                            "sizing_mode": sizing_name,
                            "fixed_position_size": sizing_mode.get("fixed_position_size"),
                            "strategy_variants": list(matrix_config.get("strategy_variants", DEFAULT_STRATEGY_VARIANTS)),
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "core_run_id",
                "group_name",
                "tickers",
                "horizon",
                "threshold",
                "cost_mode",
                "transaction_fee_bps",
                "slippage_bps",
                "sizing_mode",
                "fixed_position_size",
            ]
        )
    return pd.DataFrame(rows).sort_values(["group_name", "horizon", "threshold", "cost_mode", "sizing_mode"]).reset_index(drop=True)
