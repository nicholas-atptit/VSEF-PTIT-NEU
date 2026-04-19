"""Execution policy that turns forecasts into signals and sized positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.strategy.signal_rules import StrategyModel
from src.strategy.regime_thresholding import generate_regime_aware_signals, merge_strategy_context
from src.strategy.sizing import size_positions
from src.strategy.thresholding import generate_threshold_signals


@dataclass(frozen=True)
class PolicyConfiguration:
    """Explicit execution policy configuration for Phase 2.6 calibration."""

    policy_variant: str
    strategy_variant: str
    policy_label: str | None = None
    threshold_policy: str = "fixed"
    sizing_profile: str = "fixed_fraction_full"
    sizing_label: str | None = None
    use_risk_context: bool = False
    use_regime_context: bool = False
    use_volatility_sizing: bool = False
    use_drawdown_control: bool = False
    use_regime_sizing: bool = False
    sizing_mode: str = "fixed_fraction"
    fixed_position_size: float | None = None
    min_position_size: float = 0.0
    max_position_size: float | None = None
    volatility_target_scale: float = 1.0
    drawdown_haircut_strength: float = 1.0
    regime_multiplier_strength: float = 1.0
    regime_thresholds: dict[str, float] | None = None
    regime_threshold_multipliers: dict[str, float] | None = None
    policy_family: str | None = None
    ablation_labels: tuple[str, ...] = ()


def _annotate_policy(frame: pd.DataFrame, config: PolicyConfiguration) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["policy_variant"] = config.policy_variant
    annotated["policy_label"] = str(config.policy_label or config.policy_variant)
    annotated["strategy_variant"] = config.strategy_variant
    annotated["threshold_policy"] = str(config.threshold_policy)
    annotated["sizing_profile"] = str(config.sizing_profile)
    annotated["sizing_label"] = str(config.sizing_label or config.sizing_profile)
    annotated["use_risk_context"] = bool(config.use_risk_context)
    annotated["use_regime_context"] = bool(config.use_regime_context)
    annotated["use_volatility_sizing"] = bool(config.use_volatility_sizing)
    annotated["use_drawdown_control"] = bool(config.use_drawdown_control)
    annotated["use_regime_sizing"] = bool(config.use_regime_sizing)
    annotated["policy_family"] = str(config.policy_family or config.policy_variant)
    annotated["ablation_labels"] = ",".join(config.ablation_labels)
    return annotated


def execute_policy_configuration(
    forecast_df: pd.DataFrame,
    *,
    policy_config: PolicyConfiguration,
    threshold: float,
    allow_short: bool = False,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    capital_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute one explicit policy configuration for Phase 2.6 calibration."""

    active_risk_df = risk_df if policy_config.use_risk_context else None
    active_regime_df = regime_df if policy_config.use_regime_context else None
    if str(policy_config.threshold_policy).lower() == "regime_aware":
        signal_df = generate_regime_aware_signals(
            forecast_df,
            threshold=threshold,
            allow_short=allow_short,
            risk_df=active_risk_df,
            regime_df=active_regime_df,
            regime_thresholds=policy_config.regime_thresholds,
            regime_threshold_multipliers=policy_config.regime_threshold_multipliers,
        )
    else:
        signal_df = generate_threshold_signals(
            forecast_df,
            threshold=threshold,
            allow_short=allow_short,
        )
        signal_df = merge_strategy_context(signal_df, risk_df=active_risk_df, regime_df=active_regime_df)
        signal_df = StrategyModel.validate_signal_output(signal_df)

    merged_capital_config = dict(capital_config or {})
    merged_capital_config.update(
        {
            "sizing_mode": str(policy_config.sizing_mode),
            "use_volatility_sizing": bool(policy_config.use_volatility_sizing),
            "use_drawdown_control": bool(policy_config.use_drawdown_control),
            "use_regime_sizing": bool(policy_config.use_regime_sizing),
            "min_position_size": float(policy_config.min_position_size),
            "volatility_target_scale": float(policy_config.volatility_target_scale),
            "drawdown_haircut_strength": float(policy_config.drawdown_haircut_strength),
            "regime_multiplier_strength": float(policy_config.regime_multiplier_strength),
        }
    )
    if policy_config.fixed_position_size is not None:
        merged_capital_config["fixed_position_size"] = float(policy_config.fixed_position_size)
    if policy_config.max_position_size is not None:
        merged_capital_config["max_position_size"] = float(policy_config.max_position_size)

    position_df = size_positions(
        signal_df,
        risk_df=active_risk_df,
        capital_config=merged_capital_config,
    )
    return _annotate_policy(signal_df, policy_config), _annotate_policy(position_df, policy_config)


class BasicExecutionPolicy(StrategyModel):
    """Conservative threshold-and-size execution policy for Phase 1."""

    model_name = "basic_execution_policy"

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        allow_short: bool = False,
        capital_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.threshold = max(float(threshold), 0.0)
        self.allow_short = bool(allow_short)
        self.capital_config = dict(capital_config or {})

    def generate_signal(
        self,
        forecast_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        signal_df = generate_threshold_signals(
            forecast_df,
            threshold=self.threshold,
            allow_short=self.allow_short,
        )
        signal_df = merge_strategy_context(signal_df, risk_df=risk_df, regime_df=regime_df)
        return self.validate_signal_output(signal_df)

    def size_positions(
        self,
        signal_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        capital_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        merged_capital_config = dict(self.capital_config)
        merged_capital_config.update(dict(capital_config or {}))
        positions = size_positions(
            signal_df,
            risk_df=risk_df,
            capital_config=merged_capital_config,
        )
        return self.validate_position_output(positions)


class RegimeAwareExecutionPolicy(StrategyModel):
    """Phase 2 execution policy conditioned on regime state and forecast volatility."""

    model_name = "regime_aware_execution_policy"

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        allow_short: bool = False,
        regime_thresholds: dict[str, float] | None = None,
        regime_threshold_multipliers: dict[str, float] | None = None,
        capital_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.threshold = max(float(threshold), 0.0)
        self.allow_short = bool(allow_short)
        self.regime_thresholds = dict(regime_thresholds or {})
        self.regime_threshold_multipliers = dict(regime_threshold_multipliers or {})
        self.capital_config = dict(capital_config or {})

    def generate_signal(
        self,
        forecast_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        signal_df = generate_regime_aware_signals(
            forecast_df,
            threshold=self.threshold,
            allow_short=self.allow_short,
            risk_df=risk_df,
            regime_df=regime_df,
            regime_thresholds=self.regime_thresholds,
            regime_threshold_multipliers=self.regime_threshold_multipliers,
        )
        return self.validate_signal_output(signal_df)

    def size_positions(
        self,
        signal_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        capital_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        merged_capital_config = dict(self.capital_config)
        merged_capital_config.update(dict(capital_config or {}))
        positions = size_positions(
            signal_df,
            risk_df=risk_df,
            capital_config=merged_capital_config,
        )
        return self.validate_position_output(positions)
