from __future__ import annotations

import pandas as pd
import pytest

from src.strategy.execution_policy import PolicyConfiguration, execute_policy_configuration


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-05"),
                "ticker": "AAA",
                "y_true": 0.01,
                "y_pred": 0.02,
                "model_name": "test_model",
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
            }
        ]
    )


def _risk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-05"),
                "ticker": "AAA",
                "model_name": "test_model",
                "window_id": "w1",
                "vol_forecast": 0.08,
                "drawdown_state": "severe",
                "source_model": "garch",
            }
        ]
    )


def _regime_frame(*, label: str = "bear") -> pd.DataFrame:
    probabilities = {
        "bull": (0.80, 0.10, 0.10),
        "bear": (0.05, 0.85, 0.10),
        "sideway": (0.20, 0.20, 0.60),
    }[label]
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-05"),
                "ticker": "AAA",
                "regime_label": label,
                "regime_prob_bull": probabilities[0],
                "regime_prob_bear": probabilities[1],
                "regime_prob_sideway": probabilities[2],
                "source_model": "markov_switching",
                "window_id": "w1",
            }
        ]
    )


def test_sizing_variants_change_position_size_in_expected_direction() -> None:
    forecast_df = _forecast_frame()
    risk_df = _risk_frame()

    current_config = PolicyConfiguration(
        policy_variant="risk_only_no_regime",
        strategy_variant="forecast_plus_risk",
        sizing_mode="adaptive",
        sizing_profile="adaptive_current",
        use_risk_context=True,
        use_volatility_sizing=True,
        use_drawdown_control=True,
    )
    lighter_drawdown_config = PolicyConfiguration(
        policy_variant="risk_only_no_regime",
        strategy_variant="forecast_plus_risk",
        sizing_mode="adaptive",
        sizing_profile="adaptive_lighter_drawdown",
        use_risk_context=True,
        use_volatility_sizing=True,
        use_drawdown_control=True,
        drawdown_haircut_strength=0.5,
    )
    capped_config = PolicyConfiguration(
        policy_variant="risk_only_no_regime",
        strategy_variant="forecast_plus_risk",
        sizing_mode="adaptive",
        sizing_profile="adaptive_capped_floor",
        use_risk_context=True,
        use_volatility_sizing=True,
        use_drawdown_control=True,
        min_position_size=0.20,
        max_position_size=0.75,
    )
    fixed_config = PolicyConfiguration(
        policy_variant="fixed_threshold_fixed_fraction",
        strategy_variant="forecast_only",
        sizing_mode="fixed_fraction",
        sizing_profile="fixed_fraction_full",
        fixed_position_size=1.0,
    )

    _, current_positions = execute_policy_configuration(
        forecast_df,
        policy_config=current_config,
        threshold=0.005,
        risk_df=risk_df,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )
    _, lighter_positions = execute_policy_configuration(
        forecast_df,
        policy_config=lighter_drawdown_config,
        threshold=0.005,
        risk_df=risk_df,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )
    _, capped_positions = execute_policy_configuration(
        forecast_df,
        policy_config=capped_config,
        threshold=0.005,
        risk_df=risk_df,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )
    _, fixed_positions = execute_policy_configuration(
        forecast_df,
        policy_config=fixed_config,
        threshold=0.005,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )

    current_size = float(current_positions.loc[0, "position_size"])
    lighter_size = float(lighter_positions.loc[0, "position_size"])
    capped_size = float(capped_positions.loc[0, "position_size"])
    fixed_size = float(fixed_positions.loc[0, "position_size"])

    assert current_size < lighter_size < capped_size < fixed_size
    assert capped_size == pytest.approx(0.20)
    assert fixed_size == pytest.approx(1.0)


def test_regime_thresholding_can_veto_a_trade_that_fixed_threshold_allows() -> None:
    forecast_df = _forecast_frame()
    forecast_df.loc[0, "y_pred"] = 0.0055
    regime_df = _regime_frame(label="bear")

    fixed_config = PolicyConfiguration(
        policy_variant="fixed_threshold_fixed_fraction",
        strategy_variant="forecast_only",
        sizing_mode="fixed_fraction",
        sizing_profile="fixed_fraction_full",
        fixed_position_size=1.0,
    )
    regime_config = PolicyConfiguration(
        policy_variant="regime_threshold_fixed_fraction",
        strategy_variant="forecast_plus_regime",
        threshold_policy="regime_aware",
        sizing_mode="fixed_fraction",
        sizing_profile="fixed_fraction_full",
        fixed_position_size=1.0,
        use_regime_context=True,
    )

    fixed_signals, _ = execute_policy_configuration(
        forecast_df,
        policy_config=fixed_config,
        threshold=0.005,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )
    regime_signals, _ = execute_policy_configuration(
        forecast_df,
        policy_config=regime_config,
        threshold=0.005,
        regime_df=regime_df,
        capital_config={"risk_budget": 0.02, "max_position_size": 1.0},
    )

    assert float(fixed_signals.loc[0, "signal"]) == pytest.approx(1.0)
    assert float(regime_signals.loc[0, "signal"]) == pytest.approx(0.0)
    assert float(regime_signals.loc[0, "threshold"]) == pytest.approx(0.00625)
