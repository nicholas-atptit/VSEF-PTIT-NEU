from __future__ import annotations

import pandas as pd

from src.strategy.regime_thresholding import generate_regime_aware_signals
from src.strategy.sizing import size_positions


def test_regime_aware_thresholding_varies_barrier_by_regime() -> None:
    forecast = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "y_true": 0.0,
                "y_pred": 0.011,
                "model_name": "linear",
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
            },
            {
                "timestamp": "2024-01-03",
                "ticker": "AAA",
                "y_true": 0.0,
                "y_pred": 0.011,
                "model_name": "linear",
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
            },
        ]
    )
    regime = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "window_id": "w1",
                "regime_label": "bull",
                "regime_prob_bull": 0.8,
                "regime_prob_bear": 0.1,
                "regime_prob_sideway": 0.1,
                "source_model": "markov_switching",
            },
            {
                "timestamp": "2024-01-03",
                "ticker": "AAA",
                "window_id": "w1",
                "regime_label": "bear",
                "regime_prob_bull": 0.1,
                "regime_prob_bear": 0.8,
                "regime_prob_sideway": 0.1,
                "source_model": "markov_switching",
            },
        ]
    )

    signals = generate_regime_aware_signals(forecast, threshold=0.01, regime_df=regime)

    assert signals.loc[0, "threshold"] < signals.loc[1, "threshold"]
    assert signals.loc[0, "signal"] == 1.0
    assert signals.loc[1, "signal"] == 0.0


def test_volatility_aware_sizing_reduces_exposure_in_hostile_state() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "model_name": "linear",
                "signal": 1.0,
                "threshold": 0.01,
                "y_pred": 0.02,
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
                "regime_label": "bull",
                "vol_forecast": 0.01,
                "drawdown_state": "normal",
            },
            {
                "timestamp": "2024-01-03",
                "ticker": "AAA",
                "model_name": "linear",
                "signal": 1.0,
                "threshold": 0.01,
                "y_pred": 0.02,
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
                "regime_label": "bear",
                "vol_forecast": 0.05,
                "drawdown_state": "severe",
            },
        ]
    )

    positions = size_positions(signals)

    assert positions.loc[0, "position_size"] > positions.loc[1, "position_size"]

