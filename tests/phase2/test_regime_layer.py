from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.contracts import validate_regime_frame
from src.regime.labels import map_state_probabilities, ordered_states_from_means
from src.regime.markov_switching import MarkovSwitchingRegimeModel


def test_markov_switching_fallback_outputs_valid_regime_schema() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="B"),
            "ticker": ["AAA"] * 40,
            "close_return_1d": np.linspace(-0.02, 0.02, 40),
            "window_id": ["w1"] * 40,
        }
    )
    model = MarkovSwitchingRegimeModel(
        min_train_observations=100,
        return_column="close_return_1d",
    ).fit(frame.iloc[:25], config={"window_id": "w1"})

    result = model.predict(frame)

    assert not result.empty
    assert result["source_model"].eq("markov_switching_threshold_fallback").all()
    assert np.allclose(
        result[["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]].sum(axis=1).to_numpy(),
        1.0,
    )
    validate_regime_frame(result)


def test_state_probability_mapping_orders_bear_sideway_bull() -> None:
    returns = pd.Series([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    state_probabilities = pd.DataFrame(
        {
            2: [1.0, 1.0, 0.8, 0.0, 0.0, 0.0],
            0: [0.0, 0.0, 0.2, 1.0, 0.2, 0.0],
            1: [0.0, 0.0, 0.0, 0.0, 0.8, 1.0],
        }
    )

    state_order = ordered_states_from_means(returns, state_probabilities)
    mapped = map_state_probabilities(state_probabilities, state_order)

    assert state_order == [2, 0, 1]
    assert mapped.loc[0, "regime_label"] == "bear"
    assert mapped.loc[3, "regime_label"] == "sideway"
    assert mapped.loc[5, "regime_label"] == "bull"

