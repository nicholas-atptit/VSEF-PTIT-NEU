from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.regime.regime_detector import REGIME_TO_CODE, RegimeDetector


def test_regime_detector_transitions_and_probabilities() -> None:
    index = pd.bdate_range("2025-01-01", periods=9)
    volatility = pd.Series([0.01, 0.012, 0.015, 0.04, 0.05, 0.045, 0.02, 0.018, 0.017], index=index)
    drawdown = pd.Series([0.0, -0.01, -0.02, -0.03, -0.05, -0.06, -0.10, -0.16, -0.18], index=index)
    delta_covar = pd.Series([0.0, 0.001, 0.002, 0.004, 0.006, 0.007, 0.01, 0.02, 0.03], index=index)

    detector = RegimeDetector(
        high_vol_threshold=0.03,
        crisis_drawdown_threshold=-0.12,
        crisis_delta_covar_threshold=0.015,
    )
    result = detector.detect(
        volatility=volatility,
        drawdown=drawdown,
        delta_covar=delta_covar,
    )

    assert result.labels.iloc[0] == "NORMAL"
    assert result.labels.iloc[3] == "HIGH_VOL"
    assert result.labels.iloc[-1] == "CRISIS"
    assert result.encoded_labels.iloc[-1] == REGIME_TO_CODE["CRISIS"]
    np.testing.assert_allclose(result.probabilities.sum(axis=1).to_numpy(), np.ones(len(index)))


def test_regime_detector_handles_missing_columns_via_frame_api() -> None:
    index = pd.bdate_range("2025-02-01", periods=5)
    frame = pd.DataFrame({"rolling_volatility_20": [0.01, 0.02, 0.04, 0.01, 0.05]}, index=index)
    detector = RegimeDetector(high_vol_threshold=0.03)
    result = detector.detect_from_frame(frame)

    assert len(result.labels) == 5
    assert result.labels.iloc[2] == "HIGH_VOL"
    np.testing.assert_allclose(result.probabilities.sum(axis=1).to_numpy(), np.ones(len(index)))
