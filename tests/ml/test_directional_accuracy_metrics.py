import math

import numpy as np

from src.ml.metrics import compute_directional_accuracy_from_returns


def test_directional_accuracy_basic_correct_incorrect_sign_matching():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, -0.20, 0.30, -0.40],
        predicted_return=[0.05, -0.10, -0.20, 0.10],
    )

    assert result["n_obs"] == 4
    assert result["accuracy"] == 0.5


def test_directional_accuracy_ignores_nan_values():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, np.nan, -0.30, 0.20],
        predicted_return=[0.01, 0.20, np.nan, -0.10],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 0.5


def test_directional_accuracy_ignores_infinite_values():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.10, np.inf, -0.30, -np.inf],
        predicted_return=[0.01, 0.20, -0.10, -0.10],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 1.0


def test_directional_accuracy_ignores_zero_actual_returns():
    result = compute_directional_accuracy_from_returns(
        actual_return=[0.0, 0.10, -0.10, 0.0],
        predicted_return=[0.50, 0.20, 0.20, -0.50],
    )

    assert result["n_obs"] == 2
    assert result["accuracy"] == 0.5


def test_directional_accuracy_empty_input_returns_nan_accuracy():
    result = compute_directional_accuracy_from_returns(actual_return=[], predicted_return=[])

    assert result["n_obs"] == 0
    assert math.isnan(result["accuracy"])
