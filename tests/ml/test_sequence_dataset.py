from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.sequence_dataset import build_latest_sequence, create_sequence_dataset


def test_sequence_shapes_and_alignment() -> None:
    features = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "f2": [10, 20, 30, 40, 50],
        }
    )
    targets = pd.Series([100, 200, 300, 400, 500])

    dataset = create_sequence_dataset(features, targets, sequence_length=3)

    assert dataset.X.shape == (3, 3, 2)
    assert dataset.y.tolist() == [300, 400, 500]
    assert dataset.target_indices.tolist() == [2, 3, 4]
    np.testing.assert_array_equal(dataset.X[0], np.array([[1, 10], [2, 20], [3, 30]]))


def test_sequence_builder_preserves_past_only_windows() -> None:
    features = pd.DataFrame({"value": np.arange(6)})
    targets = pd.Series(np.arange(6) * 10)
    dataset = create_sequence_dataset(features, targets, sequence_length=4)

    assert dataset.rows_lost == 3
    np.testing.assert_array_equal(dataset.X[-1].reshape(-1), np.array([2, 3, 4, 5]))
    assert dataset.y[-1] == 50


def test_build_latest_sequence_raises_on_insufficient_history() -> None:
    features = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    try:
        build_latest_sequence(features, feature_columns=["f1", "f2"], sequence_length=3)
    except ValueError as exc:
        assert "Insufficient history" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for insufficient sequence history")
