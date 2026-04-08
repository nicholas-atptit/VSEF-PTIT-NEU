"""Sequence builders for time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    """Rolling-window representation aligned to target row indices."""

    X: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray
    feature_columns: tuple[str, ...]
    sequence_length: int
    rows_lost: int

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])


def create_sequence_dataset(
    features: pd.DataFrame | np.ndarray,
    targets: pd.Series | np.ndarray,
    *,
    sequence_length: int,
    feature_columns: list[str] | None = None,
) -> SequenceDataset:
    """Convert chronological tabular data into rolling windows.

    Each sample ending at row ``i`` uses rows ``[i-sequence_length+1, i]`` and
    predicts the target at row ``i``. This preserves chronology and makes the
    target alignment explicit.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    if isinstance(features, pd.DataFrame):
        feature_columns = feature_columns or list(features.columns)
        data = features[feature_columns].to_numpy(dtype=float, copy=False)
    else:
        if feature_columns is None:
            feature_columns = [f"f{i}" for i in range(features.shape[1])]
        data = np.asarray(features, dtype=float)

    target_array = np.asarray(targets)
    if len(data) != len(target_array):
        raise ValueError("features and targets must have the same number of rows")

    n_rows = len(data)
    rows_lost = max(sequence_length - 1, 0)
    if n_rows <= rows_lost:
        empty_y_shape = target_array.shape[1:] if target_array.ndim > 1 else ()
        return SequenceDataset(
            X=np.empty((0, sequence_length, data.shape[1]), dtype=float),
            y=np.empty((0,) + empty_y_shape, dtype=target_array.dtype),
            target_indices=np.empty((0,), dtype=int),
            feature_columns=tuple(feature_columns),
            sequence_length=sequence_length,
            rows_lost=rows_lost,
        )

    windows = []
    aligned_targets = []
    target_indices = []
    for end_idx in range(sequence_length - 1, n_rows):
        start_idx = end_idx - sequence_length + 1
        windows.append(data[start_idx : end_idx + 1])
        aligned_targets.append(target_array[end_idx])
        target_indices.append(end_idx)

    return SequenceDataset(
        X=np.asarray(windows, dtype=float),
        y=np.asarray(aligned_targets),
        target_indices=np.asarray(target_indices, dtype=int),
        feature_columns=tuple(feature_columns),
        sequence_length=sequence_length,
        rows_lost=rows_lost,
    )


def select_sequence_range(
    dataset: SequenceDataset,
    *,
    start_index: int | None = None,
    stop_index: int | None = None,
) -> SequenceDataset:
    """Filter a sequence dataset by target row index range."""

    mask = np.ones(dataset.sample_count, dtype=bool)
    if start_index is not None:
        mask &= dataset.target_indices >= start_index
    if stop_index is not None:
        mask &= dataset.target_indices < stop_index
    return SequenceDataset(
        X=dataset.X[mask],
        y=dataset.y[mask],
        target_indices=dataset.target_indices[mask],
        feature_columns=dataset.feature_columns,
        sequence_length=dataset.sequence_length,
        rows_lost=dataset.rows_lost,
    )


def build_latest_sequence(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    sequence_length: int,
) -> np.ndarray:
    """Build the most recent inference window for a sequence model."""

    if len(features) < sequence_length:
        raise ValueError(
            f"Insufficient history for sequence_length={sequence_length}: "
            f"got {len(features)} rows"
        )
    return (
        features[feature_columns]
        .tail(sequence_length)
        .to_numpy(dtype=float, copy=False)
        .reshape(1, sequence_length, len(feature_columns))
    )
