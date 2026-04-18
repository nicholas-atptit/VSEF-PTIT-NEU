"""Shared helpers for Phase 2 regime labels and probability handling."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REGIME_LABELS = ("bull", "bear", "sideway")


def dominant_regime_label(probabilities: pd.DataFrame) -> pd.Series:
    required = ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]
    missing = [column for column in required if column not in probabilities.columns]
    if missing:
        raise ValueError(f"Missing probability columns for regime labeling: {missing}")
    mapping = {
        "regime_prob_bull": "bull",
        "regime_prob_bear": "bear",
        "regime_prob_sideway": "sideway",
    }
    max_columns = probabilities[required].idxmax(axis=1)
    return max_columns.map(mapping).fillna("sideway").astype(str)


def normalize_regime_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    required = ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]
    missing = [column for column in required if column not in probabilities.columns]
    if missing:
        raise ValueError(f"Missing probability columns for regime normalization: {missing}")
    normalized = probabilities.copy()
    normalized[required] = normalized[required].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = normalized[required].sum(axis=1).replace(0.0, 1.0)
    normalized[required] = normalized[required].div(total, axis=0)
    normalized["regime_label"] = dominant_regime_label(normalized)
    return normalized


def map_state_probabilities(
    state_probabilities: pd.DataFrame,
    state_order: Iterable[int],
) -> pd.DataFrame:
    ordered_states = list(state_order)
    if len(ordered_states) != 3:
        raise ValueError("Phase 2 regime mapping expects exactly three ordered states")
    missing = [state for state in ordered_states if state not in state_probabilities.columns]
    if missing:
        raise ValueError(f"Missing state probability columns: {missing}")
    mapped = pd.DataFrame(
        {
            "regime_prob_bear": pd.to_numeric(state_probabilities[ordered_states[0]], errors="coerce").astype(float),
            "regime_prob_sideway": pd.to_numeric(state_probabilities[ordered_states[1]], errors="coerce").astype(float),
            "regime_prob_bull": pd.to_numeric(state_probabilities[ordered_states[2]], errors="coerce").astype(float),
        },
        index=state_probabilities.index,
    )
    return normalize_regime_probabilities(mapped)


def ordered_states_from_means(
    returns: pd.Series,
    state_probabilities: pd.DataFrame,
) -> list[int]:
    clean_returns = pd.to_numeric(pd.Series(returns), errors="coerce").astype(float)
    state_means: dict[int, float] = {}
    for column in state_probabilities.columns:
        weights = pd.to_numeric(state_probabilities[column], errors="coerce").fillna(0.0).astype(float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            state_means[int(column)] = 0.0
            continue
        weighted_mean = float(np.dot(clean_returns.fillna(0.0), weights) / weight_sum)
        state_means[int(column)] = weighted_mean
    return [state for state, _ in sorted(state_means.items(), key=lambda item: item[1])]
