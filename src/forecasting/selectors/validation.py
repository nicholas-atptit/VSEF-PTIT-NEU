"""Validation-only candidate selectors."""

from __future__ import annotations

import pandas as pd


def _validation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" not in frame:
        raise ValueError("selection requires a split column")
    rows = frame[frame["split"].eq("validation")].copy()
    if rows.empty:
        raise ValueError("selection requires validation rows")
    return rows


def select_direction(frame: pd.DataFrame, *, min_prediction_balance: float = 0.05, max_prediction_balance: float = 0.95) -> pd.Series:
    rows = _validation_rows(frame)
    rows = rows[rows["prediction_balance"].between(min_prediction_balance, max_prediction_balance)]
    if rows.empty:
        raise ValueError("all direction candidates have collapsed prediction balance")
    return rows.sort_values(["balanced_accuracy", "macro_f1", "mcc"], ascending=False).iloc[0]


def select_return(frame: pd.DataFrame) -> pd.Series:
    rows = _validation_rows(frame)
    if "beats_baseline" in rows:
        rows = rows[rows["beats_baseline"].astype(bool)]
    if rows.empty:
        raise ValueError("no return candidate improves on its baseline")
    return rows.sort_values(["rmse", "mae"], ascending=True).iloc[0]


def select_range(frame: pd.DataFrame, *, minimum_coverage: float = 0.8) -> pd.Series:
    rows = _validation_rows(frame)
    rows = rows[rows["interval_coverage"] >= minimum_coverage]
    if rows.empty:
        raise ValueError("no range candidate satisfies minimum coverage")
    return rows.sort_values(["winkler_score", "average_interval_width"], ascending=True).iloc[0]


def select_ranking(frame: pd.DataFrame) -> pd.Series:
    rows = _validation_rows(frame)
    return rows.sort_values(["spearman_ic", "ndcg_at_10", "ndcg_at_5"], ascending=False).iloc[0]
