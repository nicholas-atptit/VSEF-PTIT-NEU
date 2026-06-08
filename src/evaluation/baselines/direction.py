from __future__ import annotations

import pandas as pd


def always_up(index: pd.Index) -> pd.Series:
    return pd.Series(1, index=index, dtype=int)


def always_down(index: pd.Index) -> pd.Series:
    return pd.Series(0, index=index, dtype=int)


def majority_class(train_target: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(int(pd.to_numeric(train_target, errors="coerce").mean() >= 0.5), index=index, dtype=int)


def lag1_direction(returns: pd.Series) -> pd.Series:
    return (pd.to_numeric(returns, errors="coerce").shift(1) > 0).astype("Int64")
