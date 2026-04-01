"""Regression label generators — continuous return targets.

Generators:
    - ``RegNextCloseReturn`` — 1‑day forward close-to-close return
    - ``Reg5dReturn``        — 5‑day forward close-to-close return

All targets are fully time-safe: they use ``shift(-horizon)`` to access
future prices.  The last ``horizon`` rows will be ``NaN``.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.ml.labels.base import BaseLabelGenerator, _get_target_close
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegNextCloseReturn(BaseLabelGenerator):
    """1‑day forward close-to-close log return.

    Formula::

        target = close[t+1] / close[t] − 1

    Args:
        col_name: Output column name.  Default: ``target_reg_next_close_return``.
    """

    def __init__(self, col_name: str = "target_reg_next_close_return") -> None:
        self._col = col_name

    @property
    def name(self) -> str:
        return "reg_next_close_return"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _get_target_close(df)
        df[self._col] = close.shift(-1) / close - 1
        return df


class Reg5dReturn(BaseLabelGenerator):
    """5‑day forward close-to-close return.

    Formula::

        target = close[t+5] / close[t] − 1

    Args:
        col_name: Output column name.  Default: ``target_reg_5d_return``.
    """

    HORIZON = 5

    def __init__(self, col_name: str = "target_reg_5d_return") -> None:
        self._col = col_name

    @property
    def name(self) -> str:
        return "reg_5d_return"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _get_target_close(df)
        df[self._col] = close.shift(-self.HORIZON) / close - 1
        return df
