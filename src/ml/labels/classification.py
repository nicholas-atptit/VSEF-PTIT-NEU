"""Classification label generators — binary and ternary.

Binary labels (up/down):
    - ``Cls1dUpDown``  — 1‑day forward return > 0 -> 1, else 0
    - ``Cls5dUpDown``  — 5‑day forward return > 0 -> 1, else 0
    - ``Cls20dUpDown`` — 20‑day forward return > 0 -> 1, else 0

Ternary labels (up/sideways/down):
    - ``Cls1d3Class``  — uses configurable threshold (default ±1 %)
    - ``Cls5d3Class``  — uses configurable threshold (default ±2 %)

All labels are fully time-safe: they use ``shift(-horizon)`` to access
future prices.  The last ``horizon`` rows will be ``NaN``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.ml.labels.base import BaseLabelGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BINARY CLASSIFICATION (Up / Down)
# ═══════════════════════════════════════════════════════════════════════════


class Cls1dUpDown(BaseLabelGenerator):
    """1‑day binary classification: close goes up (1) or not (0).

    Label formula::

        future_return = close[t+1] / close[t] − 1
        label = 1 if future_return > 0  else 0

    Args:
        col_name: Output column name.  Default: ``label_cls_1d_updown``.
    """

    def __init__(self, col_name: str = "label_cls_1d_updown") -> None:
        self._col = col_name

    @property
    def name(self) -> str:
        return "cls_1d_updown"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        future_ret = df["close"].shift(-1) / df["close"] - 1
        df[self._col] = (future_ret > 0).astype("Int64")
        # NaN for rows where future is unknown
        df.loc[future_ret.isna(), self._col] = pd.NA
        return df


class Cls5dUpDown(BaseLabelGenerator):
    """5‑day binary classification: forward 5‑day return > 0 -> 1, else 0."""

    HORIZON = 5

    def __init__(self, col_name: str = "label_cls_5d_updown") -> None:
        self._col = col_name

    @property
    def name(self) -> str:
        return "cls_5d_updown"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        future_ret = df["close"].shift(-self.HORIZON) / df["close"] - 1
        df[self._col] = (future_ret > 0).astype("Int64")
        df.loc[future_ret.isna(), self._col] = pd.NA
        return df


class Cls20dUpDown(BaseLabelGenerator):
    """20‑day binary classification: forward 20‑day return > 0 -> 1, else 0."""

    HORIZON = 20

    def __init__(self, col_name: str = "label_cls_20d_updown") -> None:
        self._col = col_name

    @property
    def name(self) -> str:
        return "cls_20d_updown"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        future_ret = df["close"].shift(-self.HORIZON) / df["close"] - 1
        df[self._col] = (future_ret > 0).astype("Int64")
        df.loc[future_ret.isna(), self._col] = pd.NA
        return df


# ═══════════════════════════════════════════════════════════════════════════
# TERNARY CLASSIFICATION (Up / Sideways / Down)
# ═══════════════════════════════════════════════════════════════════════════


class Cls1d3Class(BaseLabelGenerator):
    """1‑day ternary classification: Up (0), Sideways (1), Down (2).

    Uses a symmetric threshold: return > +thresh -> Up,
    return < −thresh -> Down, otherwise Sideways.

    Args:
        threshold: The symmetric threshold (default 0.01 = 1 %).
            Can be overridden via ``Settings.label_cls_1d_threshold``.
        col_name: Output column name.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        col_name: str = "label_cls_1d_3class",
    ) -> None:
        self.threshold = threshold
        self._col = col_name

    @property
    def name(self) -> str:
        return "cls_1d_3class"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        future_ret = df["close"].shift(-1) / df["close"] - 1

        labels = pd.array(
            np.select(
                [future_ret > self.threshold, future_ret < -self.threshold],
                [0, 2],
                default=1,
            ),
            dtype="Int64",
        )
        df[self._col] = labels
        df.loc[future_ret.isna(), self._col] = pd.NA
        return df


class Cls5d3Class(BaseLabelGenerator):
    """5‑day ternary classification: Up (0), Sideways (1), Down (2).

    Default threshold is 2 % (wider than 1‑day to account for longer
    horizon noise).

    Args:
        threshold: Symmetric threshold (default 0.02 = 2 %).
            Can be overridden via ``Settings.label_cls_5d_threshold``.
        col_name: Output column name.
    """

    HORIZON = 5

    def __init__(
        self,
        threshold: float = 0.02,
        col_name: str = "label_cls_5d_3class",
    ) -> None:
        self.threshold = threshold
        self._col = col_name

    @property
    def name(self) -> str:
        return "cls_5d_3class"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        future_ret = df["close"].shift(-self.HORIZON) / df["close"] - 1

        labels = pd.array(
            np.select(
                [future_ret > self.threshold, future_ret < -self.threshold],
                [0, 2],
                default=1,
            ),
            dtype="Int64",
        )
        df[self._col] = labels
        df.loc[future_ret.isna(), self._col] = pd.NA
        return df
