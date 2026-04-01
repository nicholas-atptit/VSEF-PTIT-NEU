"""Volatility label generators — future realised volatility targets.

Generators:
    - ``FutureRealizedVol5d`` — annualised realised volatility over the
      *next* 5 trading days

Time-safety:
    The rolling window for realised volatility is applied to **future**
    log returns using ``shift(-horizon)``.  The last ``horizon`` rows will
    be ``NaN``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.ml.labels.base import BaseLabelGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Annualisation factor for daily returns (≈ 252 trading days/year)
_ANNUALISE = np.sqrt(252)


class FutureRealizedVol5d(BaseLabelGenerator):
    """Annualised future realised volatility over the next 5 trading days.

    Formula::

        log_ret[t] = ln(close[t] / close[t-1])
        target[t]  = std(log_ret[t+1 .. t+5]) × √252

    The forward-looking window is constructed by first computing daily
    log-returns, then shifting the rolling standard deviation backward so
    that each row ``t`` contains the volatility of the *subsequent* 5 days.

    Args:
        horizon: Number of forward trading days (default 5).
        annualise: Whether to multiply by √252 (default ``True``).
        col_name: Output column name.
    """

    def __init__(
        self,
        horizon: int = 5,
        annualise: bool = True,
        col_name: str = "target_future_realized_vol_5d",
    ) -> None:
        self.horizon = horizon
        self.annualise = annualise
        self._col = col_name

    @property
    def name(self) -> str:
        return "future_realized_vol_5d"

    @property
    def label_columns(self) -> List[str]:
        return [self._col]

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))

        # Rolling std over *future* window:
        # 1. Compute rolling std on the original series.
        # 2. Shift left so that row t gets the std of rows [t+1 .. t+horizon].
        rolling_vol = log_ret.rolling(self.horizon).std().shift(-self.horizon)

        if self.annualise:
            rolling_vol = rolling_vol * _ANNUALISE

        df[self._col] = rolling_vol
        return df
