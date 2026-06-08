import pandas as pd
import pytest

from src.evaluation.baselines import historical_mean_return, rolling_volatility_range_band


def test_baselines_use_trailing_rows_only():
    returns = pd.Series([0.1, 0.2, 9.0])
    assert historical_mean_return(returns).iloc[2] == pytest.approx(0.15)
    band = rolling_volatility_range_band(pd.Series([10.0, 11.0, 12.0, 1000.0]), window=2)
    assert band.iloc[3].notna().all()
