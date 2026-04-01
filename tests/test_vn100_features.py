"""Smoke tests for feature calculation module."""

from __future__ import annotations

import pandas as pd
from src.ml.features.technical import TechnicalFeatures
from src.ml.features.fundamental import FundamentalFeatures

def test_technical_signatures():
    """Test TechnicalFeatures static methods."""
    df = pd.DataFrame()
    # Should not crash on empty df
    result = TechnicalFeatures.add_all_indicators(df)
    assert result.empty

def test_fundamental_signatures():
    """Test FundamentalFeatures static methods."""
    df = pd.DataFrame()
    result = FundamentalFeatures.process_ratios(df)
    assert result.empty
