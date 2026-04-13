"""Smoke tests for VnstockAdapter.
"""

from importlib import import_module
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.adapters.vnstock_adapter import VnstockAdapter

@pytest.fixture
def mock_vnstock():
    with patch("src.adapters.vnstock_adapter.Vnstock") as mock:
        yield mock

def test_vnstock_adapter_init(mock_vnstock):
    """Test that VnstockAdapter initializes and sets environment variables."""
    adapter = VnstockAdapter(symbol_list=["SSI", "HPG"])
    assert adapter.symbols == ["SSI", "HPG"]
    assert "VNAI_API_KEY" in os.environ
    assert "VNSTOCK_API_KEY" in os.environ

def test_legacy_import_path_alias():
    """Legacy adapter import path should resolve to the canonical module."""
    legacy_module = import_module("src.adapters.vnstock_adapter")
    assert legacy_module.VnstockAdapter is VnstockAdapter

def test_get_ohlcv_standardization(mock_vnstock):
    """Test that get_ohlcv renames 'time' to 'date' and normalizes it."""
    # Setup mock data
    mock_df = pd.DataFrame({
        "time": ["2024-01-01"],
        "open": [30.0],
        "high": [31.0],
        "low": [29.0],
        "close": [30.5],
        "volume": [1000000]
    })
    
    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = mock_df
    mock_vnstock.return_value.stock.return_value = mock_stock
    
    adapter = VnstockAdapter()
    df = adapter.get_ohlcv("SSI", "2024-01-01", "2024-01-01")
    
    assert "date" in df.columns
    assert "time" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

def test_get_ohlc_compatibility_wrapper(mock_vnstock):
    """Legacy get_ohlc should delegate to the canonical get_ohlcv path."""
    mock_df = pd.DataFrame({
        "time": ["2024-01-01"],
        "open": [30.0],
        "high": [31.0],
        "low": [29.0],
        "close": [30.5],
        "volume": [1000000],
    })

    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = mock_df
    mock_vnstock.return_value.stock.return_value = mock_stock

    adapter = VnstockAdapter()
    df = adapter.get_ohlc("SSI", "2024-01-01", "2024-01-01")

    assert not df.empty
    assert "date" in df.columns

def test_get_financial_ratios(mock_vnstock):
    """Test that get_financial_ratios calls the correct vnstock methods."""
    mock_df = pd.DataFrame({"ticker": ["SSI"], "pe": [15.0]})
    
    mock_stock = MagicMock()
    mock_stock.finance.ratio.return_value = mock_df
    mock_vnstock.return_value.stock.return_value = mock_stock
    
    adapter = VnstockAdapter()
    df = adapter.get_financial_ratios("SSI")
    
    assert not df.empty
    assert "pe" in df.columns

def test_get_valuation_metrics_compatibility(mock_vnstock):
    """Legacy valuation method should delegate to the financial ratios endpoint."""
    mock_df = pd.DataFrame({"ticker": ["SSI"], "pb": [1.8]})

    mock_stock = MagicMock()
    mock_stock.finance.ratio.return_value = mock_df
    mock_vnstock.return_value.stock.return_value = mock_stock

    adapter = VnstockAdapter()
    df = adapter.get_valuation_metrics("SSI")

    assert not df.empty
    assert "pb" in df.columns

def test_get_news_fallback(mock_vnstock):
    """Test news fetching with fallback logic."""
    mock_df = pd.DataFrame({"title": ["News 1"], "link": ["http://ex.com"]})
    
    # First attempt (stock.news()) fails
    mock_stock = MagicMock()
    mock_stock.news.side_effect = Exception("API error")
    
    # Fallback (stock.company.news()) succeeds
    mock_company = MagicMock()
    mock_company.news.return_value = mock_df
    mock_stock.company = mock_company
    
    mock_vnstock.return_value.stock.return_value = mock_stock
    
    adapter = VnstockAdapter()
    df = adapter.get_news("SSI", count=1)
    
    assert not df.empty
    assert df.iloc[0]["title"] == "News 1"

import os # Required for os.environ check in test_vnstock_adapter_init
