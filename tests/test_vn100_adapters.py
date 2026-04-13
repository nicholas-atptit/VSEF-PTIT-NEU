"""Smoke tests for vnstock adapters."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from src.data.adapters.vnstock_adapter import VnstockAdapter

def test_adapter_init():
    """Test initialization of VnstockAdapter."""
    adapter = VnstockAdapter(symbol_list=["TCB", "SSI"])
    assert len(adapter.symbols) == 2
    assert "TCB" in adapter.symbols

def test_adapter_method_signatures():
    """Test that all required methods exist."""
    adapter = VnstockAdapter()
    assert hasattr(adapter, 'get_ohlc')
    assert hasattr(adapter, 'get_financial_ratios')
    assert hasattr(adapter, 'get_valuation_metrics')

def test_legacy_method_wrappers_delegate():
    """Legacy wrapper methods should call the canonical adapter methods."""
    adapter = VnstockAdapter()

    with patch.object(adapter, "get_ohlcv", return_value="ohlcv") as get_ohlcv:
        assert adapter.get_ohlc("SSI", "2024-01-01", "2024-01-31") == "ohlcv"
        get_ohlcv.assert_called_once_with("SSI", "2024-01-01", "2024-01-31", "1D")

    with patch.object(adapter, "get_financial_ratios", return_value="ratios") as get_ratios:
        assert adapter.get_valuation_metrics("SSI") == "ratios"
        get_ratios.assert_called_once_with("SSI")
