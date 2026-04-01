"""Smoke tests for vnstock adapters."""

from __future__ import annotations

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
