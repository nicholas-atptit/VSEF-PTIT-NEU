import datetime as dt
import pytest
from src.data.universe import get_vn100_universe, VIETTEL_TICKERS

def test_get_vn100_universe_default():
    """Test standard VN100 retrieval."""
    tickers = get_vn100_universe()
    assert isinstance(tickers, list)
    assert len(tickers) >= 100
    assert "HPG" in tickers
    assert "SSI" in tickers

def test_get_vn100_universe_plus_viettel():
    """Test extended universe mode."""
    tickers = get_vn100_universe(mode="current_plus_viettel")
    assert isinstance(tickers, list)
    # Plus Viettel should include FOX if it wasn't already there
    for t in VIETTEL_TICKERS:
        assert t in tickers
    
    # Should be sorted
    assert tickers == sorted(tickers)

def test_get_vn100_universe_historical_fallback():
    """Test that historical requests currently fallback to current."""
    past_date = dt.date(2020, 1, 1)
    tickers = get_vn100_universe(as_of_date=past_date)
    assert isinstance(tickers, list)
    assert len(tickers) >= 100

def test_get_vn100_universe_no_duplicates():
    """Ensure no duplicates return."""
    tickers = get_vn100_universe()
    assert len(tickers) == len(set(tickers))
