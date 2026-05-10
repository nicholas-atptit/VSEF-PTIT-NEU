import datetime as dt
import pytest
from src.data import universe
from src.data.universe import (
    VN100_BACKUP_ACTIVATION,
    VN100_BACKUP_AS_OF,
    VN100_BACKUP_SOURCE,
    VN100_BACKUP_TICKERS,
    VN100_MIN_EXPECTED_COUNT,
    VIETTEL_TICKERS,
    get_vn100_universe,
)

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

def test_undersized_live_universe_uses_documented_static_fallback(monkeypatch):
    """Undersized live provider output must not be silently accepted as VN100."""
    class UndersizedAdapter:
        def get_vn100_tickers(self):
            return ["HPG", "SSI"]

    monkeypatch.setattr(universe, "VnstockAdapter", lambda: UndersizedAdapter())
    tickers = get_vn100_universe()

    assert tickers == sorted(set(VN100_BACKUP_TICKERS))
    assert len(tickers) >= VN100_MIN_EXPECTED_COUNT

def test_static_fallback_provenance_is_documented():
    """Fallback source, count, as-of assumptions, and activation conditions are explicit."""
    assert VN100_BACKUP_SOURCE == "src.data.universe.VN100_BACKUP_TICKERS"
    assert len(VN100_BACKUP_TICKERS) >= VN100_MIN_EXPECTED_COUNT
    assert "historical constituents unavailable" in VN100_BACKUP_AS_OF
    assert "live universe count below 100" in VN100_BACKUP_ACTIVATION
