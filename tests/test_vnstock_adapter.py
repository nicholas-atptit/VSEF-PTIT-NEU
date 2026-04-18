"""Adapter tests for the canonical ``vnstock_data`` integration."""

from __future__ import annotations

import os
from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.adapters.vnstock_adapter import (
    DIRECT_VNSTOCK_PROVENANCE,
    STUB_PROVENANCE,
    VnstockAdapter,
)


@pytest.fixture
def fake_vnstock_module():
    class FakeQuote:
        def __init__(self, source: str, symbol: str) -> None:
            self.source = source
            self.symbol = symbol

        def history(self, start=None, end=None, interval: str = "1D", get_all: bool = True) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "time": ["2024-01-02", "2024-01-01", "2024-01-01"],
                    "open": [31.0, 30.0, 29.9],
                    "high": [32.0, 31.0, 31.0],
                    "low": [30.0, 29.0, 29.1],
                    "close": [31.5, 30.5, 30.4],
                    "volume": [1_100_000, 1_000_000, 1_000_001],
                }
            )

    class FakeListing:
        def __init__(self, source: str) -> None:
            self.source = source

        def all_symbols(self) -> pd.DataFrame:
            return pd.DataFrame({"symbol": ["SSI", "HPG", "FPT"], "organ_name": ["SSI Corp", "HPG Corp", "FPT Corp"]})

        def symbols_by_group(self, group: str) -> pd.Series:
            return pd.Series(["FPT", "HPG"], name="symbol")

        def symbols_by_industries(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "symbol": ["SSI", "HPG", "FPT"],
                    "industry_code": ["X1", "X2", "X3"],
                    "industry_name": ["Broker", "Steel", "Software"],
                }
            )

    class FakeCompany:
        def __init__(self, symbol: str, source: str) -> None:
            self.symbol = symbol
            self.source = source

        def overview(self) -> pd.DataFrame:
            return pd.DataFrame({"symbol": [self.symbol], "exchange": ["HOSE"]})

        def news(self, page: int = 1, page_size: int = 10) -> pd.DataFrame:
            return pd.DataFrame({"date": ["2024-01-02"], "description": ["Headline"], "link": ["http://x"], "name": ["Story"]})

    class FakeFinance:
        def __init__(self, source: str, symbol: str) -> None:
            self.source = source
            self.symbol = symbol

        def ratio(self, period: str = "quarter", get_all: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"report_period": [period], "pe": [15.0], "pb": [1.8]})

    class FakeTrading:
        def __init__(self, source: str, symbol: str) -> None:
            self.source = source
            self.symbol = symbol

        def foreign_trade(self, **kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "trading_date": ["2024-01-02"],
                    "fr_buy_volume_total": [1000],
                    "fr_sell_volume_total": [800],
                    "fr_buy_value_total": [2000000],
                    "fr_sell_value_total": [1500000],
                    "fr_net_volume_total": [200],
                    "fr_net_value_total": [500000],
                }
            )

    return SimpleNamespace(
        Quote=FakeQuote,
        Listing=FakeListing,
        Company=FakeCompany,
        Finance=FakeFinance,
        Trading=FakeTrading,
    )


def test_vnstock_adapter_init_sets_environment():
    adapter = VnstockAdapter(symbol_list=["SSI", "HPG"])
    assert adapter.symbols == ["SSI", "HPG"]
    assert "VNAI_API_KEY" in os.environ
    assert "VNSTOCK_API_KEY" in os.environ


def test_legacy_import_path_alias():
    legacy_module = import_module("src.adapters.vnstock_adapter")
    assert legacy_module.VnstockAdapter is VnstockAdapter


def test_get_ohlcv_standardization(monkeypatch, fake_vnstock_module):
    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        lambda name: fake_vnstock_module,
    )

    adapter = VnstockAdapter()
    df = adapter.get_ohlcv("SSI", "2024-01-01", "2024-01-02")

    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert df["ticker"].nunique() == 1
    assert df["ticker"].iloc[0] == "SSI"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.attrs["source_provenance"] == DIRECT_VNSTOCK_PROVENANCE
    assert df["date"].tolist() == sorted(df["date"].tolist())


def test_get_financial_ratios_uses_documented_finance_endpoint(monkeypatch, fake_vnstock_module):
    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        lambda name: fake_vnstock_module,
    )

    adapter = VnstockAdapter()
    df = adapter.get_financial_ratios("SSI")

    assert not df.empty
    assert "pe" in df.columns
    assert df.attrs["source_provenance"] == DIRECT_VNSTOCK_PROVENANCE


def test_get_news_uses_documented_company_endpoint(monkeypatch, fake_vnstock_module):
    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        lambda name: fake_vnstock_module,
    )

    adapter = VnstockAdapter()
    df = adapter.get_news("SSI", count=1)

    assert not df.empty
    assert df.iloc[0]["name"] == "Story"
    assert df.attrs["source_provenance"] == DIRECT_VNSTOCK_PROVENANCE


def test_audit_company_news_capability_reports_live_support(monkeypatch, fake_vnstock_module):
    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        lambda name: fake_vnstock_module,
    )

    adapter = VnstockAdapter()
    audit = adapter.audit_company_news_capability("SSI", run_live_probe=True)

    assert audit["provider_runtime_available"] is True
    assert audit["company_class_available"] is True
    assert audit["company_news_method_available"] is True
    assert audit["live_probe_attempted"] is True
    assert audit["live_probe_success"] is True
    assert audit["status"] == "live_supported"


def test_get_foreign_flow_standardizes_canonical_columns(monkeypatch, fake_vnstock_module):
    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        lambda name: fake_vnstock_module,
    )

    adapter = VnstockAdapter()
    df = adapter.get_foreign_flow("SSI", "2024-01-01", "2024-01-31")

    assert not df.empty
    assert {"foreign_buy_volume", "foreign_sell_volume", "foreign_net_volume", "foreign_net_value"} <= set(df.columns)
    assert df.attrs["source_provenance"] == DIRECT_VNSTOCK_PROVENANCE


def test_missing_vnstock_provider_returns_explicit_stub(monkeypatch):
    def _raise_import_error(name: str):
        raise ImportError("vnstock_data not installed")

    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        _raise_import_error,
    )

    adapter = VnstockAdapter()
    df = adapter.get_financial_ratios("SSI")

    assert df.empty
    assert df.attrs["source_provenance"] == STUB_PROVENANCE


def test_audit_company_news_capability_reports_unsupported_when_provider_missing(monkeypatch):
    def _raise_import_error(name: str):
        raise ImportError("vnstock_data not installed")

    monkeypatch.setattr(
        "src.data.adapters.vnstock_adapter.importlib.import_module",
        _raise_import_error,
    )

    adapter = VnstockAdapter()
    audit = adapter.audit_company_news_capability("SSI", run_live_probe=True)

    assert audit["provider_runtime_available"] is False
    assert audit["company_class_available"] is False
    assert audit["status"] == "unsupported"
