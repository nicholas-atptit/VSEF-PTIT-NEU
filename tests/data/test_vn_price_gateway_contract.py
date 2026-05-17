from __future__ import annotations

import pandas as pd
import pytest

from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName
from src.data.providers.vn_price_gateway import ProviderRequestError, fetch_price_history, normalize_ohlcv, validate_request


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": ["2026-01-02 09:00:00", "2026-01-02 10:00:00"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.5],
            "low": [9.5, 10.5],
            "close": [11.0, 12.0],
            "volume": [1000, 2000],
        }
    )


def test_vn30index_rejected_with_clear_message() -> None:
    request = FetchRequest(
        symbol="VN30INDEX",
        asset_type=AssetType.INDEX,
        start="2026-01-01",
        end="2026-01-02",
    )
    with pytest.raises(ProviderRequestError, match="Use VN30 instead of VN30INDEX"):
        validate_request(request)


def test_vnxall_rejected_as_unsupported() -> None:
    request = FetchRequest(symbol="VNXALL", asset_type=AssetType.INDEX, start="2026-01-01", end="2026-01-02")
    with pytest.raises(ProviderRequestError, match="Unsupported index code"):
        validate_request(request)


def test_hourly_request_rejects_daily_fallback_and_resampling() -> None:
    daily = FetchRequest(
        symbol="FPT",
        asset_type=AssetType.STOCK,
        start="2026-01-01",
        end="2026-01-02",
        frequency=Frequency.HOURLY,
        allow_daily=True,
    )
    resample = FetchRequest(
        symbol="FPT",
        asset_type=AssetType.STOCK,
        start="2026-01-01",
        end="2026-01-02",
        frequency=Frequency.HOURLY,
        allow_resample=True,
    )
    with pytest.raises(ProviderRequestError, match="daily fallback"):
        validate_request(daily)
    with pytest.raises(ProviderRequestError, match="daily-to-hourly resampling"):
        validate_request(resample)


def test_normalized_schema_for_stock() -> None:
    frame = normalize_ohlcv(
        _raw_frame(),
        symbol="fpt",
        asset_type=AssetType.STOCK,
        provider="repo_adapter",
        source=SourceName.KBS,
        frequency=Frequency.HOURLY,
    )
    assert list(frame.columns) == ["datetime", "ticker", "open", "high", "low", "close", "volume", "provider", "source", "frequency"]
    assert frame["ticker"].tolist() == ["fpt", "fpt"]
    assert frame["frequency"].unique().tolist() == ["1H"]


def test_normalized_schema_for_index() -> None:
    frame = normalize_ohlcv(
        _raw_frame(),
        symbol="VN30",
        asset_type=AssetType.INDEX,
        provider="repo_adapter",
        source=SourceName.KBS,
        frequency=Frequency.HOURLY,
    )
    assert list(frame.columns) == ["datetime", "index_code", "open", "high", "low", "close", "volume", "provider", "source", "frequency"]
    assert frame["index_code"].tolist() == ["VN30", "VN30"]


def test_invalid_ohlcv_rejected() -> None:
    raw = _raw_frame()
    raw.loc[0, "high"] = 8.0
    with pytest.raises(ValueError, match="high"):
        normalize_ohlcv(
            raw,
            symbol="FPT",
            asset_type=AssetType.STOCK,
            provider="repo_adapter",
            source=SourceName.KBS,
            frequency=Frequency.HOURLY,
        )


def test_gateway_import_and_request_validation_do_not_fetch_network() -> None:
    request = FetchRequest(symbol="VN30", asset_type=AssetType.INDEX, start="2026-01-01", end="2026-01-02")
    cleaned = validate_request(request)
    assert cleaned.symbol == "VN30"
    assert callable(fetch_price_history)
