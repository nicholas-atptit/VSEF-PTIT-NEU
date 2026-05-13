"""Canonical gateway for Vietnamese stock/index OHLCV price history."""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict
from typing import Any

import pandas as pd

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.data.providers.vn_provider_contract import (
    ALLOWED_INDEX_CODES,
    DEFAULT_HOURLY_INTERVAL,
    DISALLOWED_INDEX_ALIASES,
    UNSUPPORTED_INDEX_CODES,
    AssetType,
    FetchFailure,
    FetchRequest,
    FetchResponse,
    Frequency,
    ProviderName,
    SourceName,
)

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class ProviderRequestError(ValueError):
    """Raised when a provider request violates the canonical contract."""


class ProviderFetchError(RuntimeError):
    """Raised when all allowed provider attempts fail or return no rows."""

    def __init__(self, message: str, failures: list[FetchFailure], attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.failures = failures
        self.attempts = attempts


def _as_enum(value: Any, enum_type: type[AssetType] | type[Frequency] | type[SourceName]) -> Any:
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value))


def _clean_request(request: FetchRequest) -> FetchRequest:
    symbol = str(request.symbol).upper().strip()
    asset_type = _as_enum(request.asset_type, AssetType)
    frequency = _as_enum(request.frequency, Frequency)
    preferred_sources = tuple(_as_enum(source, SourceName) for source in request.preferred_sources)
    return FetchRequest(
        symbol=symbol,
        asset_type=asset_type,
        start=str(request.start),
        end=str(request.end),
        frequency=frequency,
        preferred_sources=preferred_sources,
        allow_legacy_fallback=bool(request.allow_legacy_fallback),
        allow_daily=bool(request.allow_daily),
        allow_resample=bool(request.allow_resample),
    )


def validate_request(request: FetchRequest) -> FetchRequest:
    cleaned = _clean_request(request)
    if cleaned.asset_type == AssetType.INDEX:
        if cleaned.symbol in DISALLOWED_INDEX_ALIASES:
            raise ProviderRequestError("Use VN30 instead of VN30INDEX for current provider")
        if cleaned.symbol in UNSUPPORTED_INDEX_CODES:
            raise ProviderRequestError(f"Unsupported index code for current provider: {cleaned.symbol}")
        if cleaned.symbol not in ALLOWED_INDEX_CODES:
            raise ProviderRequestError(f"Unsupported index code for current provider: {cleaned.symbol}")
    if cleaned.frequency == Frequency.HOURLY:
        if cleaned.frequency.value != DEFAULT_HOURLY_INTERVAL:
            raise ProviderRequestError('Hourly requests must use frequency="1H"')
        if cleaned.allow_daily:
            raise ProviderRequestError("Hourly requests must not allow daily fallback")
        if cleaned.allow_resample:
            raise ProviderRequestError("Hourly requests must not allow daily-to-hourly resampling")
    if not cleaned.preferred_sources:
        raise ProviderRequestError("At least one preferred source is required")
    return cleaned


def _record_attempt(
    attempts: list[dict[str, Any]],
    *,
    provider: ProviderName | str,
    source: SourceName | str,
    rows: int = 0,
    first_datetime: pd.Timestamp | None = None,
    last_datetime: pd.Timestamp | None = None,
    error: BaseException | None = None,
) -> None:
    payload = {
        "provider": str(provider),
        "source": str(source),
        "rows": int(rows),
        "first_datetime": "" if first_datetime is None else str(first_datetime),
        "last_datetime": "" if last_datetime is None else str(last_datetime),
        "error_type": "" if error is None else type(error).__name__,
        "error_message": "" if error is None else str(error),
    }
    attempts.append(payload)
    logger.info("vn_price_gateway_attempt", extra=payload)


def _time_column(frame: pd.DataFrame) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in ("datetime", "time", "timestamp", "date", "trading_date", "tradingdate"):
        if candidate in lower:
            return lower[candidate]
    if isinstance(frame.index, pd.DatetimeIndex):
        return "__index__"
    return None


def normalize_ohlcv(
    raw: pd.DataFrame,
    *,
    symbol: str,
    asset_type: AssetType,
    provider: ProviderName | str,
    source: SourceName | str,
    frequency: Frequency,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_normalized(asset_type)

    frame = raw.copy()
    time_column = _time_column(frame)
    if time_column is None:
        raise ValueError(f"{symbol}: no datetime column in provider response")
    if time_column == "__index__":
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "datetime"})
    elif time_column != "datetime":
        frame = frame.rename(columns={time_column: "datetime"})

    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    rename: dict[str, str] = {}
    for column in OHLCV_COLUMNS:
        existing = lower.get(column)
        if existing is not None and existing != column:
            rename[existing] = column
    if rename:
        frame = frame.rename(columns=rename)

    missing = [column for column in ("datetime", *OHLCV_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"{symbol}: missing OHLCV columns: {missing}")

    key = "index_code" if asset_type == AssetType.INDEX else "ticker"
    out = frame[["datetime", *OHLCV_COLUMNS]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    for column in OHLCV_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[key] = symbol
    out["provider"] = str(provider)
    out["source"] = str(source)
    out["frequency"] = frequency.value
    out = out.dropna(subset=["datetime", *OHLCV_COLUMNS])

    if out.empty:
        return _empty_normalized(asset_type)
    if out["datetime"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate datetime rows returned")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC price returned")
    if (out["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume returned")
    if (out["high"] < out["low"]).any():
        raise ValueError(f"{symbol}: high is below low")
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high is below open/close")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: low is above open/close")

    columns = ["datetime", key, *OHLCV_COLUMNS, "provider", "source", "frequency"]
    return out.sort_values("datetime")[columns].reset_index(drop=True)


def _empty_normalized(asset_type: AssetType) -> pd.DataFrame:
    key = "index_code" if asset_type == AssetType.INDEX else "ticker"
    return pd.DataFrame(columns=["datetime", key, *OHLCV_COLUMNS, "provider", "source", "frequency"])


def _response_from_frame(
    request: FetchRequest,
    frame: pd.DataFrame,
    *,
    provider: ProviderName | str,
    source: SourceName | str,
    attempts: list[dict[str, Any]],
) -> FetchResponse:
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
    first_datetime = None if timestamps.empty else pd.Timestamp(timestamps.min())
    last_datetime = None if timestamps.empty else pd.Timestamp(timestamps.max())
    return FetchResponse(
        symbol=request.symbol,
        asset_type=request.asset_type,
        frequency=request.frequency,
        provider=provider,
        source=source,
        rows=int(len(frame)),
        first_datetime=first_datetime,
        last_datetime=last_datetime,
        data=frame,
        attempts=tuple(attempts),
    )


def _fetch_with_adapter(request: FetchRequest, attempts: list[dict[str, Any]]) -> FetchResponse | None:
    source = request.preferred_sources[0]
    try:
        adapter = VnstockAdapter(symbol_list=[request.symbol])
        fetch = getattr(adapter, "get_index_ohlcv" if request.asset_type == AssetType.INDEX else "get_ohlcv")
        raw = fetch(request.symbol, request.start, request.end, request.frequency.value)
        frame = normalize_ohlcv(
            raw,
            symbol=request.symbol,
            asset_type=request.asset_type,
            provider=ProviderName.VNSTOCK_DATA,
            source=source,
            frequency=request.frequency,
        )
        first_dt = None if frame.empty else pd.Timestamp(frame["datetime"].min())
        last_dt = None if frame.empty else pd.Timestamp(frame["datetime"].max())
        _record_attempt(attempts, provider="repo_adapter", source=source, rows=len(frame), first_datetime=first_dt, last_datetime=last_dt)
        if not frame.empty:
            frame = frame.copy()
            frame["provider"] = "repo_adapter"
            return _response_from_frame(request, frame, provider="repo_adapter", source=source, attempts=attempts)
    except Exception as exc:
        _record_attempt(attempts, provider="repo_adapter", source=source, error=exc)
    return None


def _quote_history(package: str, symbol: str, source: SourceName, start: str, end: str, frequency: Frequency) -> pd.DataFrame:
    module = importlib.import_module(package)
    quote_cls = getattr(module, "Quote", None)
    if quote_cls is None:
        raise AttributeError(f"{package}.Quote is unavailable")
    quote = quote_cls(source=source.value, symbol=symbol)
    data = quote.history(start=start, end=end, interval=frequency.value)
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _fetch_direct(
    request: FetchRequest,
    attempts: list[dict[str, Any]],
    *,
    package: str,
    provider: ProviderName,
) -> FetchResponse | None:
    for source in request.preferred_sources:
        try:
            raw = _quote_history(package, request.symbol, source, request.start, request.end, request.frequency)
            frame = normalize_ohlcv(
                raw,
                symbol=request.symbol,
                asset_type=request.asset_type,
                provider=provider,
                source=source,
                frequency=request.frequency,
            )
            first_dt = None if frame.empty else pd.Timestamp(frame["datetime"].min())
            last_dt = None if frame.empty else pd.Timestamp(frame["datetime"].max())
            _record_attempt(attempts, provider=provider, source=source, rows=len(frame), first_datetime=first_dt, last_datetime=last_dt)
            if not frame.empty:
                return _response_from_frame(request, frame, provider=provider, source=source, attempts=attempts)
        except Exception as exc:
            _record_attempt(attempts, provider=provider, source=source, error=exc)
    return None


def fetch_price_history(request: FetchRequest) -> FetchResponse:
    cleaned = validate_request(request)
    attempts: list[dict[str, Any]] = []
    failures: list[FetchFailure] = []

    response = _fetch_with_adapter(cleaned, attempts)
    if response is not None:
        return response

    response = _fetch_direct(cleaned, attempts, package="vnstock_data", provider=ProviderName.VNSTOCK_DATA)
    if response is not None:
        return response

    if cleaned.allow_legacy_fallback:
        response = _fetch_direct(cleaned, attempts, package="vnstock", provider=ProviderName.LEGACY_VNSTOCK)
        if response is not None:
            return response

    for attempt in attempts:
        if attempt.get("error_type"):
            failures.append(
                FetchFailure(
                    symbol=cleaned.symbol,
                    provider=attempt["provider"],
                    source=attempt["source"],
                    error_type=attempt["error_type"],
                    error_message=attempt["error_message"],
                )
            )
    message = f"{cleaned.symbol}: no provider/source returned validated OHLCV rows"
    if failures:
        message = f"{message}; failures={[asdict(failure) for failure in failures]}"
    raise ProviderFetchError(message, failures, attempts)
