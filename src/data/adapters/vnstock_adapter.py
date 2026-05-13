"""VNStock adapter layer.

This module centralizes ``vnstock_data`` access and keeps the project honest
about what is direct provider data, what is derived locally, and what is only
an explicit stub when the provider or environment cannot supply it.
"""

from __future__ import annotations

import importlib
import os
import time
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Performance Instrumentation
# ═══════════════════════════════════════════════════════════════════════════

def _time_fetch(method_name: str) -> Callable:
    """Decorator to measure fetch method latency."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ns = time.perf_counter_ns() - start_ns
                elapsed_ms = elapsed_ns / 1e6
                logger.debug(
                    "vnstock_fetch_completed",
                    method=method_name,
                    elapsed_ms=f"{elapsed_ms:.2f}",
                )
        return wrapper
    return decorator

MAX_FETCH_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 0.75
QUOTE_SOURCE_PRIORITY = ("KBS", "VCI")
DIRECT_VNSTOCK_PROVENANCE = "direct_vnstock_data"
STUB_PROVENANCE = "stub_todo"
_VNSTOCK_EXPORTS = (
    "Quote",
    "QuoteHistory",
    "Listing",
    "Company",
    "Finance",
    "Trading",
    "CommodityPrice",
    "Fund",
    "TopStock",
)

_VNSTOCK_LAZY_EXPORTS = {
    "Market": ("vnstock_data.api.market", "Market"),
    "Macro": ("vnstock_data.api.macro", "Macro"),
    "TopStock": ("vnstock_data.api.insight", "TopStock"),
}
_VNSTOCK_SPONSOR_WARNING_PATTERN = r"\s*\*+\s*\[vnstock\].*Sponsor `vnstock_data`.*"


@dataclass(frozen=True)
class ProviderCallAttempt:
    func: Callable[[], Any]
    call_type: str
    symbol: str | None = None
    frequency: str | None = None
    source: str | None = None


class _NoopProviderRateLimiter:
    def wait(self) -> float:
        return 0.0


def _load_vnstock_namespace() -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_VNSTOCK_SPONSOR_WARNING_PATTERN,
                category=UserWarning,
            )
            module = importlib.import_module("vnstock_data")
    except (ImportError, SystemExit) as exc:
        logger.warning("vnstock_data_namespace_unavailable", error=str(exc))
        return {}
    namespace = {name: getattr(module, name, None) for name in _VNSTOCK_EXPORTS}
    for export_name, (module_name, attr_name) in _VNSTOCK_LAZY_EXPORTS.items():
        if namespace.get(export_name) is not None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_VNSTOCK_SPONSOR_WARNING_PATTERN,
                    category=UserWarning,
                )
                lazy_module = importlib.import_module(module_name)
            namespace[export_name] = getattr(lazy_module, attr_name, None)
        except Exception:
            namespace.setdefault(export_name, None)
    return namespace


class VnstockAdapter:
    """Thin adapter for ``vnstock_data`` integration.

    The adapter keeps imports lazy so the rest of the project remains importable
    in environments where ``vnstock_data`` is not installed.
    """

    def __init__(self, symbol_list: Optional[list[str]] = None, rate_limiter: Any | None = None) -> None:
        self.settings = get_settings()
        self.symbols = symbol_list or []
        self._rate_limiter = rate_limiter if rate_limiter is not None else _NoopProviderRateLimiter()
        self._vnstock_namespace: dict[str, Any] | None = None
        self._setup_env()
        logger.info(
            "vnstock_adapter_initialized",
            provider="vnstock_data",
            symbols_count=len(self.symbols),
        )

    def _setup_env(self) -> None:
        api_key = self.settings.vnstock_api_key or ""
        os.environ["VNAI_API_KEY"] = api_key
        os.environ["VNSTOCK_API_KEY"] = api_key
        if api_key:
            logger.debug("vnstock_api_keys_configured")

    def _namespace(self) -> dict[str, Any]:
        if self._vnstock_namespace is None:
            self._vnstock_namespace = self._provider_call(
                _load_vnstock_namespace,
                call_type="vnstock_data.import",
                source="vnstock_data",
            )
        return self._vnstock_namespace

    def _get_class(self, name: str) -> Any | None:
        return self._namespace().get(name)

    @staticmethod
    def _attach_attrs(
        df: pd.DataFrame,
        *,
        provenance: str,
        source_name: str,
        adjustment_status: str | None = None,
        availability: str = "available",
        notes: str | None = None,
    ) -> pd.DataFrame:
        df.attrs["source_provenance"] = provenance
        df.attrs["source_name"] = source_name
        df.attrs["source_availability"] = availability
        if adjustment_status is not None:
            df.attrs["adjustment_status"] = adjustment_status
        if notes:
            df.attrs["source_notes"] = notes
        return df

    def _empty_frame(
        self,
        reason: str,
        *,
        source_name: str,
        notes: str | None = None,
    ) -> pd.DataFrame:
        logger.warning("vnstock_dataset_unavailable", source=source_name, reason=reason)
        frame = pd.DataFrame()
        return self._attach_attrs(
            frame,
            provenance=STUB_PROVENANCE,
            source_name=source_name,
            availability="unavailable",
            notes=notes or reason,
        )

    def _is_available(self) -> bool:
        return bool(self._namespace())

    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for candidate in ("date", "time", "timestamp", "trading_date", "report_time", "reportDate", "last_updated"):
            if candidate in normalized.columns:
                normalized = normalized.rename(columns={candidate: "date"})
                normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
                return normalized
        if normalized.index.name in {"date", "time", "timestamp", "report_time", "reportDate"} or isinstance(
            normalized.index, pd.DatetimeIndex
        ):
            normalized = normalized.reset_index()
            return VnstockAdapter._normalize_dates(normalized)
        return normalized

    @staticmethod
    def _filter_date_range(
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        filtered = df.copy()
        if "date" not in filtered.columns:
            return filtered
        if start_date is not None:
            start_ts = pd.Timestamp(start_date).normalize()
            filtered = filtered[filtered["date"] >= start_ts]
        if end_date is not None:
            end_ts = pd.Timestamp(end_date).normalize()
            filtered = filtered[filtered["date"] <= end_ts]
        return filtered

    @staticmethod
    def _filter_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        filtered = df.copy()
        ticker = symbol.upper().strip()
        for candidate in ("ticker", "symbol", "code"):
            if candidate in filtered.columns:
                mask = filtered[candidate].astype(str).str.upper() == ticker
                filtered = filtered[mask]
                break
        return filtered

    @staticmethod
    def _as_dataframe(result: Any) -> pd.DataFrame:
        if isinstance(result, pd.DataFrame):
            return result.copy()
        if isinstance(result, pd.Series):
            return result.to_frame(name=result.name or "value").reset_index(drop=True)
        if isinstance(result, (list, tuple)) and result:
            return pd.DataFrame(result)
        return pd.DataFrame()

    @staticmethod
    def _result_row_count(result: Any) -> int | None:
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return int(len(result))
        if isinstance(result, (list, tuple)):
            return int(len(result))
        return None

    def _wait_for_provider(self) -> float:
        wait = getattr(self._rate_limiter, "wait", None)
        if not callable(wait):
            return 0.0
        return float(wait() or 0.0)

    def _provider_call(
        self,
        func: Callable[[], Any],
        *,
        call_type: str,
        symbol: str | None = None,
        frequency: str | None = None,
        source: str | None = None,
    ) -> Any:
        throttled_seconds = self._wait_for_provider()
        log_context: dict[str, Any] = {
            "call_type": call_type,
            "throttled_seconds": float(throttled_seconds),
        }
        if symbol is not None:
            log_context["symbol"] = symbol
        if frequency is not None:
            log_context["frequency"] = frequency
        if source is not None:
            log_context["source"] = source

        logger.info("provider_call_started", **log_context)
        start_ns = time.perf_counter_ns()
        try:
            result = func()
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            rows = self._result_row_count(result)
            completed_context = {**log_context, "elapsed_ms": f"{elapsed_ms:.2f}"}
            if rows is not None:
                completed_context["rows"] = rows
            logger.info("provider_call_completed", **completed_context)
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            logger.warning(
                "provider_call_failed",
                **log_context,
                elapsed_ms=f"{elapsed_ms:.2f}",
                error=str(exc),
            )
            raise

    def _standardize_history_frame(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        source_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        standardized = self._normalize_dates(frame)
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(standardized.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns from {source_name}: {sorted(missing)}")
        standardized = self._filter_date_range(standardized, start_date, end_date)
        standardized["ticker"] = symbol.upper().strip()
        for column in ("open", "high", "low", "close", "volume"):
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
        standardized = standardized.dropna(subset=["date", "open", "high", "low", "close"])
        standardized["volume"] = standardized["volume"].fillna(0.0)
        standardized = (
            standardized.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        return self._attach_attrs(
            standardized[["date", "ticker", "open", "high", "low", "close", "volume"]],
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name=source_name,
            adjustment_status="not_available",
            notes="Quote history returned raw OHLCV only; the live provider did not expose adjusted-price columns.",
        )

    def _standardize_time_series_frame(
        self,
        frame: pd.DataFrame,
        *,
        source_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        standardized = self._normalize_dates(frame)
        standardized = self._filter_date_range(standardized, start_date, end_date)
        if symbol:
            standardized = self._filter_symbol(standardized, symbol)
        if "date" in standardized.columns:
            standardized = standardized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        standardized = standardized.reset_index(drop=True)
        return self._attach_attrs(
            standardized,
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name=source_name,
        )

    def _call_first_success(
        self,
        attempts: Iterable[Callable[[], Any] | ProviderCallAttempt],
        *,
        source_name: str,
        call_type: str | None = None,
        symbol: str | None = None,
        frequency: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in attempts:
            if isinstance(attempt, ProviderCallAttempt):
                attempt_func = attempt.func
                attempt_call_type = attempt.call_type
                attempt_symbol = attempt.symbol
                attempt_frequency = attempt.frequency
                attempt_source = attempt.source
            else:
                attempt_func = attempt
                attempt_call_type = call_type or source_name
                attempt_symbol = symbol
                attempt_frequency = frequency
                attempt_source = source
            try:
                result = self._provider_call(
                    attempt_func,
                    call_type=attempt_call_type,
                    symbol=attempt_symbol,
                    frequency=attempt_frequency,
                    source=attempt_source,
                )
                frame = VnstockAdapter._as_dataframe(result)
                if frame is not None and not frame.empty:
                    return frame
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            logger.debug("vnstock_call_failed", source=source_name, error=str(last_error))
        return pd.DataFrame()

    @_time_fetch("get_ohlcv")
    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        ticker = symbol.upper().strip()
        if not self._is_available():
            return self._empty_frame(
                "vnstock_data_not_installed",
                source_name="Quote.history",
            )

        last_error: Exception | None = None
        for attempt_idx in range(1, MAX_FETCH_RETRIES + 1):
            for source in QUOTE_SOURCE_PRIORITY:
                try:
                    frame = self._fetch_quote_history(
                        symbol=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        source=source,
                    )
                    if frame.empty:
                        raise ValueError(f"Empty OHLCV response from vnstock_data source={source}")
                    return self._standardize_history_frame(
                        frame,
                        symbol=ticker,
                        source_name=f"Quote[{source}].history",
                        start_date=start_date,
                        end_date=end_date,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "ohlcv_fetch_source_failed",
                        symbol=ticker,
                        source=source,
                        attempt=attempt_idx,
                        max_attempts=MAX_FETCH_RETRIES,
                        error=str(exc),
                    )
            if attempt_idx < MAX_FETCH_RETRIES:
                time.sleep(BASE_RETRY_DELAY_SECONDS * attempt_idx)

        return self._empty_frame(
            str(last_error or "ohlcv_fetch_failed"),
            source_name="Quote.history",
        )

    def _fetch_quote_history(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        source: str,
    ) -> pd.DataFrame:
        Quote = self._get_class("Quote")
        QuoteHistory = self._get_class("QuoteHistory")
        attempts: list[Callable[[], Any] | ProviderCallAttempt] = []
        if Quote is not None:
            attempts.append(
                ProviderCallAttempt(
                    lambda: Quote(source=source, symbol=symbol).history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        get_all=True,
                    ),
                    call_type="Quote.history",
                    symbol=symbol,
                    frequency=interval,
                    source=source,
                )
            )
        if QuoteHistory is not None:
            attempts.append(
                ProviderCallAttempt(
                    lambda: QuoteHistory(source=source, symbol=symbol).history(
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=interval,
                    ),
                    call_type="QuoteHistory.history",
                    symbol=symbol,
                    frequency=interval,
                    source=source,
                )
            )
        return self._call_first_success(attempts, source_name=f"Quote[{source}].history")

    def get_ohlc(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        return self.get_ohlcv(symbol, start_date, end_date, interval)

    def get_index_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        return self.get_ohlcv(symbol, start_date, end_date, interval)

    def get_company_overview(self, symbol: str) -> pd.DataFrame:
        Company = self._get_class("Company")
        if Company is None:
            return self._empty_frame("Company class unavailable", source_name="Company.overview")
        ticker = symbol.upper().strip()
        frame = self._call_first_success(
            [
                ProviderCallAttempt(
                    lambda: Company(symbol=ticker, source="KBS").overview(),
                    call_type="Company.overview",
                    symbol=ticker,
                    source="KBS",
                ),
                ProviderCallAttempt(
                    lambda: Company(symbol=ticker, source="VCI").overview(),
                    call_type="Company.overview",
                    symbol=ticker,
                    source="VCI",
                ),
            ],
            source_name="Company.overview",
        )
        if frame.empty:
            return self._empty_frame(
                f"No overview returned for {ticker}",
                source_name="Company.overview",
                notes=(
                    "The installed vnstock_data package exposes Company.overview(), "
                    "but live calls failed at runtime in this environment."
                ),
            )
        return self._standardize_time_series_frame(
            frame,
            source_name="Company.overview",
        )

    def get_trading_stats(self, symbol: str) -> pd.DataFrame:
        return self._empty_frame(
            f"No stable trading-stats endpoint is mapped for {symbol.upper().strip()} in the installed vnstock_data runtime",
            source_name="Trading.trading_stats",
            notes=(
                "Live provider coverage exposed foreign_trade, trade_history, price_history, "
                "and prop_trade; a stable per-ticker trading_stats table was not verified."
            ),
        )

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        ticker = symbol.upper().strip()
        attempts: list[Callable[[], Any] | ProviderCallAttempt] = []
        Finance = self._get_class("Finance")
        if Finance is not None:
            attempts.append(
                ProviderCallAttempt(
                    lambda: Finance(source="VCI", symbol=ticker).ratio(period="quarter", get_all=False),
                    call_type="Finance.ratio",
                    symbol=ticker,
                    frequency="quarter",
                    source="VCI",
                )
            )
            attempts.append(
                ProviderCallAttempt(
                    lambda: Finance(source="KBS", symbol=ticker).ratio(period="quarter", get_all=False),
                    call_type="Finance.ratio",
                    symbol=ticker,
                    frequency="quarter",
                    source="KBS",
                )
            )

        if not attempts:
            return self._empty_frame(
                "No documented financial-ratio interface is available in this environment",
                source_name="Finance.ratio",
            )

        frame = self._call_first_success(
            attempts,
            source_name="Finance.ratio",
        )
        if frame.empty:
            return self._empty_frame(
                f"No financial ratios returned for {ticker}",
                source_name="Finance.ratio",
            )

        standardized = frame.reset_index(drop=True)
        standardized["ticker"] = ticker
        return self._attach_attrs(
            standardized,
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name="Finance.ratio",
        )

    def get_valuation_metrics(self, symbol: str) -> pd.DataFrame:
        return self.get_financial_ratios(symbol)

    def get_news(self, ticker: str, count: int = 10) -> pd.DataFrame:
        symbol = ticker.upper().strip()
        attempts: list[Callable[[], Any] | ProviderCallAttempt] = []
        Company = self._get_class("Company")

        if Company is not None:
            attempts.append(
                ProviderCallAttempt(
                    lambda: Company(symbol=symbol, source="KBS").news(page=1, page_size=max(int(count), 1)),
                    call_type="Company.news",
                    symbol=symbol,
                    source="KBS",
                )
            )
            attempts.append(
                ProviderCallAttempt(
                    lambda: Company(symbol=symbol, source="VCI").news(page=1, page_size=max(int(count), 1)),
                    call_type="Company.news",
                    symbol=symbol,
                    source="VCI",
                )
            )

        if not attempts:
            return self._empty_frame(
                "No documented company-news interface is available in this environment",
                source_name="Company.news",
            )

        frame = self._call_first_success(
            attempts,
            source_name="Company.news",
        )
        if frame.empty:
            return self._empty_frame(
                f"No news returned for {symbol}",
                source_name="Company.news",
                notes=(
                    "Company.news exists in the installed vnstock_data package, "
                    "but live calls failed at runtime in this environment."
                ),
            )

        standardized = self._standardize_time_series_frame(
            frame,
            source_name="Company.news",
        )
        if count > 0 and not standardized.empty:
            standardized = standardized.head(int(count)).reset_index(drop=True)
        return standardized

    def audit_company_news_capability(
        self,
        symbol: str = "SSI",
        *,
        count: int = 1,
        run_live_probe: bool = True,
    ) -> dict[str, Any]:
        """Audit whether the installed ``vnstock_data`` runtime exposes usable company news."""
        ticker = symbol.upper().strip()
        Company = self._get_class("Company")
        audit: dict[str, Any] = {
            "provider_name": "vnstock_data",
            "symbol": ticker,
            "provider_runtime_available": self._is_available(),
            "company_class_available": Company is not None,
            "company_overview_method_available": False,
            "company_news_method_available": False,
            "live_probe_attempted": False,
            "live_probe_success": False,
            "live_probe_rows": 0,
            "live_probe_error": None,
            "live_probe_source_name": None,
            "status": "unsupported",
            "notes": [],
        }
        if Company is None:
            audit["notes"].append("The active interpreter cannot import vnstock_data.Company.")
            return audit

        for source in ("KBS", "VCI"):
            try:
                company = self._provider_call(
                    lambda source=source: Company(symbol=ticker, source=source),
                    call_type="Company.init",
                    symbol=ticker,
                    source=source,
                )
            except Exception as exc:
                audit["live_probe_error"] = str(exc)
                continue
            audit["company_overview_method_available"] = audit["company_overview_method_available"] or hasattr(company, "overview")
            audit["company_news_method_available"] = audit["company_news_method_available"] or hasattr(company, "news")

        if not audit["company_news_method_available"]:
            audit["notes"].append("Company.news is not exposed by the installed vnstock_data runtime.")
            return audit

        if not run_live_probe:
            audit["status"] = "unstable_partial"
            audit["notes"].append("Company.news exists, but no live runtime probe was executed.")
            return audit

        audit["live_probe_attempted"] = True
        probe = self.get_news(ticker, count=max(int(count), 1))
        if probe is None or probe.empty:
            probe_attrs = getattr(probe, "attrs", {})
            audit["status"] = "unstable_partial"
            audit["live_probe_error"] = str(probe_attrs.get("source_notes", "Company.news returned no usable rows."))
            audit["live_probe_source_name"] = str(probe_attrs.get("source_name", "Company.news"))
            audit["notes"].append("Company.news exists, but the live probe did not return usable rows.")
            return audit

        audit["live_probe_success"] = True
        audit["live_probe_rows"] = int(len(probe))
        audit["live_probe_source_name"] = str(probe.attrs.get("source_name", "Company.news"))
        audit["status"] = "live_supported"
        audit["notes"].append("Company.news returned usable rows in the active runtime.")
        return audit

    def get_trade_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        Trading = self._get_class("Trading")
        if Trading is None:
            return self._empty_frame("Trading class unavailable", source_name="Trading.trade_history")

        ticker = symbol.upper().strip()
        frame = self._call_first_success(
            [
                ProviderCallAttempt(
                    lambda: Trading(source="KBS", symbol=ticker).trade_history(page=1, page_size=1000),
                    call_type="Trading.trade_history",
                    symbol=ticker,
                    source="KBS",
                )
            ],
            source_name="Trading.trade_history",
        )
        if frame.empty:
            return self._empty_frame(
                f"No trade history returned for {ticker}",
                source_name="Trading.trade_history",
                notes="KBS trade_history exposes latest intraday trades only, not a backfillable date-range series.",
            )
        standardized = frame.copy()
        if {"trading_date", "time"} <= set(standardized.columns):
            ts = standardized["trading_date"].astype(str) + " " + standardized["time"].astype(str)
            standardized["timestamp"] = pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
            standardized["date"] = standardized["timestamp"].dt.normalize()
        elif "trading_date" in standardized.columns:
            standardized["date"] = pd.to_datetime(
                standardized["trading_date"],
                format="%d/%m/%Y",
                errors="coerce",
            ).dt.normalize()
        standardized["ticker"] = ticker
        standardized = self._filter_date_range(standardized, start_date, end_date)
        return self._attach_attrs(
            standardized.reset_index(drop=True),
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name="Trading.trade_history",
            notes="KBS trade_history returns current intraday trades only.",
        )

    @_time_fetch("get_foreign_flow")
    def get_foreign_flow(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        Trading = self._get_class("Trading")
        if Trading is None:
            return self._empty_frame("Trading class unavailable", source_name="Trading.foreign_trade")

        ticker = symbol.upper().strip()
        attempts = [
            ProviderCallAttempt(
                lambda: Trading(source="VCI", symbol=ticker).foreign_trade(
                    resolution="1D",
                    start=start_date,
                    end=end_date,
                    limit=5000,
                ),
                call_type="Trading.foreign_trade",
                symbol=ticker,
                frequency="1D",
                source="VCI",
            ),
            ProviderCallAttempt(
                lambda: Trading(source="CAFEF", symbol=ticker).foreign_trade(
                    start=start_date,
                    end=end_date,
                    page=1,
                    limit=5000,
                ),
                call_type="Trading.foreign_trade",
                symbol=ticker,
                frequency="1D",
                source="CAFEF",
            ),
        ]
        frame = self._call_first_success(attempts, source_name="Trading.foreign_trade")
        if frame.empty:
            return self._empty_frame(
                f"No foreign-flow data returned for {ticker}",
                source_name="Trading.foreign_trade",
            )
        standardized = self._standardize_time_series_frame(
            frame,
            source_name="Trading.foreign_trade",
            start_date=start_date,
            end_date=end_date,
            symbol=ticker,
        )
        standardized["ticker"] = ticker
        standardized = standardized.rename(
            columns={
                "fr_buy_volume_total": "foreign_buy_volume",
                "fr_sell_volume_total": "foreign_sell_volume",
                "fr_buy_value_total": "foreign_buy_value",
                "fr_sell_value_total": "foreign_sell_value",
                "fr_net_volume_total": "foreign_net_volume",
                "fr_net_value_total": "foreign_net_value",
                "fr_buy_volume_matched": "foreign_buy_volume_matched",
                "fr_sell_volume_matched": "foreign_sell_volume_matched",
                "fr_buy_value_matched": "foreign_buy_value_matched",
                "fr_sell_value_matched": "foreign_sell_value_matched",
                "fr_buy_volume_deal": "foreign_buy_volume_deal",
                "fr_sell_volume_deal": "foreign_sell_volume_deal",
                "fr_buy_value_deal": "foreign_buy_value_deal",
                "fr_sell_value_deal": "foreign_sell_value_deal",
                "fr_net_volume_matched": "foreign_net_volume_matched",
                "fr_net_value_matched": "foreign_net_value_matched",
                "fr_net_volume_deal": "foreign_net_volume_deal",
                "fr_net_value_deal": "foreign_net_value_deal",
                "fr_room_percentage": "foreign_room_pct",
                "fr_owned_percentage": "foreign_owned_pct",
                "fr_available_percentage": "foreign_available_pct",
                "fr_current_room": "foreign_current_room",
                "fr_total_room": "foreign_total_room",
                "fr_remaining_room": "foreign_current_room",
                "fr_ownership": "foreign_owned_pct",
                "fr_buy_volume": "foreign_buy_volume",
                "fr_sell_volume": "foreign_sell_volume",
                "fr_buy_value": "foreign_buy_value",
                "fr_sell_value": "foreign_sell_value",
                "fr_net_volume": "foreign_net_volume",
                "fr_net_value": "foreign_net_value",
            }
        )
        if {"foreign_buy_volume", "foreign_sell_volume"} <= set(standardized.columns) and "foreign_net_volume" not in standardized.columns:
            standardized["foreign_net_volume"] = (
                pd.to_numeric(standardized["foreign_buy_volume"], errors="coerce")
                - pd.to_numeric(standardized["foreign_sell_volume"], errors="coerce")
            )
        if {"foreign_buy_value", "foreign_sell_value"} <= set(standardized.columns) and "foreign_net_value" not in standardized.columns:
            standardized["foreign_net_value"] = (
                pd.to_numeric(standardized["foreign_buy_value"], errors="coerce")
                - pd.to_numeric(standardized["foreign_sell_value"], errors="coerce")
            )
        return standardized

    def get_market_valuation(self, index: str = "VNINDEX", metric: str = "pe", duration: str = "5Y") -> pd.DataFrame:
        Market = self._get_class("Market")
        if Market is None:
            return self._empty_frame("Market class unavailable", source_name=f"Market.{metric}")
        metric_name = metric.lower().strip()
        if metric_name not in {"pe", "pb", "evaluation"}:
            return self._empty_frame(
                f"Unsupported market metric: {metric_name}",
                source_name=f"Market.{metric_name}",
            )
        attempts = [
            ProviderCallAttempt(
                lambda: getattr(Market(source="vnd"), metric_name)(duration=duration),
                call_type=f"Market.{metric_name}",
                symbol=index.upper(),
                frequency=duration,
                source="vnd",
            ),
        ]
        frame = self._call_first_success(attempts, source_name=f"Market.{metric_name}")
        if frame.empty:
            return self._empty_frame(
                f"No market valuation returned for {index.upper()} metric={metric_name}",
                source_name=f"Market.{metric_name}",
            )
        return self._standardize_time_series_frame(
            frame,
            source_name=f"Market.{metric_name}",
        )

    def get_macro_exchange_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        Macro = self._get_class("Macro")
        if Macro is None:
            return self._empty_frame("Macro class unavailable", source_name="Macro.exchange_rate")
        attempts = [
            ProviderCallAttempt(
                lambda: Macro(source="mbk").exchange_rate(start=start_date, end=end_date, period="day"),
                call_type="Macro.exchange_rate",
                frequency="day",
                source="mbk",
            ),
        ]
        frame = self._call_first_success(attempts, source_name="Macro.exchange_rate")
        if frame.empty:
            return self._empty_frame(
                "No exchange-rate data returned by vnstock_data",
                source_name="Macro.exchange_rate",
            )
        return self._standardize_time_series_frame(
            frame,
            source_name="Macro.exchange_rate",
            start_date=start_date,
            end_date=end_date,
        )

    def get_macro_interest_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        Macro = self._get_class("Macro")
        if Macro is None:
            return self._empty_frame("Macro class unavailable", source_name="Macro.interest_rate")
        attempts = [
            ProviderCallAttempt(
                lambda: Macro(source="mbk").interest_rate(
                    start=start_date,
                    end=end_date,
                    period="day",
                    format="long",
                ),
                call_type="Macro.interest_rate",
                frequency="day",
                source="mbk",
            ),
        ]
        frame = self._call_first_success(attempts, source_name="Macro.interest_rate")
        if frame.empty:
            return self._empty_frame(
                "No interest-rate data returned by vnstock_data",
                source_name="Macro.interest_rate",
            )
        return self._standardize_time_series_frame(
            frame,
            source_name="Macro.interest_rate",
            start_date=start_date,
            end_date=end_date,
        )

    def get_commodity_gold(self, start_date: str, end_date: str) -> pd.DataFrame:
        CommodityPrice = self._get_class("CommodityPrice")
        attempts: list[Callable[[], Any] | ProviderCallAttempt] = []
        if CommodityPrice is not None:
            attempts.extend(
                [
                    ProviderCallAttempt(
                        lambda: CommodityPrice(source="spl").gold_vn(start=start_date, end=end_date),
                        call_type="CommodityPrice.gold_vn",
                        source="spl",
                    ),
                    ProviderCallAttempt(
                        lambda: CommodityPrice(source="spl").gold_global(start=start_date, end=end_date),
                        call_type="CommodityPrice.gold_global",
                        source="spl",
                    ),
                ]
            )
        if not attempts:
            return self._empty_frame("Commodity interface unavailable", source_name="CommodityPrice.gold")
        frame = self._call_first_success(attempts, source_name="CommodityPrice.gold")
        if frame.empty:
            return self._empty_frame(
                "No gold series returned by vnstock_data",
                source_name="CommodityPrice.gold",
            )
        return self._standardize_time_series_frame(
            frame,
            source_name="CommodityPrice.gold",
            start_date=start_date,
            end_date=end_date,
        )

    def get_commodity_oil(self, start_date: str, end_date: str) -> pd.DataFrame:
        CommodityPrice = self._get_class("CommodityPrice")
        attempts: list[Callable[[], Any] | ProviderCallAttempt] = []
        if CommodityPrice is not None:
            attempts.append(
                ProviderCallAttempt(
                    lambda: CommodityPrice(source="spl").oil_crude(start=start_date, end=end_date),
                    call_type="CommodityPrice.oil_crude",
                    source="spl",
                )
            )
        if not attempts:
            return self._empty_frame("Commodity interface unavailable", source_name="CommodityPrice.oil_crude")
        frame = self._call_first_success(attempts, source_name="CommodityPrice.oil_crude")
        if frame.empty:
            return self._empty_frame(
                "No oil series returned by vnstock_data",
                source_name="CommodityPrice.oil_crude",
            )
        return self._standardize_time_series_frame(
            frame,
            source_name="CommodityPrice.oil_crude",
            start_date=start_date,
            end_date=end_date,
        )

    def get_symbols_by_industries(self) -> pd.DataFrame:
        Listing = self._get_class("Listing")
        if Listing is None:
            return self._empty_frame("Listing class unavailable", source_name="Listing.symbols_by_industries")
        frame = self._call_first_success(
            [
                ProviderCallAttempt(
                    lambda: Listing(source="KBS").symbols_by_industries(),
                    call_type="Listing.symbols_by_industries",
                    source="KBS",
                ),
                ProviderCallAttempt(
                    lambda: Listing(source="VCI").symbols_by_industries(),
                    call_type="Listing.symbols_by_industries",
                    source="VCI",
                ),
            ],
            source_name="Listing.symbols_by_industries",
        )
        if frame.empty:
            return self._empty_frame(
                "No symbol-industry mapping returned by vnstock_data",
                source_name="Listing.symbols_by_industries",
            )
        standardized = frame.reset_index(drop=True)
        if "symbol" in standardized.columns and "ticker" not in standardized.columns:
            standardized["ticker"] = standardized["symbol"].astype(str).str.upper()
        if "industry_name" in standardized.columns and "industry" not in standardized.columns:
            standardized["industry"] = standardized["industry_name"]
        return self._attach_attrs(
            standardized,
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name="Listing.symbols_by_industries",
        )

    def get_all_symbols(self) -> pd.DataFrame:
        Listing = self._get_class("Listing")
        if Listing is None:
            return self._empty_frame("Listing class unavailable", source_name="Listing.all_symbols")
        frame = self._call_first_success(
            [
                ProviderCallAttempt(
                    lambda: Listing(source="VCI").all_symbols(),
                    call_type="Listing.all_symbols",
                    source="VCI",
                ),
                ProviderCallAttempt(
                    lambda: Listing(source="KBS").all_symbols(),
                    call_type="Listing.all_symbols",
                    source="KBS",
                ),
                ProviderCallAttempt(
                    lambda: Listing(source="VND").all_symbols(),
                    call_type="Listing.all_symbols",
                    source="VND",
                ),
            ],
            source_name="Listing.all_symbols",
        )
        if frame.empty:
            return self._empty_frame(
                "No symbol listing returned by vnstock_data",
                source_name="Listing.all_symbols",
            )
        standardized = frame.copy().reset_index(drop=True)
        first_col = standardized.columns[0] if len(standardized.columns) else None
        if "symbol" in standardized.columns:
            standardized["symbol"] = standardized["symbol"].astype(str).str.upper()
        elif first_col is not None:
            standardized[first_col] = standardized[first_col].astype(str).str.upper()
            standardized = standardized.rename(columns={first_col: "symbol"})
        return self._attach_attrs(
            standardized,
            provenance=DIRECT_VNSTOCK_PROVENANCE,
            source_name="Listing.all_symbols",
        )

    def get_vn100_tickers(self) -> list[str]:
        Listing = self._get_class("Listing")
        if Listing is not None:
            frame = self._call_first_success(
                [
                    ProviderCallAttempt(
                        lambda: Listing(source="VCI").symbols_by_group("VN100"),
                        call_type="Listing.symbols_by_group",
                        symbol="VN100",
                        source="VCI",
                    ),
                    ProviderCallAttempt(
                        lambda: Listing(source="KBS").symbols_by_group("VN100"),
                        call_type="Listing.symbols_by_group",
                        symbol="VN100",
                        source="KBS",
                    ),
                ],
                source_name="Listing.symbols_by_group",
            )
            if frame is not None and not frame.empty:
                first_col = frame.columns[0]
                values = frame[first_col].dropna().astype(str).str.upper().tolist()
                if values:
                    return values
        logger.warning("vn100_listing_fallback_used")
        return [
            "ACB", "ANV", "BCM", "BID", "BMP", "BVH", "BWE", "CII", "CMG", "CTD",
            "CTG", "CTR", "DBC", "DCM", "DGW", "DHC", "DXG", "EIB", "FPT", "FRT",
            "FTS", "GAS", "GEX", "GMD", "GVR", "HCM", "HDB", "HDG", "HHV", "HPG",
            "HSG", "KBC", "KDH", "LPB", "MBB", "MSB", "MSN", "MWG", "NKG", "NLG",
            "NVL", "OCB", "PAN", "PC1", "PDR", "PET", "PHR", "PLX", "PNJ", "POW",
            "PVD", "PVT", "REE", "SAB", "SAM", "SBT", "SHB", "SJS", "SSB", "SSI",
            "STB", "SZC", "TCB", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM",
            "VIB", "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VPL", "VRE",
            "VSC", "VTP",
        ]
