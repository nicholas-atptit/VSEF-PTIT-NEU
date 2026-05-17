"""Diagnose VN30 hourly vnstock provider coverage for the listing-aware track."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import importlib
import importlib.metadata
import importlib.util
import io
import multiprocessing as mp
import queue
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    VN30_TICKERS,
    markdown_table,
    read_universe,
    rel,
    timestamp_text,
    write_csv,
)
from scripts.research.vn30_hourly_vnstock_common import (  # noqa: E402
    as_dataframe,
    standardize_provider_frame,
)


OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_listing_aware" / "provider_diagnostics"
CSV_PATH = OUTPUT_DIR / "vn30_vnstock_hourly_fetch_diagnostics.csv"
MD_PATH = OUTPUT_DIR / "vn30_vnstock_hourly_fetch_diagnostics.md"

INTERVALS = ("1H", "60m", "1h", "hourly")
WINDOWS = (
    ("2024-01-02", "2024-01-05"),
    ("2024-12-02", "2024-12-06"),
    ("2025-01-02", "2025-01-06"),
    ("2026-05-04", "2026-05-13"),
)
VNINDEX = "VNINDEX"
LOCAL_PATH_RE = re.compile(r"(?<![A-Za-z])\b[A-Za-z]:[\\/][^\s|,;)]*")

CSV_COLUMNS = [
    "ticker",
    "asset_type",
    "package",
    "package_version",
    "provider",
    "source",
    "entrypoint",
    "interval",
    "window_start",
    "window_end",
    "success",
    "rows_returned",
    "standardized_rows",
    "columns_returned",
    "first_timestamp",
    "last_timestamp",
    "returned_years",
    "exception_type",
    "exception_message",
    "diagnosis",
]


@dataclass(frozen=True)
class ProviderSpec:
    package: str
    provider: str
    source: str
    entrypoint: str
    asset_scope: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe installed vnstock hourly coverage for frozen VN30 tickers plus VNINDEX."
    )
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol subset for debugging.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown from the existing diagnostics CSV.")
    return parser.parse_args()


def sanitize_text(value: Any, *, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("file" + ":///", "[local file uri redacted]")
    text = LOCAL_PATH_RE.sub("[local path redacted]", text)
    text = " ".join(text.split())
    return text[:limit]


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception as exc:
        return f"unknown:{type(exc).__name__}"


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def discover_quote_sources() -> list[str]:
    sources: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        vnstock = importlib.import_module("vnstock")
        ensure = getattr(vnstock, "_ensure_explorer_modules_loaded", None)
        if callable(ensure):
            ensure()
        try:
            registry = importlib.import_module("vnstock.core.registry").ProviderRegistry
            sources.extend(registry.list_available("quote"))
        except Exception:
            pass
    return unique_preserve_order(sources)


def discover_client_sources() -> list[str]:
    sources: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            vnstock = importlib.import_module("vnstock")
            client = vnstock.Vnstock()
            sources.extend(getattr(client, "SUPPORTED_SOURCES", []) or [])
        except Exception:
            pass
    return unique_preserve_order(sources)


def discover_provider_specs() -> tuple[list[ProviderSpec], list[str], list[str]]:
    if importlib.util.find_spec("vnstock") is None:
        return [], [], []
    quote_sources = discover_quote_sources()
    client_sources = discover_client_sources()
    specs: list[ProviderSpec] = []
    for source in quote_sources:
        specs.append(
            ProviderSpec(
                package="vnstock",
                provider="vnstock.ProviderRegistry.quote",
                source=source,
                entrypoint="ProviderRegistry.get('quote').history",
                asset_scope="stock,index",
            )
        )
    if "MSN" in quote_sources or "MSN" in client_sources:
        specs.append(
            ProviderSpec(
                package="vnstock",
                provider="vnstock.Vnstock.world_index",
                source="MSN",
                entrypoint="Vnstock.world_index.quote.history",
                asset_scope="index",
            )
        )
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ProviderSpec] = []
    for spec in specs:
        key = (spec.provider, spec.source, spec.entrypoint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped, quote_sources, client_sources


def selected_symbols(raw: str) -> list[str]:
    universe = read_universe()
    if universe != VN30_TICKERS:
        raise ValueError("Frozen VN30 universe does not match the mandatory 30-ticker list.")
    if raw.strip():
        requested = [item.strip().upper() for item in raw.split(",") if item.strip()]
        allowed = set(universe + [VNINDEX])
        unknown = sorted(set(requested).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown symbols outside frozen VN30 plus VNINDEX: {unknown}")
        return requested
    return [*universe, VNINDEX]


def asset_type(symbol: str) -> str:
    return "index" if symbol.upper().strip() == VNINDEX else "stock"


def spec_applies(spec: ProviderSpec, symbol: str) -> bool:
    kind = asset_type(symbol)
    return kind in {item.strip() for item in spec.asset_scope.split(",")}


def execute_provider_call(spec: ProviderSpec, symbol: str, start: str, end: str, interval: str) -> Any:
    vnstock = importlib.import_module("vnstock")
    code = symbol.upper().strip()
    if spec.entrypoint == "ProviderRegistry.get('quote').history":
        ensure = getattr(vnstock, "_ensure_explorer_modules_loaded", None)
        if callable(ensure):
            ensure()
        registry = importlib.import_module("vnstock.core.registry").ProviderRegistry
        provider_class = registry.get("quote", spec.source)
        init_signature = inspect.signature(provider_class.__init__)
        init_kwargs: dict[str, Any] = {}
        for key, value in {
            "symbol": code,
            "random_agent": False,
            "show_log": False,
        }.items():
            if key in init_signature.parameters:
                init_kwargs[key] = value
        provider = provider_class(**init_kwargs)
        history_signature = inspect.signature(provider.history)
        history_kwargs: dict[str, Any] = {}
        for key, value in {
            "start": start,
            "end": end,
            "interval": interval,
            "show_log": False,
        }.items():
            if key in history_signature.parameters:
                history_kwargs[key] = value
        return provider.history(**history_kwargs)
    if spec.entrypoint == "Vnstock.world_index.quote.history":
        client = vnstock.Vnstock()
        return client.world_index(symbol=code, source=spec.source).quote.history(start=start, end=end, interval=interval)
    raise ValueError(f"Unsupported provider entrypoint: {spec.entrypoint}")


def worker(queue_out: Any, spec: ProviderSpec, symbol: str, start: str, end: str, interval: str) -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            result = execute_provider_call(spec, symbol, start, end, interval)
            raw = as_dataframe(result)
            standardized = standardize_provider_frame(raw, symbol, provider=spec.provider, source=spec.source)
            years = ""
            first_ts = ""
            last_ts = ""
            if not standardized.empty:
                timestamps = pd.to_datetime(standardized["datetime"], errors="coerce").dropna()
                if not timestamps.empty:
                    years = ",".join(str(year) for year in sorted(set(timestamps.dt.year.astype(int))))
                    first_ts = timestamp_text(timestamps.min())
                    last_ts = timestamp_text(timestamps.max())
            queue_out.put(
                {
                    "ok": True,
                    "rows_returned": int(len(raw)),
                    "standardized_rows": int(len(standardized)),
                    "columns_returned": ",".join(str(column) for column in raw.columns),
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts,
                    "returned_years": years,
                }
            )
        except BaseException as exc:
            queue_out.put(
                {
                    "ok": False,
                    "exception_type": type(exc).__name__,
                    "exception_message": sanitize_text(exc),
                }
            )


def classify_attempt(row: dict[str, Any]) -> str:
    success = row.get("success") == "yes"
    rows = int(row.get("standardized_rows") or 0)
    raw_rows = int(row.get("rows_returned") or 0)
    symbol_kind = row.get("asset_type", "")
    message = f"{row.get('exception_type', '')} {row.get('exception_message', '')}".lower()
    interval = str(row.get("interval", "")).lower()
    first = pd.to_datetime(row.get("first_timestamp", ""), errors="coerce")
    last = pd.to_datetime(row.get("last_timestamp", ""), errors="coerce")
    window_start = pd.to_datetime(row.get("window_start", ""), errors="coerce")
    window_end = pd.to_datetime(row.get("window_end", ""), errors="coerce")

    if success and rows > 0:
        if raw_rows in {1000, 2000, 5000} or rows in {1000, 2000, 5000}:
            return "limited_rows_per_request_possible"
        if not pd.isna(first) and not pd.isna(last) and not pd.isna(window_start) and not pd.isna(window_end):
            if first > window_end + pd.Timedelta(days=2) or last < window_start - pd.Timedelta(days=2):
                return "only_recent_intraday_window" if first.year >= 2025 else "returned_outside_requested_window"
            if window_start.year <= 2024 and first.year >= 2025:
                return "only_recent_intraday_window"
        return "hourly_rows_returned"

    if raw_rows > 0 and rows == 0:
        return "raw_rows_not_standard_ohlcv"
    if "timeout" in message:
        return "connection_or_timeout_error"
    if any(token in message for token in ("429", "rate", "too many request", "too many requests")):
        return "rate_limit_error"
    if any(token in message for token in ("api key", "apikey", "auth", "token", "forbidden", "unauthorized", "401", "403", "subscription")):
        return "auth_error"
    if any(token in message for token in ("connection", "connect", "dns", "proxy", "ssl", "remote", "network", "timed out", "max retries")):
        return "connection_or_timeout_error"
    if interval in {"60m", "hourly"} and any(token in message for token in ("invalid interval", "interval", "khong hop le", "không hợp lệ")):
        return "unsupported_interval"
    if any(token in message for token in ("supported sources", "not found. available", "does not support", "source")):
        return "unsupported_provider_source"
    if any(token in message for token in ("symbol", "ma chung khoan", "mã chứng khoán", "ticker", "not found", "khong tim thay", "không tìm thấy")):
        return "unsupported_index" if symbol_kind == "index" else "unsupported_ticker"
    if not row.get("exception_type") and not row.get("exception_message"):
        year = str(row.get("window_start", ""))[:4]
        return "empty_rows_for_old_window" if year in {"2024", "2025"} else "empty_rows_for_window"
    return "provider_error"


def base_attempt_row(spec: ProviderSpec, symbol: str, start: str, end: str, interval: str) -> dict[str, Any]:
    return {
        "ticker": symbol.upper().strip(),
        "asset_type": asset_type(symbol),
        "package": spec.package,
        "package_version": package_version(spec.package),
        "provider": spec.provider,
        "source": spec.source,
        "entrypoint": spec.entrypoint,
        "interval": interval,
        "window_start": start,
        "window_end": end,
        "success": "no",
        "rows_returned": 0,
        "standardized_rows": 0,
        "columns_returned": "",
        "first_timestamp": "",
        "last_timestamp": "",
        "returned_years": "",
        "exception_type": "",
        "exception_message": "",
        "diagnosis": "",
    }


def apply_attempt_payload(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok"):
        base["success"] = "yes" if int(payload.get("standardized_rows", 0) or 0) > 0 else "no"
        base["rows_returned"] = int(payload.get("rows_returned", 0) or 0)
        base["standardized_rows"] = int(payload.get("standardized_rows", 0) or 0)
        base["columns_returned"] = sanitize_text(payload.get("columns_returned", ""), limit=1200)
        base["first_timestamp"] = sanitize_text(payload.get("first_timestamp", ""))
        base["last_timestamp"] = sanitize_text(payload.get("last_timestamp", ""))
        base["returned_years"] = sanitize_text(payload.get("returned_years", ""))
    else:
        base["exception_type"] = sanitize_text(payload.get("exception_type", "ProviderError"))
        base["exception_message"] = sanitize_text(payload.get("exception_message", ""))
    base["diagnosis"] = classify_attempt(base)
    return base


def run_attempt(spec: ProviderSpec, symbol: str, start: str, end: str, interval: str, timeout_seconds: float) -> dict[str, Any]:
    base = base_attempt_row(spec, symbol, start, end, interval)
    queue_out: Any = mp.Queue()
    process = mp.Process(target=worker, args=(queue_out, spec, symbol, start, end, interval))
    process.start()
    process.join(max(1.0, float(timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(5)
        base["exception_type"] = "TimeoutError"
        base["exception_message"] = f"provider call exceeded {timeout_seconds:.1f} seconds"
    else:
        try:
            payload = queue_out.get_nowait()
        except queue.Empty:
            payload = {"ok": False, "exception_type": "NoResult", "exception_message": "provider process returned no result"}
        base = apply_attempt_payload(base, payload)
    base["diagnosis"] = classify_attempt(base)
    return base


def run_attempts_parallel(
    attempts: list[tuple[ProviderSpec, str, str, str, str]],
    *,
    timeout_seconds: float,
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers <= 1:
        rows: list[dict[str, Any]] = []
        for index, (spec, symbol, start, end, interval) in enumerate(attempts, start=1):
            rows.append(run_attempt(spec, symbol, start, end, interval, timeout_seconds))
            if index % 100 == 0 or index == len(attempts):
                print(f"completed {index}/{len(attempts)} provider diagnostic attempts")
        return rows

    pending = list(attempts)
    active: list[dict[str, Any]] = []
    rows = []
    completed = 0
    max_workers = max(1, int(max_workers))
    timeout_seconds = max(1.0, float(timeout_seconds))

    while pending or active:
        while pending and len(active) < max_workers:
            spec, symbol, start, end, interval = pending.pop(0)
            queue_out: Any = mp.Queue()
            process = mp.Process(target=worker, args=(queue_out, spec, symbol, start, end, interval))
            process.start()
            active.append(
                {
                    "process": process,
                    "queue": queue_out,
                    "started": time.monotonic(),
                    "base": base_attempt_row(spec, symbol, start, end, interval),
                }
            )

        still_active: list[dict[str, Any]] = []
        for item in active:
            process = item["process"]
            elapsed = time.monotonic() - float(item["started"])
            if process.is_alive() and elapsed <= timeout_seconds:
                still_active.append(item)
                continue
            base = item["base"]
            if process.is_alive():
                process.terminate()
                process.join(5)
                base["exception_type"] = "TimeoutError"
                base["exception_message"] = f"provider call exceeded {timeout_seconds:.1f} seconds"
                base["diagnosis"] = classify_attempt(base)
                rows.append(base)
            else:
                process.join(1)
                try:
                    payload = item["queue"].get_nowait()
                except queue.Empty:
                    payload = {
                        "ok": False,
                        "exception_type": "NoResult",
                        "exception_message": "provider process returned no result",
                    }
                rows.append(apply_attempt_payload(base, payload))
            completed += 1
            if completed % 100 == 0 or completed == len(attempts):
                print(f"completed {completed}/{len(attempts)} provider diagnostic attempts")
        active = still_active
        if active:
            time.sleep(0.05)

    return rows


def names(items: set[str] | list[str]) -> str:
    values = sorted(items)
    return ", ".join(values) if values else "none"


def success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("success") == "yes" and int(row.get("standardized_rows") or 0) > 0]


def tickers_with_rows_in_year(rows: list[dict[str, Any]], year: int) -> set[str]:
    result: set[str] = set()
    for row in success_rows(rows):
        if row.get("asset_type") != "stock":
            continue
        years = {item.strip() for item in str(row.get("returned_years", "")).split(",") if item.strip()}
        if str(year) in years:
            result.add(str(row.get("ticker", "")))
    return result


def group_success_by_provider(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("provider", "")),
            str(row.get("source", "")),
            str(row.get("entrypoint", "")),
            str(row.get("interval", "")),
        )
        item = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "source": key[1],
                "entrypoint": key[2],
                "interval": key[3],
                "successful_symbols": set(),
                "successful_vn30_tickers": set(),
                "diagnoses": {},
            },
        )
        if row.get("success") == "yes" and int(row.get("standardized_rows") or 0) > 0:
            item["successful_symbols"].add(str(row.get("ticker", "")))
            if row.get("asset_type") == "stock":
                item["successful_vn30_tickers"].add(str(row.get("ticker", "")))
        diagnosis = str(row.get("diagnosis", ""))
        item["diagnoses"][diagnosis] = item["diagnoses"].get(diagnosis, 0) + 1

    output: list[dict[str, Any]] = []
    for item in grouped.values():
        diagnoses = "; ".join(f"{key}:{value}" for key, value in sorted(item["diagnoses"].items()))
        output.append(
            {
                "provider": item["provider"],
                "source": item["source"],
                "entrypoint": item["entrypoint"],
                "interval": item["interval"],
                "successful_symbols": len(item["successful_symbols"]),
                "successful_vn30_tickers": len(item["successful_vn30_tickers"]),
                "diagnoses": diagnoses,
            }
        )
    output.sort(key=lambda row: (-int(row["successful_vn30_tickers"]), row["provider"], row["source"], row["interval"]))
    return output


def write_report(rows: list[dict[str, Any]], quote_sources: list[str], client_sources: list[str], symbols: list[str]) -> None:
    successes = success_rows(rows)
    any_stock_rows = {str(row["ticker"]) for row in successes if row.get("asset_type") == "stock"}
    rows_2024 = tickers_with_rows_in_year(rows, 2024)
    rows_2025 = tickers_with_rows_in_year(rows, 2025)
    rows_2026 = tickers_with_rows_in_year(rows, 2026)
    vnindex_rows = [row for row in successes if row.get("ticker") == VNINDEX]
    vnindex_years = sorted(
        {
            year.strip()
            for row in vnindex_rows
            for year in str(row.get("returned_years", "")).split(",")
            if year.strip()
        }
    )
    old_window_successes = rows_2024.union(rows_2025)
    recent_limit_attempts = [
        row
        for row in rows
        if row.get("diagnosis") in {"only_recent_intraday_window", "limited_rows_per_request_possible"}
    ]
    appears_recent_limited = bool(recent_limit_attempts) or (
        bool(any_stock_rows) and not old_window_successes and bool(rows_2026)
    )
    all_sample_years = len(rows_2024) == 30 and len(rows_2025) == 30 and len(rows_2026) == 30 and bool(vnindex_rows)

    if not any_stock_rows:
        feasibility = "No. No frozen VN30 ticker returned standardized hourly rows in these probes."
    elif not all_sample_years:
        feasibility = (
            "Not yet established. Sample coverage is broader than the current partial cache, but the full "
            "listing-aware benchmark is not feasible until a complete normalized cache passes the row-count "
            "and VNINDEX validation gates."
        )
    else:
        feasibility = (
            "Not established. Sample-window support exists, but sample support is not full-history support and "
            "must not be used as a benchmark gate."
        )

    if appears_recent_limited:
        limit_text = (
            "Yes for at least part of the universe: one or more attempts returned only a recent intraday window "
            "or hit a likely per-request row cap."
        )
    elif any_stock_rows:
        limit_text = (
            "Not clearly from these probes. Old-window rows were observed for most tickers and VNINDEX, but "
            "this still does not prove full listing-aware history."
        )
    else:
        limit_text = "Indeterminate because no standardized hourly stock rows were returned."

    next_action = (
        "Do not run the listing-aware benchmark from the current partial cache. Repair the listing-aware fetch "
        "path to use the registered quote providers and supported hourly intervals observed here, rerun the "
        "hourly-only cache build with conservative throttling, then validate the normalized cache. If that "
        "still fails the row-count or VNINDEX gates, acquire an external hourly source rather than using daily "
        "data or resampling."
    )

    stock_summary = [
        {
            "question": "Any hourly rows",
            "count": f"{len(any_stock_rows)}/30",
            "tickers": names(any_stock_rows),
        },
        {
            "question": "Rows in 2024",
            "count": f"{len(rows_2024)}/30",
            "tickers": names(rows_2024),
        },
        {
            "question": "Rows in 2025",
            "count": f"{len(rows_2025)}/30",
            "tickers": names(rows_2025),
        },
        {
            "question": "Rows in 2026",
            "count": f"{len(rows_2026)}/30",
            "tickers": names(rows_2026),
        },
    ]

    diagnosis_counts: dict[str, int] = {}
    for row in rows:
        diagnosis = str(row.get("diagnosis", ""))
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    diagnosis_rows = [{"diagnosis": key, "attempts": value} for key, value in sorted(diagnosis_counts.items())]

    content = [
        "# VN30 vnstock Hourly Fetch Diagnostics",
        "",
        "## Scope",
        "",
        f"- Frozen universe: `{rel(REPO_ROOT / 'configs' / 'universes' / 'vn30_constituents_frozen.csv')}`.",
        f"- Symbols probed: {len(symbols)} total ({len([s for s in symbols if s != VNINDEX])} VN30 tickers plus VNINDEX).",
        f"- Provider sources discovered from installed `vnstock`: Quote registry={', '.join(quote_sources) or 'none'}; Vnstock client={', '.join(client_sources) or 'none'}.",
        "- Provider calls use the registered quote provider classes directly; the Vnstock client is used only for the VNINDEX world-index probe.",
        f"- Intervals tested: {', '.join(INTERVALS)}.",
        f"- Windows tested: {', '.join(f'{start} to {end}' for start, end in WINDOWS)}.",
        f"- Full attempt CSV: `{rel(CSV_PATH)}`.",
        "- This diagnostic does not treat sample-window support as full-history support.",
        "",
        "## Direct Answers",
        "",
        f"- Which tickers return any hourly rows? {len(any_stock_rows)}/30: {names(any_stock_rows)}.",
        f"- Which tickers return rows in 2024? {len(rows_2024)}/30: {names(rows_2024)}.",
        f"- Which tickers return rows in 2025? {len(rows_2025)}/30: {names(rows_2025)}.",
        f"- Which tickers return rows in 2026? {len(rows_2026)}/30: {names(rows_2026)}.",
        f"- Does VNINDEX return hourly rows? {'yes' if vnindex_rows else 'no'}"
        + (f" (returned years: {', '.join(vnindex_years)})." if vnindex_rows else "."),
        f"- Does the provider appear to limit hourly history to recent dates? {limit_text}",
        f"- Is full listing-aware benchmark feasible with this provider? {feasibility}",
        f"- Next required action: {next_action}",
        "",
        "## Stock Coverage Summary",
        "",
        markdown_table(["question", "count", "tickers"], stock_summary),
        "",
        "## Provider Combination Summary",
        "",
        markdown_table(
            ["provider", "source", "entrypoint", "interval", "successful_symbols", "successful_vn30_tickers", "diagnoses"],
            group_success_by_provider(rows),
            max_rows=80,
        ),
        "",
        "## Diagnosis Counts",
        "",
        markdown_table(["diagnosis", "attempts"], diagnosis_rows),
        "",
        "## Interpretation",
        "",
        "- Success means the provider returned standardized hourly OHLCV rows for a small probe window only.",
        "- A ticker that succeeds in one sample window is not considered benchmark-usable.",
        "- Benchmark usability still requires the listing-aware normalized cache to pass the training-row, evaluation-row, and VNINDEX gates.",
        "",
    ]
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    symbols = selected_symbols(args.symbols)
    specs, quote_sources, client_sources = discover_provider_specs()
    if args.report_only:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"Diagnostics CSV does not exist: {rel(CSV_PATH)}")
        rows = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False).to_dict("records")
        write_report(rows, quote_sources, client_sources, symbols)
        print(f"Wrote {rel(MD_PATH)}")
        return 0
    if not specs:
        rows = [
            {
                "ticker": symbol,
                "asset_type": asset_type(symbol),
                "package": "vnstock",
                "package_version": "",
                "provider": "vnstock",
                "source": "not_available",
                "entrypoint": "package_import",
                "interval": "",
                "window_start": "",
                "window_end": "",
                "success": "no",
                "rows_returned": 0,
                "standardized_rows": 0,
                "columns_returned": "",
                "first_timestamp": "",
                "last_timestamp": "",
                "returned_years": "",
                "exception_type": "PackageNotFound",
                "exception_message": "vnstock is not installed",
                "diagnosis": "provider_error",
            }
            for symbol in symbols
        ]
    else:
        attempts: list[tuple[ProviderSpec, str, str, str, str]] = []
        for symbol in symbols:
            for spec in specs:
                if not spec_applies(spec, symbol):
                    continue
                for start, end in WINDOWS:
                    for interval in INTERVALS:
                        attempts.append((spec, symbol, start, end, interval))
        rows = run_attempts_parallel(
            attempts,
            timeout_seconds=args.timeout_seconds,
            max_workers=args.max_workers,
        )

    write_csv(CSV_PATH, rows, fieldnames=CSV_COLUMNS)
    write_report(rows, quote_sources, client_sources, symbols)
    print(f"Wrote {rel(CSV_PATH)}")
    print(f"Wrote {rel(MD_PATH)}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
