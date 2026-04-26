"""Curate a provider-backed foreign-flow artifact with provenance.

This script attempts to fetch real foreign-flow rows through VnstockAdapter
when the local runtime exposes the required vnstock_data Trading interface. It
does not fabricate data and refuses to overwrite the legacy local fallback path
by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import PROJECT_ROOT
from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.backtest.foreign_flow_validation import validate_foreign_flow_artifact


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "foreign_flow_curated.csv"
LEGACY_FALLBACK_PATH = PROJECT_ROOT / "data" / "foreign_flow.csv"
KEEP_COLUMNS = {
    "ticker",
    "date",
    "foreign_buy_volume",
    "foreign_sell_volume",
    "foreign_net_volume",
    "foreign_buy_value",
    "foreign_sell_value",
    "foreign_net_value",
    "foreign_room_pct",
    "foreign_owned_pct",
    "foreign_available_pct",
    "foreign_current_room",
    "foreign_total_room",
}
FIXTURE_SOURCE_MARKERS = ("fixture", "sample", "synthetic", "unit_test", "demo")


def _normalize_tickers(raw: str | list[str] | tuple[str, ...]) -> list[str]:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    return [str(value).strip().upper() for value in values if str(value).strip()]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_provider_frame(frame: pd.DataFrame, ticker: str, retrieved_at: str) -> pd.DataFrame:
    local = frame.copy()
    source_text = " ".join(
        str(value).lower()
        for column in ("source", "provider", "coverage_note")
        if column in local.columns
        for value in local[column].dropna().astype(str).unique().tolist()
    )
    fixture_like_source = any(marker in source_text for marker in FIXTURE_SOURCE_MARKERS)
    if "time" in local.columns and "date" not in local.columns:
        local = local.rename(columns={"time": "date"})
    if "date" in local.columns:
        local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.normalize()
    local["ticker"] = ticker.upper()
    present = [column for column in local.columns if column in KEEP_COLUMNS]
    local = local[present].copy()
    local = local.dropna(subset=["date"])
    local["source"] = "fixture_sample" if fixture_like_source else "vnstock_data.Trading.foreign_trade"
    local["source_date"] = local["date"]
    local["retrieved_at"] = retrieved_at
    local["provider"] = "non_real_fixture" if fixture_like_source else "vnstock_data"
    local["coverage_note"] = (
        "Provider returned fixture/sample-labeled rows; not real provider evidence."
        if fixture_like_source
        else "Provider-backed fetch through VnstockAdapter.get_foreign_flow; validate coverage before interpretation."
    )
    return local


def curate_foreign_flow_provider_artifact(
    *,
    tickers: str | list[str] | tuple[str, ...],
    start_date: str,
    end_date: str,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    adapter_factory: Callable[[list[str]], Any] = VnstockAdapter,
    retrieved_at: str | None = None,
    write_output: bool = True,
) -> dict[str, Any]:
    """Attempt provider-backed curation and return a JSON-serializable report."""

    normalized_tickers = _normalize_tickers(tickers)
    resolved_output = Path(output_path)
    if resolved_output.resolve() == LEGACY_FALLBACK_PATH.resolve():
        raise ValueError("Refusing to overwrite data/foreign_flow.csv; use an explicit curated output path.")

    report: dict[str, Any] = {
        "requested_tickers": normalized_tickers,
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "output_path": str(resolved_output),
        "provider": "vnstock_data",
        "provider_fetch_attempted": False,
        "real_data_fetched": False,
        "rows_written": 0,
        "status": "not_started",
    }

    if not normalized_tickers:
        report["status"] = "no_tickers_requested"
        report["validation"] = validate_foreign_flow_artifact(pd.DataFrame(), [], start_date, end_date)
        return report

    adapter = adapter_factory(normalized_tickers)
    trading_class = getattr(adapter, "_get_class", lambda _name: None)("Trading")
    if trading_class is None:
        report["status"] = "provider_unavailable"
        report["provider_unavailable_reason"] = "vnstock_data Trading class is unavailable in the active runtime."
        report["validation"] = validate_foreign_flow_artifact(pd.DataFrame(), normalized_tickers, start_date, end_date)
        return report

    report["provider_fetch_attempted"] = True
    retrieved_at_value = retrieved_at or _utc_timestamp()
    frames: list[pd.DataFrame] = []
    fetch_errors: dict[str, str] = {}

    for ticker in normalized_tickers:
        try:
            frame = adapter.get_foreign_flow(ticker, start_date, end_date)
        except Exception as exc:  # pragma: no cover - defensive around provider runtime variance
            fetch_errors[ticker] = str(exc)
            continue
        if frame is None or frame.empty:
            continue
        normalized = _normalize_provider_frame(frame, ticker, retrieved_at_value)
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        report["status"] = "provider_returned_no_rows"
        if fetch_errors:
            report["fetch_errors"] = fetch_errors
        report["validation"] = validate_foreign_flow_artifact(pd.DataFrame(), normalized_tickers, start_date, end_date)
        return report

    artifact = pd.concat(frames, ignore_index=True)
    artifact = artifact.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    validation = validate_foreign_flow_artifact(artifact, normalized_tickers, start_date, end_date)
    report["validation"] = validation
    report["real_data_fetched"] = bool(validation["real_provider_evidence"])
    report["status"] = "curated" if report["real_data_fetched"] else "curated_but_not_real_provider_evidence"

    if write_output:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        artifact.to_csv(resolved_output, index=False)
        report["rows_written"] = int(len(artifact))
    else:
        report["rows_written"] = 0

    if fetch_errors:
        report["fetch_errors"] = fetch_errors
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate provider-backed foreign-flow rows with provenance.")
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list, for example SSI,FPT,ACB,HPG.")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path. Defaults to data/foreign_flow_curated.csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = curate_foreign_flow_provider_artifact(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
