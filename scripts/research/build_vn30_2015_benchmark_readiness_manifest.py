"""Build VN30 2015 benchmark-readiness manifest from gateway validation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_VALIDATION = REPO_ROOT / "reports" / "generated" / "index_hourly_2015" / "validation" / "index_hourly_2015_validation.csv"
STOCK_VALIDATION = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "validation" / "vn30_hourly_2015_validation.csv"
RESET_MANIFEST = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_reset" / "reset_manifest.json"
EFFECTIVE_START = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "effective_start" / "vn30_effective_start.csv"
RECONCILIATION = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "reconciliation" / "vn30_listing_date_reconciliation.csv"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark_readiness"
JSON_PATH = REPORT_ROOT / "vn30_2015_benchmark_readiness_manifest.json"
MD_PATH = REPORT_ROOT / "vn30_2015_benchmark_readiness_report.md"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def timestamp_min(values: list[str]) -> str:
    parsed = [pd.to_datetime(value, errors="coerce") for value in values if value]
    parsed = [pd.Timestamp(value) for value in parsed if not pd.isna(value)]
    return "" if not parsed else str(min(parsed))


def timestamp_max(values: list[str]) -> str:
    parsed = [pd.to_datetime(value, errors="coerce") for value in values if value]
    parsed = [pd.Timestamp(value) for value in parsed if not pd.isna(value)]
    return "" if not parsed else str(max(parsed))


def main() -> int:
    stocks = read_csv_rows(STOCK_VALIDATION)
    indices = read_csv_rows(INDEX_VALIDATION)
    effective_rows = read_csv_rows(EFFECTIVE_START)
    reconciliation_rows = read_csv_rows(RECONCILIATION)
    universe_rows = read_csv_rows(UNIVERSE_PATH)
    active_tickers = [row.get("ticker", "") for row in universe_rows if row.get("ticker")]
    usable_stocks = [row for row in stocks if row.get("usable") == "true"]
    missing_tickers = [row.get("ticker", "") for row in stocks if row.get("usable") != "true"]
    index_by_code = {row.get("index_code", ""): row for row in indices}
    vnindex_usable = index_by_code.get("VNINDEX", {}).get("usable") == "true"
    vn30_usable = index_by_code.get("VN30", {}).get("usable") == "true"
    fetched_stocks = [row for row in stocks if row.get("file_exists") == "true" and int(row.get("row_count") or 0) > 0]
    benchmark_ready = len(usable_stocks) == 30 and vnindex_usable and vn30_usable
    reasons: list[str] = []
    if len(usable_stocks) != 30:
        reasons.append(f"usable_tickers={len(usable_stocks)}/30")
    if not vnindex_usable:
        reasons.append("VNINDEX_not_usable")
    if not vn30_usable:
        reasons.append("VN30_index_not_usable")
    reset_payload: dict[str, Any] = {}
    if RESET_MANIFEST.exists():
        reset_payload = json.loads(RESET_MANIFEST.read_text(encoding="utf-8"))
    first_by_ticker = {row.get("ticker", ""): row.get("first_datetime", "") for row in stocks}
    last_by_ticker = {row.get("ticker", ""): row.get("last_datetime", "") for row in stocks}
    effective_by_ticker = {row.get("ticker", ""): row.get("effective_start", "") for row in effective_rows}
    confirmed_listing_tickers = [row.get("ticker", "") for row in effective_rows if row.get("needs_listing_date_verification") == "no"]
    needs_listing_verification = [row.get("ticker", "") for row in effective_rows if row.get("needs_listing_date_verification") == "yes"]
    extra_user_symbols = [row.get("ticker", "") for row in reconciliation_rows if row.get("status") == "extra_user_provided_symbol"]
    usable_index_count = sum(1 for row in indices if row.get("usable") == "true")
    actual_start_any = timestamp_min([row.get("first_datetime", "") for row in stocks if row.get("first_datetime")])
    actual_latest_any = timestamp_max([row.get("last_datetime", "") for row in stocks if row.get("last_datetime")])
    common_latest_usable = timestamp_min([row.get("last_datetime", "") for row in usable_stocks])
    payload: dict[str, Any] = {
        "active_universe_name": "VN30 January 2025 review universe",
        "active_universe_source": "HOSE January 2025 VN30 review",
        "active_universe_effective_period": "03/02/2025 to 01/08/2025",
        "active_universe_count": len(active_tickers),
        "active_universe_tickers": active_tickers,
        "active_universe_includes": ["BCM", "BVH"],
        "active_universe_excludes": ["BSR", "DGC", "VPL"],
        "all_30_tickers_fetched": len(fetched_stocks) == 30,
        "all_30_tickers_usable": len(usable_stocks) == 30,
        "usable_ticker_count": len(usable_stocks),
        "fetched_ticker_count": len(fetched_stocks),
        "missing_tickers": missing_tickers,
        "confirmed_listing_date_tickers": confirmed_listing_tickers,
        "needs_listing_date_verification_tickers": needs_listing_verification,
        "extra_user_provided_symbols": extra_user_symbols,
        "effective_start_by_ticker": effective_by_ticker,
        "vnindex_usable": vnindex_usable,
        "vn30_index_usable": vn30_usable,
        "usable_index_count": usable_index_count,
        "other_indices_usable": {code: row.get("usable") == "true" for code, row in index_by_code.items() if code not in {"VNINDEX", "VN30"}},
        "actual_first_timestamp_by_ticker": first_by_ticker,
        "actual_last_timestamp_by_ticker": last_by_ticker,
        "training_period": "2015-01-01 to 2024-12-31",
        "evaluation_period": f"2025-01-01 to {actual_latest_any}" if actual_latest_any else "2025-01-01 to provider-current unavailable",
        "actual_eval_start": "2025-01-01",
        "actual_data_start_any": actual_start_any,
        "actual_latest_data_timestamp": actual_latest_any,
        "common_latest_usable_data_timestamp": common_latest_usable,
        "benchmark_can_proceed": benchmark_ready,
        "benchmark_command_later": (
            "<repo-approved-venv-python> scripts\\research\\run_vn30_hourly_benchmark_2015_from_gateway.py"
            if benchmark_ready
            else ""
        ),
        "benchmark_not_ready_reasons": reasons,
        "benchmark_run": False,
        "model_training_run": False,
        "paper_or_docx_generated": False,
        "daily_data_used": False,
        "resampling_used": False,
        "reset_manifest": reset_payload,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# VN30 Hourly 2015 Benchmark Readiness",
        "",
        "- Active universe: VN30 January 2025 review universe.",
        "- Active universe source: HOSE January 2025 VN30 review.",
        "- Active universe effective period: 03/02/2025 to 01/08/2025.",
        f"- Active universe count: {len(active_tickers)}.",
        "- Active universe includes: BCM, BVH.",
        "- Active universe excludes: BSR, DGC, VPL.",
        f"- Benchmark can proceed: {str(benchmark_ready).lower()}.",
        f"- Fetched tickers: {len(fetched_stocks)}/30.",
        f"- Usable tickers: {len(usable_stocks)}/30.",
        f"- Usable indices: {usable_index_count}/{len(indices)}.",
        f"- Missing tickers: {', '.join(missing_tickers) if missing_tickers else 'none'}.",
        f"- Confirmed listing-date tickers: {', '.join(confirmed_listing_tickers) if confirmed_listing_tickers else 'none'}.",
        f"- Tickers needing listing-date verification: {', '.join(needs_listing_verification) if needs_listing_verification else 'none'}.",
        f"- Extra user-provided symbols outside frozen universe: {', '.join(extra_user_symbols) if extra_user_symbols else 'none'}.",
        f"- VNINDEX usable: {str(vnindex_usable).lower()}.",
        f"- VN30 index usable: {str(vn30_usable).lower()}.",
        "- Training period: `2015-01-01 to 2024-12-31`.",
        f"- Evaluation period: `{payload['evaluation_period']}`.",
        f"- Actual data start, any fetched ticker: `{actual_start_any}`.",
        f"- Actual latest data timestamp, any fetched ticker: `{actual_latest_any}`.",
        f"- Common latest usable data timestamp: `{common_latest_usable}`.",
        "- Benchmark was run: no.",
        "- Model training was run: no.",
        "- Paper/DOCX generated: no.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "## Decision",
        "",
        "Benchmark may proceed later." if benchmark_ready else f"Benchmark must not proceed yet: {'; '.join(reasons)}.",
        "",
        "## Per-Ticker Actual Timestamps",
        "",
        "| ticker | first datetime | last datetime |",
        "|---|---|---|",
    ]
    for ticker, first in sorted(first_by_ticker.items()):
        lines.append(f"| `{ticker}` | {first} | {last_by_ticker.get(ticker, '')} |")
    lines.extend(["", "## Effective Starts", "", "| ticker | effective_start |", "|---|---|"])
    for ticker, effective_start in sorted(effective_by_ticker.items()):
        lines.append(f"| `{ticker}` | {effective_start} |")
    lines.extend(["", "## Index Usability", "", "| index | usable | rows | first | last |", "|---|---:|---:|---|---|"])
    for row in indices:
        lines.append(f"| `{row.get('index_code', '')}` | {row.get('usable', '')} | {row.get('row_count', '')} | {row.get('first_datetime', '')} | {row.get('last_datetime', '')} |")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"benchmark_can_proceed={str(benchmark_ready).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
