"""Build strict VN30 January 2025 benchmark-readiness manifest."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_VALIDATION = REPO_ROOT / "reports" / "generated" / "index_hourly_2015" / "validation" / "index_hourly_2015_validation.csv"
STOCK_VALIDATION = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "validation" / "vn30_hourly_2015_validation.csv"
RESET_MANIFEST = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_reset" / "reset_manifest.json"
EFFECTIVE_START = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "effective_start" / "vn30_effective_start.csv"
RECONCILIATION = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "reconciliation" / "vn30_listing_date_reconciliation.csv"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
EXCLUDED_REFERENCE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_excluded_reference_symbols.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark_readiness"
JSON_PATH = REPORT_ROOT / "vn30_2015_benchmark_readiness_manifest.json"
MD_PATH = REPORT_ROOT / "vn30_2015_benchmark_readiness_report.md"

VN30_EXPECTED_COUNT = 30
REQUIRED_INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
REQUIRED_INCLUSIONS = ("BCM", "BVH")
REQUIRED_EXCLUSIONS = ("BSR", "DGC", "VPL")
JAN2025_TICKERS = (
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
)
DESIGN_START_DATE = "2015-01-01"
TRAINING_CUTOFF = "2024-12-31"
EVALUATION_START = "2025-01-01"
BENCHMARK_COMMAND_RELATIVE = Path("scripts") / "research" / "run_vn30_hourly_benchmark_2015_from_gateway.py"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: str) -> bool:
    return str(value or "").strip().lower() == "true"


def int_value(value: str) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def ticker_list(rows: list[dict[str, str]], field: str) -> list[str]:
    return [str(row.get(field, "")).strip().upper() for row in rows if str(row.get(field, "")).strip()]


def duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def parse_timestamp(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def timestamp_min(values: list[str]) -> str:
    parsed = [ts for ts in (parse_timestamp(value) for value in values) if ts is not None]
    return "" if not parsed else min(parsed).strftime("%Y-%m-%d %H:%M:%S")


def timestamp_max(values: list[str]) -> str:
    parsed = [ts for ts in (parse_timestamp(value) for value in values) if ts is not None]
    return "" if not parsed else max(parsed).strftime("%Y-%m-%d %H:%M:%S")


def load_reset_manifest() -> dict[str, Any]:
    if not RESET_MANIFEST.exists():
        return {}
    return json.loads(RESET_MANIFEST.read_text(encoding="utf-8"))


def build_payload(
    stocks: list[dict[str, str]],
    indices: list[dict[str, str]],
    effective_rows: list[dict[str, str]],
    reconciliation_rows: list[dict[str, str]],
    universe_rows: list[dict[str, str]],
    excluded_reference_rows: list[dict[str, str]] | None = None,
    reset_payload: dict[str, Any] | None = None,
    benchmark_command_path: Path | None = None,
) -> dict[str, Any]:
    active_tickers = ticker_list(universe_rows, "ticker")
    required_tickers = set(active_tickers)
    expected_tickers = set(JAN2025_TICKERS)
    excluded_reference_rows = excluded_reference_rows or []
    configured_excluded = set(ticker_list(excluded_reference_rows, "ticker")) or set(REQUIRED_EXCLUSIONS)
    excluded_reference_tickers = sorted(configured_excluded | set(REQUIRED_EXCLUSIONS))

    stock_tickers = ticker_list(stocks, "ticker")
    validation_tickers = set(stock_tickers)
    usable_validation_tickers = {
        str(row.get("ticker", "")).strip().upper()
        for row in stocks
        if str(row.get("ticker", "")).strip() and bool_value(row.get("usable", ""))
    }
    usable_required_tickers = required_tickers & usable_validation_tickers
    missing_from_validation = sorted(required_tickers - validation_tickers)
    extra_in_validation = sorted(validation_tickers - required_tickers)
    ignored_extra_tickers = sorted(set(extra_in_validation) & set(excluded_reference_tickers))
    unclassified_extra_tickers = sorted(set(extra_in_validation) - set(excluded_reference_tickers))
    unusable_required_tickers = sorted(required_tickers - usable_required_tickers)

    stock_by_ticker = {str(row.get("ticker", "")).strip().upper(): row for row in stocks if row.get("ticker")}
    required_stock_rows = [stock_by_ticker[ticker] for ticker in sorted(required_tickers) if ticker in stock_by_ticker]
    fetched_required = [
        row
        for row in required_stock_rows
        if bool_value(row.get("file_exists", "")) and int_value(row.get("row_count", "")) > 0
    ]

    index_codes = ticker_list(indices, "index_code")
    index_by_code = {str(row.get("index_code", "")).strip().upper(): row for row in indices if row.get("index_code")}
    missing_required_indices = sorted(set(REQUIRED_INDEX_CODES) - set(index_by_code))
    unusable_required_indices = sorted(
        code for code in REQUIRED_INDEX_CODES if code in index_by_code and not bool_value(index_by_code[code].get("usable", ""))
    )
    usable_required_index_count = sum(
        1 for code in REQUIRED_INDEX_CODES if code in index_by_code and bool_value(index_by_code[code].get("usable", ""))
    )
    usable_index_count = sum(1 for row in indices if bool_value(row.get("usable", "")))
    vnindex_usable = bool_value(index_by_code.get("VNINDEX", {}).get("usable", ""))
    vn30_usable = bool_value(index_by_code.get("VN30", {}).get("usable", ""))

    first_by_ticker = {ticker: stock_by_ticker.get(ticker, {}).get("first_datetime", "") for ticker in sorted(required_tickers)}
    last_by_ticker = {ticker: stock_by_ticker.get(ticker, {}).get("last_datetime", "") for ticker in sorted(required_tickers)}
    actual_start_any = timestamp_min([row.get("first_datetime", "") for row in required_stock_rows])
    actual_start_common = timestamp_max([row.get("first_datetime", "") for row in required_stock_rows])
    actual_latest_any = timestamp_max([row.get("last_datetime", "") for row in required_stock_rows])
    actual_latest_common = timestamp_min([row.get("last_datetime", "") for row in required_stock_rows])

    effective_by_ticker = {str(row.get("ticker", "")).strip().upper(): row.get("effective_start", "") for row in effective_rows}
    confirmed_listing_tickers = [
        str(row.get("ticker", "")).strip().upper()
        for row in effective_rows
        if bool_value("true" if row.get("needs_listing_date_verification") == "no" else "false")
    ]
    needs_listing_verification = [
        str(row.get("ticker", "")).strip().upper()
        for row in effective_rows
        if row.get("needs_listing_date_verification") == "yes"
    ]
    reconciliation_reference_tickers = [
        str(row.get("ticker", "")).strip().upper()
        for row in reconciliation_rows
        if row.get("status") in {"excluded_reference_symbol", "extra_user_provided_symbol"}
    ]

    command_path = benchmark_command_path or (REPO_ROOT / BENCHMARK_COMMAND_RELATIVE)
    benchmark_command_exists = command_path.exists()
    command_relative = str(command_path.relative_to(REPO_ROOT)) if command_path.is_absolute() and command_path.is_relative_to(REPO_ROOT) else str(command_path)

    warnings: list[str] = []
    blocking_reasons: list[str] = []
    active_duplicates = duplicate_values(active_tickers)
    validation_duplicates = duplicate_values(stock_tickers)
    index_duplicates = duplicate_values(index_codes)
    active_excluded = sorted(required_tickers & set(REQUIRED_EXCLUSIONS))
    missing_required_inclusions = sorted(set(REQUIRED_INCLUSIONS) - required_tickers)
    active_universe_mismatch = sorted(required_tickers ^ expected_tickers)

    if len(active_tickers) != VN30_EXPECTED_COUNT:
        blocking_reasons.append(f"active_universe_count={len(active_tickers)}/{VN30_EXPECTED_COUNT}")
    if active_duplicates:
        blocking_reasons.append(f"active_universe_duplicate_tickers={','.join(active_duplicates)}")
    if validation_duplicates:
        blocking_reasons.append(f"stock_validation_duplicate_tickers={','.join(validation_duplicates)}")
    if index_duplicates:
        blocking_reasons.append(f"index_validation_duplicate_codes={','.join(index_duplicates)}")
    if missing_from_validation:
        blocking_reasons.append(f"required_tickers_missing_from_validation={','.join(missing_from_validation)}")
    if unusable_required_tickers:
        blocking_reasons.append(f"unusable_required_tickers={','.join(unusable_required_tickers)}")
    if unclassified_extra_tickers:
        blocking_reasons.append(f"unclassified_validation_extra_tickers={','.join(unclassified_extra_tickers)}")
    if active_excluded:
        blocking_reasons.append(f"excluded_tickers_in_active_universe={','.join(active_excluded)}")
    if missing_required_inclusions:
        blocking_reasons.append(f"required_inclusions_missing={','.join(missing_required_inclusions)}")
    if active_universe_mismatch:
        blocking_reasons.append(f"active_universe_set_mismatch={','.join(active_universe_mismatch)}")
    if missing_required_indices:
        blocking_reasons.append(f"required_indices_missing={','.join(missing_required_indices)}")
    if unusable_required_indices:
        blocking_reasons.append(f"required_indices_unusable={','.join(unusable_required_indices)}")
    if usable_required_index_count != len(REQUIRED_INDEX_CODES):
        blocking_reasons.append(f"usable_required_indices={usable_required_index_count}/{len(REQUIRED_INDEX_CODES)}")
    if not vnindex_usable:
        blocking_reasons.append("VNINDEX_not_usable")
    if not vn30_usable:
        blocking_reasons.append("VN30_index_not_usable")
    if not benchmark_command_exists:
        blocking_reasons.append(f"benchmark_command_missing={command_relative}")

    design_start = parse_timestamp(DESIGN_START_DATE)
    actual_common = parse_timestamp(actual_start_common)
    if actual_common and design_start and actual_common.date() > design_start.date():
        warnings.append(
            "benchmark design requested from 2015, but actual hourly availability begins at provider first timestamp"
        )
    if ignored_extra_tickers:
        warnings.append(f"validation extras ignored outside active universe: {','.join(ignored_extra_tickers)}")

    benchmark_ready = not blocking_reasons
    training_actual = (
        f"{actual_start_common} to {TRAINING_CUTOFF}"
        if actual_start_common
        else f"provider hourly start unavailable to {TRAINING_CUTOFF}"
    )
    evaluation_actual = (
        f"{EVALUATION_START} to {actual_latest_common}"
        if actual_latest_common
        else f"{EVALUATION_START} to provider-current unavailable"
    )

    return {
        "active_universe_name": "VN30 January 2025 review universe",
        "active_universe_source": "HOSE January 2025 VN30 review",
        "active_universe_effective_period": "03/02/2025 to 01/08/2025",
        "active_universe_expected_count": VN30_EXPECTED_COUNT,
        "active_universe_actual_count": len(active_tickers),
        "active_universe_count": len(active_tickers),
        "active_universe_tickers": active_tickers,
        "active_universe_duplicate_tickers": active_duplicates,
        "active_universe_set_matches_jan2025": not active_universe_mismatch,
        "required_inclusions": list(REQUIRED_INCLUSIONS),
        "required_exclusions": list(REQUIRED_EXCLUSIONS),
        "active_universe_includes": list(REQUIRED_INCLUSIONS),
        "active_universe_excludes": list(REQUIRED_EXCLUSIONS),
        "excluded_reference_tickers": excluded_reference_tickers,
        "reconciliation_reference_tickers": sorted(set(reconciliation_reference_tickers)),
        "validation_ticker_count": len(validation_tickers),
        "stock_validation_duplicate_tickers": validation_duplicates,
        "required_tickers_missing_from_validation": missing_from_validation,
        "validation_extra_tickers_outside_active_universe": extra_in_validation,
        "ignored_validation_extra_tickers": ignored_extra_tickers,
        "unclassified_validation_extra_tickers": unclassified_extra_tickers,
        "unusable_required_tickers": unusable_required_tickers,
        "all_30_tickers_fetched": len(fetched_required) == VN30_EXPECTED_COUNT,
        "all_30_tickers_usable": len(usable_required_tickers) == VN30_EXPECTED_COUNT,
        "usable_ticker_count": len(usable_required_tickers),
        "fetched_ticker_count": len(fetched_required),
        "missing_tickers": sorted(set(missing_from_validation) | set(unusable_required_tickers)),
        "confirmed_listing_date_tickers": sorted(confirmed_listing_tickers),
        "needs_listing_date_verification_tickers": sorted(needs_listing_verification),
        "effective_start_by_ticker": {ticker: effective_by_ticker.get(ticker, "") for ticker in sorted(required_tickers)},
        "required_index_codes": list(REQUIRED_INDEX_CODES),
        "missing_required_indices": missing_required_indices,
        "unusable_required_indices": unusable_required_indices,
        "vnindex_usable": vnindex_usable,
        "vn30_index_usable": vn30_usable,
        "usable_index_count": usable_index_count,
        "usable_required_index_count": usable_required_index_count,
        "index_usable_policy_met": usable_required_index_count == len(REQUIRED_INDEX_CODES),
        "other_indices_usable": {
            code: index_by_code.get(code, {}).get("usable") == "true"
            for code in REQUIRED_INDEX_CODES
            if code not in {"VNINDEX", "VN30"}
        },
        "actual_first_timestamp_by_ticker": first_by_ticker,
        "actual_last_timestamp_by_ticker": last_by_ticker,
        "design_start_date": DESIGN_START_DATE,
        "actual_hourly_data_start_any": actual_start_any,
        "actual_hourly_data_start_common": actual_start_common,
        "actual_hourly_data_latest_any": actual_latest_any,
        "actual_hourly_data_latest_common": actual_latest_common,
        "actual_data_start_any": actual_start_any,
        "actual_latest_data_timestamp": actual_latest_any,
        "common_latest_usable_data_timestamp": actual_latest_common,
        "training_period_claim": f"{DESIGN_START_DATE} to {TRAINING_CUTOFF}",
        "training_period_actual_available": training_actual,
        "training_period": f"{DESIGN_START_DATE} to {TRAINING_CUTOFF}",
        "evaluation_period_claim": f"{EVALUATION_START} to provider-current/latest available timestamp",
        "evaluation_period_actual_available": evaluation_actual,
        "evaluation_period": evaluation_actual,
        "actual_eval_start": EVALUATION_START,
        "data_availability_disclosure": (
            "2015 design window with provider-available hourly data beginning on "
            f"{actual_start_common or actual_start_any or 'provider first timestamp unavailable'}"
        ),
        "benchmark_command_exists": benchmark_command_exists,
        "benchmark_command_path": command_relative,
        "benchmark_can_proceed": benchmark_ready,
        "benchmark_command_later": f"<repo-approved-venv-python> {command_relative}" if benchmark_ready else "",
        "benchmark_not_ready_reasons": blocking_reasons,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
        "benchmark_run": False,
        "model_training_run": False,
        "paper_or_docx_generated": False,
        "daily_data_used": False,
        "resampling_used": False,
        "reset_manifest": reset_payload or {},
    }


def write_markdown(payload: dict[str, Any], indices: list[dict[str, str]]) -> None:
    warnings = payload["warnings"]
    blockers = payload["blocking_reasons"]
    lines = [
        "# VN30 Hourly 2015 Benchmark Readiness",
        "",
        "- Active universe: VN30 January 2025 review universe.",
        "- Active universe source: HOSE January 2025 VN30 review.",
        "- Active universe effective period: 03/02/2025 to 01/08/2025.",
        f"- Active universe count: {payload['active_universe_actual_count']}/{payload['active_universe_expected_count']}.",
        f"- Active universe tickers: {', '.join(payload['active_universe_tickers'])}.",
        "- Active universe includes: BCM, BVH.",
        "- Active universe excludes: BSR, DGC, VPL.",
        f"- Benchmark can proceed: {str(payload['benchmark_can_proceed']).lower()}.",
        f"- Benchmark command path exists: {str(payload['benchmark_command_exists']).lower()} (`{payload['benchmark_command_path']}`).",
        f"- Fetched required tickers: {payload['fetched_ticker_count']}/{payload['active_universe_expected_count']}.",
        f"- Usable required tickers: {payload['usable_ticker_count']}/{payload['active_universe_expected_count']}.",
        f"- Usable required indices: {payload['usable_required_index_count']}/{len(REQUIRED_INDEX_CODES)}.",
        f"- Missing/unusable tickers: {', '.join(payload['missing_tickers']) if payload['missing_tickers'] else 'none'}.",
        f"- Validation extras outside active universe: {', '.join(payload['validation_extra_tickers_outside_active_universe']) if payload['validation_extra_tickers_outside_active_universe'] else 'none'}.",
        f"- Confirmed listing-date tickers: {', '.join(payload['confirmed_listing_date_tickers']) if payload['confirmed_listing_date_tickers'] else 'none'}.",
        f"- Tickers needing listing-date verification: {', '.join(payload['needs_listing_date_verification_tickers']) if payload['needs_listing_date_verification_tickers'] else 'none'}.",
        f"- VNINDEX usable: {str(payload['vnindex_usable']).lower()}.",
        f"- VN30 index usable: {str(payload['vn30_index_usable']).lower()}.",
        f"- Training period claim: `{payload['training_period_claim']}`.",
        f"- Training period actual available: `{payload['training_period_actual_available']}`.",
        f"- Evaluation period claim: `{payload['evaluation_period_claim']}`.",
        f"- Evaluation period actual available: `{payload['evaluation_period_actual_available']}`.",
        f"- Data availability disclosure: {payload['data_availability_disclosure']}.",
        "- Benchmark was run: no.",
        "- Model training was run: no.",
        "- Paper/DOCX generated: no.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "## Decision",
        "",
        "Benchmark may proceed later." if payload["benchmark_can_proceed"] else "Benchmark must not proceed yet.",
        "",
        "## Blocking Reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in blockers] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(["", "## Per-Ticker Actual Timestamps", "", "| ticker | first datetime | last datetime |", "|---|---|---|"])
    for ticker, first in payload["actual_first_timestamp_by_ticker"].items():
        lines.append(f"| `{ticker}` | {first} | {payload['actual_last_timestamp_by_ticker'].get(ticker, '')} |")
    lines.extend(["", "## Effective Starts", "", "| ticker | effective_start |", "|---|---|"])
    for ticker, effective_start in payload["effective_start_by_ticker"].items():
        lines.append(f"| `{ticker}` | {effective_start} |")
    lines.extend(["", "## Index Usability", "", "| index | usable | rows | first | last |", "|---|---:|---:|---|---|"])
    for row in indices:
        lines.append(
            f"| `{row.get('index_code', '')}` | {row.get('usable', '')} | {row.get('row_count', '')} | {row.get('first_datetime', '')} | {row.get('last_datetime', '')} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    stocks = read_csv_rows(STOCK_VALIDATION)
    indices = read_csv_rows(INDEX_VALIDATION)
    effective_rows = read_csv_rows(EFFECTIVE_START)
    reconciliation_rows = read_csv_rows(RECONCILIATION)
    universe_rows = read_csv_rows(UNIVERSE_PATH)
    excluded_reference_rows = read_csv_rows(EXCLUDED_REFERENCE_PATH)
    payload = build_payload(
        stocks,
        indices,
        effective_rows,
        reconciliation_rows,
        universe_rows,
        excluded_reference_rows=excluded_reference_rows,
        reset_payload=load_reset_manifest(),
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(payload, indices)
    print(f"benchmark_can_proceed={str(payload['benchmark_can_proceed']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
