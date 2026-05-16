"""Build listing-date reconciliation and 2015 effective starts for active VN30."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
LISTING_PATH = REPO_ROOT / "configs" / "universes" / "vn30_listing_dates.csv"
RECON_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "reconciliation"
EFFECTIVE_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "effective_start"
RECON_CSV = RECON_ROOT / "vn30_listing_date_reconciliation.csv"
RECON_MD = RECON_ROOT / "vn30_listing_date_reconciliation.md"
EFFECTIVE_CSV = EFFECTIVE_ROOT / "vn30_effective_start.csv"
EFFECTIVE_MD = EFFECTIVE_ROOT / "vn30_effective_start.md"
BASE_START = date(2015, 1, 1)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: str) -> date | None:
    value = str(value or "").strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_universe() -> list[str]:
    return [row["ticker"].strip().upper() for row in read_csv_rows(UNIVERSE_PATH) if row.get("ticker")]


def read_listing() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(LISTING_PATH)
    return {row["ticker"].strip().upper(): row for row in rows if row.get("ticker")}


def build_reconciliation(universe: list[str], listing: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    active = set(universe)
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(universe) | set(listing)):
        meta = listing.get(ticker, {})
        in_active = ticker in active
        first = str(meta.get("first_trading_date", "")).strip()
        source_note = str(meta.get("source_note", "")).strip()
        metadata_active = str(meta.get("active_in_jan2025_universe", "")).strip().lower()
        needs_verification = str(meta.get("needs_verification", "")).strip().lower()
        if in_active and first:
            status = "confirmed_listing_date"
        elif in_active:
            status = "needs_verification"
        elif metadata_active == "true":
            status = "metadata_active_mismatch"
        else:
            status = "excluded_reference_symbol"
        rows.append(
            {
                "ticker": ticker,
                "in_active_universe": str(in_active).lower(),
                "in_listing_metadata": str(ticker in listing).lower(),
                "active_in_jan2025_universe_metadata": metadata_active,
                "first_trading_date": first,
                "source_note": source_note,
                "needs_verification": needs_verification,
                "status": status,
                "active_universe_changed": "false",
            }
        )
    return rows


def build_effective_starts(universe: list[str], listing: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in universe:
        meta = listing.get(ticker, {})
        first = parse_date(meta.get("first_trading_date", ""))
        source_note = str(meta.get("source_note", "") or "needs_verification").strip()
        if first is None:
            effective = BASE_START
            reason = "missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp"
            needs = "yes"
        elif first > BASE_START:
            effective = first
            reason = "listed_after_2015_use_first_trading_date"
            needs = "no"
        else:
            effective = BASE_START
            reason = "listed_before_2015_use_2015_start"
            needs = "no"
        rows.append(
            {
                "ticker": ticker,
                "first_trading_date": "" if first is None else first.isoformat(),
                "effective_start": effective.isoformat(),
                "effective_start_reason": reason,
                "source_note": source_note if first is not None else f"{source_note}/provider_first_timestamp",
                "needs_listing_date_verification": needs,
            }
        )
    return rows


def write_reconciliation_report(rows: list[dict[str, Any]], universe: list[str], listing: dict[str, dict[str, str]]) -> None:
    confirmed = [row["ticker"] for row in rows if row["status"] == "confirmed_listing_date"]
    needs = [row["ticker"] for row in rows if row["status"] == "needs_verification"]
    excluded = [row["ticker"] for row in rows if row["status"] == "excluded_reference_symbol"]
    mismatches = [row["ticker"] for row in rows if row["status"] == "metadata_active_mismatch"]
    missing_from_user = [ticker for ticker in universe if listing.get(ticker, {}).get("source_note") != "user_provided"]
    lines = [
        "# VN30 Listing-Date Reconciliation",
        "",
        "- Active universe: VN30 January 2025 review universe.",
        f"- Active universe ticker count: {len(universe)}.",
        f"- Listing metadata ticker count: {len(listing)}.",
        f"- Tickers with confirmed listing dates: {len(confirmed)}.",
        f"- Tickers needing verification: {len(needs)}.",
        f"- Reference symbols outside active universe: {', '.join(excluded) if excluded else 'none'}.",
        f"- Metadata active mismatches: {', '.join(mismatches) if mismatches else 'none'}.",
        f"- Active-universe tickers missing from user-provided table: {', '.join(missing_from_user) if missing_from_user else 'none'}.",
        "- Active universe changed: no.",
        "",
        "| ticker | status | first_trading_date | source_note | in_active_universe | metadata_active | needs_verification |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['ticker']}` | `{row['status']}` | {row['first_trading_date']} | `{row['source_note']}` | {row['in_active_universe']} | {row['active_in_jan2025_universe_metadata']} | {row['needs_verification']} |"
        )
    RECON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_effective_report(rows: list[dict[str, Any]]) -> None:
    needs = [row["ticker"] for row in rows if row["needs_listing_date_verification"] == "yes"]
    lines = [
        "# VN30 2015 Effective Starts",
        "",
        "- Active universe: VN30 January 2025 review universe.",
        "- Rule: `effective_start(ticker) = max(2015-01-01, first_trading_date)`.",
        f"- Active tickers needing listing-date verification: {', '.join(needs) if needs else 'none'}.",
        "",
        "| ticker | first_trading_date | effective_start | reason | needs_verification |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['ticker']}` | {row['first_trading_date']} | {row['effective_start']} | `{row['effective_start_reason']}` | {row['needs_listing_date_verification']} |"
        )
    EFFECTIVE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    universe = read_universe()
    listing = read_listing()
    reconciliation = build_reconciliation(universe, listing)
    effective = build_effective_starts(universe, listing)
    write_csv(
        RECON_CSV,
        reconciliation,
        [
            "ticker",
            "in_active_universe",
            "in_listing_metadata",
            "active_in_jan2025_universe_metadata",
            "first_trading_date",
            "source_note",
            "needs_verification",
            "status",
            "active_universe_changed",
        ],
    )
    write_csv(
        EFFECTIVE_CSV,
        effective,
        ["ticker", "first_trading_date", "effective_start", "effective_start_reason", "source_note", "needs_listing_date_verification"],
    )
    write_reconciliation_report(reconciliation, universe, listing)
    write_effective_report(effective)
    print(f"active_universe_count={len(universe)}")
    print(f"listing_metadata_count={len(listing)}")
    print("active_universe_changed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
