"""Audit VN100 cache coverage for evidence-gap closure.

This script is read-only with respect to official benchmark artifacts. It uses
the existing official cache summaries and local cache files to document which
VN100 tickers can enter a 2025 benchmark without fetching or rerunning the
heavy benchmark pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn100"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "evidence_gap_closure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit VN100 cache coverage from existing local artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def cache_file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "first_date": "", "last_date": ""}

    rows = read_csv(path)
    date_values: list[str] = []
    for row in rows:
        for field in ("timestamp", "date", "time", "trading_date"):
            value = str(row.get(field, "")).strip()
            if value:
                date_values.append(value[:10])
                break
    date_values = sorted(value for value in date_values if value)
    return {
        "exists": True,
        "rows": len(rows),
        "first_date": date_values[0] if date_values else "",
        "last_date": date_values[-1] if date_values else "",
    }


def rows_by_ticker_frequency(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        frequency = str(row.get("frequency", "")).strip().lower()
        if ticker and frequency:
            indexed[(ticker, frequency)] = row
    return indexed


def missing_reason(daily: dict[str, str] | None, hourly: dict[str, str] | None) -> str:
    reasons: list[str] = []
    for frequency, row in (("daily", daily), ("hourly", hourly)):
        if not row:
            reasons.append(f"{frequency}:not_listed_in_official_cache_summary")
            continue
        if not as_bool(row.get("benchmark_usable")):
            reason = str(row.get("benchmark_usable_reason") or row.get("invalid_reason") or "not_usable")
            reasons.append(f"{frequency}:{reason}")
    return " | ".join(reasons) if reasons else "usable_in_at_least_one_frequency"


def build_audit_rows(artifact_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    usable_rows = read_csv(artifact_dir / "usable_cache_summary.csv")
    fetch_rows = read_csv(artifact_dir / "fetch_summary.csv")
    indexed = rows_by_ticker_frequency(usable_rows or fetch_rows)
    tickers = sorted({ticker for ticker, _frequency in indexed})

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        daily_row = indexed.get((ticker, "daily"))
        hourly_row = indexed.get((ticker, "hourly"))
        daily_file = cache_dir / "daily" / f"{ticker}.csv"
        hourly_file = cache_dir / "hourly" / f"{ticker}.csv"
        daily_stats = cache_file_stats(daily_file)
        hourly_stats = cache_file_stats(hourly_file)
        daily_usable = as_bool((daily_row or {}).get("benchmark_usable"))
        hourly_usable = as_bool((hourly_row or {}).get("benchmark_usable"))

        rows.append(
            {
                "ticker": ticker,
                "has_daily_cache_file": daily_stats["exists"],
                "has_hourly_cache_file": hourly_stats["exists"],
                "daily_cache_rows": daily_stats["rows"],
                "hourly_cache_rows": hourly_stats["rows"],
                "daily_cache_first_date": daily_stats["first_date"],
                "daily_cache_last_date": daily_stats["last_date"],
                "hourly_cache_first_date": hourly_stats["first_date"],
                "hourly_cache_last_date": hourly_stats["last_date"],
                "official_daily_actual_start": (daily_row or {}).get("actual_start", ""),
                "official_daily_actual_end": (daily_row or {}).get("actual_end", ""),
                "official_hourly_actual_start": (hourly_row or {}).get("actual_start", ""),
                "official_hourly_actual_end": (hourly_row or {}).get("actual_end", ""),
                "daily_pre_eval_rows": as_int((daily_row or {}).get("pre_eval_rows")),
                "daily_eval_rows": as_int((daily_row or {}).get("eval_rows")),
                "hourly_pre_eval_rows": as_int((hourly_row or {}).get("pre_eval_rows")),
                "hourly_eval_rows": as_int((hourly_row or {}).get("eval_rows")),
                "daily_benchmark_usable_2025": daily_usable,
                "hourly_benchmark_usable_2025": hourly_usable,
                "benchmark_usable_2025": daily_usable or hourly_usable,
                "benchmark_usable_frequency": ",".join(
                    freq for freq, usable in (("daily", daily_usable), ("hourly", hourly_usable)) if usable
                ),
                "effective_daily_start": (daily_row or {}).get("effective_start", ""),
                "effective_daily_end": (daily_row or {}).get("effective_end", ""),
                "effective_hourly_start": (hourly_row or {}).get("effective_start", ""),
                "effective_hourly_end": (hourly_row or {}).get("effective_end", ""),
                "missing_data_reason": missing_reason(daily_row, hourly_row),
                "can_enter_expanded_benchmark": daily_usable or hourly_usable,
            }
        )
    return rows


def evaluated_tickers(artifact_dir: Path) -> set[str]:
    tickers: set[str] = set()
    for frequency in ("daily", "hourly"):
        summary = read_json(artifact_dir / frequency / "benchmark_summary.json")
        tickers.update(str(item).upper() for item in summary.get("evaluated_tickers", []))
    return tickers


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def write_markdown_report(path: Path, artifact_dir: Path, cache_dir: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tickers = {str(row["ticker"]) for row in rows}
    daily_files = [row for row in rows if row["has_daily_cache_file"]]
    hourly_files = [row for row in rows if row["has_hourly_cache_file"]]
    daily_usable = [row for row in rows if row["daily_benchmark_usable_2025"]]
    hourly_usable = [row for row in rows if row["hourly_benchmark_usable_2025"]]
    usable = [row for row in rows if row["benchmark_usable_2025"]]
    current_tickers = evaluated_tickers(artifact_dir)
    new_usable = sorted(str(row["ticker"]) for row in usable if str(row["ticker"]) not in current_tickers)

    top_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["missing_data_reason"])
        top_reasons[reason] = top_reasons.get(reason, 0) + 1
    reason_rows = [
        {"reason": reason, "ticker_count": count}
        for reason, count in sorted(top_reasons.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]

    content = [
        "# VN100 Cache Coverage Audit",
        "",
        "## Source",
        "",
        f"- Official artifact directory: `{rel(artifact_dir)}`.",
        f"- Local cache directory inspected: `{rel(cache_dir)}`.",
        "- No provider fetch, training run, or benchmark rerun was performed.",
        "",
        "## Summary",
        "",
        f"- VN100 tickers considered in official cache summary: {len(tickers)}.",
        f"- Tickers with local daily cache files: {len(daily_files)}.",
        f"- Tickers with local hourly cache files: {len(hourly_files)}.",
        f"- Standalone daily benchmark-usable tickers for 2025: {len(daily_usable)}.",
        f"- Hourly benchmark-usable tickers for 2025: {len(hourly_usable)}.",
        f"- Tickers benchmark-usable in at least one frequency: {len(usable)}.",
        f"- New benchmark-usable tickers beyond the official evaluated set: {len(new_usable)}.",
        "",
        "## Benchmark-Usable Tickers",
        "",
        ", ".join(sorted(str(row["ticker"]) for row in usable)) if usable else "None.",
        "",
        "## Missing/Unusable Reason Concentration",
        "",
        markdown_table(["reason", "ticker_count"], reason_rows),
        "",
        "## Expanded Benchmark Readiness Verdict",
        "",
    ]

    if new_usable:
        content.extend(
            [
                "Existing local summaries show additional benchmark-usable tickers. A heavy official rerun would still be",
                "required to produce expanded benchmark artifacts, and that rerun is outside this phase's constraints.",
                "",
                f"Additional usable tickers: {', '.join(new_usable)}.",
            ]
        )
    else:
        content.extend(
            [
                "The existing official cache summary does not show additional 2025 benchmark-usable tickers beyond the",
                "seven already evaluated tickers. The expanded 2025 benchmark is therefore not generated in this phase.",
            ]
        )

    content.extend(
        [
            "",
            "## Output Files",
            "",
            f"- CSV audit: `{rel(path.with_suffix('.csv'))}`.",
            f"- Missing-evidence note for expanded benchmark: `{rel(path.parent / 'vn100_expanded_benchmark_missing_evidence.md')}`.",
            "",
            "## Claim Boundary",
            "",
            "This audit can support a coverage-readiness statement only. It does not add predictions, improve accuracy,",
            "or establish representativeness for the full VN100 universe.",
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def write_expanded_missing_report(path: Path, artifact_dir: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current_tickers = evaluated_tickers(artifact_dir)
    usable_tickers = {str(row["ticker"]) for row in rows if row["benchmark_usable_2025"]}
    new_usable = sorted(usable_tickers - current_tickers)
    content = [
        "# VN100 Expanded Benchmark Missing Evidence",
        "",
        "## Verdict",
        "",
        "The expanded official 2025 benchmark was not generated in this phase.",
        "",
        "## Reason",
        "",
        "- The phase explicitly disallows heavy benchmark reruns.",
        "- The official cache audit found no additional benchmark-usable 2025 tickers beyond the already evaluated set."
        if not new_usable
        else "- Additional locally usable tickers appear in the cache audit, but producing official expanded artifacts requires a heavy rerun.",
        "- No new provider fetch was performed and no synthetic data source was introduced.",
        "",
        "## Current Evidence",
        "",
        f"- Official evaluated tickers: {', '.join(sorted(current_tickers)) if current_tickers else 'none'}.",
        f"- Existing benchmark-usable tickers from cache audit: {', '.join(sorted(usable_tickers)) if usable_tickers else 'none'}.",
        f"- Additional usable tickers available without a rerun: {', '.join(new_usable) if new_usable else 'none'}.",
        "",
        "## Missing Before Closure",
        "",
        "- A documented, approved cache-expansion or fetch run that increases usable ticker coverage.",
        "- A fresh official benchmark run preserving train_cutoff=2024-12-31, eval_start=2025-01-01, eval_end=2025-12-31, and target_timestamp <= train_cutoff.",
        "- New expanded prediction, benchmark, baseline, regime, and confidence-sweep artifacts.",
        "",
        "## Claim Boundary",
        "",
        "The seven-ticker coverage gap remains open unless a new official expanded benchmark artifact family is produced.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = build_audit_rows(args.artifact_dir, args.cache_dir)
    csv_path = args.output_dir / "vn100_cache_coverage_audit.csv"
    md_path = args.output_dir / "vn100_cache_coverage_audit.md"
    missing_path = args.output_dir / "vn100_expanded_benchmark_missing_evidence.md"
    write_csv(csv_path, rows)
    write_markdown_report(md_path, args.artifact_dir, args.cache_dir, rows)
    write_expanded_missing_report(missing_path, args.artifact_dir, rows)
    print(f"Wrote {rel(csv_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(missing_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
