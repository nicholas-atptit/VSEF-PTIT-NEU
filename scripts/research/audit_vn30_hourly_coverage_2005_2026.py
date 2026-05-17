"""Audit frozen VN30 hourly data coverage for the 2005-2026 NCKH design."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    EVAL_END_TEXT,
    EVAL_START_TEXT,
    REPORT_ROOT,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START_TEXT,
    audit_hourly_coverage,
    failed_tickers_from_audit,
    markdown_table,
    rel,
    usable_tickers_from_audit,
    write_csv,
)


DEFAULT_OUTPUT_DIR = REPORT_ROOT / "audit"
AUDIT_COLUMNS = [
    "ticker",
    "first_available_hourly_timestamp",
    "last_available_hourly_timestamp",
    "hourly_rows",
    "training_rows_2005_2024",
    "evaluation_rows_2025_2026",
    "missing_training_coverage",
    "missing_evaluation_coverage",
    "benchmark_usable",
    "missing_reason",
    "raw_hourly_sources",
    "raw_hourly_files",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen VN30 hourly coverage for 2005-2026.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete-exit-zero",
        action="store_true",
        help="Write the audit but return zero even when fewer than 30 tickers are benchmark-usable.",
    )
    return parser.parse_args()


def write_markdown_report(path: Path, rows: list[dict[str, object]]) -> None:
    usable = usable_tickers_from_audit(rows)
    failed = failed_tickers_from_audit(rows)
    reason_counts: dict[str, int] = {}
    for row in failed:
        reason = str(row.get("missing_reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_rows = [
        {"missing_reason": reason, "ticker_count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    headers = [
        "ticker",
        "first_available_hourly_timestamp",
        "last_available_hourly_timestamp",
        "hourly_rows",
        "missing_training_coverage",
        "missing_evaluation_coverage",
        "benchmark_usable",
        "missing_reason",
    ]
    content = [
        "# VN30 Hourly Coverage Audit 2005-2026",
        "",
        "## Scope",
        "",
        "- Universe: frozen VN30, exactly 30 tickers from `configs/universes/vn30_constituents_frozen.csv`.",
        "- Frequency: hourly only.",
        f"- Historical/training period: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation/comparison period: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Daily data and daily-to-hourly resampling are not used.",
        "",
        "## Summary",
        "",
        f"- Benchmark-usable tickers: {len(usable)} of 30.",
        f"- Missing/unusable tickers: {len(failed)} of 30.",
        f"- Full requested VN30 hourly design feasible: {str(len(usable) == 30).lower()}.",
        "",
        "## Benchmark-Usable Tickers",
        "",
        ", ".join(usable) if usable else "None.",
        "",
        "## Missing Reason Concentration",
        "",
        markdown_table(["missing_reason", "ticker_count"], reason_rows) if reason_rows else "None.",
        "",
        "## Per-Ticker Audit",
        "",
        markdown_table(headers, rows),
        "",
        "## Boundary",
        "",
        "If fewer than 30 tickers are benchmark-usable, the benchmark and paper must stop before final VN30 claims.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = audit_hourly_coverage()
    output_dir = args.output_dir
    csv_path = output_dir / "vn30_hourly_coverage_audit.csv"
    md_path = output_dir / "vn30_hourly_coverage_audit.md"
    write_csv(csv_path, rows, fieldnames=AUDIT_COLUMNS)
    write_markdown_report(md_path, rows)
    usable_count = len(usable_tickers_from_audit(rows))
    print(
        "VN30 hourly coverage audit complete: "
        f"usable={usable_count}/30 csv={rel(csv_path)} report={rel(md_path)}"
    )
    if usable_count < 30 and not args.allow_incomplete_exit_zero:
        print(
            "VN30 hourly coverage audit failed: fewer than 30 tickers are benchmark-usable "
            "for the requested 2005-2026 hourly design.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
