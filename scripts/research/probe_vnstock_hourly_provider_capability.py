"""Probe vnstock/vnstock_data hourly fetch capability for VN30 full design."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import markdown_table, rel, write_csv  # noqa: E402
from scripts.research.vn30_hourly_vnstock_common import (  # noqa: E402
    FETCH_REPORT_ROOT,
    PROBE_SYMBOLS,
    PROBE_WINDOWS,
    attempt_provider_fetch,
    package_status_rows,
)


FIELDNAMES = [
    "symbol",
    "asset_type",
    "sample_start",
    "sample_end",
    "package",
    "package_version",
    "provider",
    "source",
    "function_used",
    "returned_rows",
    "standardized_rows",
    "returned_columns",
    "first_timestamp",
    "last_timestamp",
    "success",
    "exception_type",
    "exception_message",
]


def probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in PROBE_SYMBOLS:
        for start_date, end_date in PROBE_WINDOWS:
            for result in attempt_provider_fetch(symbol, start_date, end_date):
                row = result.to_log_row()
                row["sample_start"] = row.pop("start_date")
                row["sample_end"] = row.pop("end_date")
                rows.append(row)
    return rows


def support_flag(rows: list[dict[str, Any]], symbol: str) -> bool:
    return any(row.get("symbol") == symbol and str(row.get("success", "")).lower() == "true" for row in rows)


def best_provider(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "none"
    frame["success_bool"] = frame["success"].astype(str).str.lower().eq("true")
    frame["standardized_rows_num"] = pd.to_numeric(frame["standardized_rows"], errors="coerce").fillna(0)
    grouped = (
        frame.groupby(["package", "provider", "source", "function_used"], dropna=False)
        .agg(successes=("success_bool", "sum"), rows=("standardized_rows_num", "sum"))
        .reset_index()
        .sort_values(["successes", "rows"], ascending=[False, False])
    )
    if grouped.empty or int(grouped.iloc[0]["successes"]) == 0:
        return "none"
    top = grouped.iloc[0]
    return f"{top['package']} / {top['source']} / {top['function_used']}"


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    package_rows = package_status_rows()
    stock_supported = support_flag(rows, "ACB") or support_flag(rows, "HPG")
    vnindex_supported = support_flag(rows, "VNINDEX")
    vn30index_supported = support_flag(rows, "VN30INDEX")
    vnxall_supported = support_flag(rows, "VNXALL")
    successes = [row for row in rows if str(row.get("success", "")).lower() == "true"]
    content = [
        "# vnstock Hourly Provider Probe",
        "",
        "## Package Detection",
        "",
        markdown_table(["package", "installed", "version", "origin"], package_rows),
        "",
        "## Probe Decision",
        "",
        f"- Can provider fetch hourly stock data: {stock_supported}.",
        f"- Can provider fetch hourly VNINDEX: {vnindex_supported}.",
        f"- VN30INDEX exact-code support: {vn30index_supported}.",
        f"- VNXALL exact-code support: {vnxall_supported}.",
        f"- Best provider/source/function: {best_provider(rows)}.",
        "- Success requires actual standardized hourly OHLCV rows, not provider availability claims.",
        "",
        "## Successful Attempts",
        "",
        markdown_table(
            [
                "symbol",
                "sample_start",
                "sample_end",
                "package",
                "source",
                "function_used",
                "standardized_rows",
                "first_timestamp",
                "last_timestamp",
            ],
            successes,
            max_rows=80,
        )
        if successes
        else "No provider attempt returned usable hourly OHLCV rows.",
        "",
        "## Attempt Log Preview",
        "",
        markdown_table(
            [
                "symbol",
                "sample_start",
                "sample_end",
                "package",
                "source",
                "function_used",
                "standardized_rows",
                "success",
                "exception_type",
                "exception_message",
            ],
            rows,
            max_rows=120,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    output_dir = FETCH_REPORT_ROOT / "provider_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = probe_rows()
    csv_path = output_dir / "vnstock_hourly_provider_probe.csv"
    report_path = output_dir / "vnstock_hourly_provider_probe.md"
    write_csv(csv_path, rows, fieldnames=FIELDNAMES)
    write_report(report_path, rows)
    print(f"vnstock hourly provider probe complete: attempts={len(rows)} report={rel(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
