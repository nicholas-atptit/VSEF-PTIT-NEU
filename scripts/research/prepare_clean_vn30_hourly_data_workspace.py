"""Archive old VN30/index hourly generated artifacts before gateway refetch."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_clean_workspace"

REPORT_OUTPUT_PATHS = [
    "reports/generated/vn30_hourly_2005_2026",
    "reports/generated/vn30_hourly_available_window",
    "reports/generated/vn30_hourly_listing_aware",
    "reports/generated/vn30_hourly_vnstock_fetch",
    "reports/generated/vn30_hourly_vnstock_full",
    "reports/generated/index_hourly_fetch",
    "outputs/vn30_hourly_official_2005_2026_traincutoff",
    "outputs/vn30_hourly_available_window_benchmark",
    "outputs/vn30_hourly_listing_aware_traincutoff",
    "outputs/vn30_hourly_vnstock_full_2005_2026_traincutoff",
    "outputs/vn30_hourly_provider_available_traincutoff",
]
CACHE_PATHS = [
    "data/market_cache/vnstock_data/vn30/hourly_listing_aware",
    "data/market_cache/vnstock_data/indices/hourly",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", help="Copy sources into archive instead of moving them.")
    parser.add_argument("--archive-cache", action="store_true", help="Also archive generated normalized cache directories.")
    return parser.parse_args()


def _archive_one(source_rel: str, archive_root: Path, *, copy: bool, reason: str, timestamp: str) -> dict[str, Any]:
    source = REPO_ROOT / source_rel
    destination = archive_root / source_rel
    row = {
        "source_path": source_rel,
        "archive_path": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
        "action": "skipped",
        "reason": "",
        "timestamp": timestamp,
    }
    if not source.exists():
        row["reason"] = "source path does not exist"
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        row["action"] = "copied"
    else:
        shutil.move(str(source), str(destination))
        row["action"] = "moved"
    row["reason"] = reason
    return row


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "clean_workspace_manifest.json"
    md_path = REPORT_DIR / "clean_workspace_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = payload["items"]
    lines = [
        "# VN30 Hourly Clean Workspace Manifest",
        "",
        f"- Archive directory: `{payload['archive_directory']}`",
        f"- Cache archived: {str(payload['cache_archived']).lower()}",
        f"- Raw data touched: {str(payload['raw_data_touched']).lower()}",
        f"- Archive mode: `{payload['mode']}`",
        "",
        "| source | action | archive path | reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['source_path']}` | `{row['action']}` | `{row['archive_path']}` | {row['reason']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = REPO_ROOT / "archive" / "generated_data_snapshots" / f"vn30_hourly_pre_benchmark_{timestamp}"
    items: list[dict[str, Any]] = []
    for path in REPORT_OUTPUT_PATHS:
        items.append(
            _archive_one(
                path,
                archive_root,
                copy=args.copy,
                reason="old generated report/output artifact archived before gateway refetch",
                timestamp=timestamp,
            )
        )
    if args.archive_cache:
        for path in CACHE_PATHS:
            items.append(
                _archive_one(
                    path,
                    archive_root,
                    copy=args.copy,
                    reason="generated normalized cache archived because --archive-cache was passed",
                    timestamp=timestamp,
                )
            )
    payload = {
        "timestamp": timestamp,
        "archive_directory": str(archive_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "mode": "copy" if args.copy else "move",
        "cache_archived": bool(args.archive_cache),
        "raw_data_touched": False,
        "items": items,
    }
    write_outputs(payload)
    print(f"archive_directory={payload['archive_directory']}")
    print(f"items={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
