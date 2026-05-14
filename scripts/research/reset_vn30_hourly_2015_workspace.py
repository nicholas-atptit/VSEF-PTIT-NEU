"""Archive and remove superseded VN30/index hourly generated working artifacts."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_reset"

OLD_GENERATED_PATHS = [
    "reports/generated/vn30_hourly_2005_2026",
    "reports/generated/vn30_hourly_available_window",
    "reports/generated/vn30_hourly_listing_aware",
    "reports/generated/vn30_hourly_vnstock_fetch",
    "reports/generated/vn30_hourly_vnstock_full",
    "reports/generated/vn30_hourly_gateway",
    "reports/generated/index_hourly_fetch",
    "reports/generated/index_hourly_gateway",
    "reports/generated/vn30_gateway_benchmark_readiness",
    "outputs/vn30_hourly_official_2005_2026_traincutoff",
    "outputs/vn30_hourly_available_window_benchmark",
    "outputs/vn30_hourly_listing_aware_traincutoff",
    "outputs/vn30_hourly_vnstock_full_2005_2026_traincutoff",
    "outputs/vn30_hourly_provider_available_traincutoff",
]

OLD_CACHE_PATHS = [
    "data/market_cache/vnstock_data/vn30/hourly_listing_aware",
    "data/market_cache/vnstock_data/vn30/hourly_gateway",
    "data/market_cache/vnstock_data/indices/hourly",
]


def archive_remove(source_rel: str, archive_root: Path, *, timestamp: str, reason: str) -> dict[str, Any]:
    source = REPO_ROOT / source_rel
    destination = archive_root / source_rel
    row = {
        "source_path": source_rel,
        "archive_path": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
        "action": "skipped_missing",
        "reason": reason if not source.exists() else "",
        "timestamp": timestamp,
    }
    if not source.exists():
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination, onexc=clear_readonly)
        else:
            destination.unlink()
    shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
    grant_delete_access(source)
    try:
        shutil.rmtree(source, onexc=clear_readonly) if source.is_dir() else source.unlink()
    except PermissionError as exc:
        row["action"] = "kept"
        row["reason"] = f"{reason}; archived copy created but removal denied by filesystem ACL: {type(exc).__name__}"
        return row
    row["action"] = "archived_removed"
    row["reason"] = reason
    return row


def clear_readonly(function: Any, path: str, exc_info: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def grant_delete_access(path: Path) -> None:
    if os.name != "nt":
        return
    user = os.environ.get("USERNAME")
    if not user:
        return
    subprocess.run(
        ["icacls", str(path), "/grant", f"{user}:(OI)(CI)F", "/T", "/C"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "reset_manifest.json"
    md_path = REPORT_DIR / "reset_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# VN30 Hourly 2015 Workspace Reset",
        "",
        f"- Archive directory: `{payload['archive_directory']}`",
        f"- Raw data touched: {str(payload['raw_data_touched']).lower()}",
        f"- Old cache removed: {str(payload['old_cache_removed']).lower()}",
        f"- Old 2005/2006 generated artifacts removed: {str(payload['old_generated_artifacts_removed']).lower()}",
        "",
        "| source | action | archive path | reason |",
        "|---|---|---|---|",
    ]
    for row in payload["items"]:
        lines.append(f"| `{row['source_path']}` | `{row['action']}` | `{row['archive_path']}` | {row['reason']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = REPO_ROOT / "archive" / "generated_data_snapshots" / f"vn30_hourly_pre_2015_reset_{timestamp}"
    items: list[dict[str, Any]] = []
    for path in OLD_GENERATED_PATHS:
        items.append(
            archive_remove(
                path,
                archive_root,
                timestamp=timestamp,
                reason="superseded generated report/output artifact from pre-2015 design",
            )
        )
    for path in OLD_CACHE_PATHS:
        items.append(
            archive_remove(
                path,
                archive_root,
                timestamp=timestamp,
                reason="superseded normalized cache from pre-2015 VN30/index hourly design",
            )
        )
    old_cache_removed = any(row["action"] == "archived_removed" and row["source_path"] in OLD_CACHE_PATHS for row in items)
    old_generated_removed = any(row["action"] == "archived_removed" and row["source_path"] in OLD_GENERATED_PATHS for row in items)
    payload = {
        "timestamp": timestamp,
        "archive_directory": str(archive_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "raw_data_touched": False,
        "old_cache_removed": old_cache_removed,
        "old_generated_artifacts_removed": old_generated_removed,
        "items": items,
    }
    write_outputs(payload)
    print(f"archive_directory={payload['archive_directory']}")
    print(f"old_generated_artifacts_removed={str(old_generated_removed).lower()}")
    print(f"old_cache_removed={str(old_cache_removed).lower()}")
    print("raw_data_touched=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
