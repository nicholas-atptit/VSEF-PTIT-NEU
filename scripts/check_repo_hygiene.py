"""Check tracked repository files for generated artifacts and local residue."""

from __future__ import annotations

import re
import subprocess
import sys
import ast
from pathlib import PurePosixPath


ROOT_FORBIDDEN_DIRS = {"artifacts", "models", "tmp"}
FULL_DATA_BACKUP_ROOTS = {"data", "outputs"}
FULL_DATA_BACKUP_PREFIXES = {
    "archive/generated_data_snapshots/",
    "archive/reports_superseded/",
    "reports/generated/",
}
GENERATED_COMPONENTS = {"__pycache__", "node_modules"}
PYTHON_BYTECODE_SUFFIXES = {".pyc", ".pyo", ".pyd"}
INVALID_FILENAME_CHARS = set('<>:"|?*')
FILE_URI_TOKEN = "file" + ":///"
LOCAL_PATH_RE = re.compile(r"(?<![A-Za-z])\b[A-Za-z]:[\\/]")


CONTENT_ALLOWLIST_PATHS = {
    "reports/CODE_AUDIT_REPORT.md": "Phase 0 immutable baseline records the local execution environment.",
    "reports/CODE_AUDIT_REMEDIATION_PLAN.md": "Phase 0 immutable baseline records the local execution environment.",
}

CONTENT_ALLOWLIST_PREFIXES = {
    "docs/archive/": "historical cleanup/archive evidence retained for audit trail.",
    "docs/audits/": "historical audit evidence retained with original command context.",
    "docs/experiments/": "historical experiment evidence retained with original command context.",
    "docs/reports/": "historical technical report evidence retained with original command context.",
    "reports/forecasting_core/": "governed forecasting evidence pack retained as historical benchmark output.",
    "reports/repeated_seed_1000_smoke_report_pack/": "governed repeated-seed evidence pack retained as historical benchmark output.",
    "reports/risk_aware/": "governed risk-aware evidence pack retained as historical benchmark output.",
}

CONTENT_ALLOWLIST_FILES = {
    "reports/VNSTOCK_AGENT_DATA_GUIDE.md": "documents exact approved local provider interpreter for future agents.",
    "reports/VNSTOCK_AGENT_DATA_GUIDE_SUMMARY.md": "documents exact approved local provider interpreter for future agents.",
    "reports/VNSTOCK_PROVIDER_STANDARDIZATION_GUIDE.md": "documents exact approved local provider interpreter for future agents.",
    "reports/VNSTOCK_DATA_INTERPRETER_FIX_PLAN.md": "documents exact local interpreter remediation commands requested for environment repair.",
    "reports/VSEF_1000_SEED_SMOKE_STABILITY_REPORT.md": "historical benchmark evidence retained.",
    "reports/stress_test_report.md": "historical benchmark evidence retained until report migration is approved.",
    "reports/system_benchmark.md": "historical benchmark evidence retained until report migration is approved.",
}


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_content_allowlisted(path: str) -> bool:
    if path in CONTENT_ALLOWLIST_PATHS or path in CONTENT_ALLOWLIST_FILES:
        return True
    return any(path.startswith(prefix) for prefix in CONTENT_ALLOWLIST_PREFIXES)


def is_full_data_backup_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if parts and parts[0] in FULL_DATA_BACKUP_ROOTS:
        return True
    return any(path.startswith(prefix) for prefix in FULL_DATA_BACKUP_PREFIXES)


def lfs_filter_paths(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    result = subprocess.run(
        ["git", "check-attr", "--stdin", "filter"],
        check=False,
        input="\n".join(paths),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return set()

    lfs_paths: set[str] = set()
    for line in result.stdout.splitlines():
        path, separator, value = line.partition(": filter: ")
        if separator and value.strip() == "lfs":
            if path.startswith('"') and path.endswith('"'):
                try:
                    path = ast.literal_eval(path)
                except (SyntaxError, ValueError):
                    path = path.strip('"')
            lfs_paths.add(path.rstrip("\r\n"))
    return lfs_paths


def filename_violations(path: str) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for component in PurePosixPath(path).parts:
        if component in {"", ".", ".."}:
            violations.append(("malformed_filename", "contains empty or relative path component"))
        if component.strip() != component:
            violations.append(("malformed_filename", "path component has leading or trailing whitespace"))
        if any(char in INVALID_FILENAME_CHARS for char in component):
            violations.append(("malformed_filename", "path component contains Windows-invalid characters"))
        if any(ord(char) < 32 for char in component):
            violations.append(("malformed_filename", "path component contains control characters"))
        if any(0xE000 <= ord(char) <= 0xF8FF for char in component):
            violations.append(("malformed_filename", "path component contains private-use Unicode"))
    return violations


def tracked_artifact_violations(path: str, lfs_paths: set[str]) -> list[tuple[str, str]]:
    parts = PurePosixPath(path).parts
    if not parts:
        return []

    violations: list[tuple[str, str]] = []
    root = parts[0]
    suffix = PurePosixPath(path).suffix.lower()

    if root in ROOT_FORBIDDEN_DIRS:
        violations.append(("tracked_generated_artifact", f"root generated directory is tracked: {root}/"))
    if is_full_data_backup_path(path) and path not in lfs_paths:
        violations.append(
            (
                "tracked_backup_without_lfs",
                "full-data-backup artifact is tracked without Git LFS filter",
            )
        )
    if any(part in GENERATED_COMPONENTS for part in parts):
        category = "tracked_node_modules" if "node_modules" in parts else "tracked_python_cache"
        violations.append((category, "generated dependency/cache directory is tracked"))
    if suffix in PYTHON_BYTECODE_SUFFIXES:
        violations.append(("tracked_python_bytecode", "Python bytecode file is tracked"))
    if any(part.endswith(".egg-info") for part in parts):
        violations.append(("tracked_egg_info", "Python egg-info metadata is tracked"))

    return violations


def scan_content(path: str) -> list[tuple[str, str]]:
    if is_content_allowlisted(path):
        return []
    try:
        raw = open(path, "rb").read()
    except OSError as exc:
        return [("unreadable_tracked_file", str(exc))]
    if b"\0" in raw:
        return []
    text = raw.decode("utf-8", errors="replace")

    violations: list[tuple[str, str]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if FILE_URI_TOKEN in line:
            violations.append(("local_file_uri", f"line {index} contains a local file URI"))
        if LOCAL_PATH_RE.search(line):
            violations.append(("local_absolute_path", f"line {index} contains a local absolute path"))
    return violations


def main() -> int:
    violations: list[tuple[str, str, str]] = []
    paths = git_ls_files()
    lfs_paths = lfs_filter_paths(paths)
    for path in paths:
        for category, message in tracked_artifact_violations(path, lfs_paths):
            violations.append((category, path, message))
        for category, message in filename_violations(path):
            violations.append((category, path, message))
        for category, message in scan_content(path):
            violations.append((category, path, message))

    for category, path, message in violations:
        print(f"{category}: {path}: {message}")

    if violations:
        print(f"Repository hygiene check failed: {len(violations)} violation(s).")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
