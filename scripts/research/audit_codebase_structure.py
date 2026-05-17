"""Inventory repository code layout for cleanup planning.

This script performs a read-only scan of code directories and writes a
structure inventory report. It does not delete or move files.
"""

from __future__ import annotations

import ast
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "scripts", "tests", "configs", "config")
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "code_cleanup"
CSV_PATH = OUTPUT_DIR / "codebase_structure_inventory.csv"
MD_PATH = OUTPUT_DIR / "codebase_structure_inventory.md"

RAW_PROVIDER_PATTERNS = (
    "vnstock",
    "vnstock_data",
)
GATEWAY_IMPORT = "src.data.providers.vn_price_gateway"
LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"([A-Za-z]:\\(?:Users|Repos|Data|Temp|tmp)\\|/(?:Users|home|mnt|tmp)/)"
)
TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b", re.IGNORECASE)
LEGACY_NAME_RE = re.compile(
    r"(2005_2026|failed|target60|target65|final65|optimization|sweep|paper|"
    r"vnstock_|from_vnstock|probe|diagnose|reset_|refetch_)"
)
ACTIVE_NAME_RE = re.compile(
    r"(canonical_eval|available_window|run_vn30_daily_2015_benchmark|"
    r"run_supported_indices_directional_benchmark|index_benchmark_common|"
    r"validate_supported_indices_benchmark_readiness|audit_supported_indices|"
    r"fetch_supported_indices_daily_gateway_2015|fetch_vn30_daily_gateway_2015|"
    r"vn30_hourly_2015_effective_start|vn30_hourly_2015_fetch_plan)"
)


@dataclass(frozen=True)
class FileRecord:
    path: Path
    rel: str
    directory: str
    line_count: int
    imports: tuple[str, ...]
    imported_names: tuple[str, ...]
    has_raw_provider_import: bool
    has_gateway_import: bool
    has_local_absolute_path: bool
    has_todo_marker: bool
    over_500_lines: bool
    has_main_entrypoint: bool
    likely_legacy: bool
    likely_active: bool


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        files.extend(
            p
            for p in root.rglob("*.py")
            if "__pycache__" not in p.parts and ".venv" not in p.parts
        )
    return sorted(files)


def module_name_for(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_imports(path: Path, text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return (), ()

    modules: list[str] = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
                names.append(node.module.split(".")[0])
            for alias in node.names:
                if alias.name != "*":
                    names.append(alias.name)
    return tuple(sorted(set(modules))), tuple(sorted(set(names)))


def has_main_entrypoint(text: str) -> bool:
    return '__name__ == "__main__"' in text or "__name__ == '__main__'" in text


def read_record(path: Path) -> FileRecord:
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    imports, imported_names = parse_imports(path, text)
    line_count = text.count("\n") + (1 if text else 0)
    import_blob = "\n".join(imports)
    has_raw_provider_import = any(
        module in import_blob
        for module in RAW_PROVIDER_PATTERNS
    ) and GATEWAY_IMPORT not in import_blob

    return FileRecord(
        path=path,
        rel=rel,
        directory=path.parent.relative_to(REPO_ROOT).as_posix(),
        line_count=line_count,
        imports=imports,
        imported_names=imported_names,
        has_raw_provider_import=has_raw_provider_import,
        has_gateway_import=GATEWAY_IMPORT in import_blob,
        has_local_absolute_path=bool(LOCAL_ABSOLUTE_PATH_RE.search(text)),
        has_todo_marker=bool(TODO_RE.search(text)),
        over_500_lines=line_count > 500,
        has_main_entrypoint=has_main_entrypoint(text),
        likely_legacy=bool(LEGACY_NAME_RE.search(path.name))
        or "legacy" in path.parts,
        likely_active=bool(ACTIVE_NAME_RE.search(path.name))
        or rel.startswith("src/data/providers/")
        or rel.startswith("src/data/adapters/")
        or rel in {
            "scripts/check_provider_usage_policy.py",
            "scripts/check_repo_hygiene.py",
            "scripts/check_runtime_preflight.py",
        },
    )


def import_graph(records: list[FileRecord]) -> tuple[set[str], dict[str, set[str]]]:
    by_module = {module_name_for(record.path): record.rel for record in records}
    by_leaf = defaultdict(set)
    for module, rel in by_module.items():
        by_leaf[module.rsplit(".", 1)[-1]].add(rel)

    active_roots = [record for record in records if record.likely_active]
    referenced: set[str] = set()
    reverse_refs: dict[str, set[str]] = defaultdict(set)
    for record in active_roots:
        for imported in record.imports:
            candidates = []
            if imported in by_module:
                candidates.append(by_module[imported])
            leaf = imported.rsplit(".", 1)[-1]
            candidates.extend(sorted(by_leaf.get(leaf, set())))
            for candidate in candidates:
                referenced.add(candidate)
                reverse_refs[candidate].add(record.rel)
    for record in active_roots:
        referenced.add(record.rel)
        reverse_refs[record.rel].add(record.rel)
    return referenced, reverse_refs


def write_csv(records: list[FileRecord], reverse_refs: dict[str, set[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "path",
                "directory",
                "line_count",
                "imports_count",
                "referenced_by_active_code",
                "active_referrers",
                "has_raw_provider_import",
                "has_gateway_import",
                "has_local_absolute_path",
                "has_todo_fixme_hack",
                "over_500_lines",
                "has_main_entrypoint",
                "likely_legacy",
                "likely_active",
            ],
        )
        writer.writeheader()
        for record in records:
            referrers = sorted(reverse_refs.get(record.rel, set()))
            writer.writerow(
                {
                    "path": record.rel,
                    "directory": record.directory,
                    "line_count": record.line_count,
                    "imports_count": len(record.imports),
                    "referenced_by_active_code": bool(referrers),
                    "active_referrers": "; ".join(referrers),
                    "has_raw_provider_import": record.has_raw_provider_import,
                    "has_gateway_import": record.has_gateway_import,
                    "has_local_absolute_path": record.has_local_absolute_path,
                    "has_todo_fixme_hack": record.has_todo_marker,
                    "over_500_lines": record.over_500_lines,
                    "has_main_entrypoint": record.has_main_entrypoint,
                    "likely_legacy": record.likely_legacy,
                    "likely_active": record.likely_active,
                }
            )


def bullet_paths(records: list[FileRecord], predicate, limit: int = 40) -> list[str]:
    paths = [record.rel for record in records if predicate(record)]
    if not paths:
        return ["- None found."]
    lines = [f"- `{path}`" for path in paths[:limit]]
    if len(paths) > limit:
        lines.append(f"- ... {len(paths) - limit} more in CSV.")
    return lines


def write_markdown(records: list[FileRecord], reverse_refs: dict[str, set[str]]) -> None:
    directory_counts = Counter(record.directory for record in records)
    duplicate_names = {
        name: sorted(record.rel for record in group)
        for name, group in group_by_name(records).items()
        if len(group) > 1
    }
    top_level_scripts = [
        r for r in records if r.path.parent == REPO_ROOT / "scripts"
    ]
    research_scripts = [
        r for r in records if r.rel.startswith("scripts/research/")
    ]
    legacy_scripts = [
        r for r in records if r.rel.startswith("scripts/legacy/")
    ]
    no_active_imports = [
        r for r in records if r.rel not in reverse_refs and r.rel.startswith("scripts/")
    ]

    lines: list[str] = [
        "# Codebase Structure Inventory",
        "",
        "Generated by `scripts/research/audit_codebase_structure.py`.",
        "",
        "## Summary",
        "",
        f"- Python files scanned: {len(records)}",
        f"- Top-level scripts count: {len(top_level_scripts)}",
        f"- Research scripts count: {len(research_scripts)}",
        f"- Legacy scripts count: {len(legacy_scripts)}",
        f"- Duplicate module basenames: {len(duplicate_names)}",
        f"- Scripts with no imports from likely active code: {len(no_active_imports)}",
        f"- Files importing raw vnstock/vnstock_data directly: {sum(r.has_raw_provider_import for r in records)}",
        f"- Files importing provider gateway correctly: {sum(r.has_gateway_import for r in records)}",
        f"- Files containing local absolute paths: {sum(r.has_local_absolute_path for r in records)}",
        f"- Files containing TODO/FIXME/HACK: {sum(r.has_todo_marker for r in records)}",
        f"- Files over 500 lines: {sum(r.over_500_lines for r in records)}",
        f"- Files with `__main__` entrypoints: {sum(r.has_main_entrypoint for r in records)}",
        f"- Likely legacy scripts/files: {sum(r.likely_legacy for r in records)}",
        f"- Likely active scripts/files: {sum(r.likely_active for r in records)}",
        "",
        "## Python File Count by Directory",
        "",
    ]
    lines.extend(
        f"- `{directory}`: {count}"
        for directory, count in sorted(directory_counts.items())
    )

    lines.extend(["", "## Duplicate Module Names", ""])
    if duplicate_names:
        for name, paths in sorted(duplicate_names.items()):
            lines.append(f"- `{name}`: {', '.join(f'`{p}`' for p in paths)}")
    else:
        lines.append("- None found.")

    sections = [
        (
            "Files With No Imports From Likely Active Code",
            no_active_imports,
        ),
        (
            "Files Importing Raw vnstock/vnstock_data Directly",
            [r for r in records if r.has_raw_provider_import],
        ),
        (
            "Files Importing Provider Gateway Correctly",
            [r for r in records if r.has_gateway_import],
        ),
        (
            "Files Containing Local Absolute Paths",
            [r for r in records if r.has_local_absolute_path],
        ),
        (
            "Files Containing TODO/FIXME/HACK",
            [r for r in records if r.has_todo_marker],
        ),
        (
            "Files Over 500 Lines",
            [r for r in records if r.over_500_lines],
        ),
        (
            "Files With __main__ Entrypoints",
            [r for r in records if r.has_main_entrypoint],
        ),
        (
            "Likely Legacy Scripts",
            [r for r in records if r.likely_legacy],
        ),
        (
            "Likely Active Scripts",
            [r for r in records if r.likely_active],
        ),
    ]
    for heading, section_records in sections:
        lines.extend(["", f"## {heading}", ""])
        if not section_records:
            lines.append("- None found.")
        else:
            lines.extend(f"- `{record.rel}`" for record in section_records[:80])
            if len(section_records) > 80:
                lines.append(f"- ... {len(section_records) - 80} more in CSV.")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_by_name(records: list[FileRecord]) -> dict[str, list[FileRecord]]:
    grouped: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        grouped[record.path.stem].append(record)
    return grouped


def main() -> int:
    records = [read_record(path) for path in iter_python_files()]
    _, reverse_refs = import_graph(records)
    write_csv(records, reverse_refs)
    write_markdown(records, reverse_refs)
    print(f"Wrote {CSV_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote {MD_PATH.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
