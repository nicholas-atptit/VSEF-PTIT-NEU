"""Static guardrail for VN price-provider usage.

Normal stock/index OHLCV fetching must go through
src.data.providers.vn_price_gateway.fetch_price_history.
"""

from __future__ import annotations

import ast
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUGGESTED_REPLACEMENT = "use src.data.providers.vn_price_gateway.fetch_price_history"

DEFAULT_SCAN_ROOTS = ("src", "scripts", "tests")
ALLOWED_EXACT_PATHS = {
    "src/data/adapters/vnstock_adapter.py": "canonical low-level adapter owns raw vnstock_data access",
    "src/data/providers/vn_price_gateway.py": "canonical gateway owns direct provider fallback",
    "src/data/historical/backdate.py": "legacy daily backdate path; not part of VN30 hourly migration in this task",
    "src/api/streaming/fallback.py": "legacy streaming fallback outside VN30 hourly provider-standardization scope",
    "src/api/streaming/producers/market_data_producer.py": "legacy streaming producer outside VN30 hourly provider-standardization scope",
    "scripts/research/diagnose_vn30_vnstock_hourly_fetch_failures.py": "provider diagnosis script; raw calls are allowed probe behavior",
    "scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py": "legacy benchmark path; must not be run before data gate",
    "scripts/discover_vn100.py": "legacy listing discovery, not OHLCV price history",
    "scripts/extract_market_csv_single.py": "legacy listing extraction, not OHLCV price history",
    "scripts/extract_market_csv_per_ticker.py": "legacy listing extraction, not OHLCV price history",
    "scripts/extract_llm_jsonl.py": "legacy listing extraction, not OHLCV price history",
    "src/api/ui/dashboard.py": "legacy UI fallback outside provider-standardization scope",
    "tests/test_vnstock_news.py": "provider behavior probe test",
}
ALLOWED_GLOBS = (
    "scripts/research/probe_*provider*.py",
    "scripts/research/probe_vnstock*.py",
    "scripts/research/verify_*vnstock*.py",
    "tests/**/*provider*.py",
    "tests/**/*vnstock*.py",
    "tests/*provider*.py",
    "tests/*vnstock*.py",
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    pattern: str
    text: str


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_allowed(path: Path) -> bool:
    rel = _repo_relative(path)
    if rel in ALLOWED_EXACT_PATHS:
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in ALLOWED_GLOBS)


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *DEFAULT_SCAN_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line.endswith(".py")]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def scan_file(path: Path) -> list[Violation]:
    rel = _repo_relative(path)
    if is_allowed(path):
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Violation(rel, exc.lineno or 1, "syntax_error", str(exc))]

    lines = text.splitlines()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"vnstock", "vnstock_data"}:
                    violations.append(Violation(rel, node.lineno, f"import {alias.name}", lines[node.lineno - 1].strip()))
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"vnstock", "vnstock_data"}:
                violations.append(Violation(rel, node.lineno, f"from {node.module} import", lines[node.lineno - 1].strip()))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"Quote", "Vnstock"}:
                violations.append(Violation(rel, node.lineno, f"{name}(", lines[node.lineno - 1].strip()))
            if isinstance(node.func, ast.Attribute) and node.func.attr == "history":
                violations.append(Violation(rel, node.lineno, ".history(", lines[node.lineno - 1].strip()))
    return violations


def scan_paths(paths: list[Path] | None = None) -> list[Violation]:
    files = paths if paths is not None else _tracked_python_files()
    return [violation for path in files for violation in scan_file(path)]


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    paths = [Path(arg) if Path(arg).is_absolute() else REPO_ROOT / arg for arg in argv] if argv else None
    violations = scan_paths(paths)
    for violation in violations:
        print(f"{violation.path}:{violation.line}: forbidden {violation.pattern}: {violation.text}")
        print(f"  suggested replacement: {SUGGESTED_REPLACEMENT}")
    if violations:
        print(f"Provider usage policy failed: {len(violations)} violation(s).")
        return 1
    print("Provider usage policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
