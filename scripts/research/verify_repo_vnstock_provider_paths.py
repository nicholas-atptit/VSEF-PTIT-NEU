"""Verify repo vnstock provider import paths and adapter usage.

This script does not fetch market data. It records interpreter, import, local
shadowing, and repository provider-path observations for VN30 hourly work.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "environment"
MD_REPORT = REPORT_DIR / "repo_vnstock_provider_path_verification.md"
JSON_REPORT = REPORT_DIR / "repo_vnstock_provider_path_verification.json"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _find_shadow_paths() -> list[str]:
    names = ("vnstock_data.py", "vnstock_data", "vnstock.py", "vnstock")
    found: list[str] = []
    for name in names:
        candidate = REPO_ROOT / name
        if candidate.exists():
            found.append(_rel(candidate))
    return found


def _import_status(module_name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(module_name)
    result: dict[str, object] = {
        "module": module_name,
        "spec_found": spec is not None,
        "spec_origin": getattr(spec, "origin", None) if spec else None,
        "import_success": False,
        "module_file": None,
        "version": None,
        "traceback": None,
    }
    try:
        module = importlib.import_module(module_name)
        result["import_success"] = True
        result["module_file"] = getattr(module, "__file__", None)
        result["version"] = getattr(module, "__version__", None)
    except BaseException:
        result["traceback"] = traceback.format_exc()
    return result


def _adapter_status() -> dict[str, object]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    result: dict[str, object] = {
        "path": "src/data/adapters/vnstock_adapter.py",
        "exists": (REPO_ROOT / "src" / "data" / "adapters" / "vnstock_adapter.py").exists(),
        "import_success": False,
        "module_file": None,
        "traceback": None,
    }
    try:
        module = importlib.import_module("src.data.adapters.vnstock_adapter")
        result["import_success"] = True
        result["module_file"] = getattr(module, "__file__", None)
    except BaseException:
        result["traceback"] = traceback.format_exc()
    return result


def _vn30_scripts_bypass_adapter() -> dict[str, object]:
    scripts = sorted((REPO_ROOT / "scripts" / "research").glob("*vn30*hourly*.py"))
    observations: list[dict[str, object]] = []
    bypass = False
    for path in scripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        uses_adapter = "vnstock_adapter" in text or "VnstockAdapter" in text
        imports_direct = (
            "vnstock_data" in text
            or "import vnstock" in text
            or "from vnstock" in text
            or "_load_vnstock" in text
        )
        if imports_direct and not uses_adapter:
            bypass = True
        observations.append(
            {
                "path": _rel(path),
                "uses_repo_adapter": uses_adapter,
                "uses_direct_vnstock_provider": imports_direct,
            }
        )
    return {"bypass_detected": bypass, "scripts": observations}


def _write_reports(report: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Repo vnstock Provider Path Verification",
        "",
        f"- sys.executable: `{report['sys_executable']}`",
        f"- cwd: `{report['cwd']}`",
        f"- local shadowing detected: {'yes' if report['local_shadow_paths'] else 'no'}",
        f"- vnstock_data import success: {'yes' if report['vnstock_data']['import_success'] else 'no'}",
        f"- vnstock import success: {'yes' if report['vnstock']['import_success'] else 'no'}",
        f"- repo adapter file exists: {'yes' if report['repo_adapter']['exists'] else 'no'}",
        f"- repo adapter import success: {'yes' if report['repo_adapter']['import_success'] else 'no'}",
        f"- VN30 scripts bypass repo adapter: {'yes' if report['vn30_scripts']['bypass_detected'] else 'no'}",
        "",
        "## sys.path First 10",
        "",
    ]
    lines.extend(f"{idx + 1}. `{entry}`" for idx, entry in enumerate(report["sys_path_first_10"]))
    lines.extend(
        [
            "",
            "## Recommended Provider Path",
            "",
            "A. repo adapter first",
            "",
            "B. vnstock_data direct second",
            "",
            "C. legacy vnstock fallback third",
            "",
            "## VN30 Hourly Script Observations",
            "",
            "| script | uses repo adapter | uses direct provider |",
            "|---|---:|---:|",
        ]
    )
    for item in report["vn30_scripts"]["scripts"]:
        lines.append(
            f"| `{item['path']}` | {'yes' if item['uses_repo_adapter'] else 'no'} | "
            f"{'yes' if item['uses_direct_vnstock_provider'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Local Shadow Paths",
            "",
        ]
    )
    if report["local_shadow_paths"]:
        lines.extend(f"- `{path}`" for path in report["local_shadow_paths"])
    else:
        lines.append("- none")
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = {
        "sys_executable": sys.executable,
        "cwd": os.getcwd(),
        "sys_path_first_10": sys.path[:10],
        "local_shadow_paths": _find_shadow_paths(),
        "vnstock_data": _import_status("vnstock_data"),
        "vnstock": _import_status("vnstock"),
        "repo_adapter": _adapter_status(),
        "vn30_scripts": _vn30_scripts_bypass_adapter(),
        "recommended_provider_path": [
            "repo adapter first",
            "vnstock_data direct second",
            "legacy vnstock fallback third",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    _write_reports(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
