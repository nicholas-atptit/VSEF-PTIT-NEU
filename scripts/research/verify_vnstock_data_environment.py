"""Verify vnstock_data importability in the current Python environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "environment"
MARKDOWN_PATH = OUTPUT_DIR / "verify_vnstock_data_environment.md"
JSON_PATH = OUTPUT_DIR / "verify_vnstock_data_environment.json"
SHADOW_CANDIDATES = ("vnstock_data.py", "vnstock_data", "vnstock.py", "vnstock")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "args": args,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def pip_list_vnstock() -> dict[str, Any]:
    result = run_command([sys.executable, "-m", "pip", "list"])
    lines = []
    for line in str(result.get("stdout", "")).splitlines():
        if "vnstock" in line.lower():
            lines.append(line)
    result["stdout"] = "\n".join(lines)
    result["matching_lines"] = lines
    return result


def spec_details(package_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return {"found": False, "origin": "", "submodule_search_locations": []}
    return {
        "found": True,
        "origin": getattr(spec, "origin", "") or "",
        "submodule_search_locations": [
            str(item) for item in (getattr(spec, "submodule_search_locations", None) or [])
        ],
    }


def import_attempt(package_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "")
        if not version:
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                version = ""
        return {
            "success": True,
            "module_file": getattr(module, "__file__", "") or "",
            "version": version,
            "traceback": "",
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "module_file": "",
            "version": "",
            "traceback": traceback.format_exc(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def shadow_checks() -> list[dict[str, Any]]:
    rows = []
    for name in SHADOW_CANDIDATES:
        path = REPO_ROOT / name
        rows.append(
            {
                "candidate": name,
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file" if path.is_file() else "",
                "path": rel(path) if path.exists() else "",
            }
        )
    return rows


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        text = value.replace(str(REPO_ROOT), "<repo>").replace(str(REPO_ROOT).replace("\\", "/"), "<repo>")
        executable_parent = str(Path(sys.executable).resolve().parent)
        text = text.replace(executable_parent, "<python-dir>")
        text = text.replace(executable_parent.replace("\\", "/"), "<python-dir>")
        return "\n".join(line.rstrip() for line in text.splitlines())
    return value


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def build_payload() -> dict[str, Any]:
    return {
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "current_working_directory": os.getcwd(),
        "sys_path_first_10": sys.path[:10],
        "pip_version": run_command([sys.executable, "-m", "pip", "--version"]),
        "pip_show_vnstock_data": run_command([sys.executable, "-m", "pip", "show", "vnstock_data"]),
        "pip_show_vnstock": run_command([sys.executable, "-m", "pip", "show", "vnstock"]),
        "pip_list_vnstock_entries": pip_list_vnstock(),
        "find_spec_vnstock_data": spec_details("vnstock_data"),
        "find_spec_vnstock": spec_details("vnstock"),
        "import_vnstock_data": import_attempt("vnstock_data"),
        "import_vnstock": import_attempt("vnstock"),
        "local_shadow_checks": shadow_checks(),
    }


def write_reports(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = sanitized(payload)
    JSON_PATH.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    shadow_rows = report_payload["local_shadow_checks"]
    content = [
        "# vnstock_data Environment Verification",
        "",
        "## Interpreter",
        "",
        f"- sys.executable: `{report_payload['sys_executable']}`",
        f"- sys.version: `{report_payload['sys_version']}`",
        f"- current working directory: `{report_payload['current_working_directory']}`",
        "",
        "## sys.path First 10",
        "",
        *[f"{index + 1}. `{entry}`" for index, entry in enumerate(report_payload["sys_path_first_10"])],
        "",
        "## pip",
        "",
        "### python -m pip --version",
        "",
        f"- returncode: `{report_payload['pip_version']['returncode']}`",
        "```text",
        report_payload["pip_version"]["stdout"],
        report_payload["pip_version"]["stderr"],
        "```",
        "",
        "### python -m pip show vnstock_data",
        "",
        f"- returncode: `{report_payload['pip_show_vnstock_data']['returncode']}`",
        "```text",
        report_payload["pip_show_vnstock_data"]["stdout"],
        report_payload["pip_show_vnstock_data"]["stderr"],
        "```",
        "",
        "### python -m pip show vnstock",
        "",
        f"- returncode: `{report_payload['pip_show_vnstock']['returncode']}`",
        "```text",
        report_payload["pip_show_vnstock"]["stdout"],
        report_payload["pip_show_vnstock"]["stderr"],
        "```",
        "",
        "### pip list entries matching vnstock",
        "",
        "```text",
        "\n".join(report_payload["pip_list_vnstock_entries"]["matching_lines"]) or "(none)",
        "```",
        "",
        "## importlib Specs",
        "",
        f"- importlib.util.find_spec(\"vnstock_data\"): `{report_payload['find_spec_vnstock_data']}`",
        f"- importlib.util.find_spec(\"vnstock\"): `{report_payload['find_spec_vnstock']}`",
        "",
        "## Imports",
        "",
        f"- import vnstock_data success: `{report_payload['import_vnstock_data']['success']}`",
        f"- import vnstock_data error type: `{report_payload['import_vnstock_data']['error_type']}`",
        f"- import vnstock_data error message: `{report_payload['import_vnstock_data']['error_message']}`",
        "",
        "### vnstock_data traceback",
        "",
        "```text",
        report_payload["import_vnstock_data"]["traceback"] or "(none)",
        "```",
        "",
        f"- import vnstock success: `{report_payload['import_vnstock']['success']}`",
        f"- vnstock module file: `{report_payload['import_vnstock']['module_file']}`",
        f"- vnstock version: `{report_payload['import_vnstock']['version']}`",
        "",
        "## Local Shadow Checks",
        "",
        markdown_table(["candidate", "exists", "kind", "path"], shadow_rows),
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(str(line).rstrip() for line in content), encoding="utf-8")


def print_summary(payload: dict[str, Any]) -> None:
    print("sys.executable:", payload["sys_executable"])
    print("sys.version:", payload["sys_version"])
    print("current working directory:", payload["current_working_directory"])
    print("sys.path first 10:")
    for index, entry in enumerate(payload["sys_path_first_10"], start=1):
        print(f"  {index}. {entry}")
    print("python -m pip --version:")
    print(payload["pip_version"]["stdout"])
    if payload["pip_version"]["stderr"]:
        print(payload["pip_version"]["stderr"])
    print("pip show vnstock_data returncode:", payload["pip_show_vnstock_data"]["returncode"])
    print(payload["pip_show_vnstock_data"]["stdout"] or payload["pip_show_vnstock_data"]["stderr"])
    print("pip show vnstock returncode:", payload["pip_show_vnstock"]["returncode"])
    print(payload["pip_show_vnstock"]["stdout"] or payload["pip_show_vnstock"]["stderr"])
    print("pip list entries matching vnstock:")
    print("\n".join(payload["pip_list_vnstock_entries"]["matching_lines"]) or "(none)")
    print('find_spec("vnstock_data"):', payload["find_spec_vnstock_data"])
    print('find_spec("vnstock"):', payload["find_spec_vnstock"])
    print("import vnstock_data success:", payload["import_vnstock_data"]["success"])
    if not payload["import_vnstock_data"]["success"]:
        print("import vnstock_data traceback:")
        print(payload["import_vnstock_data"]["traceback"])
    print("import vnstock success:", payload["import_vnstock"]["success"])
    if payload["import_vnstock"]["success"]:
        print("vnstock module file:", payload["import_vnstock"]["module_file"])
        print("vnstock version:", payload["import_vnstock"]["version"])
    print("local shadow checks:")
    for row in payload["local_shadow_checks"]:
        print(f"  {row['candidate']}: exists={row['exists']} kind={row['kind']} path={row['path']}")
    print("wrote:", rel(MARKDOWN_PATH))
    print("wrote:", rel(JSON_PATH))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    payload = build_payload()
    write_reports(payload)
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
