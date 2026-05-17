"""Runtime reproducibility preflight for local VSEF workspaces.

The script is intentionally read-only. It checks package availability, configured
service endpoints, and expected local artifact roots, then reports explicit
warnings for missing optional or externally managed dependencies.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 11)
SOCKET_TIMEOUT_SECONDS = 0.75

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class CheckResult:
    category: str
    name: str
    status: str
    detail: str


def _status_icon(status: str) -> str:
    return {"OK": "OK", "WARN": "WARN", "FAIL": "FAIL"}.get(status, status)


def _print_results(results: Iterable[CheckResult]) -> tuple[int, int, int]:
    ok = warn = fail = 0
    for result in results:
        if result.status == "OK":
            ok += 1
        elif result.status == "WARN":
            warn += 1
        elif result.status == "FAIL":
            fail += 1
        print(f"[{_status_icon(result.status)}] {result.category}: {result.name} - {result.detail}")
    print(f"SUMMARY: ok={ok} warn={warn} fail={fail}")
    return ok, warn, fail


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _check_python() -> list[CheckResult]:
    version = sys.version_info
    version_text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) < MIN_PYTHON:
        return [
            CheckResult(
                "python",
                "version",
                "FAIL",
                f"Python {version_text}; requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            )
        ]
    return [CheckResult("python", "version", "OK", f"Python {version_text}")]


def _check_imports() -> list[CheckResult]:
    required = [
        ("sqlalchemy", "sqlalchemy"),
        ("asyncpg", "asyncpg"),
        ("psycopg2", "psycopg2"),
        ("alembic", "alembic"),
        ("chromadb", "chromadb"),
        ("pandas", "pandas"),
        ("pandas-ta", "pandas_ta"),
        ("numpy", "numpy"),
        ("statsmodels", "statsmodels"),
        ("arch", "arch"),
        ("httpx", "httpx"),
        ("aiohttp", "aiohttp"),
        ("redis", "redis"),
        ("msgpack", "msgpack"),
        ("pydantic-settings", "pydantic_settings"),
        ("python-dotenv", "dotenv"),
        ("structlog", "structlog"),
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml"),
        ("vnstock_data", "vnstock_data"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("joblib", "joblib"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("openai", "openai"),
        ("langchain-text-splitters", "langchain_text_splitters"),
        ("rich", "rich"),
        ("torch", "torch"),
    ]
    optional = [
        ("dnse", "dnse"),
        ("sentence-transformers", "sentence_transformers"),
        ("gymnasium", "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("legacy vnstock", "vnstock"),
        ("google-generativeai", "google.generativeai"),
    ]

    results: list[CheckResult] = []
    for label, module in required:
        status = "OK" if _module_available(module) else "WARN"
        detail = "import spec found" if status == "OK" else "not importable in this environment"
        results.append(CheckResult("required_import", label, status, detail))

    for label, module in optional:
        status = "OK" if _module_available(module) else "WARN"
        detail = "import spec found" if status == "OK" else "optional package not importable"
        results.append(CheckResult("optional_import", label, status, detail))

    return results


def _parse_host_port(raw: str, default_port: int | None = None) -> tuple[str | None, int | None]:
    if not raw:
        return None, None
    if "://" in raw:
        parsed = urlparse(raw)
        return parsed.hostname, parsed.port or default_port
    if ":" in raw:
        host, port = raw.rsplit(":", 1)
        try:
            return host, int(port)
        except ValueError:
            return host, default_port
    return raw, default_port


def _socket_check(name: str, host: str | None, port: int | None) -> CheckResult:
    if not host or not port:
        return CheckResult("service", name, "WARN", "not configured")
    try:
        with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT_SECONDS):
            return CheckResult("service", name, "OK", f"reachable at {host}:{port}")
    except OSError as exc:
        return CheckResult("service", name, "WARN", f"configured at {host}:{port}, not reachable: {exc}")


def _check_services() -> list[CheckResult]:
    try:
        from config.settings import get_settings

        settings = get_settings()
        db_host = getattr(settings, "timescale_host", None)
        db_port = getattr(settings, "timescale_port", None)
        redis_host, redis_port = _parse_host_port(getattr(settings, "redis_url", ""), default_port=6379)
        kafka_host, kafka_port = _parse_host_port(getattr(settings, "kafka_broker_url", ""), default_port=9092)
        chroma_host = getattr(settings, "chroma_host", None)
        chroma_port = getattr(settings, "chroma_port", None)
        ollama_host, ollama_port = _parse_host_port(getattr(settings, "ollama_base_url", ""), default_port=11434)
    except Exception as exc:
        return [CheckResult("service", "settings", "WARN", f"could not load config.settings: {exc}")]

    return [
        _socket_check("database", db_host, int(db_port) if db_port else None),
        _socket_check("redis", redis_host, redis_port),
        _socket_check("kafka", kafka_host, kafka_port),
        _socket_check("vector_store_chroma", chroma_host, int(chroma_port) if chroma_port else None),
        _socket_check("local_llm_ollama", ollama_host, ollama_port),
    ]


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def _check_artifacts() -> list[CheckResult]:
    checks = [
        ("data_raw", ROOT / "data" / "raw", "*.csv", "local raw OHLCV cache"),
        ("data_daily_market_split", ROOT / "data" / "daily_market_split_data", "*.csv", "daily market CSV cache"),
        ("data_hourly_market_split", ROOT / "data" / "hourly_market_split_data", "*.csv", "hourly market CSV cache"),
        ("data_bctc", ROOT / "data" / "bctc", "*", "BCTC/RAG source directory"),
        ("data_listing", ROOT / "data" / "listing", "*", "listing/universe source directory"),
        ("market_proxy", ROOT / "data" / "market_proxy.csv", "*", "derived market proxy CSV"),
        ("market_breadth", ROOT / "data" / "market_breadth.csv", "*", "market breadth context CSV"),
        ("macro_context", ROOT / "data" / "macro_context.csv", "*", "macro context CSV"),
        ("foreign_flow", ROOT / "data" / "foreign_flow_curated.csv", "*", "curated foreign-flow CSV"),
        ("sector_proxies", ROOT / "data" / "sector_proxies.csv", "*", "sector proxy CSV"),
        ("model_artifacts", ROOT / "models", "*.joblib", "trained model bundles"),
        ("generated_artifacts", ROOT / "artifacts", "*", "ignored workflow outputs"),
    ]

    results: list[CheckResult] = []
    for name, path, pattern, detail in checks:
        count = _count_files(path, pattern)
        if path.exists() and count > 0:
            results.append(CheckResult("artifact", name, "OK", f"{detail}; present ({count} files)"))
        elif path.exists():
            results.append(CheckResult("artifact", name, "WARN", f"{detail}; path exists but no matching files"))
        else:
            results.append(CheckResult("artifact", name, "WARN", f"{detail}; missing at {path.relative_to(ROOT)}"))
    return results


def _check_environment_flags() -> list[CheckResult]:
    flags = [
        "TIMESCALE_URL",
        "REDIS_URL",
        "KAFKA_BROKER_URL",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
    ]
    results: list[CheckResult] = []
    for flag in flags:
        if os.getenv(flag):
            results.append(CheckResult("environment", flag, "OK", "set"))
        else:
            results.append(CheckResult("environment", flag, "WARN", "not set; default settings or local-only mode may apply"))
    return results


def main() -> int:
    results: list[CheckResult] = []
    results.extend(_check_python())
    results.extend(_check_imports())
    results.extend(_check_services())
    results.extend(_check_artifacts())
    results.extend(_check_environment_flags())

    _, _, fail = _print_results(results)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
