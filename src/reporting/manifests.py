"""Run-manifest helpers for the Phase 1 benchmark path."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def _run_git(args: list[str], repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def collect_git_metadata(repo_root: str | Path = ".") -> dict[str, Any]:
    """Collect branch and commit metadata for manifest writing."""

    root = Path(repo_root).resolve()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    commit_hash = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    return {
        "branch": branch,
        "commit_hash": commit_hash,
        "is_dirty": bool(status),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_runtime_metadata(python_executable: str | None = None) -> dict[str, Any]:
    executable = python_executable or sys.executable
    pip_version: str | None = None
    try:
        result = subprocess.run(
            [executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            pip_version = result.stdout.strip() or None
    except Exception:
        pip_version = None
    return {
        "python_executable": executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_version": pip_version,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_dependency_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[str(package)] = importlib_metadata.version(str(package))
        except importlib_metadata.PackageNotFoundError:
            versions[str(package)] = None
    return versions


def build_run_manifest(
    *,
    git_metadata: dict[str, Any],
    command: str,
    tickers: list[str],
    requested_models: list[str],
    evaluated_models: list[str],
    skipped_models: list[dict[str, str]],
    target_type: str,
    horizon: int,
    seed: int,
    costs: dict[str, float],
    evaluation_config: dict[str, Any],
    artifact_paths: dict[str, str],
    started_at: str,
    completed_at: str,
    manifest_type: str = "phase1_run_manifest_v1",
    runtime: dict[str, Any] | None = None,
    dependency_versions: dict[str, str | None] | None = None,
    benchmark_modes: list[str] | None = None,
    regime_context: dict[str, Any] | None = None,
    risk_context: dict[str, Any] | None = None,
    strategy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable run manifest for Phase 1 or Phase 2 benchmarks."""

    manifest = {
        "manifest_type": manifest_type,
        "created_at": completed_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "git": dict(git_metadata),
        "command": command,
        "tickers": list(tickers),
        "requested_models": list(requested_models),
        "evaluated_models": list(evaluated_models),
        "skipped_models": list(skipped_models),
        "target_type": target_type,
        "horizon": int(horizon),
        "seed": int(seed),
        "costs": {key: float(value) for key, value in costs.items()},
        "evaluation_config": dict(evaluation_config),
        "artifact_paths": dict(artifact_paths),
    }
    if runtime:
        manifest["runtime"] = dict(runtime)
    if dependency_versions:
        manifest["dependency_versions"] = dict(dependency_versions)
    if benchmark_modes:
        manifest["benchmark_modes"] = list(benchmark_modes)
    if regime_context:
        manifest["regime"] = dict(regime_context)
    if risk_context:
        manifest["risk"] = dict(risk_context)
    if strategy_context:
        manifest["strategy"] = dict(strategy_context)
    return manifest


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any], filename: str = "run_manifest.json") -> Path:
    """Write a run manifest into the requested output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
