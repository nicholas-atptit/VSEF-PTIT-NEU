"""Run-manifest helpers for the Phase 1 benchmark path."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
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
) -> dict[str, Any]:
    """Build a serializable run manifest for Phase 1 benchmarks."""

    return {
        "manifest_type": "phase1_run_manifest_v1",
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


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any], filename: str = "run_manifest.json") -> Path:
    """Write a run manifest into the requested output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
