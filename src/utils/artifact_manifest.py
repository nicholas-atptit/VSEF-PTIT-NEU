"""Artifact manifest writing with claim and protection metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.governance.artifact_policy import is_protected_evidence_path
from src.governance.claim_boundary import claim_statement

from .research_io import write_json


def write_artifact_manifest(path: Path, artifacts: list[str | Path], **metadata: Any) -> None:
    rows = [
        {"path": Path(artifact).as_posix(), "protected": is_protected_evidence_path(artifact)}
        for artifact in artifacts
    ]
    write_json(path, {"claim_boundary": claim_statement(), "artifacts": rows, **metadata})
