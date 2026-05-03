"""Manifest helpers for Phase 3 Router v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.phase3_router.schema import (
    PHASE3_ROUTER_VERSION,
    ROUTE_DECISIONS,
    ROUTER_ARTIFACT_FILENAMES,
    ROUTER_DECISION_COLUMNS,
    Phase3RouterConfig,
)


def _deterministic_generated_at(router_decisions: pd.DataFrame) -> str | None:
    if router_decisions.empty or "timestamp" not in router_decisions.columns:
        return None
    timestamps = pd.to_datetime(router_decisions["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return pd.Timestamp(timestamps.max()).isoformat()


def _route_counts(router_decisions: pd.DataFrame) -> dict[str, int]:
    if router_decisions.empty or "route_decision" not in router_decisions.columns:
        return {decision: 0 for decision in ROUTE_DECISIONS}
    counts = router_decisions["route_decision"].astype(str).value_counts().to_dict()
    return {decision: int(counts.get(decision, 0)) for decision in ROUTE_DECISIONS}


def build_router_manifest(
    *,
    config: Phase3RouterConfig,
    input_row_counts: dict[str, int],
    router_decisions: pd.DataFrame,
    router_summary: pd.DataFrame,
    artifact_paths: dict[str, str] | None = None,
    allocator_manifest: dict[str, Any] | None = None,
    missing_allocator_outputs: bool = False,
) -> dict[str, Any]:
    """Build a deterministic Phase 3 Router v1 manifest."""

    allocator_context = allocator_manifest if isinstance(allocator_manifest, dict) else {}
    return {
        "manifest_type": "phase3_router_v1_manifest",
        "version": PHASE3_ROUTER_VERSION,
        "deterministic": True,
        "config": config.to_dict(),
        "thresholds": config.thresholds,
        "input_row_counts": dict(input_row_counts),
        "output_row_counts": {
            "router_decisions": int(len(router_decisions)),
            "router_summary": int(len(router_summary)),
        },
        "artifact_filenames": dict(ROUTER_ARTIFACT_FILENAMES),
        "artifact_paths": dict(artifact_paths or {}),
        "route_decisions": list(ROUTE_DECISIONS),
        "route_decision_counts": _route_counts(router_decisions),
        "required_router_decision_fields": list(ROUTER_DECISION_COLUMNS),
        "allocator_manifest_context_available": bool(allocator_context),
        "allocator_manifest_type": allocator_context.get("manifest_type", ""),
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
        "missing_allocator_outputs": bool(missing_allocator_outputs),
        "generated_at": _deterministic_generated_at(router_decisions),
    }


def write_router_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / ROUTER_ARTIFACT_FILENAMES["router_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
