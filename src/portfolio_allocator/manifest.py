"""Manifest helpers for Portfolio Allocator v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.portfolio_allocator.schema import (
    PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES,
    PORTFOLIO_ALLOCATOR_VERSION,
    REQUIRED_ALLOCATION_COLUMNS,
    PortfolioAllocatorConfig,
)


def _deterministic_generated_at(allocation_df: pd.DataFrame) -> str | None:
    if allocation_df.empty or "timestamp" not in allocation_df.columns:
        return None
    timestamps = pd.to_datetime(allocation_df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None
    return pd.Timestamp(timestamps.max()).isoformat()


def build_allocator_manifest(
    *,
    config: PortfolioAllocatorConfig,
    input_row_counts: dict[str, int],
    portfolio_allocation: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_risk_summary: pd.DataFrame,
    decision_card_count: int,
    artifact_paths: dict[str, str] | None = None,
    missing_enriched_candidates: bool = False,
) -> dict[str, Any]:
    """Build a deterministic Portfolio Allocator v1 manifest."""

    return {
        "manifest_type": "portfolio_allocator_v1_manifest",
        "version": PORTFOLIO_ALLOCATOR_VERSION,
        "config": config.to_dict(),
        "thresholds": config.thresholds,
        "input_row_counts": dict(input_row_counts),
        "output_row_counts": {
            "portfolio_allocation": int(len(portfolio_allocation)),
            "portfolio_summary": int(len(portfolio_summary)),
            "portfolio_risk_summary": int(len(portfolio_risk_summary)),
            "portfolio_decision_cards": int(decision_card_count),
        },
        "artifact_filenames": dict(PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES),
        "artifact_paths": dict(artifact_paths or {}),
        "required_allocation_fields": list(REQUIRED_ALLOCATION_COLUMNS),
        "allocation_statuses": ["allocation_candidate", "no_allocation"],
        "diagnostic_only_authority": True,
        "no_buy_sell_recommendation_authority": True,
        "no_forced_trade_rule": True,
        "missing_enriched_candidates": bool(missing_enriched_candidates),
        "generated_at": _deterministic_generated_at(portfolio_allocation),
    }


def write_allocator_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["allocator_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
