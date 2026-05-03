"""File IO helpers for Portfolio Allocator v1 artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio_allocator.allocation import run_portfolio_allocator
from src.portfolio_allocator.manifest import write_allocator_manifest
from src.portfolio_allocator.schema import PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES, PortfolioAllocatorConfig, PortfolioAllocatorResult


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_portfolio_allocator_outputs(output_dir: str | Path, result: PortfolioAllocatorResult) -> dict[str, str]:
    """Write all Portfolio Allocator v1 artifacts and update the result manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    allocation_path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["portfolio_allocation"]
    summary_path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["portfolio_summary"]
    risk_summary_path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["portfolio_risk_summary"]
    cards_path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["portfolio_decision_cards"]
    manifest_path = destination / PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES["allocator_manifest"]

    result.allocation.to_csv(allocation_path, index=False)
    result.portfolio_summary.to_csv(summary_path, index=False)
    result.portfolio_risk_summary.to_csv(risk_summary_path, index=False)
    cards_path.write_text(
        "\n".join(json.dumps(card, default=_json_default, sort_keys=True) for card in result.decision_cards)
        + ("\n" if result.decision_cards else ""),
        encoding="utf-8",
    )

    paths = {
        "portfolio_allocation": str(allocation_path),
        "portfolio_summary": str(summary_path),
        "portfolio_risk_summary": str(risk_summary_path),
        "portfolio_decision_cards": str(cards_path),
        "allocator_manifest": str(manifest_path),
    }
    result.manifest = {**result.manifest, "artifact_paths": paths}
    write_allocator_manifest(destination, result.manifest)
    result.output_paths = {name: Path(path) for name, path in paths.items()}
    return paths


def run_portfolio_allocator_from_files(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: PortfolioAllocatorConfig | None = None,
) -> PortfolioAllocatorResult:
    """Run Portfolio Allocator v1 from saved Quant Core artifacts."""

    source = Path(input_dir)
    enriched_path = source / "decision_lane_enriched_candidates.csv"
    enriched = _read_csv(enriched_path)
    risk_adjusted = _read_csv(source / "risk_adjusted_candidates.csv")
    dominance = _read_csv(source / "scenario_dominance_summary.csv")
    result = run_portfolio_allocator(
        enriched,
        risk_adjusted_candidates_df=risk_adjusted,
        scenario_dominance_df=dominance,
        config=config,
        missing_enriched_candidates=not enriched_path.exists(),
    )
    write_portfolio_allocator_outputs(output_dir, result)
    return result
