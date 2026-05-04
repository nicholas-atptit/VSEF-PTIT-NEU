"""File IO helpers for Phase 3 Router v1 artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.phase3_router.manifest import write_router_manifest
from src.phase3_router.routing import run_phase3_router
from src.phase3_router.schema import (
    LEGACY_ROUTER_ARTIFACT_FILENAMES,
    ROUTER_ARTIFACT_FILENAMES,
    Phase3RouterConfig,
    Phase3RouterResult,
)


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _legacy_cards(router_decisions: pd.DataFrame) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in router_decisions.to_dict(orient="records"):
        cards.append(
            {
                "card_type": "Phase3RouteDecisionCard",
                "label": row.get("route_decision"),
                "ticker": row.get("ticker"),
                "allocation_id": row.get("allocation_id"),
                "final_weight": row.get("final_weight"),
                "route_reason": row.get("route_reason"),
                "route_reason_codes": row.get("route_reason_codes"),
                "diagnostic_only_authority": True,
                "no_buy_sell_recommendation_authority": True,
            }
        )
    return cards


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, default=_json_default, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_phase3_router_outputs(
    output_dir: str | Path,
    result: Phase3RouterResult,
    *,
    write_legacy_aliases: bool = False,
) -> dict[str, str]:
    """Write Phase 3 Router v1 artifacts and update the result manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    decisions_path = destination / ROUTER_ARTIFACT_FILENAMES["router_decisions"]
    summary_path = destination / ROUTER_ARTIFACT_FILENAMES["router_summary"]
    manifest_path = destination / ROUTER_ARTIFACT_FILENAMES["router_manifest"]

    paths = {
        "router_decisions": str(decisions_path),
        "router_summary": str(summary_path),
        "router_manifest": str(manifest_path),
    }
    result.manifest = {**result.manifest, "artifact_paths": paths}

    result.router_decisions.to_csv(decisions_path, index=False)
    result.router_summary.to_csv(summary_path, index=False)
    write_router_manifest(destination, result.manifest)

    if write_legacy_aliases:
        legacy_paths = {
            name: destination / filename for name, filename in LEGACY_ROUTER_ARTIFACT_FILENAMES.items()
        }
        result.router_decisions.to_csv(legacy_paths["route_decision"], index=False)
        _write_jsonl(legacy_paths["phase3_decision_cards"], _legacy_cards(result.router_decisions))
        result.router_summary.to_csv(legacy_paths["routing_summary"], index=False)
        legacy_paths["routing_manifest"].write_text(
            json.dumps(result.manifest, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        paths.update({name: str(path) for name, path in legacy_paths.items()})

    result.output_paths = {name: Path(path) for name, path in paths.items()}
    return paths


def run_phase3_router_from_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: Phase3RouterConfig | None = None,
    write_legacy_aliases: bool = False,
) -> Phase3RouterResult:
    """Run Phase 3 Router v1 from saved Portfolio Allocator artifacts."""

    source = Path(input_dir)
    destination = Path(output_dir) if output_dir is not None else source
    allocation_path = source / "portfolio_allocation.csv"
    allocation = _read_csv(allocation_path)
    result = run_phase3_router(
        allocation,
        portfolio_summary_df=_read_csv(source / "portfolio_summary.csv"),
        portfolio_risk_summary_df=_read_csv(source / "portfolio_risk_summary.csv"),
        allocator_manifest=_read_json(source / "allocator_manifest.json"),
        config=config,
        missing_allocator_outputs=not allocation_path.exists(),
    )
    write_phase3_router_outputs(destination, result, write_legacy_aliases=write_legacy_aliases)
    return result
