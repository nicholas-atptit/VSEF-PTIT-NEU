from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.phase3_router import run_phase3_router, run_phase3_router_from_files, write_phase3_router_outputs
from src.phase3_router.schema import LEGACY_ROUTER_ARTIFACT_FILENAMES, ROUTER_DECISION_COLUMNS
from tests.phase3_router.test_router_decisions import allocation_frame, portfolio_summary


def test_router_artifacts_are_written_with_required_columns(tmp_path: Path) -> None:
    result = run_phase3_router(allocation_frame(), portfolio_summary_df=portfolio_summary())
    paths = write_phase3_router_outputs(tmp_path, result)

    assert Path(paths["router_decisions"]).exists()
    assert Path(paths["router_summary"]).exists()
    assert Path(paths["router_manifest"]).exists()

    decisions = pd.read_csv(tmp_path / "router_decisions.csv")
    assert list(decisions.columns) == list(ROUTER_DECISION_COLUMNS)


def test_legacy_aliases_are_optional_and_not_canonical_manifest_entries(tmp_path: Path) -> None:
    result = run_phase3_router(allocation_frame(), portfolio_summary_df=portfolio_summary())
    write_phase3_router_outputs(tmp_path, result)

    for filename in LEGACY_ROUTER_ARTIFACT_FILENAMES.values():
        assert not (tmp_path / filename).exists()

    legacy_dir = tmp_path / "legacy"
    result = run_phase3_router(allocation_frame(), portfolio_summary_df=portfolio_summary())
    paths = write_phase3_router_outputs(legacy_dir, result, write_legacy_aliases=True)

    for name, filename in LEGACY_ROUTER_ARTIFACT_FILENAMES.items():
        assert name in paths
        assert (legacy_dir / filename).exists()

    manifest = json.loads((legacy_dir / "router_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_filenames"] == {
        "router_decisions": "router_decisions.csv",
        "router_summary": "router_summary.csv",
        "router_manifest": "router_manifest.json",
    }
    assert set(manifest["artifact_paths"]) == {"router_decisions", "router_summary", "router_manifest"}


def test_manifest_records_diagnostic_authority_and_counts(tmp_path: Path) -> None:
    result = run_phase3_router(allocation_frame(), portfolio_summary_df=portfolio_summary())
    write_phase3_router_outputs(tmp_path, result)

    manifest = json.loads((tmp_path / "router_manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_type"] == "phase3_router_v1_manifest"
    assert manifest["diagnostic_only_authority"]
    assert manifest["no_buy_sell_recommendation_authority"]
    assert manifest["route_decision_counts"]["route_allocation_candidate"] == 1


def test_file_runner_reads_allocator_artifacts_and_writes_router_outputs(tmp_path: Path) -> None:
    allocation_frame().to_csv(tmp_path / "portfolio_allocation.csv", index=False)
    portfolio_summary().to_csv(tmp_path / "portfolio_summary.csv", index=False)
    pd.DataFrame([{"portfolio_status": "allocation_candidate", "ticker": "AAA", "risk_score": 0.20}]).to_csv(
        tmp_path / "portfolio_risk_summary.csv",
        index=False,
    )
    (tmp_path / "allocator_manifest.json").write_text(
        json.dumps({"manifest_type": "portfolio_allocator_v1_manifest"}, sort_keys=True),
        encoding="utf-8",
    )

    result = run_phase3_router_from_files(tmp_path)

    assert result.router_decisions.loc[0, "route_decision"] == "route_allocation_candidate"
    assert (tmp_path / "router_decisions.csv").exists()
    assert result.manifest["allocator_manifest_context_available"]


def test_router_outputs_are_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    result = run_phase3_router(allocation_frame({"ticker": "AAA"}, {"ticker": "BBB"}), portfolio_summary_df=portfolio_summary(total_exposure=0.10, cash_weight=0.90))
    write_phase3_router_outputs(tmp_path, result)
    first = {path.name: path.read_text(encoding="utf-8") for path in sorted(tmp_path.iterdir())}

    result = run_phase3_router(allocation_frame({"ticker": "AAA"}, {"ticker": "BBB"}), portfolio_summary_df=portfolio_summary(total_exposure=0.10, cash_weight=0.90))
    write_phase3_router_outputs(tmp_path, result)
    second = {path.name: path.read_text(encoding="utf-8") for path in sorted(tmp_path.iterdir())}

    assert first == second
