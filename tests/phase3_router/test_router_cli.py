from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.phase3_router.schema import LEGACY_ROUTER_ARTIFACT_FILENAMES, ROUTER_ARTIFACT_FILENAMES
from tests.phase3_router.test_router_decisions import allocation_frame, portfolio_summary


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_router_inputs(input_dir: Path, *, risk_score: float = 0.20) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    allocation_frame({"risk_score": risk_score}).to_csv(input_dir / "portfolio_allocation.csv", index=False)
    portfolio_summary().to_csv(input_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "portfolio_status": "allocation_candidate",
                "ticker": "AAA",
                "risk_score": risk_score,
                "dominant_scenario": "bull",
            }
        ]
    ).to_csv(input_dir / "portfolio_risk_summary.csv", index=False)
    (input_dir / "allocator_manifest.json").write_text(
        json.dumps({"manifest_type": "portfolio_allocator_v1_manifest"}, sort_keys=True),
        encoding="utf-8",
    )


def _run_router_cli(input_dir: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_phase3_router.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            *extra_args,
        ],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_cli_writes_canonical_outputs_and_manifest_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_router_inputs(input_dir)

    completed = _run_router_cli(input_dir, output_dir)

    assert completed.returncode == 0, completed.stderr
    for filename in ROUTER_ARTIFACT_FILENAMES.values():
        assert (output_dir / filename).exists()
        assert filename in completed.stdout
    for filename in LEGACY_ROUTER_ARTIFACT_FILENAMES.values():
        assert not (output_dir / filename).exists()

    manifest = json.loads((output_dir / "router_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_filenames"] == ROUTER_ARTIFACT_FILENAMES
    assert {Path(path).name for path in manifest["artifact_paths"].values()} == set(
        ROUTER_ARTIFACT_FILENAMES.values()
    )


def test_cli_max_risk_score_is_wired_to_router_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    default_output = tmp_path / "default"
    raised_output = tmp_path / "raised"
    _write_router_inputs(input_dir, risk_score=0.80)

    default_completed = _run_router_cli(input_dir, default_output)
    raised_completed = _run_router_cli(input_dir, raised_output, "--max-risk-score", "0.90")

    assert default_completed.returncode == 0, default_completed.stderr
    assert raised_completed.returncode == 0, raised_completed.stderr

    default_decisions = pd.read_csv(default_output / "router_decisions.csv")
    raised_decisions = pd.read_csv(raised_output / "router_decisions.csv")
    assert default_decisions.loc[0, "route_decision"] == "reject"
    assert raised_decisions.loc[0, "route_decision"] == "hold"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--min-candidate-score", "0.1"),
        ("--min-model-agreement", "0.6"),
        ("--min-allocation-weight", "0.02"),
        ("--low-agreement-action", "hold_for_review"),
        ("--no-allow-no-allocation", None),
    ],
)
def test_cli_rejects_removed_legacy_flags(tmp_path: Path, flag: str, value: str | None) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_router_inputs(input_dir)
    extra_args = (flag,) if value is None else (flag, value)

    completed = _run_router_cli(input_dir, output_dir, *extra_args)

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
