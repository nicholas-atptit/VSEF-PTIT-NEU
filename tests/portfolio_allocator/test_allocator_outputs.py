from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.portfolio_allocator import run_portfolio_allocator, write_portfolio_allocator_outputs
from src.portfolio_allocator.schema import REQUIRED_ALLOCATION_COLUMNS
from tests.portfolio_allocator.conftest import allocation_candidates


def test_output_artifacts_are_written(tmp_path: Path) -> None:
    result = run_portfolio_allocator(allocation_candidates())
    paths = write_portfolio_allocator_outputs(tmp_path, result)

    for name in (
        "portfolio_allocation",
        "portfolio_summary",
        "portfolio_risk_summary",
        "portfolio_decision_cards",
        "allocator_manifest",
    ):
        assert Path(paths[name]).exists()

    allocation = pd.read_csv(tmp_path / "portfolio_allocation.csv")
    assert set(REQUIRED_ALLOCATION_COLUMNS).issubset(allocation.columns)


def test_manifest_records_no_buy_sell_authority(tmp_path: Path) -> None:
    result = run_portfolio_allocator(allocation_candidates())
    write_portfolio_allocator_outputs(tmp_path, result)

    manifest = json.loads((tmp_path / "allocator_manifest.json").read_text(encoding="utf-8"))

    assert manifest["diagnostic_only_authority"]
    assert manifest["no_buy_sell_recommendation_authority"]
    assert manifest["no_forced_trade_rule"]


def test_decision_cards_include_required_authority_flags(tmp_path: Path) -> None:
    result = run_portfolio_allocator(allocation_candidates())
    write_portfolio_allocator_outputs(tmp_path, result)

    cards = [
        json.loads(line)
        for line in (tmp_path / "portfolio_decision_cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert cards
    assert cards[0]["diagnostic_only_authority"]
    assert cards[0]["no_buy_sell_recommendation_authority"]
    assert cards[0]["allocation_status"] in {"allocation_candidate", "no_allocation"}


def test_deterministic_reproducibility_for_identical_inputs(tmp_path: Path) -> None:
    frame = allocation_candidates({"ticker": "AAA"}, {"ticker": "BBB"})

    first_result = run_portfolio_allocator(frame)
    write_portfolio_allocator_outputs(tmp_path, first_result)
    first = {path.name: path.read_text(encoding="utf-8") for path in sorted(tmp_path.iterdir())}

    second_result = run_portfolio_allocator(frame)
    write_portfolio_allocator_outputs(tmp_path, second_result)
    second = {path.name: path.read_text(encoding="utf-8") for path in sorted(tmp_path.iterdir())}

    assert first == second
