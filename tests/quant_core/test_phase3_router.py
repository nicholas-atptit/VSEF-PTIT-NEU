from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.routing.phase3_router import Phase3RouterConfig, run_phase3_router


REPO_ROOT = Path(__file__).resolve().parents[2]


def _allocation_row(
    *,
    ticker: str = "ACB",
    allocation_weight: float = 0.05,
    candidate_score: float = 0.08,
    agreement: float = 0.8,
    risk_score: float = 0.2,
    health_status: str = "healthy",
) -> dict[str, object]:
    return {
        "allocation_id": f"alloc_0001_{ticker}",
        "decision_label": "allocation_candidate",
        "ticker": ticker,
        "allocation_weight": allocation_weight,
        "invested_weight": allocation_weight,
        "cash_weight": 1.0 - allocation_weight,
        "candidate_score": candidate_score,
        "adjusted_score": candidate_score,
        "model_agreement_score": agreement,
        "risk_score": risk_score,
        "primary_model_name": "xgboost",
        "health_status": health_status,
        "horizon": 5,
        "target_type": "forward_return",
        "run_mode": "research_core",
        "timestamp": "2026-01-02",
        "packet_id": "packet_001",
        "regime_label": "bull",
        "volatility_bucket": "low",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_allocator_outputs(
    input_dir: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    no_allocation: bool = False,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    if no_allocation:
        allocation = pd.DataFrame(
            [
                {
                    "allocation_id": "cash_no_allocation",
                    "decision_label": "no_allocation",
                    "ticker": "CASH",
                    "allocation_weight": 1.0,
                    "invested_weight": 0.0,
                    "cash_weight": 1.0,
                    "reason": "all_candidates_rejected",
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "portfolio_label": "no_allocation",
                    "allocation_count": 0,
                    "invested_exposure": 0.0,
                    "cash_weight": 1.0,
                    "no_allocation_reason": "all_candidates_rejected",
                }
            ]
        )
        risk = pd.DataFrame(
            [
                {
                    "portfolio_label": "no_allocation",
                    "ticker": "CASH",
                    "allocation_weight": 1.0,
                    "risk_score": 0.0,
                    "drawdown_state": "",
                }
            ]
        )
        cards = [{"card_type": "PortfolioAllocationCard", "label": "no_allocation", "ticker": "CASH"}]
        manifest = {
            "manifest_type": "portfolio_allocator_v1_manifest",
            "no_allocation_reason": "all_candidates_rejected",
            "decision_authority": "allocation_candidates_only",
        }
    else:
        allocation = pd.DataFrame(rows if rows is not None else [_allocation_row()])
        summary = pd.DataFrame(
            [
                {
                    "portfolio_label": "allocation_candidate",
                    "allocation_count": len(allocation),
                    "invested_exposure": float(allocation["allocation_weight"].sum()),
                    "cash_weight": 1.0 - float(allocation["allocation_weight"].sum()),
                    "no_allocation_reason": "",
                }
            ]
        )
        risk = pd.DataFrame(
            [
                {
                    "portfolio_label": "allocation_candidate",
                    "ticker": row["ticker"],
                    "allocation_weight": row["allocation_weight"],
                    "risk_score": row.get("risk_score", 0.0),
                    "drawdown_state": row.get("drawdown_state", ""),
                }
                for row in allocation.to_dict(orient="records")
            ]
        )
        cards = [
            {
                "card_type": "PortfolioAllocationCard",
                "label": "allocation_candidate",
                "ticker": row["ticker"],
                "allocation_weight": row["allocation_weight"],
            }
            for row in allocation.to_dict(orient="records")
        ]
        manifest = {
            "manifest_type": "portfolio_allocator_v1_manifest",
            "no_allocation_reason": "",
            "decision_authority": "allocation_candidates_only",
        }

    allocation.to_csv(input_dir / "portfolio_allocation.csv", index=False)
    summary.to_csv(input_dir / "portfolio_summary.csv", index=False)
    risk.to_csv(input_dir / "portfolio_risk_summary.csv", index=False)
    _write_jsonl(input_dir / "portfolio_decision_cards.jsonl", cards)
    (input_dir / "allocator_manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _read_cards(output_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_dir / "phase3_decision_cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_no_recommendation_tokens(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "final_recommendation" not in text
            assert "BUY" not in text
            assert "SELL" not in text
            assert "production_ready" not in text


def test_missing_allocator_outputs_produces_missing_data_route(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = run_phase3_router(input_dir, output_dir)

    assert result.route_decision.loc[0, "route_label"] == "rejected_missing_required_data"
    assert (output_dir / "route_decision.csv").exists()
    assert (output_dir / "phase3_decision_cards.jsonl").exists()
    assert (output_dir / "routing_summary.csv").exists()
    assert (output_dir / "routing_manifest.json").exists()


def test_allocator_no_allocation_produces_no_candidate(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, no_allocation=True)

    result = run_phase3_router(input_dir, output_dir)

    assert result.route_decision.loc[0, "route_label"] == "no_candidate"
    assert result.route_decision.loc[0, "reason"] == "all_candidates_rejected"


def test_allocation_below_min_weight_is_held(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, rows=[_allocation_row(allocation_weight=0.005)])

    result = run_phase3_router(input_dir, output_dir, config=Phase3RouterConfig(min_allocation_weight=0.01))

    assert result.route_decision.loc[0, "route_label"] == "hold_for_review"
    assert result.route_decision.loc[0, "reason"] == "allocation_weight_below_threshold"


def test_low_confidence_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, rows=[_allocation_row(candidate_score=0.001)])

    result = run_phase3_router(input_dir, output_dir, config=Phase3RouterConfig(min_candidate_score=0.01))

    assert result.route_decision.loc[0, "route_label"] == "reject_low_confidence"


def test_low_agreement_uses_configured_action(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, rows=[_allocation_row(agreement=0.2)])

    held = run_phase3_router(input_dir, output_dir, config=Phase3RouterConfig(low_agreement_action="hold_for_review"))
    rejected = run_phase3_router(input_dir, output_dir, config=Phase3RouterConfig(low_agreement_action="reject_low_agreement"))

    assert held.route_decision.loc[0, "route_label"] == "hold_for_review"
    assert rejected.route_decision.loc[0, "route_label"] == "reject_low_agreement"


def test_high_risk_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    row = _allocation_row(risk_score=1.5)
    row["drawdown_state"] = "severe"
    _write_allocator_outputs(input_dir, rows=[row])

    result = run_phase3_router(input_dir, output_dir, config=Phase3RouterConfig(max_risk_score=1.0))

    assert result.route_decision.loc[0, "route_label"] == "reject_high_risk"


def test_unhealthy_model_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, rows=[_allocation_row(health_status="failing")])

    result = run_phase3_router(input_dir, output_dir)

    assert result.route_decision.loc[0, "route_label"] == "reject_unhealthy_model"


def test_valid_allocation_candidate_is_routed(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir)

    result = run_phase3_router(input_dir, output_dir)

    assert result.route_decision.loc[0, "route_label"] == "route_allocation_candidate"


def test_jsonl_cards_do_not_emit_recommendation_tokens(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir)

    run_phase3_router(input_dir, output_dir)
    cards = _read_cards(output_dir)

    assert {card["label"] for card in cards} == {"route_allocation_candidate"}
    _assert_no_recommendation_tokens(output_dir)


def test_deterministic_reproducibility_for_same_inputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir, rows=[_allocation_row(ticker="ACB"), _allocation_row(ticker="BID")])

    run_phase3_router(input_dir, output_dir)
    first = {path.name: path.read_text(encoding="utf-8") for path in sorted(output_dir.iterdir()) if path.is_file()}
    run_phase3_router(input_dir, output_dir)
    second = {path.name: path.read_text(encoding="utf-8") for path in sorted(output_dir.iterdir()) if path.is_file()}

    assert first == second


def test_missing_optional_diagnostic_files_do_not_crash(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir)

    result = run_phase3_router(input_dir, output_dir)

    assert result.route_decision.loc[0, "route_label"] == "route_allocation_candidate"
    assert not (input_dir / "model_consensus_summary.csv").exists()
    assert not (input_dir / "model_health_summary.csv").exists()


def test_cli_works_on_synthetic_fixtures(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_allocator_outputs(input_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_phase3_router.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--min-allocation-weight",
            "0.01",
            "--min-model-agreement",
            "0.5",
            "--max-risk-score",
            "1.0",
        ],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "route_allocation_candidate" in completed.stdout
    assert (output_dir / "route_decision.csv").exists()
    assert (output_dir / "phase3_decision_cards.jsonl").exists()
