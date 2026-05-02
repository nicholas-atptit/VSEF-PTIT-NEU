from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.allocation.portfolio_allocator import PortfolioAllocatorConfig, run_portfolio_allocator


def _candidate_rows(*, tickers: list[str] | None = None, score: float = 0.05, agreement: float = 0.8) -> pd.DataFrame:
    tickers = tickers or ["ACB"]
    rows = []
    for index, ticker in enumerate(tickers, start=1):
        rows.append(
            {
                "packet_id": f"packet_{index:03d}",
                "timestamp": "2026-01-02",
                "ticker": ticker,
                "group_name": "small_banks",
                "horizon": 5,
                "target_type": "forward_return",
                "run_mode": "research_core",
                "primary_model_name": "xgboost",
                "primary_prediction": score,
                "model_agreement_score": agreement,
                "agreement_bucket": "high",
                "regime_label": "bull",
                "volatility_bucket": "low",
                "active_signal_count": 2,
                "top_policy_model": "xgboost",
                "top_policy_sharpe": 0.5,
                "candidate_score": score,
            }
        )
    return pd.DataFrame(rows)


def _write_candidates(input_dir: Path, frame: pd.DataFrame | None = None) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else _candidate_rows()).to_csv(input_dir / "decision_lane_candidates.csv", index=False)


def _read_cards(output_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_dir / "portfolio_decision_cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_no_final_recommendation(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file():
            assert "final_recommendation" not in path.read_text(encoding="utf-8")


def test_no_candidates_produces_no_allocation_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, pd.DataFrame(columns=_candidate_rows().columns))

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.allocation.loc[0, "ticker"] == "CASH"
    assert (output_dir / "portfolio_allocation.csv").exists()
    assert (output_dir / "portfolio_summary.csv").exists()
    assert (output_dir / "portfolio_risk_summary.csv").exists()
    assert (output_dir / "portfolio_decision_cards.jsonl").exists()
    assert (output_dir / "allocator_manifest.json").exists()


def test_missing_optional_files_do_not_crash(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir)

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "allocation_candidate"
    assert set(result.allocation["decision_label"]) == {"allocation_candidate"}


def test_missing_required_candidate_file_produces_no_allocation_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.manifest["no_allocation_reason"] == "missing_required_candidate_file"
    assert result.manifest["rejection_counts"]["rejected_missing_required_data"] == 1


def test_low_confidence_candidate_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(score=0.001))

    result = run_portfolio_allocator(
        input_dir,
        output_dir,
        config=PortfolioAllocatorConfig(min_candidate_score=0.01),
    )

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.manifest["rejection_counts"]["rejected_low_confidence"] == 1


def test_low_agreement_candidate_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(score=0.05, agreement=0.2))

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.manifest["rejection_counts"]["rejected_low_agreement"] == 1


def test_high_risk_candidate_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-02",
                "ticker": "ACB",
                "group_name": "small_banks",
                "horizon": 5,
                "target_type": "forward_return",
                "run_mode": "research_core",
                "vol_forecast": 0.95,
                "drawdown_state": "severe",
                "risk_model": "var_cvar_drawdown",
            }
        ]
    ).to_csv(input_dir / "risk_summary.csv", index=False)

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.manifest["rejection_counts"]["rejected_high_risk"] == 1


def test_unhealthy_model_candidate_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir)
    pd.DataFrame([{"model_name": "xgboost", "health_status": "failing", "run_success_rate": 0.0}]).to_csv(
        input_dir / "model_health_summary.csv",
        index=False,
    )

    result = run_portfolio_allocator(input_dir, output_dir)

    assert result.summary.loc[0, "portfolio_label"] == "no_allocation"
    assert result.manifest["rejection_counts"]["rejected_unhealthy_model"] == 1


def test_max_ticker_weight_is_enforced(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(tickers=["ACB", "BID", "CTG"], score=0.05))

    result = run_portfolio_allocator(
        input_dir,
        output_dir,
        config=PortfolioAllocatorConfig(max_ticker_weight=0.05, max_total_exposure=0.60, cash_buffer=0.40),
    )

    assert (result.allocation["allocation_weight"] <= 0.05 + 1e-12).all()


def test_max_total_exposure_and_cash_buffer_are_preserved(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(tickers=[f"T{i:02d}" for i in range(10)], score=0.05))

    result = run_portfolio_allocator(
        input_dir,
        output_dir,
        config=PortfolioAllocatorConfig(max_ticker_weight=0.20, max_total_exposure=0.50, cash_buffer=0.40),
    )

    invested = float(result.summary.loc[0, "invested_exposure"])
    cash = float(result.summary.loc[0, "cash_weight"])
    assert invested <= 0.50 + 1e-12
    assert cash >= 0.40
    assert abs((invested + cash) - 1.0) < 1e-12


def test_deterministic_reproducibility_for_same_input_and_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(tickers=["ACB", "BID"], score=0.05))
    config = PortfolioAllocatorConfig(max_ticker_weight=0.10, max_total_exposure=0.20, cash_buffer=0.50)

    run_portfolio_allocator(input_dir, output_dir, config=config)
    first = {path.name: path.read_text(encoding="utf-8") for path in sorted(output_dir.iterdir()) if path.is_file()}
    run_portfolio_allocator(input_dir, output_dir, config=config)
    second = {path.name: path.read_text(encoding="utf-8") for path in sorted(output_dir.iterdir()) if path.is_file()}

    assert first == second


def test_decision_cards_use_only_allocation_or_no_allocation_labels(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(tickers=["ACB", "BID"], score=0.05))

    run_portfolio_allocator(input_dir, output_dir)
    cards = _read_cards(output_dir)

    assert {card["label"] for card in cards} <= {"allocation_candidate", "no_allocation"}
    _assert_no_final_recommendation(output_dir)


def test_no_allocation_card_uses_no_allocation_label_only(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_candidates(input_dir, _candidate_rows(score=0.001))

    run_portfolio_allocator(
        input_dir,
        output_dir,
        config=PortfolioAllocatorConfig(min_candidate_score=0.01),
    )
    cards = _read_cards(output_dir)

    assert [card["label"] for card in cards] == ["no_allocation"]
    _assert_no_final_recommendation(output_dir)
