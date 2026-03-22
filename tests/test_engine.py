"""Tests for Phase 4 Decision Engine.

Validates the Decision Matrix rules and Hard-Cap Risk constraints.
"""

from __future__ import annotations

import pytest

from src.api.schemas import ActionPlan, ExpectedRange, QualitativeAnalysis, QuantitativeSignals, TrendProbabilities
from src.engine.matrix import evaluate_decision_matrix
from src.engine.risk import apply_risk_constraints


@pytest.fixture
def mock_quant_buy():
    return QuantitativeSignals(
        trend_probabilities=TrendProbabilities(up=0.8, down=0.1, sideways=0.1),
        expected_range=ExpectedRange(bottom_10th=34.0, median_50th=35.0, ceiling_90th=36.0),
        action_plan=ActionPlan(
            recommendation="BUY",
            entry_zone=[34.5, 35.0],
            stop_loss=32.0,  # Note: (32 - 34.5) / 34.5 = -7.2% -> This should trigger override
            take_profit=38.0,
        )
    )


@pytest.fixture
def mock_qual_positive():
    return QualitativeAnalysis(
        analysis_status="success",
        sentiment="positive",
        risk_factor="low",
        reasoning="Good news",
        system_parameters={
            "applied_risk_tolerance": 0.70,
            "confidence_metrics": {"stock_quantitative_data": 0.95, "rag_context_data": 0.70}
        },
        sources_used=["zone_1"]
    )


class TestDecisionMatrix:
    def test_rule1_perfect_match(self, mock_quant_buy, mock_qual_positive):
        decision, consensus = evaluate_decision_matrix(mock_quant_buy, mock_qual_positive)
        assert decision == "EXECUTE_BUY"
        assert consensus.veto_triggered is False

    def test_rule2_veto_rule(self, mock_quant_buy, mock_qual_positive):
        mock_qual_positive.sentiment = "negative"
        decision, consensus = evaluate_decision_matrix(mock_quant_buy, mock_qual_positive)
        assert decision == "CANCEL_ORDER"
        assert consensus.veto_triggered is True

    def test_rule3_null_rule(self, mock_quant_buy):
        decision, consensus = evaluate_decision_matrix(mock_quant_buy, None)
        assert decision == "STANDBY"
        assert consensus.llm_sentiment == "N/A"

    def test_rule3_insufficient_data(self, mock_quant_buy, mock_qual_positive):
        mock_qual_positive.analysis_status = "insufficient_data"
        decision, consensus = evaluate_decision_matrix(mock_quant_buy, mock_qual_positive)
        assert decision == "STANDBY"


class TestRiskManagement:
    def test_stop_loss_hard_cap(self, mock_quant_buy):
        # Entry range: [34.5, 35.0]. SL = 32.0. Proposed SL % = -7.24%
        payload, override = apply_risk_constraints(
            ticker="SSI",
            action_plan=mock_quant_buy.action_plan,
            real_time_price=34.8,  # Valid entry
            atr_14=1.5,
            applied_risk_tolerance=0.70,
            portfolio_risk_capital=100000.0,
        )
        assert payload is not None
        assert override.original_stop_loss_pct < -0.07
        assert override.applied_stop_loss_pct == -0.07  # Capped exactly to -7%
        
        # Original SL was 32.0. Min entry is 34.5. New SL should be 34.5 * (1 - 0.07) = 32.085 -> 32.09
        assert payload.hard_stop_loss_price > 32.0
        assert payload.hard_stop_loss_price == 32.09

    def test_anti_fomo_block(self, mock_quant_buy):
        payload, override = apply_risk_constraints(
            ticker="SSI",
            action_plan=mock_quant_buy.action_plan,
            real_time_price=36.0,  # Max entry is 35.0. 36.0 is > 35.0 * 1.015 (35.525). FOMO trigger!
            atr_14=1.5,
            applied_risk_tolerance=0.70,
            portfolio_risk_capital=100000.0,
        )
        assert payload is None
        assert override.fomo_check_passed is False

    def test_position_sizing(self, mock_quant_buy):
        payload, override = apply_risk_constraints(
            ticker="SSI",
            action_plan=mock_quant_buy.action_plan,
            real_time_price=34.8,
            atr_14=2.0,  # ATR 2.0 -> Risk per share = 2.0 * 1.5 = 3.0
            applied_risk_tolerance=0.70,
            portfolio_risk_capital=100000.0,  # Risk Budget = 70,000
        )
        assert payload is not None
        assert override.fomo_check_passed is True
        # Volume = 70,000 / 3.0 = 23333.33 -> Rounded to 23300
        assert payload.volume == 23300
