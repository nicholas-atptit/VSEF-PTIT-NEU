"""Tests for Module 3: Signal Generator.

Validates trading signal logic, risk cap enforcement, and confidence routing.
"""

from __future__ import annotations

import pytest

from src.ml.signal_generator import (
    ACTION_BUY,
    ACTION_RANGE_TRADE,
    ACTION_SELL,
    SignalGenerator,
)


@pytest.fixture
def sg() -> SignalGenerator:
    """Create a SignalGenerator instance."""
    return SignalGenerator()


# ── Shared test data ──────────────────────────────────────────

MOCK_RANGE = {
    "bottom_10th": 34.50,
    "median_50th": 35.80,
    "ceiling_90th": 36.90,
}


class TestUptrendSignal:
    """When P_Up > 60%, system should recommend BUY."""

    def test_buy_recommendation(self, sg: SignalGenerator):
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(
            ticker="SSI",
            current_close=35.0,
            model_output=model_output,
        )
        assert result["quantitative_signals"]["action_plan"]["recommendation"] == ACTION_BUY

    def test_buy_entry_zone(self, sg: SignalGenerator):
        """Entry zone should be [Q10, current_close]."""
        model_output = {
            "trend_probabilities": {"up": 0.70, "sideways": 0.10, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        entry = result["quantitative_signals"]["action_plan"]["entry_zone"]
        assert entry[0] == MOCK_RANGE["bottom_10th"]
        assert entry[1] == 35.0

    def test_buy_stop_loss(self, sg: SignalGenerator):
        """Stop loss should be Q10 - 1.5%."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        expected_sl = round(MOCK_RANGE["bottom_10th"] * (1 - 0.015), 2)
        assert result["quantitative_signals"]["action_plan"]["stop_loss"] == expected_sl

    def test_buy_take_profit(self, sg: SignalGenerator):
        """Take profit should target Q90."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        assert result["quantitative_signals"]["action_plan"]["take_profit"] == MOCK_RANGE["ceiling_90th"]


class TestDowntrendSignal:
    """When P_Down > 60%, system should recommend SELL."""

    def test_sell_recommendation(self, sg: SignalGenerator):
        model_output = {
            "trend_probabilities": {"up": 0.10, "sideways": 0.20, "down": 0.70},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        assert result["quantitative_signals"]["action_plan"]["recommendation"] == ACTION_SELL


class TestSidewaysSignal:
    """When neither up nor down dominates, recommend RANGE_TRADE."""

    def test_range_trade_recommendation(self, sg: SignalGenerator):
        model_output = {
            "trend_probabilities": {"up": 0.30, "sideways": 0.50, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        assert result["quantitative_signals"]["action_plan"]["recommendation"] == ACTION_RANGE_TRADE

    def test_range_trade_entry_zone(self, sg: SignalGenerator):
        """Range trade entry should be [Q10, Q90]."""
        model_output = {
            "trend_probabilities": {"up": 0.30, "sideways": 0.50, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        entry = result["quantitative_signals"]["action_plan"]["entry_zone"]
        assert entry[0] == MOCK_RANGE["bottom_10th"]
        assert entry[1] == MOCK_RANGE["ceiling_90th"]


class TestRiskCapOverride:
    """max_risk_tolerance must ALWAYS be ≤ 0.70 (Core Rule)."""

    def test_risk_cap_default(self, sg: SignalGenerator):
        """Default should be 0.70."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        assert result["system_parameters"]["max_risk_tolerance"] == 0.70

    def test_risk_cap_higher_input_clamped(self, sg: SignalGenerator):
        """Even if client requests 1.0, it must be clamped to 0.70."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(
            ticker="SSI", current_close=35.0, model_output=model_output,
            risk_tolerance=1.0,
        )
        assert result["system_parameters"]["max_risk_tolerance"] <= 0.70

    def test_risk_cap_lower_input_preserved(self, sg: SignalGenerator):
        """If client requests 0.50, it should be preserved (below cap)."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(
            ticker="SSI", current_close=35.0, model_output=model_output,
            risk_tolerance=0.50,
        )
        assert result["system_parameters"]["max_risk_tolerance"] == 0.50


class TestConfidenceRouting:
    """Confidence metrics must follow the routing rules."""

    def test_stock_data_confidence(self, sg: SignalGenerator):
        """stock_quantitative_data must be 0.95."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        conf = result["system_parameters"]["confidence_metrics"]
        assert conf["stock_quantitative_data"] == 0.95

    def test_context_confidence(self, sg: SignalGenerator):
        """general_market_context must be 0.70."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        conf = result["system_parameters"]["confidence_metrics"]
        assert conf["general_market_context"] == 0.70


class TestOutputPayload:
    """Test overall payload structure matches JSON contract."""

    def test_required_keys(self, sg: SignalGenerator):
        """Payload must have all top-level keys from the contract."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)

        assert "ticker" in result
        assert "timestamp" in result
        assert "quantitative_signals" in result
        assert "system_parameters" in result

    def test_ticker_uppercase(self, sg: SignalGenerator):
        """Ticker should be uppercased."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="ssi", current_close=35.0, model_output=model_output)
        assert result["ticker"] == "SSI"

    def test_quantitative_signals_structure(self, sg: SignalGenerator):
        """quantitative_signals should have all 3 sub-keys."""
        model_output = {
            "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
            "expected_range": MOCK_RANGE,
        }
        result = sg.generate(ticker="SSI", current_close=35.0, model_output=model_output)
        qs = result["quantitative_signals"]
        assert "trend_probabilities" in qs
        assert "expected_range" in qs
        assert "action_plan" in qs
