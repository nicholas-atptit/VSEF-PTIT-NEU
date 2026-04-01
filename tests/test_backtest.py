"""Tests for Phase 5 Backtesting & Paper Trading.

Validates Time-Travel prevention, Walk-Forward chunking, Slippage,
and Performance Metrics engines.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.ml.backtest.event_driven import get_safe_rag_context, simulate_execution_cost, walk_forward_chunks
from src.ml.backtest.metrics import calculate_risk_adjusted_returns, calculate_veto_standby_rates
from src.ml.backtest.paper import PaperTradingEngine, LatencyProfile, track_execution_slippage, validate_latency


class TestEventDriven:
    def test_safe_rag_context_timestamp(self):
        """Proof of concept test ensuring time gets injected into Vector DB query."""
        freeze_time = dt.datetime(2026, 3, 15, tzinfo=dt.UTC)
        context = get_safe_rag_context("SSI", ["zone_1"], freeze_time)
        assert "SSI" in context
        assert "2026-03-15" in context

    def test_simulate_execution_cost(self):
        """Test slippage and fees calculation (Buy)."""
        # Buy 1000 shares at 35.0 = 35000 nominal
        # Slippage: 0.20% -> spread is 35.0 * 1.002 = 35.07
        # Total Fees: 0.15% of 35000 = 52.5. 
        # Slippage Cost = 0.07 * 1000 = 70.0
        adjusted_price, total_fees, slippage_cost = simulate_execution_cost(
            entry_price=35.0,
            volume=1000,
            action="BUY"
        )
        assert adjusted_price == 35.07
        assert total_fees == 52.5
        assert round(slippage_cost, 2) == 70.0

    def test_walk_forward_chunks(self):
        """Test sliding window chunking."""
        chunks = walk_forward_chunks(2018, 2026, train_years=5, test_years=1)
        assert len(chunks) == 3
        
        # Chunk 1: Train 2018-2023, Test 2023-2024
        assert chunks[0]["train_start"] == 2018
        assert chunks[0]["test_start"] == 2023
        assert chunks[0]["test_end"] == 2024
        
        # Chunk 3: Train 2020-2025, Test 2025-2026
        assert chunks[2]["train_start"] == 2020
        assert chunks[2]["test_start"] == 2025
        assert chunks[2]["test_end"] == 2026


class TestMetrics:
    def test_calculate_veto_standby_rates_warning(self):
        """Test threshold warning logic."""
        stats = calculate_veto_standby_rates(
            total_signals=100,
            standby_count=96,  # 96%
            veto_count=2,
        )
        assert stats["standby_rate"] == 0.96
        assert "CRITICAL WARNING" in stats["warning"]

    def test_calculate_risk_adjusted_returns(self):
        """Test Sharpe, Sortino ratios."""
        # Simple up and down returns
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, -0.002]
        
        metrics = calculate_risk_adjusted_returns(returns)
        
        assert "annualized_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics


class TestPaperTrading:
    def test_validate_latency_pass(self):
        profile = LatencyProfile(total_latency_seconds=4.5)
        assert validate_latency(profile) is True

    def test_validate_latency_breach(self):
        profile = LatencyProfile(total_latency_seconds=5.2)  # > 5.0 is breach
        assert validate_latency(profile) is False

    def test_track_execution_slippage(self):
        ideal = 35.0
        actual = 35.5 # Bought 0.5 higher than intended
        pct = track_execution_slippage(ideal, actual, "BUY")
        assert round(pct, 5) == round(-0.5 / 35.0, 5)  # Negative slippage (loss)

    def test_apply_execution_updates_positions_and_cash(self):
        engine = PaperTradingEngine(initial_capital=1_000_000.0)
        now = dt.datetime(2026, 3, 15, tzinfo=dt.UTC)
        latency = LatencyProfile()

        opened = engine._apply_execution(
            ticker="SSI",
            decision_action="EXECUTE_BUY",
            order_payload={"entry_price": 10.0, "volume": 1_000},
            current_price=10.0,
            as_of=now,
            latency=latency,
        )
        assert opened["status"] == "opened"
        assert engine.get_portfolio_summary()["open_positions"] == 1

        closed = engine._apply_execution(
            ticker="SSI",
            decision_action="EXECUTE_SELL",
            order_payload={"entry_price": 11.0, "volume": 1_000},
            current_price=11.0,
            as_of=now + dt.timedelta(minutes=5),
            latency=latency,
        )
        assert closed["status"] == "closed"
        summary = engine.get_portfolio_summary()
        assert summary["open_positions"] == 0
        assert summary["cash"] > 1_000_000.0
