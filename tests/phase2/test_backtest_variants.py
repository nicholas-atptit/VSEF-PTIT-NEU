from __future__ import annotations

import pandas as pd

from src.evaluation.backtest import BacktestConfig, CostAwareBacktester


def test_backtester_keeps_strategy_variants_separate() -> None:
    positions = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "model_name": "linear",
                "strategy_variant": "forecast_only",
                "signal": 1.0,
                "position_size": 1.0,
                "target_timestamp": "2024-01-05",
            },
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "model_name": "linear",
                "strategy_variant": "forecast_plus_risk",
                "signal": 1.0,
                "position_size": 0.5,
                "target_timestamp": "2024-01-05",
            },
        ]
    )
    market = {
        "AAA": pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=5, freq="B"),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000] * 5,
            }
        )
    }

    result = CostAwareBacktester(BacktestConfig(horizon=2, transaction_fee_bps=0.0, slippage_bps=0.0)).run(positions, market)

    assert set(result["strategy_metrics"]["strategy_variant"]) == {"forecast_only", "forecast_plus_risk"}
    assert len(result["strategy_metrics"]) == 2
    assert len(result["trades"]) == 2

