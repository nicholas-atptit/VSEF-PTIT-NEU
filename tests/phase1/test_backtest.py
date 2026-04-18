from __future__ import annotations

import pandas as pd

from src.evaluation.backtest import BacktestConfig, CostAwareBacktester


def test_backtest_applies_costs_explicitly_to_trade_returns() -> None:
    positions = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "model_name": "linear",
                "signal": 1.0,
                "position_size": 1.0,
                "target_timestamp": "2024-01-05",
            }
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
                "volume": [1000, 1000, 1000, 1000, 1000],
            }
        )
    }

    no_cost = CostAwareBacktester(
        BacktestConfig(horizon=2, transaction_fee_bps=0.0, slippage_bps=0.0)
    ).run(positions, market)["trades"]["net_trade_return"].iloc[0]
    with_cost = CostAwareBacktester(
        BacktestConfig(horizon=2, transaction_fee_bps=15.0, slippage_bps=20.0)
    ).run(positions, market)["trades"]["net_trade_return"].iloc[0]

    assert with_cost < no_cost
