from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.backtest.strategy_backtest import (
    BENCHMARK_STRATEGY_TYPE,
    BUY_AND_HOLD_MODEL_NAME,
    MODEL_STRATEGY_TYPE,
    NAIVE_FLAT_STRATEGY_NAME,
    PORTFOLIO_TICKER,
    StrategyBacktestConfig,
    StrategyBacktestRunner,
    build_trade_daily_returns,
    calculate_net_trade_return,
    compute_strategy_metrics,
    generate_signal_label,
)


def _history_frame(ticker: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": [10.0, 10.1, 10.2, 10.5, 10.4, 10.3, 10.6, 10.8, 11.0, 11.1],
            "high": [10.2, 10.3, 10.4, 10.7, 10.6, 10.5, 10.8, 11.0, 11.2, 11.3],
            "low": [9.9, 10.0, 10.1, 10.3, 10.2, 10.1, 10.4, 10.6, 10.8, 10.9],
            "close": [10.1, 10.2, 10.5, 10.4, 10.3, 10.6, 10.8, 11.0, 11.1, 11.25],
            "volume": [1000, 1100, 1200, 1050, 1150, 1250, 1300, 1280, 1400, 1450],
        }
    )


class _FakeAdapter:
    def __init__(self, symbol_list=None) -> None:
        self.symbol_list = symbol_list or []
        self._histories = {ticker: _history_frame(ticker) for ticker in self.symbol_list}

    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, interval: str = "1D") -> pd.DataFrame:
        df = self._histories[symbol.upper()].copy()
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def _forecast_result() -> dict:
    comparison = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "model_name": "cart",
                "prediction_date": "2026-01-02",
                "target_date": "2026-01-07",
                "actual_return": 0.03,
                "predicted_return": 0.04,
                "momentum_predicted_return": 0.02,
            },
            {
                "ticker": "AAA",
                "model_name": "cart",
                "prediction_date": "2026-01-05",
                "target_date": "2026-01-08",
                "actual_return": 0.01,
                "predicted_return": 0.03,
                "momentum_predicted_return": -0.01,
            },
            {
                "ticker": "AAA",
                "model_name": "cart",
                "prediction_date": "2026-01-08",
                "target_date": "2026-01-13",
                "actual_return": -0.01,
                "predicted_return": -0.02,
                "momentum_predicted_return": 0.01,
            },
            {
                "ticker": "BBB",
                "model_name": "cart",
                "prediction_date": "2026-01-02",
                "target_date": "2026-01-07",
                "actual_return": 0.02,
                "predicted_return": 0.05,
                "momentum_predicted_return": 0.03,
            },
            {
                "ticker": "BBB",
                "model_name": "cart",
                "prediction_date": "2026-01-08",
                "target_date": "2026-01-13",
                "actual_return": 0.015,
                "predicted_return": 0.025,
                "momentum_predicted_return": 0.005,
            },
        ]
    )
    return {
        "horizons": {
            "3d": {
                "comparison": comparison,
                "paths": {
                    "predicted_vs_actual": "artifacts/backtest_forward_return/3d/predicted_vs_actual.csv",
                },
            }
        },
        "available_algorithms": ["cart"],
        "skipped_algorithms": [{"algorithm": "sarimax", "reason": "statsmodels missing"}],
    }


def test_generate_signal_label_respects_threshold() -> None:
    assert generate_signal_label(0.02, 0.01) == "buy"
    assert generate_signal_label(0.009, 0.01) == "hold"
    assert generate_signal_label(-0.02, 0.01) == "stay_out"


def test_build_trade_daily_returns_matches_round_trip_return() -> None:
    trade_history = _history_frame("AAA").iloc[0:3][["date", "open", "close"]].copy()
    side_cost_rate = 0.0035

    daily_returns = build_trade_daily_returns(trade_history, side_cost_rate)
    net_trade_return = float((1.0 + daily_returns).prod() - 1.0)

    assert net_trade_return == calculate_net_trade_return(
        float(trade_history["open"].iloc[0]),
        float(trade_history["close"].iloc[-1]),
        side_cost_rate,
    )


def test_non_overlapping_trade_logic_enforces_next_open_and_skips_overlap(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.ml.backtest.strategy_backtest.VnstockAdapter", _FakeAdapter)
    runner = StrategyBacktestRunner(
        StrategyBacktestConfig(
            tickers=["AAA"],
            train_start="2020-01-01",
            train_end="2025-12-31",
            eval_start="2026-01-02",
            eval_end="2026-01-15",
            output_dir=str(tmp_path / "artifacts" / "strategy_backtest"),
            forecast_output_dir=str(tmp_path / "artifacts" / "backtest_forward_return"),
            horizons=["3d"],
            algorithms=["cart"],
            thresholds=[0.0],
        )
    )

    history = _history_frame("AAA")
    predictions = _forecast_result()["horizons"]["3d"]["comparison"]
    aaa_predictions = predictions[predictions["ticker"] == "AAA"].copy()

    trades_df, diagnostics = runner._build_trades_for_group(
        prediction_rows=aaa_predictions,
        history=history,
        ticker="AAA",
        horizon_name="3d",
        model_name="cart",
        strategy_type=MODEL_STRATEGY_TYPE,
        threshold=0.0,
        predicted_column="predicted_return",
    )

    assert diagnostics["buy_signals"] == 2
    assert diagnostics["skipped_overlap_signals"] == 1
    assert len(trades_df) == 1
    assert trades_df.loc[0, "prediction_date"] == "2026-01-02"
    assert trades_df.loc[0, "entry_date"] == "2026-01-05"
    assert trades_df.loc[0, "exit_date"] == "2026-01-07"


def test_compute_strategy_metrics_uses_trade_and_equity_series() -> None:
    daily_returns = pd.Series([0.01, -0.005, 0.0, 0.007])
    positions = pd.Series([1, 1, 0, 1])
    trade_returns = pd.Series([0.015, -0.003, 0.009])

    metrics = compute_strategy_metrics(daily_returns, positions, trade_returns)

    assert metrics["number_of_trades"] == 3
    assert metrics["win_rate"] == 2 / 3
    assert metrics["total_return"] > 0
    assert metrics["turnover"] >= 2.0


def test_strategy_runner_writes_artifacts_and_benchmark_summaries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.ml.backtest.strategy_backtest.VnstockAdapter", _FakeAdapter)
    monkeypatch.setattr(
        "src.ml.backtest.strategy_backtest.StrategyBacktestRunner._ensure_forecasts",
        lambda self: _forecast_result(),
    )

    runner = StrategyBacktestRunner(
        StrategyBacktestConfig(
            tickers=["AAA", "BBB"],
            train_start="2020-01-01",
            train_end="2025-12-31",
            eval_start="2026-01-02",
            eval_end="2026-01-15",
            output_dir=str(tmp_path / "artifacts" / "strategy_backtest"),
            forecast_output_dir=str(tmp_path / "artifacts" / "backtest_forward_return"),
            horizons=["3d"],
            algorithms=["cart", "sarimax"],
            thresholds=[0.0, 0.01],
        )
    )
    result = runner.run()

    horizon_result = result["horizons"]["3d"]
    for output_path in horizon_result["paths"].values():
        assert Path(output_path).exists()

    portfolio_metrics = horizon_result["portfolio_metrics"]
    assert (portfolio_metrics["ticker"] == PORTFOLIO_TICKER).all()
    assert BUY_AND_HOLD_MODEL_NAME in set(portfolio_metrics["model_name"])
    assert NAIVE_FLAT_STRATEGY_NAME in set(portfolio_metrics["model_name"])
    assert "beats_buy_and_hold" in portfolio_metrics.columns
    assert "positive_net_return_after_costs" in portfolio_metrics.columns

    strategy_metrics = horizon_result["strategy_metrics"]
    cart_rows = strategy_metrics[
        (strategy_metrics["model_name"] == "cart") & (strategy_metrics["strategy_type"] == MODEL_STRATEGY_TYPE)
    ]
    assert not cart_rows.empty
    assert cart_rows["number_of_trades"].max() >= 1

    overall_ranking = result["overall_ranking"]
    assert not overall_ranking.empty
    assert overall_ranking.loc[0, "horizon"] == "3d"
    assert result["skipped_algorithms"] == [{"algorithm": "sarimax", "reason": "statsmodels missing"}]
