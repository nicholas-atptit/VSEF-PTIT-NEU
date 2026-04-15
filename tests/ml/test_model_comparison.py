from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.backtest.model_comparison import BacktestModelComparisonRunner, ModelComparisonConfig
from src.ml.data_loader import generate_mock_data
from src.ml.feature_engineering import FeatureEngineer


def _mock_history(ticker: str, start: str = "2019-07-01", end: str = "2021-01-29") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="B")
    df = generate_mock_data(ticker=ticker, num_days=len(dates), seed=77 + len(ticker))
    df["date"] = dates
    df["ticker"] = ticker
    return df


def test_feature_engineer_adds_daily_forecasting_features() -> None:
    engineer = FeatureEngineer()
    df = _mock_history("AAA", start="2023-01-01", end="2023-06-30")

    transformed = engineer.transform(df, drop_na=False)

    expected = {
        "close_return_2d",
        "close_return_5d",
        "close_mean_10",
        "close_std_20",
        "volume_ma_10",
        "volume_shock_10",
        "high_low_range",
        "open_close_spread",
        "atr_proxy_10",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
    }
    assert expected <= set(transformed.columns)


def test_model_comparison_runner_writes_expected_outputs(tmp_path, monkeypatch) -> None:
    histories = {
        "AAA": _mock_history("AAA"),
        "BBB": _mock_history("BBB"),
    }
    benchmark = _mock_history("VNINDEX")

    class _FakeAdapter:
        def __init__(self, symbol_list=None) -> None:
            self.symbol_list = symbol_list or []

        def get_ohlcv(self, symbol: str, start_date: str, end_date: str, interval: str = "1D") -> pd.DataFrame:
            df = histories[symbol.upper()].copy()
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

        def get_index_ohlcv(self, symbol: str, start_date: str, end_date: str, interval: str = "1D") -> pd.DataFrame:
            df = benchmark.copy()
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()
            return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

    monkeypatch.setattr("src.ml.backtest.real_data.VnstockAdapter", _FakeAdapter)

    config = ModelComparisonConfig(
        tickers=["AAA", "BBB"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "backtest_model_comparison"),
        algorithms=["cart"],
        max_depth=3,
    )
    result = BacktestModelComparisonRunner(config).run()

    comparison = result["comparison"]
    assert not comparison.empty
    assert {"ticker", "model_name", "predicted_close", "predicted_close_baseline"} <= set(comparison.columns)

    model_comparison = result["model_comparison"]
    assert {"ticker", "model_name", "rmse", "mape", "directional_accuracy", "beats_naive_baseline"} <= set(
        model_comparison.columns
    )
    assert "naive_previous_close" in set(model_comparison["model_name"])

    ranking = result["ranking"]
    assert {"model_name", "rank_rmse", "rank_mape", "rank_directional_accuracy"} <= set(ranking.columns)
    assert {"cart", "naive_previous_close"} <= set(ranking["model_name"])

    for output_path in result["paths"].values():
        assert Path(output_path).exists()
