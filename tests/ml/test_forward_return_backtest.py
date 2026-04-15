from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.backtest.forward_return import (
    ForwardReturnBacktestConfig,
    ForwardReturnBacktestRunner,
    _compute_error_metrics,
)
from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer


def _mock_history(ticker: str, start: str = "2019-07-01", end: str = "2021-03-31") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="B")
    df = generate_mock_data(ticker=ticker, num_days=len(dates), seed=91 + len(ticker))
    df["date"] = dates
    df["ticker"] = ticker
    return df


def test_forward_return_target_creation_uses_trading_day_shift() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30, freq="B")
    frame = pd.DataFrame({"date": dates, "close": pd.Series(range(10, 40), dtype=float)})

    labeled = DualModelTrainer._add_targets(frame, {"3d": 3, "5d": 5, "20d": 20})

    assert labeled.loc[0, "target_date_3d"] == dates[3]
    assert labeled.loc[0, "target_date_5d"] == dates[5]
    assert labeled.loc[0, "target_return_3d"] == (frame.loc[3, "close"] / frame.loc[0, "close"]) - 1.0
    assert labeled.loc[4, "target_return_20d"] == (frame.loc[24, "close"] / frame.loc[4, "close"]) - 1.0


def test_compute_error_metrics_uses_return_sign_for_directional_accuracy() -> None:
    actual = pd.Series([0.10, -0.05, 0.00])
    predicted = pd.Series([0.08, 0.02, 0.00])

    metrics = _compute_error_metrics(actual, predicted)

    assert metrics["observations"] == 3
    assert metrics["directional_accuracy"] == 2 / 3


def test_forward_return_runner_writes_horizon_artifacts_and_keeps_target_dates_in_eval_window(
    tmp_path,
    monkeypatch,
) -> None:
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

    config = ForwardReturnBacktestConfig(
        tickers=["AAA", "BBB"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "backtest_forward_return"),
        algorithms=["cart"],
        horizons=["3d", "5d"],
        max_depth=3,
    )
    result = ForwardReturnBacktestRunner(config).run()

    for horizon in ("3d", "5d"):
        horizon_result = result["horizons"][horizon]
        for output_path in horizon_result["paths"].values():
            assert Path(output_path).exists()

        comparison = horizon_result["comparison"]
        target_dates = pd.to_datetime(comparison["target_date"]).dt.normalize()
        prediction_dates = pd.to_datetime(comparison["prediction_date"]).dt.normalize()
        assert target_dates.between(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-29")).all()
        assert (prediction_dates < target_dates).all()

    assert Path(result["overall_paths"]["overall_horizon_summary"]).exists()
    assert Path(result["overall_paths"]["overall_horizon_ranking"]).exists()


def test_forward_return_runner_gracefully_skips_unavailable_model_families(monkeypatch, tmp_path) -> None:
    def _fake_create_model(algorithm: str, **kwargs):
        if algorithm == "sarimax":
            raise ImportError("statsmodels missing")
        return object()

    monkeypatch.setattr("src.ml.backtest.forward_return.create_model", _fake_create_model)

    config = ForwardReturnBacktestConfig(
        tickers=["AAA"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "backtest_forward_return"),
        algorithms=["cart", "sarimax"],
        horizons=["3d"],
    )
    runner = ForwardReturnBacktestRunner(config)
    available, skipped = runner._resolve_available_algorithms()

    assert available == ["cart"]
    assert skipped == [{"algorithm": "sarimax", "reason": "statsmodels missing"}]
