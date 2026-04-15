from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.backtest.dual_task import DualTaskBacktestConfig, DualTaskBacktestRunner
from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer


def _mock_history(ticker: str, start: str = "2019-07-01", end: str = "2021-03-31") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="B")
    df = generate_mock_data(ticker=ticker, num_days=len(dates), seed=101 + len(ticker))
    df["date"] = dates
    df["ticker"] = ticker
    return df


def test_add_targets_creates_cost_adjusted_profit_labels() -> None:
    dates = pd.bdate_range("2026-01-02", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "open": [10.0, 10.2, 10.4, 10.5, 10.7, 10.9],
            "high": [10.1, 10.3, 10.5, 10.6, 10.8, 11.0],
            "low": [9.9, 10.1, 10.3, 10.4, 10.6, 10.8],
            "close": [10.0, 10.3, 10.5, 10.8, 10.9, 11.1],
            "volume": [100, 110, 120, 130, 140, 150],
        }
    )

    labeled = DualModelTrainer._add_targets(
        frame,
        {"3d": 3},
        transaction_fee_bps=15,
        slippage_bps=20,
    )

    expected_net = DualModelTrainer.calculate_net_trade_return(
        entry_open=frame.loc[1, "open"],
        exit_close=frame.loc[3, "close"],
        transaction_fee_bps=15,
        slippage_bps=20,
    )
    assert labeled.loc[0, "target_date_3d"] == dates[3]
    assert labeled.loc[0, "entry_date_3d"] == dates[1]
    assert labeled.loc[0, "target_net_return_3d"] == expected_net
    assert labeled.loc[0, "profit_label_3d"] == int(expected_net > 0.0)


def test_train_explicit_split_predict_includes_profit_outputs(tmp_path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    history = _mock_history("DUAL", start="2019-07-01", end="2026-04-10")
    result = trainer.train_explicit_split(
        ticker="DUAL",
        df=history[history["date"] <= pd.Timestamp("2025-12-31")].reset_index(drop=True),
        train_start="2020-01-01",
        train_end="2025-12-31",
        algorithms=["cart"],
        horizon_name="3d",
        horizon_days=3,
        max_depth=3,
    )

    assert result["horizon_name"] == "3d"
    features = trainer.compute_features_for_ticker(
        "DUAL",
        history,
        window_start="2020-01-01",
        window_end="2026-04-10",
    )
    feature_slice = features[features["date"] <= pd.Timestamp("2026-01-15")].reset_index(drop=True)
    prediction = trainer.predict("DUAL", feature_slice, horizon="3d")

    assert "predicted_return" in prediction
    assert "predicted_profit_label" in prediction
    assert "predicted_profit_probability" in prediction


def test_dual_task_runner_writes_task_artifacts_and_joined_summary(tmp_path, monkeypatch) -> None:
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

    config = DualTaskBacktestConfig(
        tickers=["AAA", "BBB"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "dual_task"),
        algorithms=["cart"],
        horizons=["3d"],
        max_depth=3,
    )
    result = DualTaskBacktestRunner(config).run()

    horizon_result = result["horizons"]["3d"]
    for output_path in horizon_result["regression_paths"].values():
        assert Path(output_path).exists()
    for output_path in horizon_result["classification_paths"].values():
        assert Path(output_path).exists()

    joined_path = Path(result["summary_paths"]["joined_evaluation"])
    assert joined_path.exists()
    joined_df = pd.read_csv(joined_path)
    assert {
        "date",
        "ticker",
        "horizon",
        "model_name",
        "actual_return",
        "predicted_return",
        "actual_profit_label",
        "predicted_profit_label",
        "predicted_profit_probability",
    } <= set(joined_df.columns)

    classification_metrics = horizon_result["classification_metrics"]
    assert {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "positive_class_precision",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    } <= set(classification_metrics.columns)


def test_dual_task_runner_gracefully_skips_unavailable_model_families(monkeypatch, tmp_path) -> None:
    def _fake_create_model(algorithm: str, **kwargs):
        if algorithm == "sarimax":
            raise ImportError("statsmodels missing")
        return object()

    monkeypatch.setattr("src.ml.backtest.forward_return.create_model", _fake_create_model)

    config = DualTaskBacktestConfig(
        tickers=["AAA"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-29",
        output_dir=str(tmp_path / "artifacts" / "dual_task"),
        algorithms=["cart", "sarimax"],
        horizons=["3d"],
    )
    runner = DualTaskBacktestRunner(config)
    available, skipped = runner._resolve_available_algorithms()

    assert available == ["cart"]
    assert skipped == [{"algorithm": "sarimax", "reason": "statsmodels missing"}]
