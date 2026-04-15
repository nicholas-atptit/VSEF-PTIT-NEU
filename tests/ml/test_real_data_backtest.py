from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.real_data import FixedWindowBacktestConfig, RealDataBacktestRunner
from src.ml.data_loader import generate_mock_data
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer import DualModelTrainer


def _mock_history(ticker: str, start: str = "2019-07-01", end: str = "2026-04-10") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="B")
    df = generate_mock_data(ticker=ticker, num_days=len(dates), seed=42 + len(ticker))
    df["date"] = dates
    df["ticker"] = ticker
    return df


def test_delta_features_ignore_non_numeric_columns() -> None:
    engineer = FeatureEngineer()
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "ticker": ["AAA", "AAA", "AAA"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 110, 120],
            "sentiment_avg": [0.1, 0.2, 0.3],
            "headline_bucket": ["a", "b", "c"],
        }
    )

    transformed = engineer._add_delta_features(df)

    assert "d_sentiment_avg" in transformed.columns
    assert "d_headline_bucket" not in transformed.columns


def test_train_explicit_split_supports_custom_daily_horizon(tmp_path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    history = _mock_history("MOCK")

    result = trainer.train_explicit_split(
        ticker="MOCK",
        df=history[history["date"] <= pd.Timestamp("2026-01-31")].reset_index(drop=True),
        train_start="2020-01-01",
        train_end="2026-01-31",
        algorithms=["cart"],
        horizon_name="daily",
        horizon_days=1,
        max_depth=3,
    )

    assert result["ticker"] == "MOCK"
    assert result["horizon_name"] == "daily"
    assert result["horizon_days"] == 1

    manifest_path = tmp_path / "models" / "MOCK" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["horizons"]["daily"]["days"] == 1

    features = trainer.compute_features_for_ticker(
        "MOCK",
        history,
        window_start="2020-01-01",
        window_end="2026-04-10",
    )
    feature_slice = features[features["date"] <= pd.Timestamp("2026-01-30")].reset_index(drop=True)
    prediction = trainer.predict("MOCK", feature_slice, horizon="daily")

    assert prediction["horizon"] == "daily"
    assert prediction["horizon_days"] == 1
    assert "predicted_return" in prediction


def test_real_data_backtest_runner_writes_eval_artifacts(tmp_path, monkeypatch) -> None:
    histories = {
        "AAA": _mock_history("AAA", start="2019-07-01", end="2021-01-29"),
        "BBB": _mock_history("BBB", start="2019-07-01", end="2021-01-29"),
    }
    benchmark = _mock_history("VNINDEX", start="2019-07-01", end="2021-01-29")

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

    config = FixedWindowBacktestConfig(
        tickers=["AAA", "BBB"],
        train_start="2020-01-01",
        train_end="2020-12-31",
        eval_start="2021-01-01",
        eval_end="2021-01-30",
        output_dir=str(tmp_path / "artifacts" / "backtest"),
        algorithms=["cart"],
        max_depth=3,
    )
    result = RealDataBacktestRunner(config).run()

    comparison = result["comparison"]
    assert not comparison.empty
    assert {
        "date",
        "ticker",
        "actual_close",
        "predicted_close",
        "predicted_close_baseline",
        "absolute_error",
        "absolute_error_baseline",
        "pct_error",
        "pct_error_baseline",
    } <= set(comparison.columns)
    comparison_dates = pd.to_datetime(comparison["date"]).dt.normalize()
    assert comparison_dates.min() >= pd.Timestamp("2021-01-01")
    assert comparison_dates.max() <= pd.Timestamp("2021-01-30")

    metrics = result["metrics"]
    assert "OVERALL" in set(metrics["ticker"])
    assert {
        "model_mae",
        "baseline_mae",
        "model_rmse",
        "baseline_rmse",
        "model_mape",
        "baseline_mape",
        "model_directional_accuracy",
        "baseline_directional_accuracy",
        "beats_baseline_overall",
    } <= set(metrics.columns)
    assert result["run_config"]["leakage_checks"]["comparison_rows_only_in_eval_window"] is True
    assert result["run_config"]["leakage_checks"]["prediction_dates_before_target_dates"] is True
    assert set(result["chart_files"]) == {"AAA", "BBB"}
    assert {"eval_last_trading_date", "eval_tail_gap_days"} <= set(result["fetch_summary"].columns)

    for output_path in result["paths"].values():
        assert Path(output_path).exists()

    for ticker, chart_map in result["chart_files"].items():
        assert Path(chart_map["actual_vs_predicted"]).exists()
        assert Path(chart_map["absolute_error"]).exists()
