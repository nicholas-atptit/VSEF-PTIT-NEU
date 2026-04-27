from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_walkforward_all_models_stacking_eval import parse_args
from src.ml.backtest.walk_forward_all_models_stacking import (
    WalkForwardAllModelsStackingConfig,
    WalkForwardAllModelsStackingRunner,
)


def _synthetic_ohlcv(ticker: str = "AAA", periods: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = pd.Series([100.0 + float(idx) for idx in range(periods)])
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker.upper(),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [10_000 + idx for idx in range(periods)],
        }
    )


def _write_ohlcv(path: Path, ticker: str = "AAA") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_ohlcv(ticker=ticker).to_csv(path, index=False)
    return path


class _ProviderShouldNotBeUsed:
    def __init__(self, symbol_list=None) -> None:
        self.symbol_list = symbol_list or []

    def get_ohlcv(self, *_args, **_kwargs) -> pd.DataFrame:
        raise AssertionError("provider should not be used when ohlcv_data_dir is explicit")

    def get_index_ohlcv(self, *_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame()


def test_cli_accepts_ohlcv_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "ohlcv_data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_walkforward_all_models_stacking_eval.py",
            "--tickers",
            "AAA",
            "--ohlcv-data-dir",
            str(data_dir),
        ],
    )

    args = parse_args()

    assert args.ohlcv_data_dir == str(data_dir)


def test_explicit_ohlcv_data_dir_is_propagated_to_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "staged_ohlcv"
    _write_ohlcv(data_dir / "AAA.csv")
    calls: list[Path | None] = []

    def _fake_load_ohlcv_from_csv(ticker, csv_dir=None, start_date=None, end_date=None):
        calls.append(Path(csv_dir) if csv_dir is not None else None)
        return _synthetic_ohlcv(ticker=ticker)

    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _ProviderShouldNotBeUsed,
    )
    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.load_ohlcv_from_csv",
        _fake_load_ohlcv_from_csv,
    )

    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        ohlcv_data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)
    history = runner._fetch_history(
        "AAA",
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-04-30"),
    )

    assert calls == [data_dir]
    assert history.attrs["source"] == "csv_explicit_ohlcv_data_dir"
    assert not history.empty


def test_missing_explicit_ohlcv_data_dir_fails_clearly(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_ohlcv"
    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        ohlcv_data_dir=str(missing_dir),
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(FileNotFoundError, match="ohlcv_data_dir does not exist"):
        WalkForwardAllModelsStackingRunner(config)


def test_missing_ticker_file_in_explicit_ohlcv_data_dir_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "staged_ohlcv"
    data_dir.mkdir()
    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _ProviderShouldNotBeUsed,
    )
    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        ohlcv_data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)

    with pytest.raises(FileNotFoundError, match="missing OHLCV file for AAA"):
        runner._fetch_history("AAA", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-30"))


def test_default_history_loading_behavior_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeProvider:
        def __init__(self, symbol_list=None) -> None:
            self.symbol_list = symbol_list or []

        def get_ohlcv(self, symbol: str, *_args, **_kwargs) -> pd.DataFrame:
            return _synthetic_ohlcv(ticker=symbol)

        def get_index_ohlcv(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    def _csv_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("default CSV fallback should not be used when provider returns data")

    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _FakeProvider,
    )
    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.load_ohlcv_from_csv",
        _csv_should_not_be_called,
    )

    config = WalkForwardAllModelsStackingConfig(tickers=["AAA"], output_dir=str(tmp_path / "out"))
    runner = WalkForwardAllModelsStackingRunner(config)
    history = runner._fetch_history(
        "AAA",
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-04-30"),
    )

    assert history.attrs["source"] == "vnstock_unknown"
    assert not history.empty


def test_run_metadata_records_explicit_ohlcv_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "staged_ohlcv"
    _write_ohlcv(data_dir / "AAA.csv")
    captured: dict[str, object] = {}
    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        history_start="2020-01-01",
        history_end="2020-04-30",
        initial_train_start="2020-01-01",
        initial_train_end="2020-03-31",
        forecast_start="2020-04-01",
        forecast_end="2020-04-30",
        horizons=["short_5d"],
        step_sizes=[1],
        algorithms=["cart"],
        ohlcv_data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)

    base_df = pd.DataFrame(
        {
            "step_size": [1],
            "ticker": ["AAA"],
            "prediction_date": [pd.Timestamp("2020-04-01")],
            "feature_date": [pd.Timestamp("2020-04-01")],
            "target_date": [pd.Timestamp("2020-04-08")],
            "horizon": ["short_5d"],
            "model_name": ["cart"],
            "predicted_return": [0.01],
            "predicted_direction": [1],
            "actual_return": [0.02],
            "actual_direction": [1],
            "absolute_error": [0.01],
            "squared_error": [0.0001],
            "direction_correct": [1],
            "evaluation_eligible": [True],
        }
    )
    empty = pd.DataFrame()
    monkeypatch.setattr(runner, "_resolve_available_algorithms", lambda: (["cart"], []))
    monkeypatch.setattr(
        runner,
        "_fetch_histories",
        lambda *_args, **_kwargs: (
            {"AAA": _synthetic_ohlcv()},
            pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "source": "csv_explicit_ohlcv_data_dir",
                        "rows": 80,
                        "fetched_min_date": "2020-01-01",
                        "fetched_max_date": "2020-04-21",
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(runner, "_build_context_sources", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_generate_base_predictions", lambda *_args, **_kwargs: (base_df, empty, empty, empty))
    monkeypatch.setattr(runner, "_build_stacking_predictions", lambda _base_df: base_df.copy())
    monkeypatch.setattr(runner, "_build_actual_comparison_summary", lambda *_args, **_kwargs: (empty, empty))
    monkeypatch.setattr(runner, "_build_summary_tables", lambda *_args, **_kwargs: (empty, empty, empty))
    monkeypatch.setattr(runner, "_build_stacking_vs_models", lambda *_args, **_kwargs: empty)
    monkeypatch.setattr(runner, "_build_backtest_tables", lambda *_args, **_kwargs: (empty, empty))
    monkeypatch.setattr(runner, "_render_charts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_write_report", lambda *_args, **_kwargs: "")

    def _capture_metadata(*_args, metadata, **_kwargs):
        captured["metadata"] = metadata
        return {}

    monkeypatch.setattr(runner, "_write_csv_outputs", _capture_metadata)

    runner.run()

    metadata = captured["metadata"]
    assert metadata["config"]["ohlcv_data_dir"] == str(data_dir)
    assert metadata["ohlcv_data"]["ohlcv_data_dir"] == str(data_dir)
    assert metadata["ohlcv_data"]["ohlcv_data_dir_explicit"] is True
    assert metadata["ohlcv_data"]["source_mode"] == "explicit_csv_dir"


def test_explicit_ohlcv_data_dir_does_not_use_default_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "staged_ohlcv"
    _write_ohlcv(data_dir / "AAA.csv")
    used_default_cache = False

    def _fake_load_ohlcv_from_csv(ticker, csv_dir=None, start_date=None, end_date=None):
        nonlocal used_default_cache
        if csv_dir is None:
            used_default_cache = True
        return _synthetic_ohlcv(ticker=ticker)

    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.VnstockAdapter",
        _ProviderShouldNotBeUsed,
    )
    monkeypatch.setattr(
        "src.ml.backtest.walk_forward_all_models_stacking.load_ohlcv_from_csv",
        _fake_load_ohlcv_from_csv,
    )

    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        ohlcv_data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)
    runner._fetch_history("AAA", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-30"))

    assert used_default_cache is False
