from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_walkforward_all_models_stacking_eval import parse_args
from src.ml.backtest.context_coverage_diagnostics import build_context_coverage_rows
from src.ml.backtest.walk_forward_all_models_stacking import (
    WalkForwardAllModelsStackingConfig,
    WalkForwardAllModelsStackingRunner,
)
from src.ml.data_loader import apply_context_features
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer import DualModelTrainer


def _patch_empty_context_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ml.trainer.load_market_proxy", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_sector_proxies", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_ticker_sectors", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_market_breadth", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_macro_context", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_sentiment", lambda **_kwargs: pd.DataFrame())


def _base_ohlcv(periods: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    close = pd.Series(np.linspace(100.0, 110.0, len(dates)))
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, len(dates)),
        }
    )


def _disabled_foreign_flow_frame() -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["foreign_flow_context_mode"] = "disabled"
    frame.attrs["foreign_flow_coverage_status"] = "disabled"
    frame.attrs["source_name"] = "disabled"
    return frame


def test_cli_accepts_foreign_flow_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_walkforward_all_models_stacking_eval.py",
            "--tickers",
            "AAA",
            "--foreign-flow-mode",
            "disabled",
        ],
    )

    args = parse_args()

    assert args.foreign_flow_mode == "disabled"
    assert args.foreign_flow_path is None


def test_default_context_loading_preserves_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_empty_context_loaders(monkeypatch)
    calls: list[dict[str, object]] = []

    def _fake_load_foreign_flow(*_args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr("src.ml.trainer.load_foreign_flow", _fake_load_foreign_flow)

    context_sources = DualModelTrainer(model_dir=tmp_path / "models")._load_context_sources()

    assert calls == [{}]
    assert context_sources["_foreign_flow_mode"] == "auto"
    assert context_sources["_foreign_flow_path_explicit"] is False


def test_foreign_flow_mode_path_requires_foreign_flow_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_walkforward_all_models_stacking_eval.py",
            "--tickers",
            "AAA",
            "--foreign-flow-mode",
            "path",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_foreign_flow_mode_disabled_rejects_foreign_flow_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_walkforward_all_models_stacking_eval.py",
            "--tickers",
            "AAA",
            "--foreign-flow-mode",
            "disabled",
            "--foreign-flow-path",
            "data/foreign_flow.csv",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_disabled_mode_does_not_load_default_foreign_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_empty_context_loaders(monkeypatch)

    def _fail_load_foreign_flow(*_args, **_kwargs):
        raise AssertionError("disabled mode must not load default foreign_flow.csv")

    monkeypatch.setattr("src.ml.trainer.load_foreign_flow", _fail_load_foreign_flow)
    trainer = DualModelTrainer(model_dir=tmp_path / "models")

    context_sources = trainer._load_context_sources(foreign_flow_mode="disabled")
    contextual = trainer._ensure_context_features(_base_ohlcv(5), "AAA", context_sources)

    assert context_sources["_foreign_flow_mode"] == "disabled"
    assert context_sources["_foreign_flow_path_explicit"] is False
    assert context_sources["foreign_flow_df"].attrs["foreign_flow_context_mode"] == "disabled"
    assert set(contextual["foreign_flow_context_mode"]) == {"disabled"}
    assert set(contextual["foreign_flow_coverage_status"]) == {"disabled"}
    assert "foreign_flow_context_available" not in contextual.columns
    assert "foreign_net_value" not in contextual.columns


def test_runner_records_disabled_foreign_flow_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_empty_context_loaders(monkeypatch)

    class _FakeAdapter:
        def get_index_ohlcv(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        history_start="2024-01-02",
        history_end="2024-01-05",
        forecast_start="2024-01-02",
        forecast_end="2024-01-05",
        foreign_flow_mode="disabled",
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)
    runner.adapter = _FakeAdapter()

    context_sources = runner._build_context_sources(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"))
    metadata = runner._foreign_flow_context_metadata

    assert asdict(config)["foreign_flow_mode"] == "disabled"
    assert metadata["enabled"] is False
    assert metadata["mode"] == "disabled"
    assert metadata["path"] is None
    assert metadata["row_count"] == 0
    assert metadata["source_name"] == "disabled"
    assert metadata["artifact_validation"] is None
    assert metadata["reason"] == "foreign-flow context intentionally disabled"
    assert context_sources["foreign_flow_df"].attrs["foreign_flow_context_mode"] == "disabled"


def test_disabled_foreign_flow_does_not_trigger_weak_coverage_when_breadth_is_available() -> None:
    base = _base_ohlcv(10)
    breadth = pd.DataFrame({"date": base["date"], "market_breadth": 0.0})
    contextual = apply_context_features(
        base,
        "AAA",
        breadth_df=breadth,
        foreign_flow_df=_disabled_foreign_flow_frame(),
    )

    rows = build_context_coverage_rows(
        feature_frame=contextual,
        fold_context={
            "ticker": "AAA",
            "fold_id": "fold_001",
            "step_size": 1,
            "forecast_sequence_index": 0,
            "prediction_date": "2024-01-15",
            "horizon": "short_5d",
            "train_start": "2024-01-02",
            "train_end": "2024-01-15",
            "eval_start": "2024-01-15",
            "eval_end": "2024-01-22",
        },
    )
    row = rows.iloc[0]

    assert row["breadth_available_count"] == len(base)
    assert row["breadth_missing_rate"] == 0.0
    assert row["foreign_flow_available_count"] == 0
    assert row["foreign_flow_missing_count"] == 0
    assert pd.isna(row["foreign_flow_missing_rate"])
    assert row["foreign_flow_context_mode"] == "disabled"
    assert row["foreign_flow_coverage_status"] == "disabled"
    assert row["coverage_warning_level"] == "ok"
    assert row["coverage_metadata_status"] == "available"


def test_disabled_foreign_flow_support_columns_are_not_active_features() -> None:
    base = _base_ohlcv()
    breadth = pd.DataFrame({"date": base["date"], "market_breadth": 0.0})
    contextual = apply_context_features(
        base,
        "AAA",
        breadth_df=breadth,
        foreign_flow_df=_disabled_foreign_flow_frame(),
    )
    feature_engineer = FeatureEngineer()
    feature_frame = feature_engineer.build_feature_frame(contextual, build_mode="fast_core_mode")
    feature_columns = feature_engineer.get_feature_columns(feature_frame)

    assert "breadth_context_available" in feature_frame.columns
    assert "foreign_flow_context_mode" in feature_frame.columns
    assert "foreign_flow_coverage_status" in feature_frame.columns
    assert "foreign_flow_context_mode" not in feature_columns
    assert "foreign_flow_coverage_status" not in feature_columns
    assert not any(column.startswith("foreign_net_value") for column in feature_columns)
