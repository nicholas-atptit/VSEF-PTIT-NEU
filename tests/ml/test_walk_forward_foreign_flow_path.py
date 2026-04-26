from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_walkforward_all_models_stacking_eval import parse_args
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


def _write_foreign_flow_artifact(path: Path, ticker: str = "AAA") -> Path:
    dates = pd.bdate_range("2024-01-02", "2024-01-05")
    pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "foreign_net_value": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
            "source": "vnstock_data.Trading.foreign_trade",
            "source_date": dates,
            "retrieved_at": "2026-04-27T00:00:00Z",
            "provider": "vnstock_data",
            "coverage_note": "provider-backed unit test artifact",
        }
    ).to_csv(path, index=False)
    return path


def test_cli_accepts_foreign_flow_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    foreign_path = tmp_path / "foreign_flow_curated.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_walkforward_all_models_stacking_eval.py",
            "--tickers",
            "AAA",
            "--foreign-flow-path",
            str(foreign_path),
        ],
    )

    args = parse_args()

    assert args.foreign_flow_path == str(foreign_path)


def test_default_context_loading_preserves_default_foreign_flow_behavior(
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
    assert context_sources["_foreign_flow_path_explicit"] is False
    assert context_sources["_foreign_flow_path"] is None


def test_custom_foreign_flow_path_is_passed_to_context_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_empty_context_loaders(monkeypatch)
    foreign_path = _write_foreign_flow_artifact(tmp_path / "foreign_flow_curated.csv")
    calls: list[Path | None] = []

    def _fake_load_foreign_flow(*_args, path=None, **_kwargs):
        calls.append(Path(path) if path is not None else None)
        return pd.read_csv(path)

    monkeypatch.setattr("src.ml.trainer.load_foreign_flow", _fake_load_foreign_flow)

    context_sources = DualModelTrainer(model_dir=tmp_path / "models")._load_context_sources(
        foreign_flow_path=foreign_path
    )

    assert calls == [foreign_path]
    assert context_sources["_foreign_flow_path_explicit"] is True
    assert context_sources["_foreign_flow_path"] == str(foreign_path)
    assert set(context_sources["foreign_flow_df"]["ticker"]) == {"AAA"}


def test_missing_custom_foreign_flow_path_fails_clearly(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_foreign_flow.csv"

    with pytest.raises(FileNotFoundError, match="foreign_flow_path"):
        DualModelTrainer(model_dir=tmp_path / "models")._load_context_sources(
            foreign_flow_path=missing_path
        )


def test_runner_records_custom_foreign_flow_path_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_empty_context_loaders(monkeypatch)
    foreign_path = _write_foreign_flow_artifact(tmp_path / "foreign_flow_curated.csv")

    class _FakeAdapter:
        def get_index_ohlcv(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    config = WalkForwardAllModelsStackingConfig(
        tickers=["AAA"],
        history_start="2024-01-02",
        history_end="2024-01-05",
        forecast_start="2024-01-02",
        forecast_end="2024-01-05",
        foreign_flow_path=str(foreign_path),
        output_dir=str(tmp_path / "out"),
    )
    runner = WalkForwardAllModelsStackingRunner(config)
    runner.adapter = _FakeAdapter()

    context_sources = runner._build_context_sources(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"))
    metadata = runner._foreign_flow_context_metadata

    assert asdict(config)["foreign_flow_path"] == str(foreign_path)
    assert metadata["foreign_flow_path"] == str(foreign_path)
    assert metadata["foreign_flow_path_explicit"] is True
    assert metadata["source_name"] == foreign_path.name
    assert metadata["artifact_validation"]["artifact_classification"] == "usable_for_requested_window"
    assert metadata["artifact_validation"]["real_provider_evidence"] is True
    assert not context_sources["foreign_flow_df"].empty


def test_custom_foreign_flow_path_changes_availability_without_polluting_features(
    tmp_path: Path,
) -> None:
    foreign_path = _write_foreign_flow_artifact(tmp_path / "foreign_flow_curated.csv")
    foreign_flow = pd.read_csv(foreign_path)
    close = pd.Series([100.0 + value for value in range(70)])
    base = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.bdate_range("2024-01-02", periods=70),
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1_000.0 + value for value in range(70)],
        }
    )

    contextual = apply_context_features(base, "AAA", foreign_flow_df=foreign_flow)
    feature_frame = FeatureEngineer().build_feature_frame(contextual, build_mode="fast_core_mode")
    feature_columns = FeatureEngineer().get_feature_columns(feature_frame)

    assert int(contextual["foreign_flow_context_available"].sum()) == 4
    assert bool(contextual.loc[0, "foreign_flow_context_available"]) is True
    assert bool(contextual.loc[4, "foreign_flow_context_missing"]) is True
    assert {"source", "source_date", "retrieved_at", "provider", "coverage_note"}.isdisjoint(contextual.columns)
    assert "foreign_flow_context_available" not in feature_columns
    assert "foreign_flow_context_missing" not in feature_columns
