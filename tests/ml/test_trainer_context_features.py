from __future__ import annotations

import pandas as pd

from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer


def test_ensure_context_features_does_not_short_circuit_partial_context(tmp_path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000, 1100, 1200],
            "m_ret": [0.0, 0.01, -0.02],
        }
    )
    context_sources = {
        "market_df": pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "m_ret": [0.0, 0.01, -0.02]}),
        "sector_df": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "industry": ["Software", "Software", "Software"],
                "ret": [0.0, 0.02, -0.01],
            }
        ),
        "ticker_sectors": pd.DataFrame({"ticker": ["AAA"], "industry": ["Software"]}),
        "breadth_df": pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "market_breadth": [0.1, 0.2, -0.1]}),
        "macro_df": pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-03"]), "fx_usdvnd": [24_000, 24_100]}),
        "foreign_flow_df": pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "foreign_net_volume": [50, 25, -10],
            }
        ),
        "sentiment_df": pd.DataFrame(),
    }

    result = trainer._ensure_context_features(df, "AAA", context_sources)

    assert "s_ret" in result.columns
    assert "market_breadth" in result.columns
    assert "fx_usdvnd" in result.columns
    assert "foreign_net_volume" in result.columns


def test_prepare_ticker_data_reuses_context_buffer_for_warmup_stats(
    monkeypatch,
    tmp_path,
) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = generate_mock_data(ticker="AAA", num_days=320, seed=7)

    context_sources = {
        "market_df": pd.DataFrame(),
        "sector_df": pd.DataFrame(),
        "ticker_sectors": pd.DataFrame(),
        "breadth_df": pd.DataFrame(),
        "macro_df": pd.DataFrame(),
        "foreign_flow_df": pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime([df["date"].iloc[0]]),
                "foreign_net_value": [0.0],
            }
        ),
        "sentiment_df": pd.DataFrame(),
    }

    call_count = 0
    original = trainer._ensure_context_features

    def _counted(df_in, ticker_in, context_in):
        nonlocal call_count
        call_count += 1
        return original(df_in, ticker_in, context_in)

    monkeypatch.setattr(trainer, "_ensure_context_features", _counted)

    prepared = trainer.prepare_ticker_data(
        ticker="AAA",
        df=df,
        max_sequence_length=20,
        context_sources=context_sources,
    )

    assert call_count == 1
    assert not prepared.feature_frame.empty


def test_load_context_sources_disables_sentiment_by_default(monkeypatch, tmp_path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    monkeypatch.setattr("src.ml.trainer.load_market_proxy", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_sector_proxies", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_ticker_sectors", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_market_breadth", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_macro_context", lambda: pd.DataFrame())
    monkeypatch.setattr("src.ml.trainer.load_foreign_flow", lambda: pd.DataFrame())

    captured: dict[str, object] = {}

    def _fake_load_sentiment(*, enabled=True, require_validated_source=False, **kwargs):
        captured["enabled"] = enabled
        captured["require_validated_source"] = require_validated_source
        frame = pd.DataFrame()
        frame.attrs["sentiment_integration_status"] = "disabled" if not enabled else "loaded_unvalidated"
        return frame

    monkeypatch.setattr("src.ml.trainer.load_sentiment", _fake_load_sentiment)

    context_sources = trainer._load_context_sources()

    assert captured["enabled"] is False
    assert captured["require_validated_source"] is True
    assert context_sources["sentiment_df"].attrs["sentiment_integration_status"] == "disabled"
