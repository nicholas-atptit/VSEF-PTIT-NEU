"""Smoke tests for VN100 data loader extensions.

These tests exercise the new batch-loading functions added to
``src.ml.data_loader`` without requiring a live database or API connection.
They rely entirely on the CSV files already present in
``data/daily_market_split_data/``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import warnings

import pandas as pd
import pytest

# ── Ensure project root is on sys.path ──────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.data_loader import (
    apply_context_features,
    audit_sentiment_capability,
    build_data_quality_contract,
    build_foreign_flow_incremental,
    build_market_breadth_from_csv,
    build_macro_context_incremental,
    build_sector_proxies_from_csv,
    clear_artifact_frame_cache,
    generate_mock_data,
    load_foreign_flow,
    load_macro_context,
    load_market_breadth,
    load_ohlcv_from_csv,
    load_market_proxy,
    load_fundamentals,
    load_sentiment,
    VN100DataLoader,
    load_vn100_daily_dataset,
)

# ── Constants ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV_DIR = PROJECT_ROOT / "data" / "daily_market_split_data"

# Pick tickers that we know exist in the CSV data directory
SAMPLE_TICKERS = ["FPT", "HPG", "VNM"]
# Sanity: at least one should exist
_TICKERS_AVAILABLE = [t for t in SAMPLE_TICKERS if (DAILY_CSV_DIR / f"{t}.csv").exists()]


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — original interfaces still work
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure the original public API is unchanged."""

    def test_generate_mock_data_still_works(self):
        df = generate_mock_data(ticker="MOCK", num_days=100)
        assert len(df) == 100
        assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_generate_mock_data_declares_provenance(self):
        df = generate_mock_data(ticker="MOCK", num_days=100, runtime_mode="demo")
        provenance = df.attrs["data_provenance"]
        assert provenance["source"] == "synthetic_mock_data"
        assert provenance["uses_mock_data"] is True
        assert provenance["fallback_triggered"] is False
        assert provenance["runtime_mode"] == "demo"

    def test_load_ohlcv_from_csv_exists(self):
        """The new CSV loader is importable."""
        assert callable(load_ohlcv_from_csv)

    def test_vn100_data_loader_class_exists(self):
        """The VN100DataLoader class is importable."""
        assert callable(VN100DataLoader)


# ═══════════════════════════════════════════════════════════════════
# load_ohlcv_from_csv
# ═══════════════════════════════════════════════════════════════════


class TestLoadOhlcvFromCsv:
    """Test file-backed CSV loading."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_load_known_ticker(self):
        ticker = _TICKERS_AVAILABLE[0]
        df = load_ohlcv_from_csv(ticker, csv_dir=DAILY_CSV_DIR)
        assert not df.empty
        assert "date" in df.columns
        assert "close" in df.columns
        assert "ticker" in df.columns
        assert df["ticker"].iloc[0] == ticker.upper()

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_date_filtering(self):
        ticker = _TICKERS_AVAILABLE[0]
        start = dt.date(2023, 1, 1)
        end = dt.date(2023, 12, 31)
        df = load_ohlcv_from_csv(ticker, csv_dir=DAILY_CSV_DIR, start_date=start, end_date=end)
        if not df.empty:
            assert df["date"].min() >= pd.Timestamp(start)
            assert df["date"].max() <= pd.Timestamp(end)

    def test_missing_ticker_returns_empty(self):
        df = load_ohlcv_from_csv("ZZZZNOTEXIST", csv_dir=DAILY_CSV_DIR)
        assert df.empty


# ═══════════════════════════════════════════════════════════════════
# Auxiliary loaders
# ═══════════════════════════════════════════════════════════════════


class TestAuxiliaryLoaders:
    """Test market proxy, fundamentals, and sentiment loaders."""

    def test_load_market_proxy(self):
        market_path = PROJECT_ROOT / "data" / "market_proxy.csv"
        if market_path.exists():
            df = load_market_proxy()
            assert not df.empty
            assert "date" in df.columns
            assert "m_ret" in df.columns
            assert df.attrs["source_provenance"] in {"derived_from_vnstock_data", "direct_vnstock_data"}
        else:
            # Should degrade gracefully even without the cached artifact.
            df = load_market_proxy(path="/nonexistent/path.csv")
            assert isinstance(df, pd.DataFrame)

    def test_load_fundamentals_no_crash(self):
        """Fundamentals loader never crashes even if file missing."""
        df = load_fundamentals(path="/nonexistent/fund.csv")
        assert isinstance(df, pd.DataFrame)

    def test_load_sentiment_no_crash(self):
        """Sentiment loader never crashes even if file missing."""
        df = load_sentiment(path="/nonexistent/sent.csv")
        assert isinstance(df, pd.DataFrame)

    def test_load_macro_context_stub_when_live_build_fails(self, monkeypatch, tmp_path: Path):
        clear_artifact_frame_cache()
        def _raise(*args, **kwargs):
            raise RuntimeError("live build unavailable")

        monkeypatch.setattr("src.ml.data_loader.build_macro_context_from_vnstock", _raise)
        macro_path = tmp_path / "missing_macro.csv"
        df = load_macro_context(path=macro_path)
        assert df.empty
        assert df.attrs["source_provenance"] == "stub_todo"

    def test_load_foreign_flow_stub(self):
        df = load_foreign_flow(path="/nonexistent/foreign.csv")
        assert df.empty
        assert df.attrs["source_provenance"] == "stub_todo"

    def test_load_macro_context_uses_live_builder_when_artifact_missing(self, monkeypatch, tmp_path: Path):
        clear_artifact_frame_cache()
        expected = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "fx_usdvnd": [24_100]})
        expected.attrs["source_provenance"] = "direct_vnstock_data"
        expected.attrs["source_name"] = "vnstock_data.live_macro_context"

        monkeypatch.setattr(
            "src.ml.data_loader.build_macro_context_from_vnstock",
            lambda start_date=None, end_date=None: expected,
        )

        macro_path = tmp_path / "missing_macro.csv"
        df = load_macro_context(path=macro_path)
        assert not df.empty
        assert df.attrs["source_provenance"] == "direct_vnstock_data"


class TestTimeSafety:
    """Check loader-level alignment and lagging behavior."""

    def test_sentiment_loader_applies_default_lag(self, tmp_path: Path):
        sentiment_path = tmp_path / "sentiment.csv"
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "sentiment_avg": [0.1, 0.4, -0.2],
                "news_volume": [1, 2, 1],
            }
        ).to_csv(sentiment_path, index=False)

        loaded = load_sentiment(path=sentiment_path, tickers=["AAA"])

        assert loaded["sentiment_avg"].tolist() == [0.0, 0.1, 0.4]
        assert loaded["news_volume"].tolist() == [0.0, 1.0, 2.0]

    def test_sentiment_capability_audit_marks_local_artifact_no_go(self, tmp_path: Path, monkeypatch):
        sentiment_path = tmp_path / "sentiment.csv"
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": ["2024-01-02", "2024-01-02"],
                "sentiment_avg": [0.1, 0.2],
                "news_volume": [1, 1],
            }
        ).to_csv(sentiment_path, index=False)

        monkeypatch.setattr(
            "src.data.adapters.vnstock_adapter.VnstockAdapter.audit_company_news_capability",
            lambda self, symbol="SSI", count=1, run_live_probe=True: {
                "provider_runtime_available": False,
                "status": "unsupported",
            },
        )

        audit = audit_sentiment_capability(path=sentiment_path, run_live_probe=False)

        assert audit["live_news_endpoint_status"] == "unsupported"
        assert audit["artifact_status"] == "unstable_partial"
        assert audit["main_pipeline_recommendation"] == "no_go"
        assert audit["artifact_exists"] is True

    def test_sentiment_loader_rejects_unvalidated_artifact_when_required(self, tmp_path: Path, monkeypatch):
        sentiment_path = tmp_path / "sentiment.csv"
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": ["2024-01-02"],
                "sentiment_avg": [0.1],
                "news_volume": [1],
            }
        ).to_csv(sentiment_path, index=False)

        monkeypatch.setattr(
            "src.ml.data_loader.audit_sentiment_capability",
            lambda path=None, live_probe_symbol="SSI", run_live_probe=True: {
                "main_pipeline_recommendation": "no_go",
                "artifact_exists": True,
                "artifact_status": "unstable_partial",
            },
        )

        loaded = load_sentiment(path=sentiment_path, require_validated_source=True)

        assert loaded.empty
        assert loaded.attrs["source_provenance"] == "stub_todo"
        assert loaded.attrs["sentiment_integration_status"] == "rejected_unvalidated"

    def test_apply_context_features_merges_macro_with_backward_asof(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "open": [10.0, 10.2, 10.1],
                "high": [10.3, 10.4, 10.3],
                "low": [9.9, 10.0, 9.8],
                "close": [10.1, 10.3, 10.0],
                "volume": [1000, 1200, 1100],
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "fx_usdvnd": [24_000, 24_100],
            }
        )

        merged = apply_context_features(df, "AAA", macro_df=macro_df)

        assert merged["fx_usdvnd"].tolist() == [24_000, 24_100, 24_100]

    def test_apply_context_features_merges_enriched_breadth_and_sector_context(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "open": [10.0, 10.2, 10.1],
                "high": [10.3, 10.4, 10.3],
                "low": [9.9, 10.0, 9.8],
                "close": [10.1, 10.3, 10.0],
                "volume": [1000, 1200, 1100],
            }
        )
        sector_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "industry": ["Software", "Software", "Software"],
                "ret": [0.01, -0.02, 0.03],
                "sector_dispersion": [0.02, 0.03, 0.01],
                "sector_member_count": [5, 5, 5],
            }
        )
        ticker_sectors = pd.DataFrame({"ticker": ["AAA"], "industry": ["Software"]})
        breadth_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "advancers": [60, 40, 55],
                "decliners": [30, 50, 35],
                "market_breadth": [0.33, -0.11, 0.22],
                "advancing_share": [0.60, 0.44, 0.58],
                "pct_above_ma20": [0.55, 0.48, 0.51],
                "pct_above_ma50": [0.50, 0.46, 0.49],
                "up_volume": [1000.0, 800.0, 900.0],
                "down_volume": [500.0, 900.0, 700.0],
            }
        )

        merged = apply_context_features(
            df,
            "AAA",
            sector_df=sector_df,
            ticker_sectors=ticker_sectors,
            breadth_df=breadth_df,
        )

        assert "sector_dispersion" in merged.columns
        assert "sector_member_count" in merged.columns
        assert "pct_above_ma20" in merged.columns
        assert "up_volume" in merged.columns
        assert merged["sector_dispersion"].tolist() == [0.02, 0.03, 0.01]


class TestDataQualityContracts:
    def test_quality_contract_detects_duplicate_keys_and_missing_ratio(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "close": [10.0, 10.0, None],
            }
        )
        df.attrs["source_provenance"] = "derived_from_vnstock_data"

        contract = build_data_quality_contract(df, dataset_name="test_frame")

        assert contract["duplicate_key_count"] == 1
        assert contract["missing_ratio"] > 0.0
        assert contract["source_provenance_present"] is True
        assert contract["unsupported_source"] is False

    def test_quality_contract_flags_stale_artifact(self, tmp_path: Path):
        artifact_path = tmp_path / "artifact.csv"
        artifact_path.write_text("date,close\n2024-01-01,10\n", encoding="utf-8")
        stale_time = dt.datetime.now() - dt.timedelta(days=10)
        timestamp = stale_time.timestamp()
        import os

        os.utime(artifact_path, (timestamp, timestamp))

        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "close": [10.0]})
        contract = build_data_quality_contract(
            df,
            dataset_name="artifact.csv",
            artifact_path=artifact_path,
            stale_after_days=1,
        )

        assert contract["artifact_exists"] is True
        assert contract["stale"] is True
        assert contract["artifact_age_days"] >= 1

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_loaded_ohlcv_has_quality_contract(self):
        df = load_ohlcv_from_csv(_TICKERS_AVAILABLE[0], csv_dir=DAILY_CSV_DIR)
        contract = df.attrs.get("data_quality_contract")

        assert isinstance(contract, dict)
        assert contract["source_provenance_present"] is True
        assert contract["date_monotonic"] is True


class TestBreadthBuilder:
    """Validate breadth construction from local vnstock-derived OHLCV caches."""

    def test_build_market_breadth_from_csv(self, tmp_path: Path):
        clear_artifact_frame_cache()
        csv_dir = tmp_path / "daily"
        csv_dir.mkdir()

        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "close": [10.0, 11.0],
                "high": [10.0, 11.2],
                "low": [10.0, 10.8],
                "volume": [100.0, 250.0],
            }
        ).to_csv(csv_dir / "AAA.csv", index=False)
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "close": [10.0, 9.5],
                "high": [10.0, 9.7],
                "low": [10.0, 9.2],
                "volume": [120.0, 300.0],
            }
        ).to_csv(csv_dir / "BBB.csv", index=False)

        breadth = build_market_breadth_from_csv(csv_dir=csv_dir)
        breadth = breadth.sort_values("date").reset_index(drop=True)

        assert not breadth.empty
        assert {
            "advancers",
            "decliners",
            "advance_decline_ratio",
            "market_breadth",
            "up_volume",
            "down_volume",
            "pct_above_ma20",
            "pct_above_ma50",
            "new_highs_252",
            "new_lows_252",
        } <= set(breadth.columns)
        assert breadth.iloc[-1]["advancers"] == 1
        assert breadth.iloc[-1]["decliners"] == 1
        assert breadth.iloc[-1]["up_volume"] == 250.0
        assert breadth.iloc[-1]["down_volume"] == 300.0

    def test_market_breadth_builder_supports_incremental_rebuild(self, tmp_path: Path):
        clear_artifact_frame_cache()
        csv_dir = tmp_path / "daily"
        csv_dir.mkdir()
        output_path = tmp_path / "market_breadth.csv"

        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "close": [10.0, 11.0],
                "high": [10.1, 11.2],
                "low": [9.9, 10.8],
                "volume": [100.0, 250.0],
            }
        ).to_csv(csv_dir / "AAA.csv", index=False)
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "close": [10.0, 9.5],
                "high": [10.1, 9.7],
                "low": [9.9, 9.2],
                "volume": [120.0, 300.0],
            }
        ).to_csv(csv_dir / "BBB.csv", index=False)

        first = build_market_breadth_from_csv(csv_dir=csv_dir, output_path=output_path)
        assert first["date"].max() == pd.Timestamp("2024-01-02")

        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "close": [10.0, 11.0, 11.3],
                "high": [10.1, 11.2, 11.5],
                "low": [9.9, 10.8, 11.0],
                "volume": [100.0, 250.0, 275.0],
            }
        ).to_csv(csv_dir / "AAA.csv", index=False)
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "close": [10.0, 9.5, 9.8],
                "high": [10.1, 9.7, 9.9],
                "low": [9.9, 9.2, 9.6],
                "volume": [120.0, 300.0, 310.0],
            }
        ).to_csv(csv_dir / "BBB.csv", index=False)

        clear_artifact_frame_cache()
        second = build_market_breadth_from_csv(csv_dir=csv_dir, output_path=output_path)
        assert second["date"].min() == pd.Timestamp("2024-01-02")
        assert second["date"].max() == pd.Timestamp("2024-01-03")

    def test_build_sector_proxies_from_csv_adds_dispersion(self, tmp_path: Path):
        clear_artifact_frame_cache()
        csv_dir = tmp_path / "daily"
        csv_dir.mkdir()

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        pd.DataFrame(
            {
                "date": dates,
                "close": [10.0, 10.5, 10.8, 11.0, 11.3],
            }
        ).to_csv(csv_dir / "AAA.csv", index=False)
        pd.DataFrame(
            {
                "date": dates,
                "close": [20.0, 19.8, 20.2, 20.4, 20.1],
            }
        ).to_csv(csv_dir / "BBB.csv", index=False)

        ticker_sectors = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "industry": ["Software", "Software"],
            }
        )

        sector = build_sector_proxies_from_csv(csv_dir=csv_dir, ticker_sectors=ticker_sectors)

        assert not sector.empty
        assert {"date", "industry", "ret", "sector_dispersion", "sector_member_count"} <= set(sector.columns)
        assert sector["sector_member_count"].max() >= 2
        assert (sector["sector_dispersion"] >= 0).all()

    def test_context_artifact_loaders_use_cache_hits(self, tmp_path: Path):
        clear_artifact_frame_cache()

        breadth_path = tmp_path / "market_breadth.csv"
        pd.DataFrame({"date": ["2024-01-02"], "market_breadth": [0.1]}).to_csv(breadth_path, index=False)
        first_breadth = load_market_breadth(path=breadth_path)
        second_breadth = load_market_breadth(path=breadth_path)
        assert first_breadth.attrs.get("artifact_cache_status") == "miss"
        assert second_breadth.attrs.get("artifact_cache_status") == "hit"

        macro_path = tmp_path / "macro_context.csv"
        pd.DataFrame({"date": ["2024-01-02"], "fx_usdvnd": [24_100]}).to_csv(macro_path, index=False)
        first_macro = load_macro_context(path=macro_path)
        second_macro = load_macro_context(path=macro_path)
        assert first_macro.attrs.get("artifact_cache_status") == "miss"
        assert second_macro.attrs.get("artifact_cache_status") == "hit"

        foreign_path = tmp_path / "foreign_flow.csv"
        pd.DataFrame(
            {"date": ["2024-01-02"], "ticker": ["AAA"], "foreign_net_value": [1000.0]}
        ).to_csv(foreign_path, index=False)
        first_foreign = load_foreign_flow(path=foreign_path, tickers=["AAA"])
        second_foreign = load_foreign_flow(path=foreign_path, tickers=["AAA"])
        assert first_foreign.attrs.get("artifact_cache_status") == "miss"
        assert second_foreign.attrs.get("artifact_cache_status") == "hit"

    def test_load_foreign_flow_missing_artifact_caches_stub_result(self, tmp_path: Path):
        clear_artifact_frame_cache()

        missing_path = tmp_path / "missing_foreign_flow.csv"

        first = load_foreign_flow(path=missing_path)
        second = load_foreign_flow(path=missing_path)

        assert first.attrs.get("artifact_cache_status") == "miss"
        assert second.attrs.get("artifact_cache_status") == "hit"
        assert second.attrs.get("source_provenance") == "stub_todo"

    def test_macro_incremental_builder_preserves_new_columns_from_recent_rebuild(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        clear_artifact_frame_cache()
        output_path = tmp_path / "macro_context.csv"
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "fx_usdvnd": [24_000.0, 24_010.0],
            }
        ).to_csv(output_path, index=False)

        def _fake_build_macro_context_from_vnstock(*, start_date=None, end_date=None):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "fx_usdvnd": [24_010.0, 24_025.0],
                    "interest_rate": [4.0, 4.1],
                }
            )

        monkeypatch.setattr(
            "src.ml.data_loader.build_macro_context_from_vnstock",
            _fake_build_macro_context_from_vnstock,
        )

        result = build_macro_context_incremental(
            output_path=output_path,
            incremental_update=True,
            lookback_days=1,
        )

        assert {"fx_usdvnd", "interest_rate"} <= set(result.columns)
        assert result["date"].max() == pd.Timestamp("2024-01-03")

    def test_foreign_flow_incremental_preserves_multi_key_prefix_and_new_columns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        clear_artifact_frame_cache()
        output_path = tmp_path / "foreign_flow.csv"
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "ticker": ["AAA", "BBB", "AAA", "BBB"],
                "foreign_net_value": [100.0, 200.0, 150.0, 250.0],
            }
        ).to_csv(output_path, index=False)

        class _FakeAdapter:
            def __init__(self, symbol_list=None):
                self.symbol_list = symbol_list or []

            def get_foreign_flow(self, ticker: str, start_date: str, end_date: str):
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                        "ticker": [ticker, ticker],
                        "foreign_net_value": [300.0, 350.0],
                        "foreign_net_volume": [30.0, 35.0],
                    }
                )

        monkeypatch.setattr("src.data.adapters.vnstock_adapter.VnstockAdapter", _FakeAdapter)

        result = build_foreign_flow_incremental(
            tickers=["AAA", "BBB"],
            output_path=output_path,
            incremental_update=True,
            lookback_days=0,
        )

        historical_prefix = result[result["date"] == pd.Timestamp("2024-01-01")]
        assert set(historical_prefix["ticker"]) == {"AAA", "BBB"}
        assert "foreign_net_volume" in result.columns
        assert result["date"].max() == pd.Timestamp("2024-01-03")

    def test_foreign_flow_incremental_defaults_full_rebuild_start_when_start_date_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        clear_artifact_frame_cache()
        output_path = tmp_path / "foreign_flow.csv"
        observed_calls: list[tuple[str, str, str]] = []

        class _FakeAdapter:
            def __init__(self, symbol_list=None):
                self.symbol_list = symbol_list or []

            def get_foreign_flow(self, ticker: str, start_date: str, end_date: str):
                observed_calls.append((ticker, start_date, end_date))
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02"]),
                        "ticker": [ticker],
                        "foreign_net_value": [300.0],
                        "foreign_net_volume": [30.0],
                    }
                )

        monkeypatch.setattr("src.data.adapters.vnstock_adapter.VnstockAdapter", _FakeAdapter)

        result = build_foreign_flow_incremental(
            tickers=["AAA"],
            output_path=output_path,
            start_date=None,
            end_date=dt.date(2024, 1, 31),
            incremental_update=True,
            lookback_days=400,
        )

        assert not result.empty
        assert observed_calls == [("AAA", "2018-02-01", "2024-01-31")]
        assert output_path.exists()

    def test_apply_context_features_main_path_avoids_fragmentation_warning(self):
        df = generate_mock_data(ticker="AAA", num_days=220)
        dates = pd.to_datetime(df["date"])
        breadth_df = pd.DataFrame(
            {
                "date": dates,
                "market_breadth": 0.1,
                "advancing_share": 0.55,
                "advance_decline_ratio": 1.1,
            }
        )
        sector_df = pd.DataFrame(
            {
                "date": dates,
                "industry": "Software",
                "ret": 0.01,
                "sector_dispersion": 0.02,
                "sector_member_count": 12,
            }
        )
        ticker_sectors = pd.DataFrame({"ticker": ["AAA"], "industry": ["Software"]})
        foreign_flow_df = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAA",
                "foreign_net_value": 1000.0,
                "foreign_net_volume": 100.0,
            }
        )
        macro_df = pd.DataFrame(
            {
                "date": dates,
                "fx_usdvnd": 24_100.0,
                "interest_rate": 4.1,
            }
        )
        market_df = pd.DataFrame({"date": dates, "m_ret": 0.005})

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", category=pd.errors.PerformanceWarning)
            result = apply_context_features(
                df,
                "AAA",
                market_df=market_df,
                sector_df=sector_df,
                ticker_sectors=ticker_sectors,
                breadth_df=breadth_df,
                foreign_flow_df=foreign_flow_df,
                macro_df=macro_df,
            )

        performance_warnings = [
            warning
            for warning in captured
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]

        assert performance_warnings == []
        assert {"s_ret", "market_breadth", "fx_usdvnd", "foreign_net_value"} <= set(result.columns)


# ═══════════════════════════════════════════════════════════════════
# VN100DataLoader class
# ═══════════════════════════════════════════════════════════════════


class TestVN100DataLoader:
    """Test the batch data loader."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_basic(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:2],
            join_market=False,
        )
        assert not df.empty
        assert "ticker" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns
        # Should have data for at least one ticker
        assert df["ticker"].nunique() >= 1

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_with_market_join(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=True,
        )
        market_path = PROJECT_ROOT / "data" / "market_proxy.csv"
        if market_path.exists() and not df.empty:
            assert "m_ret" in df.columns

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_with_date_filter(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        start = dt.date(2023, 6, 1)
        end = dt.date(2023, 12, 31)
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            start_date=start,
            end_date=end,
            join_market=False,
        )
        if not df.empty:
            assert df["date"].min() >= pd.Timestamp(start)
            assert df["date"].max() <= pd.Timestamp(end)

    def test_build_dataset_empty_tickers(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR)
        df = loader.build_dataset(tickers=[])
        assert df.empty

    def test_build_dataset_unknown_tickers(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR)
        df = loader.build_dataset(tickers=["ZZZZNOTEXIST1", "ZZZZNOTEXIST2"])
        assert df.empty

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_inference_dataset(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_inference_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            lookback_days=365,
        )
        assert not df.empty
        assert "ticker" in df.columns

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_dataset_sorted_by_ticker_date(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:2],
            join_market=False,
        )
        if len(df) > 1:
            # Verify sorted by [ticker, date]
            is_sorted = (
                df["ticker"].is_monotonic_increasing
                or all(
                    df.groupby("ticker")["date"]
                    .apply(lambda s: s.is_monotonic_increasing)
                )
            )
            assert is_sorted

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_dataset_has_no_duplicate_ticker_date_keys(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:2],
            join_market=False,
        )
        assert not df.duplicated(subset=["ticker", "date"]).any()


# ═══════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════


class TestConvenienceFunction:
    """Test the load_vn100_daily_dataset one-liner."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_explicit_tickers(self):
        df = load_vn100_daily_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            start_date=dt.date(2023, 1, 1),
            join_market=False,
            prefer_source="csv",
        )
        assert not df.empty


# ═══════════════════════════════════════════════════════════════════
# DataFrame schema validation
# ═══════════════════════════════════════════════════════════════════


class TestDataFrameSchema:
    """Verify the output DataFrame matches the ML pipeline expectations."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_output_columns_match_pipeline(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=False,
        )
        if not df.empty:
            required = {"date", "open", "high", "low", "close", "volume", "ticker"}
            assert required.issubset(set(df.columns)), (
                f"Missing columns: {required - set(df.columns)}"
            )

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_numeric_types(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=False,
        )
        if not df.empty:
            assert pd.api.types.is_float_dtype(df["close"])
            assert pd.api.types.is_float_dtype(df["open"])
            assert pd.api.types.is_integer_dtype(df["volume"])
