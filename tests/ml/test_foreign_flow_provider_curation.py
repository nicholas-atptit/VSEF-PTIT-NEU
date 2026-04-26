from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.curate_foreign_flow_provider_artifact import (
    DEFAULT_OUTPUT_PATH,
    LEGACY_FALLBACK_PATH,
    curate_foreign_flow_provider_artifact,
)


class _ProviderUnavailableAdapter:
    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    def _get_class(self, name: str):
        return None


class _MockTrading:
    pass


class _MockProviderAdapter:
    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    def _get_class(self, name: str):
        return _MockTrading if name == "Trading" else None

    def get_foreign_flow(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(start_date, end_date)
        return pd.DataFrame(
            {
                "date": dates,
                "foreign_net_volume": [10.0 + idx for idx in range(len(dates))],
                "foreign_net_value": [100_000.0 + idx for idx in range(len(dates))],
                "foreign_buy_volume": [60.0 + idx for idx in range(len(dates))],
                "foreign_sell_volume": [50.0 for _ in dates],
            }
        )


class _FixtureLabelProviderAdapter(_MockProviderAdapter):
    def get_foreign_flow(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        frame = super().get_foreign_flow(ticker, start_date, end_date)
        frame["source"] = "fixture_sample"
        frame["provider"] = "non_real_fixture"
        return frame


def test_provider_unavailable_skips_gracefully(tmp_path: Path) -> None:
    output = tmp_path / "foreign_flow_curated.csv"

    report = curate_foreign_flow_provider_artifact(
        tickers="SSI,FPT",
        start_date="2025-01-02",
        end_date="2025-01-10",
        output_path=output,
        adapter_factory=_ProviderUnavailableAdapter,
    )

    assert report["status"] == "provider_unavailable"
    assert report["provider_fetch_attempted"] is False
    assert report["real_data_fetched"] is False
    assert report["validation"]["artifact_classification"] == "empty_or_missing"
    assert not output.exists()


def test_mocked_provider_writes_provenance_columns(tmp_path: Path) -> None:
    output = tmp_path / "foreign_flow_curated.csv"

    report = curate_foreign_flow_provider_artifact(
        tickers=["SSI", "FPT"],
        start_date="2025-01-02",
        end_date="2025-01-10",
        output_path=output,
        adapter_factory=_MockProviderAdapter,
        retrieved_at="2026-04-26T00:00:00Z",
    )
    artifact = pd.read_csv(output)

    assert report["status"] == "curated"
    assert report["provider_fetch_attempted"] is True
    assert report["real_data_fetched"] is True
    assert {"source", "source_date", "retrieved_at", "provider", "coverage_note"} <= set(artifact.columns)
    assert set(artifact["provider"]) == {"vnstock_data"}
    assert set(artifact["source"]) == {"vnstock_data.Trading.foreign_trade"}


def test_mocked_full_coverage_classifies_as_usable(tmp_path: Path) -> None:
    report = curate_foreign_flow_provider_artifact(
        tickers="SSI,FPT",
        start_date="2025-01-02",
        end_date="2025-01-10",
        output_path=tmp_path / "foreign_flow_curated.csv",
        adapter_factory=_MockProviderAdapter,
        retrieved_at="2026-04-26T00:00:00Z",
    )

    assert report["validation"]["artifact_classification"] == "usable_for_requested_window"
    assert report["validation"]["requested_ticker_date_coverage_rate"] == 1.0
    assert report["validation"]["real_provider_evidence"] is True


def test_fixture_sample_provider_labels_are_not_real_provider_evidence(tmp_path: Path) -> None:
    output = tmp_path / "foreign_flow_curated.csv"

    report = curate_foreign_flow_provider_artifact(
        tickers="SSI",
        start_date="2025-01-02",
        end_date="2025-01-10",
        output_path=output,
        adapter_factory=_FixtureLabelProviderAdapter,
        retrieved_at="2026-04-26T00:00:00Z",
    )
    artifact = pd.read_csv(output)

    assert report["status"] == "curated_but_not_real_provider_evidence"
    assert report["validation"]["artifact_classification"] == "usable_for_requested_window"
    assert report["validation"]["fixture_or_sample_source"] is True
    assert report["validation"]["real_provider_evidence"] is False
    assert set(artifact["source"]) == {"fixture_sample"}
    assert set(artifact["provider"]) == {"non_real_fixture"}


def test_default_output_path_does_not_overwrite_legacy_foreign_flow() -> None:
    assert DEFAULT_OUTPUT_PATH.name == "foreign_flow_curated.csv"
    assert DEFAULT_OUTPUT_PATH != LEGACY_FALLBACK_PATH


def test_explicit_legacy_foreign_flow_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        curate_foreign_flow_provider_artifact(
            tickers="SSI",
            start_date="2025-01-02",
            end_date="2025-01-10",
            output_path=LEGACY_FALLBACK_PATH,
            adapter_factory=_ProviderUnavailableAdapter,
        )
