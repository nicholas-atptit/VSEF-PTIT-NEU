from pathlib import Path
import csv
import inspect

from scripts.research import build_vn30_2015_benchmark_readiness_manifest as readiness


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_UNIVERSE = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"


def read_active_tickers() -> list[str]:
    with ACTIVE_UNIVERSE.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["ticker"].strip().upper() for row in csv.DictReader(handle)]


def stock_rows(tickers: list[str]) -> list[dict[str, str]]:
    return [
        {
            "ticker": ticker,
            "file_exists": "true",
            "row_count": "1500",
            "first_datetime": "2023-09-11 10:00:00",
            "last_datetime": "2026-05-15 00:00:00",
            "training_rows_before_2025": "1321",
            "evaluation_rows_from_2025": "176",
            "usable": "true",
        }
        for ticker in tickers
    ]


def index_rows() -> list[dict[str, str]]:
    return [
        {
            "index_code": code,
            "usable": "true",
            "row_count": "1000",
            "first_datetime": "2022-05-19 00:00:00",
            "last_datetime": "2026-05-15 00:00:00",
        }
        for code in readiness.REQUIRED_INDEX_CODES
    ]


def universe_rows(tickers: list[str]) -> list[dict[str, str]]:
    return [{"ticker": ticker} for ticker in tickers]


def effective_rows(tickers: list[str]) -> list[dict[str, str]]:
    return [{"ticker": ticker, "effective_start": "2015-01-01", "needs_listing_date_verification": "no"} for ticker in tickers]


def excluded_rows() -> list[dict[str, str]]:
    return [{"ticker": ticker} for ticker in readiness.REQUIRED_EXCLUSIONS]


def test_active_universe_is_jan2025_counted_and_deduped():
    tickers = read_active_tickers()
    assert len(tickers) == readiness.VN30_EXPECTED_COUNT
    assert len(tickers) == len(set(tickers))
    assert {"BCM", "BVH"}.issubset(tickers)
    assert {"BSR", "DGC", "VPL"}.isdisjoint(tickers)


def test_readiness_fails_if_validation_ticker_set_mismatches_active_universe(tmp_path):
    tickers = list(readiness.JAN2025_TICKERS)
    command = tmp_path / "benchmark.py"
    command.write_text("# placeholder for path-existence test\n", encoding="utf-8")

    payload = readiness.build_payload(
        stock_rows([ticker for ticker in tickers if ticker != "BCM"]),
        index_rows(),
        effective_rows(tickers),
        [],
        universe_rows(tickers),
        excluded_reference_rows=excluded_rows(),
        benchmark_command_path=command,
    )

    assert payload["benchmark_can_proceed"] is False
    assert payload["required_tickers_missing_from_validation"] == ["BCM"]
    assert "required_tickers_missing_from_validation=BCM" in payload["blocking_reasons"]


def test_benchmark_command_path_must_exist_for_readiness(tmp_path):
    tickers = list(readiness.JAN2025_TICKERS)
    missing_command = tmp_path / "missing_benchmark.py"

    payload = readiness.build_payload(
        stock_rows(tickers),
        index_rows(),
        effective_rows(tickers),
        [],
        universe_rows(tickers),
        excluded_reference_rows=excluded_rows(),
        benchmark_command_path=missing_command,
    )

    assert payload["benchmark_can_proceed"] is False
    assert payload["benchmark_command_exists"] is False
    assert any(reason.startswith("benchmark_command_missing=") for reason in payload["blocking_reasons"])


def test_readiness_can_pass_when_sets_and_command_are_valid(tmp_path):
    tickers = list(readiness.JAN2025_TICKERS)
    command = tmp_path / "benchmark.py"
    command.write_text("# placeholder for path-existence test\n", encoding="utf-8")

    payload = readiness.build_payload(
        stock_rows(tickers),
        index_rows(),
        effective_rows(tickers),
        [],
        universe_rows(tickers),
        excluded_reference_rows=excluded_rows(),
        benchmark_command_path=command,
    )

    assert payload["benchmark_can_proceed"] is True
    assert payload["benchmark_command_exists"] is True
    assert payload["active_universe_actual_count"] == readiness.VN30_EXPECTED_COUNT


def test_readiness_uses_policy_constant_not_legacy_count_check():
    source = inspect.getsource(readiness)
    assert "VN30_EXPECTED_COUNT = 30" in source
    assert "len(usable_stocks) == 30" not in source
