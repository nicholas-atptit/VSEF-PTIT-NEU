from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.ml.backtest.signal_effectiveness import (
    BENCHMARK_COLUMNS,
    FRONTIER_COLUMNS,
    POLICY_RETURN_THRESHOLD,
    SIGNAL_BUY,
    SIGNAL_ROW_COLUMNS,
    STRATEGY_PROXY_COLUMNS,
    SUCCESS_COST_ADJUSTED_POSITIVE,
    SUCCESS_RAW_POSITIVE,
    SUCCESS_TARGET_RETURN,
    SUMMARY_COLUMNS,
    SignalEffectivenessConfig,
    SignalEffectivenessRunner,
    generate_signal_rows,
    normalize_prediction_frame,
    summarize_signal_effectiveness,
)


def _raw_predictions(rows: list[tuple[str, str, str, float, float, int | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_date": date,
                "target_date": pd.Timestamp(date) + pd.Timedelta(days=5),
                "ticker": ticker,
                "horizon": horizon,
                "model_name": model,
                "predicted_return": predicted,
                "predicted_direction": direction,
                "actual_return": realized,
            }
            for date, ticker, model, predicted, realized, direction in rows
            for horizon in ["short_5d"]
        ]
    )


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    normalized, _metadata = normalize_prediction_frame(frame)
    return normalized


def test_buy_precision_is_computed_correctly() -> None:
    frame = _normalize(
        _raw_predictions(
            [
                ("2026-01-02", "AAA", "cart", 0.020, 0.030, 1),
                ("2026-01-03", "BBB", "cart", 0.015, -0.010, 1),
                ("2026-01-04", "CCC", "cart", -0.020, -0.030, 0),
                ("2026-01-05", "DDD", "cart", 0.005, 0.020, 1),
            ]
        )
    )

    signals = generate_signal_rows(
        frame,
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_threshold=0.01,
        cost_per_trade=0.0,
        slippage=0.0,
        success_definition=SUCCESS_RAW_POSITIVE,
    )
    summary = summarize_signal_effectiveness(signals, minimum_signal_counts=[1])
    row = summary.iloc[0]

    assert row["buy_signal_count"] == 2
    assert row["hold_signal_count"] == 1
    assert row["avoid_signal_count"] == 1
    assert row["buy_precision"] == pytest.approx(0.5)
    assert row["buy_recall"] == pytest.approx(0.5)


def test_cost_adjusted_success_requires_return_above_round_trip_cost() -> None:
    frame = _normalize(
        _raw_predictions(
            [
                ("2026-01-02", "AAA", "cart", 0.020, 0.002, 1),
                ("2026-01-03", "BBB", "cart", 0.030, 0.005, 1),
            ]
        )
    )

    signals = generate_signal_rows(
        frame,
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_threshold=0.01,
        cost_per_trade=0.001,
        slippage=0.0005,
        success_definition=SUCCESS_COST_ADJUSTED_POSITIVE,
    )

    assert signals["estimated_round_trip_cost"].iloc[0] == pytest.approx(0.003)
    assert signals["buy_success"].tolist() == [False, True]


def test_target_return_success_uses_configured_threshold() -> None:
    frame = _normalize(
        _raw_predictions(
            [
                ("2026-01-02", "AAA", "cart", 0.020, 0.019, 1),
                ("2026-01-03", "BBB", "cart", 0.030, 0.025, 1),
            ]
        )
    )

    signals = generate_signal_rows(
        frame,
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_threshold=0.01,
        cost_per_trade=0.0,
        slippage=0.0,
        success_definition=SUCCESS_TARGET_RETURN,
        target_return_threshold=0.02,
    )

    assert signals["buy_success"].tolist() == [False, True]


def test_threshold_frontier_fewer_signals_and_higher_precision(tmp_path) -> None:
    rows: list[tuple[str, str, str, float, float, int | None]] = []
    date = pd.Timestamp("2026-01-02")
    for idx in range(30):
        rows.append((str((date + pd.Timedelta(days=idx)).date()), f"A{idx:02d}", "cart", 0.006, 0.01 if idx < 15 else -0.01, 1))
    for idx in range(20):
        rows.append((str((date + pd.Timedelta(days=40 + idx)).date()), f"B{idx:02d}", "cart", 0.020, 0.01 if idx < 16 else -0.01, 1))
    for idx in range(10):
        rows.append((str((date + pd.Timedelta(days=80 + idx)).date()), f"C{idx:02d}", "cart", 0.040, 0.02, 1))

    config = SignalEffectivenessConfig(
        output_dir=str(tmp_path / "signal_effectiveness"),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.005, 0.015, 0.03],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=[1],
    )
    result = SignalEffectivenessRunner(config).run(_raw_predictions(rows))
    frontier = result["precision_coverage_frontier"].sort_values("predicted_return_threshold")

    counts = frontier["buy_signal_count"].tolist()
    precision = frontier["buy_precision"].tolist()
    assert counts == [60, 30, 10]
    assert precision[0] < precision[1] < precision[2]


def test_minimum_signal_count_filter_controls_buy_precision_table(tmp_path) -> None:
    frame = _raw_predictions(
        [
            ("2026-01-02", "AAA", "cart", 0.020, 0.030, 1),
            ("2026-01-03", "BBB", "cart", 0.020, 0.030, 1),
            ("2026-01-04", "CCC", "cart", 0.020, 0.030, 1),
            ("2026-01-05", "DDD", "cart", 0.020, 0.030, 1),
        ]
    )
    config = SignalEffectivenessConfig(
        output_dir=str(tmp_path / "signal_effectiveness"),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.01],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=[3, 5],
    )

    result = SignalEffectivenessRunner(config).run(frame)
    summary = result["signal_effectiveness_summary"].sort_values("minimum_signal_count")
    precision = result["buy_precision_by_model_horizon"]

    assert summary["passes_minimum_signal_count"].tolist() == [True, False]
    assert precision["minimum_signal_count"].tolist() == [3]


def test_signal_creation_does_not_use_realized_return() -> None:
    base = _raw_predictions(
        [
            ("2026-01-02", "AAA", "cart", 0.020, 0.030, 1),
            ("2026-01-03", "BBB", "cart", -0.020, -0.030, 0),
            ("2026-01-04", "CCC", "cart", 0.001, -0.050, 1),
        ]
    )
    modified = base.copy()
    modified["actual_return"] = [-0.99, 0.99, 0.99]

    base_signals = generate_signal_rows(
        _normalize(base),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_threshold=0.01,
        cost_per_trade=0.0,
        slippage=0.0,
        success_definition=SUCCESS_RAW_POSITIVE,
    )
    modified_signals = generate_signal_rows(
        _normalize(modified),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_threshold=0.01,
        cost_per_trade=0.0,
        slippage=0.0,
        success_definition=SUCCESS_RAW_POSITIVE,
    )

    assert base_signals["signal"].tolist() == modified_signals["signal"].tolist()


def test_output_schema_is_stable(tmp_path) -> None:
    frame = _raw_predictions(
        [
            ("2026-01-02", "AAA", "cart", 0.020, 0.030, 1),
            ("2026-01-03", "BBB", "cart", -0.020, -0.030, 0),
        ]
    )
    config = SignalEffectivenessConfig(
        output_dir=str(tmp_path / "signal_effectiveness"),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.01],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=[1],
    )
    result = SignalEffectivenessRunner(config).run(frame)

    assert list(result["signal_rows"].columns) == SIGNAL_ROW_COLUMNS
    assert list(result["signal_effectiveness_summary"].columns) == SUMMARY_COLUMNS
    assert list(result["precision_coverage_frontier"].columns) == FRONTIER_COLUMNS
    assert list(result["strategy_proxy_metrics"].columns) == STRATEGY_PROXY_COLUMNS
    assert list(result["benchmark_comparison"].columns) == BENCHMARK_COLUMNS
    metadata = json.loads(Path(result["paths"]["run_metadata"]).read_text(encoding="utf-8"))
    assert metadata["analysis_only"] is True
    assert metadata["leakage_guard"]["realized_return_used_for_signal_creation"] is False


def test_empty_no_buy_cases_do_not_crash(tmp_path) -> None:
    frame = _raw_predictions(
        [
            ("2026-01-02", "AAA", "cart", 0.001, 0.030, 1),
            ("2026-01-03", "BBB", "cart", -0.001, -0.030, 0),
        ]
    )
    config = SignalEffectivenessConfig(
        output_dir=str(tmp_path / "signal_effectiveness"),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.10],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=[1],
    )
    result = SignalEffectivenessRunner(config).run(frame)
    summary = result["signal_effectiveness_summary"].iloc[0]

    assert (result["signal_rows"]["signal"] != SIGNAL_BUY).all()
    assert summary["buy_signal_count"] == 0
    assert pd.isna(summary["buy_precision"])


def test_cli_runs_on_synthetic_csv(tmp_path) -> None:
    input_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / "signal_effectiveness"
    _raw_predictions(
        [
            ("2026-01-02", "AAA", "cart", 0.020, 0.030, 1),
            ("2026-01-03", "BBB", "cart", -0.020, -0.030, 0),
        ]
    ).to_csv(input_path, index=False)

    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_signal_effectiveness_backtest.py",
            "--predictions-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--policy",
            POLICY_RETURN_THRESHOLD,
            "--threshold-grid",
            "0.01",
            "--cost-per-trade",
            "0.0",
            "--slippage",
            "0.0",
            "--success-definition",
            SUCCESS_RAW_POSITIVE,
            "--minimum-signal-count",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "signal_rows.csv").exists()
    assert (output_dir / "buy_precision_by_model_horizon.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
