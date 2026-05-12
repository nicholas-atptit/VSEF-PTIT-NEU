import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.research

OUTPUT_DIR = Path("outputs/vn100_hybrid_accuracy_benchmark")
DAILY_SUMMARY_PATH = OUTPUT_DIR / "daily" / "benchmark_summary.json"
HOURLY_SUMMARY_PATH = OUTPUT_DIR / "hourly" / "benchmark_summary.json"
BENCHMARK_COMMAND = "python scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py"
REQUIRED_DAILY_ACCURACY = 0.60
REQUIRED_DAILY_PREDICTIONS = 5000
EVALUATION_TYPE = "walk_forward_out_of_sample"


def _read_summary(path: Path) -> dict:
    if not path.exists():
        pytest.fail(
            f"Benchmark summary does not exist: {path}. "
            f"Run the benchmark first with: {BENCHMARK_COMMAND}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_daily_vn100_hybrid_benchmark_passes_60pct_gate():
    if os.getenv("VSEF_SKIP_BENCHMARK") == "1":
        pytest.skip("VSEF_SKIP_BENCHMARK=1 explicitly skips the VN100 daily benchmark gate.")

    summary = _read_summary(DAILY_SUMMARY_PATH)
    accuracy = float(summary.get("overall_accuracy", 0.0))
    n_predictions = int(summary.get("n_predictions", 0))

    assert summary.get("evaluation_type") == EVALUATION_TYPE, (
        f"Daily VN100 benchmark has invalid evaluation_type={summary.get('evaluation_type')!r}; "
        f"expected {EVALUATION_TYPE!r}."
    )
    assert n_predictions >= REQUIRED_DAILY_PREDICTIONS, (
        f"Daily VN100 benchmark has too few predictions: "
        f"n_predictions={n_predictions}, required_min={REQUIRED_DAILY_PREDICTIONS}, "
        f"accuracy={accuracy:.4f}, threshold={REQUIRED_DAILY_ACCURACY:.4f}"
    )
    assert accuracy >= REQUIRED_DAILY_ACCURACY, (
        f"Daily VN100 benchmark failed: overall_accuracy={accuracy:.4f}, "
        f"threshold={REQUIRED_DAILY_ACCURACY:.4f}, n_predictions={n_predictions}"
    )


def test_hourly_vn100_hybrid_benchmark_gate_when_enabled():
    if os.getenv("VSEF_STRICT_HOURLY_BENCHMARK") != "1":
        pytest.skip("Set VSEF_STRICT_HOURLY_BENCHMARK=1 to run the VN100 strict hourly benchmark gate.")

    summary = _read_summary(HOURLY_SUMMARY_PATH)
    accuracy = float(summary.get("overall_accuracy", 0.0))
    threshold = float(summary.get("threshold", 0.60))
    n_predictions = int(summary.get("n_predictions", 0))

    assert summary.get("evaluation_type") == EVALUATION_TYPE
    assert accuracy >= threshold, (
        f"Hourly VN100 benchmark failed: overall_accuracy={accuracy:.4f}, "
        f"threshold={threshold:.4f}, n_predictions={n_predictions}"
    )
    assert n_predictions > 0, (
        f"Hourly VN100 benchmark produced no predictions: "
        f"n_predictions={n_predictions}, accuracy={accuracy:.4f}, threshold={threshold:.4f}"
    )
