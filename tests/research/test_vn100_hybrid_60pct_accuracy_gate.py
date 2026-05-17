import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.research

BENCHMARK_OUTPUT_DIR_ENV = "VSEF_BENCHMARK_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR = Path("outputs/vn100_hybrid_accuracy_benchmark")
BENCHMARK_COMMAND = "python scripts/run_vn100_hybrid_frequency_accuracy_benchmark.py"
REQUIRED_DAILY_ACCURACY = 0.60
REQUIRED_DAILY_PREDICTIONS = 5000
EVALUATION_TYPE = "walk_forward_out_of_sample"


def _benchmark_output_dir() -> Path:
    override = os.getenv(BENCHMARK_OUTPUT_DIR_ENV)
    if override and override.strip():
        return Path(override)
    return DEFAULT_OUTPUT_DIR


def _summary_path(frequency: str) -> Path:
    return _benchmark_output_dir() / frequency / "benchmark_summary.json"


def _read_summary(path: Path) -> dict:
    if not path.exists():
        pytest.fail(
            f"Benchmark summary does not exist at checked path: {path}. "
            f"Run the benchmark first with: {BENCHMARK_COMMAND}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_output_dir_defaults_to_standard_path(monkeypatch):
    monkeypatch.delenv(BENCHMARK_OUTPUT_DIR_ENV, raising=False)

    assert _benchmark_output_dir() == DEFAULT_OUTPUT_DIR
    assert _summary_path("daily") == DEFAULT_OUTPUT_DIR / "daily" / "benchmark_summary.json"


def test_benchmark_output_dir_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(BENCHMARK_OUTPUT_DIR_ENV, str(tmp_path))

    assert _benchmark_output_dir() == tmp_path
    assert _summary_path("hourly") == tmp_path / "hourly" / "benchmark_summary.json"


def test_daily_vn100_hybrid_benchmark_passes_60pct_gate():
    if os.getenv("VSEF_SKIP_BENCHMARK") == "1":
        pytest.skip("VSEF_SKIP_BENCHMARK=1 explicitly skips the VN100 daily benchmark gate.")

    summary = _read_summary(_summary_path("daily"))
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

    summary = _read_summary(_summary_path("hourly"))
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
