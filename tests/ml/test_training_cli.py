from __future__ import annotations

import sys
from pathlib import Path

from scripts.train_ml_tickers import parse_args
import scripts.train_ml_tickers as training_cli


def test_cli_parses_new_algorithm_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ml_tickers.py",
            "--tickers",
            "SSI",
            "--algorithms",
            "cart,lstm,bilstm",
            "--primary-algorithm",
            "lstm",
            "--sequence-length",
            "20",
            "--epochs",
            "50",
            "--batch-size",
            "32",
            "--enable-covar",
            "--enable-risk-engine",
            "--enable-regime",
            "--enable-regime-switching",
            "--enable-allocation",
            "--covar-quantile",
            "0.1",
            "--covar-window",
            "45",
            "--risk-penalty-strength",
            "1.5",
            "--enable-benchmark",
            "--enable-stress-test",
            "--enable-risk-tuning",
            "--risk-tuning-trials",
            "3",
        ],
    )

    args = parse_args()

    assert args.algorithms == "cart,lstm,bilstm"
    assert args.primary_algorithm == "lstm"
    assert args.sequence_length == 20
    assert args.epochs == 50
    assert args.batch_size == 32
    assert args.enable_covar is True
    assert args.enable_risk_engine is True
    assert args.enable_regime is True
    assert args.enable_regime_switching is True
    assert args.enable_allocation is True
    assert args.covar_quantile == 0.1
    assert args.covar_window == 45
    assert args.risk_penalty_strength == 1.5
    assert args.enable_benchmark is True
    assert args.enable_stress_test is True
    assert args.enable_risk_tuning is True
    assert args.risk_tuning_trials == 3


def test_cli_advanced_workflow_runs_sequentially(monkeypatch, tmp_path, capsys) -> None:
    calls: list[str] = []
    benchmark_path = tmp_path / "benchmark.csv"
    stress_path = tmp_path / "stress.csv"
    tuning_path = tmp_path / "tuning.csv"
    final_report_path = tmp_path / "full_system_report.md"

    class _Runner:
        def __init__(self, *, model_root) -> None:
            self.model_root = model_root

        def run(self, **kwargs):
            output_root = kwargs["output_root"]
            if isinstance(self, _BenchmarkRunner):
                calls.append("benchmark")
                return {
                    "summary": [],
                    "detail_path": Path(kwargs["report_path"]),
                }
            if isinstance(self, _StressRunner):
                calls.append("stress")
                return {
                    "summary": [],
                    "detail_path": Path(kwargs["report_path"]),
                }
            calls.append("tuning")
            return {
                "best_score": 1.0,
                "best_params": {"covar_window": 60},
                "csv_path": Path(kwargs["report_path"]),
            }

    class _BenchmarkRunner(_Runner):
        pass

    class _StressRunner(_Runner):
        pass

    class _TuningRunner(_Runner):
        pass

    def _write_report(**kwargs):
        calls.append("report")
        final_report_path.write_text("ok", encoding="utf-8")
        return final_report_path

    monkeypatch.setattr(training_cli, "resolve_files", lambda args: [tmp_path / "AAA.csv"])
    monkeypatch.setattr(training_cli, "SystemBenchmarkRunner", _BenchmarkRunner)
    monkeypatch.setattr(training_cli, "StressTestRunner", _StressRunner)
    monkeypatch.setattr(training_cli, "RiskTuningRunner", _TuningRunner)
    monkeypatch.setattr(training_cli, "write_full_system_report", _write_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ml_tickers.py",
            "--tickers",
            "AAA",
            "--algorithms",
            "cart",
            "--enable-benchmark",
            "--enable-stress-test",
            "--enable-risk-tuning",
            "--benchmark-report",
            str(benchmark_path),
            "--stress-report",
            str(stress_path),
            "--tuning-report",
            str(tuning_path),
            "--risk-tuning-trials",
            "2",
        ],
    )

    training_cli.main()
    output = capsys.readouterr().out

    assert calls == ["benchmark", "stress", "tuning", "report"]
    assert str(final_report_path) in output
    assert str(benchmark_path) in output
    assert str(stress_path) in output
    assert str(tuning_path) in output
