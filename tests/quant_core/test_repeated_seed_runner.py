from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from src.evaluation.repeated_seed_runner import (
    ROOT_OUTPUT_FILES,
    RepeatedSeedConfig,
    build_seed_command,
    is_seed_complete,
    run_repeated_seed_stability,
    seed_directory,
    write_repeated_seed_outputs,
)


def _write_required_outputs(seed_dir: Path, seed: int) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "run_manifest.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model_name": "naive",
                "target_type": "forward_return",
                "rmse": 0.1 + seed,
                "mae": 0.05 + seed,
                "directional_accuracy": 0.5,
                "f1": 0.4,
                "observations": 12,
            }
        ]
    ).to_csv(seed_dir / "forecast_summary.csv", index=False)
    pd.DataFrame([{"model_name": "naive", "status": "success"}]).to_csv(
        seed_dir / "model_execution_log.csv",
        index=False,
    )


def _write_fake_quant_core_script(path: Path, *, fail_seed_one: bool = False) -> None:
    path.write_text(
        f"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--preset", required=True)
parser.add_argument("--run-mode", required=True)
parser.add_argument("--max-workers", type=int, default=1)
parser.add_argument("--models", nargs="+", default=None)
parser.add_argument("--model-roles", nargs="+", default=None)
parser.add_argument("--horizons", nargs="+", type=int, default=None)
parser.add_argument("--target-types", nargs="+", default=None)
parser.add_argument("--no-ensemble", action="store_true")
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
if {str(bool(fail_seed_one))} and args.seed == 1:
    (out / "partial.txt").write_text("failed before required outputs", encoding="utf-8")
    sys.exit(7)
(out / "run_manifest.json").write_text(
    json.dumps({{"seed": args.seed, "run_mode": args.run_mode, "preset": args.preset}}),
    encoding="utf-8",
)
pd.DataFrame([{{"model_name": "naive", "target_type": "forward_return", "rmse": args.seed + 0.1, "mae": args.seed + 0.05, "directional_accuracy": 0.5, "f1": 0.4, "observations": 10}}]).to_csv(out / "forecast_summary.csv", index=False)
pd.DataFrame([{{"model_name": "naive", "horizon": 5, "target_type": "forward_return", "rmse": args.seed + 0.2, "mae": args.seed + 0.1, "directional_accuracy": 0.6, "f1": 0.45, "observations": 10}}]).to_csv(out / "forecast_summary_by_horizon.csv", index=False)
pd.DataFrame([{{"model_name": "naive", "status": "success"}}]).to_csv(out / "model_execution_log.csv", index=False)
pd.DataFrame([{{"model_name": "naive", "horizon": 5, "target_type": "forward_return", "run_mode": args.run_mode, "policy_variant": "baseline", "sharpe": 0.1, "cagr": 0.02, "max_drawdown": -0.03, "total_return": 0.04, "win_rate": 0.5, "trade_count": 3}}]).to_csv(out / "strategy_metrics.csv", index=False)
pd.DataFrame([{{"model_name": "naive", "run_success_rate": 1.0, "forecast_observations_total": 10, "mean_rmse": args.seed + 0.1, "mean_directional_accuracy": 0.5, "policy_eval_count": 1, "mean_sharpe": 0.1, "positive_policy_frequency": 1.0}}]).to_csv(out / "model_health_summary.csv", index=False)
pd.DataFrame([{{"horizon": 5, "target_type": "forward_return", "run_mode": args.run_mode, "agreement_score": 1.0, "disagreement_score": 0.0, "dispersion_score": 0.1, "sign_conflict_rate": 0.0, "active_signal_share": 1.0}}]).to_csv(out / "model_consensus_summary.csv", index=False)
pd.DataFrame([{{"horizon": 5, "target_type": "forward_return", "run_mode": args.run_mode, "candidate_score": 0.2, "primary_prediction": 0.01, "model_agreement_score": 1.0, "active_signal_count": 1, "top_policy_sharpe": 0.1}}]).to_csv(out / "decision_lane_candidates.csv", index=False)
""",
        encoding="utf-8",
    )


def test_build_seed_command_uses_real_quant_core_script_and_forwarded_args(tmp_path: Path) -> None:
    config = RepeatedSeedConfig(
        output_dir=tmp_path / "runs",
        seeds=(1,),
        preset="smoke",
        run_mode="baseline_only",
        models=("naive",),
        model_roles=("baseline",),
        horizons=(5, 10),
        target_types=("forward_return",),
        max_workers=2,
        python_executable="python",
        no_ensemble=True,
    )

    command = build_seed_command(config, 1)

    assert command[0] == "python"
    assert Path(command[1]) == Path("scripts/run_quant_core.py")
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--output-dir") + 1].endswith("seed_000001")
    assert command[command.index("--preset") + 1] == "smoke"
    assert command[command.index("--run-mode") + 1] == "baseline_only"
    assert command[command.index("--max-workers") + 1] == "2"
    assert command[command.index("--models") + 1] == "naive"
    assert command[command.index("--model-roles") + 1] == "baseline"
    assert command[command.index("--horizons") + 1 : command.index("--target-types")] == ["5", "10"]
    assert command[command.index("--target-types") + 1] == "forward_return"
    assert "--no-ensemble" in command


def test_is_seed_complete_requires_manifest_forecast_summary_and_execution_log(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_000001"
    seed_dir.mkdir()
    assert not is_seed_complete(seed_dir)
    (seed_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (seed_dir / "forecast_summary.csv").write_text("model_name\nnaive\n", encoding="utf-8")
    assert not is_seed_complete(seed_dir)
    (seed_dir / "model_execution_log.csv").write_text("model_name,status\nnaive,success\n", encoding="utf-8")
    assert is_seed_complete(seed_dir)


def test_dry_run_does_not_create_seed_dirs(tmp_path: Path) -> None:
    config = RepeatedSeedConfig(
        output_dir=tmp_path / "dry",
        seeds=(1, 2, 3),
        dry_run=True,
        python_executable=sys.executable,
    )

    result = run_repeated_seed_stability(config)

    assert set(result.execution_log["status"]) == {"dry_run"}
    assert result.output_paths == {}
    assert not config.output_dir.exists()


def test_resume_skips_complete_seed_and_reruns_incomplete_seed(tmp_path: Path) -> None:
    fake_script = tmp_path / "fake_quant_core.py"
    _write_fake_quant_core_script(fake_script)
    output_dir = tmp_path / "runs"
    _write_required_outputs(seed_directory(output_dir, 1), 1)
    config = RepeatedSeedConfig(
        output_dir=output_dir,
        seeds=(1, 2),
        run_mode="baseline_only",
        resume=True,
        python_executable=sys.executable,
        quant_core_script=fake_script,
        cwd=tmp_path,
    )

    result = run_repeated_seed_stability(config)

    statuses = dict(zip(result.execution_log["seed"], result.execution_log["status"]))
    assert statuses == {1: "skipped_complete", 2: "completed"}
    assert is_seed_complete(seed_directory(output_dir, 2))
    assert (output_dir / "seed_execution_log.csv").exists()


def test_failed_seed_is_recorded_and_execution_continues_by_default(tmp_path: Path) -> None:
    fake_script = tmp_path / "fake_quant_core.py"
    _write_fake_quant_core_script(fake_script, fail_seed_one=True)
    config = RepeatedSeedConfig(
        output_dir=tmp_path / "runs",
        seeds=(1, 2),
        run_mode="baseline_only",
        stop_on_failure=False,
        python_executable=sys.executable,
        quant_core_script=fake_script,
        cwd=tmp_path,
    )

    result = run_repeated_seed_stability(config)

    statuses = dict(zip(result.execution_log["seed"], result.execution_log["status"]))
    assert statuses == {1: "failed", 2: "completed"}
    assert (seed_directory(config.output_dir, 1) / "partial.txt").exists()
    failure_summary = pd.read_csv(config.output_dir / "failure_summary.csv")
    assert "failed" in set(failure_summary["status"])


def test_stop_on_failure_stops_after_first_failed_seed(tmp_path: Path) -> None:
    fake_script = tmp_path / "fake_quant_core.py"
    _write_fake_quant_core_script(fake_script, fail_seed_one=True)
    config = RepeatedSeedConfig(
        output_dir=tmp_path / "runs",
        seeds=(1, 2),
        stop_on_failure=True,
        python_executable=sys.executable,
        quant_core_script=fake_script,
        cwd=tmp_path,
    )

    result = run_repeated_seed_stability(config)

    assert result.execution_log["seed"].tolist() == [1]
    assert result.execution_log.loc[0, "status"] == "failed"
    assert not seed_directory(config.output_dir, 2).exists()


def test_aggregation_handles_missing_optional_seed_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    _write_required_outputs(seed_directory(output_dir, 1), 1)
    config = RepeatedSeedConfig(output_dir=output_dir, seeds=(1,))
    execution_log = pd.DataFrame(
        [
            {
                "seed": 1,
                "seed_dir": str(seed_directory(output_dir, 1)),
                "command": "fake",
                "status": "completed",
                "return_code": 0,
                "started_at": "2026-04-30T00:00:00+00:00",
                "completed_at": "2026-04-30T00:00:01+00:00",
                "duration_seconds": 1.0,
                "was_complete_before_run": False,
                "required_files_present": True,
                "error_message": "",
            }
        ]
    )

    paths = write_repeated_seed_outputs(config, execution_log)

    for filename in ROOT_OUTPUT_FILES:
        assert filename in paths
        assert paths[filename].exists()
    stability = pd.read_csv(output_dir / "model_seed_stability.csv")
    assert set(["model_name", "target_type", "rmse_mean", "rmse_p95"]).issubset(stability.columns)
