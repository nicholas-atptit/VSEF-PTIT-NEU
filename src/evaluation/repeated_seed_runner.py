"""Repeated-seed Quant Core stability runner.

This module orchestrates real per-seed calls to ``scripts/run_quant_core.py`` and
aggregates the CSV outputs that those runs produce. It does not synthesize model
metrics and should not be used to choose a single best seed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEED_COMPLETION_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "forecast_summary.csv",
    "model_execution_log.csv",
)

EXPECTED_SEED_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "summary.md",
    "scenario_matrix.csv",
    "model_governance.csv",
    "full_model_predictions.csv",
    "forecast_summary.csv",
    "forecast_summary_by_horizon.csv",
    "window_summary.csv",
    "risk_summary.csv",
    "regime_summary.csv",
    "signals.csv",
    "positions.csv",
    "trades.csv",
    "strategy_metrics.csv",
    "equity_curve.csv",
    "policy_summary.csv",
    "model_execution_log.csv",
    "model_consensus_summary.csv",
    "model_health_summary.csv",
    "analysis_packets.jsonl",
    "decision_lane_candidates.csv",
)

ROOT_OUTPUT_FILES: tuple[str, ...] = (
    "repeated_seed_manifest.json",
    "seed_execution_log.csv",
    "seed_artifact_inventory.csv",
    "seed_forecast_summary.csv",
    "model_seed_stability.csv",
    "model_horizon_seed_stability.csv",
    "strategy_seed_stability.csv",
    "model_health_seed_stability.csv",
    "consensus_seed_stability.csv",
    "decision_candidate_seed_stability.csv",
    "failure_summary.csv",
    "summary.md",
)

FORECAST_METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "directional_accuracy",
    "f1",
    "observations",
)

STRATEGY_METRICS: tuple[str, ...] = (
    "sharpe",
    "cagr",
    "max_drawdown",
    "total_return",
    "win_rate",
    "trade_count",
)

CONSENSUS_METRICS: tuple[str, ...] = (
    "agreement_score",
    "disagreement_score",
    "dispersion_score",
    "sign_conflict_rate",
    "active_signal_share",
)

DECISION_CANDIDATE_METRICS: tuple[str, ...] = (
    "candidate_count",
    "candidate_score",
    "primary_prediction",
    "model_agreement_score",
    "active_signal_count",
    "top_policy_sharpe",
)

MODEL_HEALTH_METRICS: tuple[str, ...] = (
    "run_success_rate",
    "forecast_observations_total",
    "mean_rmse",
    "mean_directional_accuracy",
    "policy_eval_count",
    "mean_sharpe",
    "positive_policy_frequency",
)


@dataclass(frozen=True)
class RepeatedSeedConfig:
    """Configuration for repeated-seed Quant Core stability runs."""

    output_dir: Path = Path("artifacts/quant_core_repeated_seed")
    seeds: tuple[int, ...] = tuple(range(1, 11))
    preset: str = "smoke"
    run_mode: str = "research_core"
    models: tuple[str, ...] = ()
    model_roles: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    target_types: tuple[str, ...] = ()
    max_workers: int = 1
    resume: bool = True
    stop_on_failure: bool = False
    dry_run: bool = False
    save_seed_outputs: bool = True
    python_executable: str = sys.executable
    no_ensemble: bool = False
    quant_core_script: Path = Path("scripts/run_quant_core.py")
    cwd: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_seed_range(
        cls,
        *,
        seed_start: int = 1,
        seed_count: int = 10,
        **kwargs: Any,
    ) -> "RepeatedSeedConfig":
        if int(seed_count) < 1:
            raise ValueError("seed_count must be >= 1")
        seeds = tuple(range(int(seed_start), int(seed_start) + int(seed_count)))
        return cls(seeds=seeds, **kwargs)


@dataclass(frozen=True)
class SeedExecutionResult:
    seed: int
    seed_dir: Path
    command: list[str]
    command_text: str
    status: str
    return_code: int | None
    started_at: str | None
    completed_at: str | None
    duration_seconds: float
    was_complete_before_run: bool
    required_files_present: bool
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "seed_dir": str(self.seed_dir),
            "command": self.command_text,
            "status": self.status,
            "return_code": self.return_code,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": float(self.duration_seconds),
            "was_complete_before_run": bool(self.was_complete_before_run),
            "required_files_present": bool(self.required_files_present),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RepeatedSeedRunResult:
    config: RepeatedSeedConfig
    execution_log: pd.DataFrame
    output_paths: dict[str, Path]


def resolve_seeds(
    *,
    seed_start: int = 1,
    seed_count: int = 10,
    seeds: Iterable[int] | None = None,
    seeds_file: str | Path | None = None,
) -> tuple[int, ...]:
    """Resolve explicit seeds, file seeds, or a seed range."""

    resolved: list[int] = []
    if seeds:
        resolved.extend(int(seed) for seed in seeds)
    if seeds_file is not None:
        path = Path(seeds_file)
        if not path.exists():
            raise FileNotFoundError(f"seeds_file does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            for token in clean.replace(",", " ").split():
                resolved.append(int(token))
    if not resolved:
        if int(seed_count) < 1:
            raise ValueError("seed_count must be >= 1")
        resolved = list(range(int(seed_start), int(seed_start) + int(seed_count)))
    deduped = tuple(dict.fromkeys(int(seed) for seed in resolved))
    if not deduped:
        raise ValueError("No seeds were resolved")
    return deduped


def seed_directory(output_dir: str | Path, seed: int) -> Path:
    """Return the canonical seed-specific output directory."""

    return Path(output_dir) / f"seed_{int(seed):06d}"


def is_seed_complete(seed_dir: str | Path) -> bool:
    """A seed is complete only when the minimum required files exist."""

    path = Path(seed_dir)
    return all((path / filename).exists() for filename in SEED_COMPLETION_FILES)


def build_seed_command(config: RepeatedSeedConfig, seed: int) -> list[str]:
    """Build the real per-seed Quant Core command."""

    command = [
        str(config.python_executable),
        str(config.quant_core_script),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(seed_directory(config.output_dir, seed)),
        "--preset",
        str(config.preset),
        "--run-mode",
        str(config.run_mode),
        "--max-workers",
        str(int(config.max_workers)),
    ]
    if config.models:
        command.extend(["--models", *[str(model) for model in config.models]])
    if config.model_roles:
        command.extend(["--model-roles", *[str(role) for role in config.model_roles]])
    if config.horizons:
        command.extend(["--horizons", *[str(int(horizon)) for horizon in config.horizons]])
    if config.target_types:
        command.extend(["--target-types", *[str(target_type) for target_type in config.target_types]])
    if config.no_ensemble:
        command.append("--no-ensemble")
    return command


def command_to_text(command: list[str]) -> str:
    """Return a Windows-friendly command string for logs and dry-run output."""

    return subprocess.list2cmdline([str(part) for part in command])


def _tail_output(stdout: str, stderr: str, *, limit: int = 800) -> str:
    text = "\n".join(part for part in [stderr.strip(), stdout.strip()] if part)
    return text[-limit:] if len(text) > limit else text


def _run_one_seed(config: RepeatedSeedConfig, seed: int) -> SeedExecutionResult:
    seed_dir = seed_directory(config.output_dir, seed)
    command = build_seed_command(config, seed)
    command_text = command_to_text(command)
    complete_before = is_seed_complete(seed_dir)
    if config.resume and complete_before:
        return SeedExecutionResult(
            seed=seed,
            seed_dir=seed_dir,
            command=command,
            command_text=command_text,
            status="skipped_complete",
            return_code=None,
            started_at=None,
            completed_at=None,
            duration_seconds=0.0,
            was_complete_before_run=True,
            required_files_present=True,
        )
    if config.dry_run:
        return SeedExecutionResult(
            seed=seed,
            seed_dir=seed_dir,
            command=command,
            command_text=command_text,
            status="dry_run",
            return_code=None,
            started_at=None,
            completed_at=None,
            duration_seconds=0.0,
            was_complete_before_run=complete_before,
            required_files_present=complete_before,
        )

    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(config.cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    completed = datetime.now(timezone.utc)
    duration = time.perf_counter() - start_clock
    complete_after = is_seed_complete(seed_dir)
    status = "completed" if process.returncode == 0 and complete_after else "failed"
    error_message = ""
    if status == "failed":
        if process.returncode == 0 and not complete_after:
            error_message = "Quant Core command exited 0 but required completion files are missing"
        else:
            error_message = _tail_output(process.stdout or "", process.stderr or "")
    return SeedExecutionResult(
        seed=seed,
        seed_dir=seed_dir,
        command=command,
        command_text=command_text,
        status=status,
        return_code=int(process.returncode),
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        duration_seconds=duration,
        was_complete_before_run=complete_before,
        required_files_present=complete_after,
        error_message=error_message,
    )


def run_repeated_seed_stability(config: RepeatedSeedConfig) -> RepeatedSeedRunResult:
    """Run repeated-seed Quant Core jobs and write aggregate outputs."""

    if not config.seeds:
        raise ValueError("At least one seed is required")
    max_workers = max(1, int(config.max_workers))

    if config.dry_run:
        rows = [_run_one_seed(config, seed).to_dict() for seed in config.seeds]
        return RepeatedSeedRunResult(
            config=config,
            execution_log=pd.DataFrame(rows),
            output_paths={},
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SeedExecutionResult] = []
    if max_workers == 1 or config.stop_on_failure:
        for seed in config.seeds:
            result = _run_one_seed(config, seed)
            results.append(result)
            if config.stop_on_failure and result.status == "failed":
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_seed, config, seed): seed for seed in config.seeds}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item.seed)

    execution_log = pd.DataFrame([result.to_dict() for result in results]).sort_values("seed").reset_index(drop=True)
    output_paths = write_repeated_seed_outputs(config, execution_log)
    return RepeatedSeedRunResult(config=config, execution_log=execution_log, output_paths=output_paths)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_jsonl_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return int(len(_read_csv_if_exists(path)))
    if path.suffix.lower() == ".jsonl":
        return int(len(_read_jsonl_if_exists(path)))
    if path.name == "run_manifest.json":
        return 1
    return None


def build_seed_artifact_inventory(config: RepeatedSeedConfig, execution_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = execution_log["seed"].astype(int).tolist() if "seed" in execution_log.columns else list(config.seeds)
    for seed in seeds:
        seed_dir = seed_directory(config.output_dir, seed)
        for artifact in EXPECTED_SEED_ARTIFACTS:
            path = seed_dir / artifact
            rows.append(
                {
                    "seed": int(seed),
                    "seed_dir": str(seed_dir),
                    "artifact": artifact,
                    "path": str(path),
                    "exists": bool(path.exists()),
                    "row_count": _row_count(path),
                }
            )
    return pd.DataFrame(rows)


def _seed_frames(config: RepeatedSeedConfig, execution_log: pd.DataFrame, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if execution_log.empty or "seed" not in execution_log.columns:
        return pd.DataFrame()
    for row in execution_log.to_dict(orient="records"):
        if str(row.get("status")) not in {"completed", "skipped_complete"}:
            continue
        seed = int(row["seed"])
        path = seed_directory(config.output_dir, seed) / filename
        frame = _read_csv_if_exists(path)
        if frame.empty:
            continue
        frame.insert(0, "seed", seed)
        frame.insert(1, "seed_dir", str(seed_directory(config.output_dir, seed)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _metric_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "seed_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    return {
        "seed_count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
    }


def aggregate_seed_stability(
    frame: pd.DataFrame,
    *,
    dimensions: Iterable[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Aggregate numeric metrics across seeds using available columns only."""

    if frame.empty or "seed" not in frame.columns:
        return pd.DataFrame()
    dims = _available(dimensions, frame)
    metric_columns = _available(metrics, frame)
    if not metric_columns:
        return pd.DataFrame(columns=[*dims, "group_seed_count"])
    if dims:
        per_seed = frame.groupby(["seed", *dims], dropna=False)[metric_columns].mean(numeric_only=True).reset_index()
        grouped = per_seed.groupby(dims, dropna=False)
    else:
        per_seed = frame.groupby(["seed"], dropna=False)[metric_columns].mean(numeric_only=True).reset_index()
        grouped = [((), per_seed)]
    metric_columns = [metric for metric in metric_columns if metric in per_seed.columns]
    if not metric_columns:
        return pd.DataFrame(columns=[*dims, "group_seed_count"])

    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        if dims:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(dims, keys))
        else:
            row = {}
        row["group_seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            stats = _metric_stats(group[metric])
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(dims).reset_index(drop=True) if dims and rows else pd.DataFrame(rows)


def _build_failure_summary(execution_log: pd.DataFrame) -> pd.DataFrame:
    if execution_log.empty:
        return pd.DataFrame(columns=["status", "count"])
    status_summary = execution_log.groupby("status", dropna=False).size().reset_index(name="count")
    failures = execution_log[execution_log["status"].astype(str) == "failed"].copy()
    if failures.empty:
        return status_summary
    reason_summary = (
        failures.assign(error_message=failures["error_message"].fillna("").astype(str).str[:160])
        .groupby(["status", "error_message"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return pd.concat([status_summary, reason_summary], ignore_index=True, sort=False)


def _decision_candidate_stability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    dims = _available(["horizon", "target_type", "run_mode"], frame)
    if not dims:
        dims = []
    count_frame = frame.groupby(["seed", *dims], dropna=False).size().reset_index(name="candidate_count")
    metric_columns = [
        column
        for column in DECISION_CANDIDATE_METRICS
        if column != "candidate_count" and column in frame.columns
    ]
    if metric_columns:
        metric_frame = frame.groupby(["seed", *dims], dropna=False)[metric_columns].mean(numeric_only=True).reset_index()
        count_frame = count_frame.merge(metric_frame, on=["seed", *dims], how="left")
    return aggregate_seed_stability(
        count_frame,
        dimensions=dims,
        metrics=[column for column in DECISION_CANDIDATE_METRICS if column in count_frame.columns],
    )


def build_summary_markdown(config: RepeatedSeedConfig, execution_log: pd.DataFrame) -> str:
    counts = execution_log["status"].value_counts().to_dict() if "status" in execution_log.columns else {}
    lines = [
        "# Quant Core Repeated-Seed Stability Summary",
        "",
        f"- Preset: `{config.preset}`",
        f"- Run mode: `{config.run_mode}`",
        f"- Seed count requested: `{len(config.seeds)}`",
        f"- Seeds: `{', '.join(str(seed) for seed in config.seeds)}`",
        f"- Max workers: `{config.max_workers}`",
        f"- Resume: `{config.resume}`",
        f"- Stop on failure: `{config.stop_on_failure}`",
        f"- Save seed outputs: `{config.save_seed_outputs}`",
        f"- Output directory: `{config.output_dir}`",
        "",
        "## Execution Status",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {int(count)} |")
    lines.extend(
        [
            "",
            "Repeated-seed training is a stability diagnostic. It must not be used to select the single best random seed based on test-period performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_repeated_seed_outputs(config: RepeatedSeedConfig, execution_log: pd.DataFrame) -> dict[str, Path]:
    """Write root-level repeated-seed logs, inventories, aggregates, and summary."""

    output_paths: dict[str, Path] = {}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_artifact_inventory = build_seed_artifact_inventory(config, execution_log)
    seed_forecast_summary = _seed_frames(config, execution_log, "forecast_summary.csv")
    seed_forecast_by_horizon = _seed_frames(config, execution_log, "forecast_summary_by_horizon.csv")
    seed_strategy = _seed_frames(config, execution_log, "strategy_metrics.csv")
    seed_health = _seed_frames(config, execution_log, "model_health_summary.csv")
    seed_consensus = _seed_frames(config, execution_log, "model_consensus_summary.csv")
    if "sign_conflict" in seed_consensus.columns and "sign_conflict_rate" not in seed_consensus.columns:
        seed_consensus["sign_conflict_rate"] = seed_consensus["sign_conflict"].astype(bool).astype(float)
    seed_candidates = _seed_frames(config, execution_log, "decision_lane_candidates.csv")

    outputs: dict[str, pd.DataFrame] = {
        "seed_execution_log.csv": execution_log,
        "seed_artifact_inventory.csv": seed_artifact_inventory,
        "seed_forecast_summary.csv": seed_forecast_summary,
        "model_seed_stability.csv": aggregate_seed_stability(
            seed_forecast_summary,
            dimensions=["model_name", "target_type"],
            metrics=FORECAST_METRICS,
        ),
        "model_horizon_seed_stability.csv": aggregate_seed_stability(
            seed_forecast_by_horizon,
            dimensions=["model_name", "horizon", "target_type"],
            metrics=FORECAST_METRICS,
        ),
        "strategy_seed_stability.csv": aggregate_seed_stability(
            seed_strategy,
            dimensions=["model_name", "horizon", "target_type", "run_mode", "policy_variant"],
            metrics=STRATEGY_METRICS,
        ),
        "model_health_seed_stability.csv": aggregate_seed_stability(
            seed_health,
            dimensions=["model_name"],
            metrics=MODEL_HEALTH_METRICS,
        ),
        "consensus_seed_stability.csv": aggregate_seed_stability(
            seed_consensus,
            dimensions=["horizon", "target_type", "run_mode"],
            metrics=CONSENSUS_METRICS,
        ),
        "decision_candidate_seed_stability.csv": _decision_candidate_stability(seed_candidates),
        "failure_summary.csv": _build_failure_summary(execution_log),
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        output_paths[filename] = path

    manifest_path = output_dir / "repeated_seed_manifest.json"
    summary_path = output_dir / "summary.md"
    manifest_output_files = {
        **{filename: str(path) for filename, path in output_paths.items()},
        "repeated_seed_manifest.json": str(manifest_path),
        "summary.md": str(summary_path),
    }
    manifest = {
        "manifest_type": "quant_core_repeated_seed_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quant_core_script": str(config.quant_core_script),
        "real_quant_core_execution_path": True,
        "dry_run": bool(config.dry_run),
        "config": {
            "output_dir": str(config.output_dir),
            "seeds": list(config.seeds),
            "preset": config.preset,
            "run_mode": config.run_mode,
            "models": list(config.models),
            "model_roles": list(config.model_roles),
            "horizons": list(config.horizons),
            "target_types": list(config.target_types),
            "max_workers": int(config.max_workers),
            "resume": bool(config.resume),
            "stop_on_failure": bool(config.stop_on_failure),
            "save_seed_outputs": bool(config.save_seed_outputs),
            "no_ensemble": bool(config.no_ensemble),
        },
        "status_counts": execution_log["status"].value_counts().to_dict() if not execution_log.empty else {},
        "output_files": manifest_output_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    output_paths["repeated_seed_manifest.json"] = manifest_path

    summary_path.write_text(build_summary_markdown(config, execution_log), encoding="utf-8")
    output_paths["summary.md"] = summary_path
    return output_paths
