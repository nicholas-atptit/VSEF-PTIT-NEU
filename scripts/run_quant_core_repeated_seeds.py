"""Run repeated-seed Quant Core stability diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.model_governance import RUN_MODES
from src.evaluation.quant_core import PRESET_CONFIGS
from src.evaluation.repeated_seed_runner import (
    RepeatedSeedConfig,
    build_seed_command,
    command_to_text,
    resolve_seeds,
    run_repeated_seed_stability,
)
from src.evaluation.targets import supported_target_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the governed Quant Core runner across repeated random seeds "
            "and aggregate stability diagnostics."
        )
    )
    parser.add_argument("--output-dir", default="artifacts/quant_core_repeated_seed")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds-file", default=None)
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), default="smoke")
    parser.add_argument("--run-mode", choices=list(RUN_MODES), default="research_core")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model-roles", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--target-types", nargs="+", default=None, choices=supported_target_specs())
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-seed-outputs",
        dest="save_seed_outputs",
        action="store_true",
        default=True,
        help="Retain seed directories for auditability and resume support. Enabled by default.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--no-ensemble", action="store_true")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> RepeatedSeedConfig:
    seeds = resolve_seeds(
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        seeds=args.seeds,
        seeds_file=args.seeds_file,
    )
    return RepeatedSeedConfig(
        output_dir=Path(args.output_dir),
        seeds=seeds,
        preset=args.preset,
        run_mode=args.run_mode,
        models=tuple(args.models or ()),
        model_roles=tuple(args.model_roles or ()),
        horizons=tuple(args.horizons or ()),
        target_types=tuple(args.target_types or ()),
        max_workers=max(1, int(args.max_workers)),
        resume=bool(args.resume),
        stop_on_failure=bool(args.stop_on_failure),
        dry_run=bool(args.dry_run),
        save_seed_outputs=bool(args.save_seed_outputs),
        python_executable=str(args.python_executable),
        no_ensemble=bool(args.no_ensemble),
        cwd=REPO_ROOT,
    )


def _print_run_header(config: RepeatedSeedConfig) -> None:
    print(f"Output directory: {config.output_dir}")
    print(f"Seed count: {len(config.seeds)}")
    print(f"Seeds: {', '.join(str(seed) for seed in config.seeds)}")
    print(f"Preset: {config.preset}")
    print(f"Run mode: {config.run_mode}")
    print(f"Max workers: {config.max_workers}")
    print(f"Resume: {config.resume}")
    print(f"Dry run: {config.dry_run}")


def main() -> int:
    config = _build_config(parse_args())
    _print_run_header(config)
    if config.dry_run:
        print("Commands:")
        for seed in config.seeds:
            print(command_to_text(build_seed_command(config, seed)))
        result = run_repeated_seed_stability(config)
        print(f"Dry-run rows: {len(result.execution_log)}")
        return 0

    result = run_repeated_seed_stability(config)
    status_counts = result.execution_log["status"].value_counts().to_dict()
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {int(count)}")
    print("Root outputs:")
    for filename, path in sorted(result.output_paths.items()):
        print(f"  {filename}: {path}")
    return 1 if int(status_counts.get("failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
