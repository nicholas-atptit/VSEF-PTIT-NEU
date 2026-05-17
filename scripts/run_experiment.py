"""Run a Phase 1 standardized VSEF experiment from YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.experiment_orchestrator import ExperimentOrchestrator


PROJECT_VENV_EXAMPLE = (
    "python scripts/run_experiment.py --config configs/experiments/EXP-SMOKE-001.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-driven VSEF experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Active Python executable: {sys.executable}")
    try:
        import_module("vnstock_data")
    except Exception as exc:
        print(f"vnstock_data import failed: {exc}")
        print("Run with the project venv, for example:")
        print(f" {PROJECT_VENV_EXAMPLE}")
    orchestrator = ExperimentOrchestrator(args.config)
    result = orchestrator.run()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"completed", "completed_with_errors"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
