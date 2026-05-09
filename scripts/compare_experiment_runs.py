"""Compare two Phase 1 experiment output folders for reproducibility evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two standardized experiment run folders.")
    parser.add_argument("--left", required=True, help="First experiment output directory")
    parser.add_argument("--right", required=True, help="Second experiment output directory")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_schema(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return list(pd.read_csv(path, nrows=0).columns)


def compare_runs(left: Path, right: Path) -> dict[str, Any]:
    left_config = _load_yaml(left / "config" / "resolved_config.yaml")
    right_config = _load_yaml(right / "config" / "resolved_config.yaml")
    left_manifest = _load_json(left / "manifests" / "run_manifest.json")
    right_manifest = _load_json(right / "manifests" / "run_manifest.json")
    left_metrics_schema = _csv_schema(left / "metrics" / "metrics.csv")
    right_metrics_schema = _csv_schema(right / "metrics" / "metrics.csv")

    left_artifacts = set((left / "artifacts").glob("*")) if (left / "artifacts").exists() else set()
    right_artifacts = set((right / "artifacts").glob("*")) if (right / "artifacts").exists() else set()
    left_artifact_names = sorted(path.name for path in left_artifacts)
    right_artifact_names = sorted(path.name for path in right_artifacts)

    left_manifest_schema = sorted(left_manifest.keys()) if left_manifest else None
    right_manifest_schema = sorted(right_manifest.keys()) if right_manifest else None

    return {
        "left": str(left),
        "right": str(right),
        "resolved_config_equal": left_config == right_config,
        "metrics_schema_equal": left_metrics_schema == right_metrics_schema,
        "manifest_schema_equal": left_manifest_schema == right_manifest_schema,
        "artifact_presence_equal": left_artifact_names == right_artifact_names,
        "left_metrics_schema": left_metrics_schema,
        "right_metrics_schema": right_metrics_schema,
        "left_manifest_schema": left_manifest_schema,
        "right_manifest_schema": right_manifest_schema,
        "left_artifacts": left_artifact_names,
        "right_artifacts": right_artifact_names,
    }


def main() -> int:
    args = parse_args()
    result = compare_runs(Path(args.left), Path(args.right))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
