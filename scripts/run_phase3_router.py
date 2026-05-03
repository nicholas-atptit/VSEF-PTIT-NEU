"""Run deterministic Phase 3 Router v1 on saved allocator outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase3_router import Phase3RouterConfig, run_phase3_router_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create auditable Phase 3 route-decision artifacts from saved allocator outputs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing allocator and Quant Core diagnostic outputs")
    parser.add_argument("--output-dir", default=None, help="Directory for router outputs; defaults to input-dir")
    parser.add_argument("--max-risk-score", type=float, default=0.70)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--min-model-agreement", type=float, default=0.5)
    parser.add_argument("--min-allocation-weight", type=float, default=0.01)
    parser.add_argument(
        "--low-agreement-action",
        choices=["hold_for_review", "reject_low_agreement"],
        default="hold_for_review",
    )
    parser.add_argument("--allow-no-allocation", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Phase3RouterConfig(
        risk_reject_threshold=args.max_risk_score,
    )
    result = run_phase3_router_from_files(
        args.input_dir,
        args.output_dir,
        config=config,
        write_legacy_aliases=True,
    )
    decision_counts = result.router_decisions["route_decision"].value_counts().sort_index().to_dict()
    print("Route decisions:")
    for decision, count in decision_counts.items():
        print(f"  {decision}: {count}")
    print("Outputs:")
    for name, path in sorted(result.output_paths.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
