"""Run the context-conditioned meta-selector analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.context_meta_selector import (
    ContextConditionedMetaSelectorRunner,
    ContextMetaSelectorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-conditioned meta-selector analysis")
    parser.add_argument("--walk-forward-dir", type=str, default="artifacts/walk_forward_regime_robustness")
    parser.add_argument("--meta-selector-dir", type=str, default="artifacts/meta_selector")
    parser.add_argument("--audit-output-dir", type=str, default="artifacts/meta_selector_audit")
    parser.add_argument("--output-dir", type=str, default="artifacts/context_meta_selector")
    parser.add_argument(
        "--selector-modes",
        type=str,
        default="context_knn_selector,context_bin_lookup,context_meta_score",
    )
    parser.add_argument("--minimum-prior-samples-for-context-match", type=int, default=30)
    parser.add_argument("--minimum-prior-folds", type=int, default=2)
    parser.add_argument("--primary-top-k", type=int, default=3)
    parser.add_argument("--knn-neighbors", type=int, default=40)
    parser.add_argument("--meta-score-ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ContextMetaSelectorConfig(
        walk_forward_dir=args.walk_forward_dir,
        meta_selector_dir=args.meta_selector_dir,
        audit_output_dir=args.audit_output_dir,
        output_dir=args.output_dir,
        selector_modes=[item.strip() for item in args.selector_modes.split(",") if item.strip()],
        minimum_prior_samples_for_context_match=int(args.minimum_prior_samples_for_context_match),
        minimum_prior_folds=int(args.minimum_prior_folds),
        primary_top_k=int(args.primary_top_k),
        knn_neighbors=int(args.knn_neighbors),
        meta_score_ridge_alpha=float(args.meta_score_ridge_alpha),
    )
    result = ContextConditionedMetaSelectorRunner(config).run()

    print("Context-conditioned meta-selector analysis completed.")
    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nContext selector overview:")
    print(result["context_selector_overview"].to_string(index=False))

    print("\nOverall context selector report:")
    print(result["overall_context_selector_report"].to_string(index=False))


if __name__ == "__main__":
    main()
