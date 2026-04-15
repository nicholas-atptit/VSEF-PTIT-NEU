"""Run the regime-conditioned meta-selector analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.meta_selector import MetaSelectorConfig, RegimeConditionedMetaSelectorRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-conditioned meta-selector analysis")
    parser.add_argument("--walk-forward-dir", type=str, default="artifacts/walk_forward_regime_robustness")
    parser.add_argument("--output-dir", type=str, default="artifacts/meta_selector")
    parser.add_argument(
        "--selector-modes",
        type=str,
        default="simple_regime_lookup,regime_weighted_rank,fallback_global",
    )
    parser.add_argument("--minimum-prior-folds-per-regime", type=int, default=2)
    parser.add_argument("--minimum-samples-per-regime", type=int, default=30)
    parser.add_argument("--primary-top-k", type=int, default=3)
    parser.add_argument("--utility-weight-topk-avg-return", type=float, default=0.40)
    parser.add_argument("--utility-weight-topk-profit-rate", type=float, default=0.30)
    parser.add_argument("--utility-weight-positive-class-precision", type=float, default=0.20)
    parser.add_argument("--utility-weight-directional-accuracy", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MetaSelectorConfig(
        walk_forward_dir=args.walk_forward_dir,
        output_dir=args.output_dir,
        selector_modes=[item.strip() for item in args.selector_modes.split(",") if item.strip()],
        minimum_prior_folds_per_regime=int(args.minimum_prior_folds_per_regime),
        minimum_samples_per_regime=int(args.minimum_samples_per_regime),
        primary_top_k=int(args.primary_top_k),
        utility_weight_topk_avg_return=float(args.utility_weight_topk_avg_return),
        utility_weight_topk_profit_rate=float(args.utility_weight_topk_profit_rate),
        utility_weight_positive_class_precision=float(args.utility_weight_positive_class_precision),
        utility_weight_directional_accuracy=float(args.utility_weight_directional_accuracy),
    )
    result = RegimeConditionedMetaSelectorRunner(config).run()

    print("Regime-conditioned meta-selector analysis completed.")
    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nMeta-selector overview:")
    print(result["meta_selector_overview"].to_string(index=False))

    print("\nOverall meta-selector report:")
    print(result["overall_meta_selector_report"].to_string(index=False))


if __name__ == "__main__":
    main()
