"""Run combined signal analysis on top of dual-task forecast outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.combined_signal import CombinedSignalAnalysisRunner, CombinedSignalConfig


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined signal analysis for dual-task outputs")
    parser.add_argument("--dual-task-dir", type=str, default="artifacts/dual_task", help="Source dual-task artifact directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/combined_signal", help="Combined signal output directory")
    parser.add_argument("--horizons", type=str, default="3d,5d,20d", help="Comma-separated horizons to analyze")
    parser.add_argument(
        "--return-thresholds",
        type=str,
        default="0.0,0.005,0.01,0.02",
        help="Comma-separated return thresholds for labeling and gating",
    )
    parser.add_argument(
        "--probability-thresholds",
        type=str,
        default="0.50,0.55,0.60,0.65",
        help="Comma-separated probability thresholds for labeling and gating",
    )
    parser.add_argument("--w-return", type=float, default=0.5, help="Weighted-linear return strength weight")
    parser.add_argument("--w-profit", type=float, default=0.5, help="Weighted-linear profit confidence weight")
    parser.add_argument("--top-k-values", type=str, default="1,3,5", help="Comma-separated top-k values for ranking analysis")
    parser.add_argument("--ranking-group", type=str, default="date", choices=["date", "week"], help="Ranking group granularity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CombinedSignalConfig(
        dual_task_dir=args.dual_task_dir,
        output_dir=args.output_dir,
        horizons=[item.strip().lower() for item in args.horizons.split(",") if item.strip()],
        return_thresholds=_parse_float_list(args.return_thresholds),
        probability_thresholds=_parse_float_list(args.probability_thresholds),
        w_return=float(args.w_return),
        w_profit=float(args.w_profit),
        top_k_values=[int(float(item)) for item in _parse_float_list(args.top_k_values)],
        ranking_group=args.ranking_group,
    )
    result = CombinedSignalAnalysisRunner(config).run()

    print("Combined signal analysis completed.")
    print("\nPer-horizon artifacts:")
    for horizon, horizon_result in result["horizons"].items():
        print(f"\n[{horizon}]")
        for name, path in horizon_result["paths"].items():
            print(f"{name}: {path}")
        ranking_preview = horizon_result["combined_ranking_summary"][
            ["model_name", "ranking_method", "top_k", "average_actual_return", "profit_rate", "lift_vs_base_rate"]
        ].head(12)
        print(ranking_preview.round(6).to_string(index=False))

    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nCross-horizon combined ranking:")
    print(result["cross_horizon_combined_ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
