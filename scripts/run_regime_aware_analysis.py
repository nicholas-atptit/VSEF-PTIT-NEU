"""Run regime-aware analysis on saved dual-task and combined-signal artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.regime_aware_analysis import RegimeAwareAnalysisConfig, RegimeAwareAnalysisRunner


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-aware analysis for saved forecasting artifacts")
    parser.add_argument("--dual-task-dir", type=str, default="artifacts/dual_task", help="Source dual-task artifact directory")
    parser.add_argument("--combined-signal-dir", type=str, default="artifacts/combined_signal", help="Source combined-signal artifact directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/regime_aware_analysis", help="Regime-aware output directory")
    parser.add_argument("--horizons", type=str, default="3d,5d,20d", help="Comma-separated horizons to analyze")
    parser.add_argument("--benchmark-symbol", type=str, default="VNINDEX", help="Benchmark symbol for regime inference")
    parser.add_argument(
        "--benchmark-source",
        type=str,
        default="vnindex_or_market_proxy",
        choices=["vnindex", "market_proxy", "vnindex_or_market_proxy"],
        help="Benchmark source selection",
    )
    parser.add_argument("--benchmark-path", type=str, default=None, help="Optional market proxy CSV fallback path")
    parser.add_argument("--regime-lookback-days", type=int, default=20, help="Lookback window for regime returns")
    parser.add_argument("--bull-threshold", type=float, default=0.03, help="Bull regime threshold for benchmark lookback return")
    parser.add_argument("--bear-threshold", type=float, default=-0.03, help="Bear regime threshold for benchmark lookback return")
    parser.add_argument("--return-thresholds", type=str, default="0.0,0.005,0.01,0.02", help="Comma-separated return thresholds for gated combined ranking")
    parser.add_argument("--probability-thresholds", type=str, default="0.50,0.55,0.60,0.65", help="Comma-separated probability thresholds for gated combined ranking")
    parser.add_argument("--top-k-values", type=str, default="1,3,5", help="Comma-separated top-k values for ranking analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegimeAwareAnalysisConfig(
        dual_task_dir=args.dual_task_dir,
        combined_signal_dir=args.combined_signal_dir,
        output_dir=args.output_dir,
        horizons=[item.strip().lower() for item in args.horizons.split(",") if item.strip()],
        benchmark_symbol=args.benchmark_symbol.upper().strip(),
        benchmark_source=args.benchmark_source,
        benchmark_path=args.benchmark_path,
        regime_lookback_days=int(args.regime_lookback_days),
        bull_threshold=float(args.bull_threshold),
        bear_threshold=float(args.bear_threshold),
        return_thresholds=_parse_float_list(args.return_thresholds),
        probability_thresholds=_parse_float_list(args.probability_thresholds),
        top_k_values=_parse_int_list(args.top_k_values),
    )
    result = RegimeAwareAnalysisRunner(config).run()

    print("Regime-aware analysis completed.")
    print(f"Benchmark source used: {result['benchmark_source_used']}")
    print("\nPer-horizon artifacts:")
    for horizon, horizon_result in result["horizons"].items():
        print(f"\n[{horizon}]")
        for name, path in horizon_result["paths"].items():
            print(f"{name}: {path}")
        preview = horizon_result["ranking_by_regime"][
            ["regime", "model_name", "ranking_method", "top_k", "average_actual_return", "profit_rate"]
        ].head(12)
        print(preview.round(6).to_string(index=False))

    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nOverall regime summary:")
    print(result["overall_regime_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
