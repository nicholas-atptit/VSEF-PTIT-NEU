"""Run walk-forward regime-aware robustness analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.walk_forward_regime_robustness import (
    WalkForwardRegimeRobustnessConfig,
    WalkForwardRegimeRobustnessRunner,
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward regime-aware robustness analysis")
    parser.add_argument("--tickers", nargs="+", default=["DGC", "ACB", "MWG", "HPG"])
    parser.add_argument("--train-start", type=str, default="2020-01-01")
    parser.add_argument("--first-eval-start", type=str, default="2023-01-01")
    parser.add_argument("--last-eval-end", type=str, default="2026-04-10")
    parser.add_argument("--eval-window-days", type=int, default=60)
    parser.add_argument("--step-size-days", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--horizons", type=str, default="3d,5d,20d")
    parser.add_argument("--algorithms", type=str, default="cart,xgboost,lightgbm,sarimax,ets")
    parser.add_argument("--training-window-mode", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--rolling-train-window-days", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/walk_forward_regime_robustness")
    parser.add_argument("--benchmark-symbol", type=str, default="VNINDEX")
    parser.add_argument("--benchmark-source", type=str, default="vnindex_or_market_proxy", choices=["vnindex", "market_proxy", "vnindex_or_market_proxy"])
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--regime-lookback-days", type=int, default=20)
    parser.add_argument("--bull-threshold", type=float, default=0.03)
    parser.add_argument("--bear-threshold", type=float, default=-0.03)
    parser.add_argument("--return-thresholds", type=str, default="0.0,0.005,0.01,0.02")
    parser.add_argument("--probability-thresholds", type=str, default="0.50,0.55,0.60,0.65")
    parser.add_argument("--w-return", type=float, default=0.5)
    parser.add_argument("--w-profit", type=float, default=0.5)
    parser.add_argument("--top-k-values", type=str, default="1,3,5")
    parser.add_argument("--fold-retry-count", type=int, default=2)
    parser.add_argument("--fold-retry-backoff-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WalkForwardRegimeRobustnessConfig(
        tickers=[ticker.upper().strip() for ticker in args.tickers if ticker.strip()],
        train_start=args.train_start,
        first_eval_start=args.first_eval_start,
        last_eval_end=args.last_eval_end,
        eval_window_days=int(args.eval_window_days),
        step_size_days=int(args.step_size_days),
        max_folds=int(args.max_folds),
        horizons=[item.strip().lower() for item in args.horizons.split(",") if item.strip()],
        algorithms=[item.strip().lower() for item in args.algorithms.split(",") if item.strip()],
        training_window_mode=args.training_window_mode,
        rolling_train_window_days=args.rolling_train_window_days,
        output_dir=args.output_dir,
        benchmark_symbol=args.benchmark_symbol.upper().strip(),
        benchmark_source=args.benchmark_source,
        benchmark_path=args.benchmark_path,
        regime_lookback_days=int(args.regime_lookback_days),
        bull_threshold=float(args.bull_threshold),
        bear_threshold=float(args.bear_threshold),
        return_thresholds=_parse_float_list(args.return_thresholds),
        probability_thresholds=_parse_float_list(args.probability_thresholds),
        w_return=float(args.w_return),
        w_profit=float(args.w_profit),
        top_k_values=_parse_int_list(args.top_k_values),
        fold_retry_count=int(args.fold_retry_count),
        fold_retry_backoff_seconds=float(args.fold_retry_backoff_seconds),
    )
    result = WalkForwardRegimeRobustnessRunner(config).run()

    print("Walk-forward regime robustness analysis completed.")
    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nFold overview:")
    print(result["fold_overview"].to_string(index=False))

    print("\nOverall robustness report:")
    print(result["overall_robustness_report"].to_string(index=False))


if __name__ == "__main__":
    main()
