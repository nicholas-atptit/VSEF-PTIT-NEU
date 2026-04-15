"""Run the strategy backtest layer on top of forward-return forecasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.strategy_backtest import StrategyBacktestConfig, StrategyBacktestRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy backtest on forward-return forecast outputs")
    parser.add_argument("--tickers", nargs="+", default=["DGC", "ACB", "MWG", "HPG"], help="Ticker symbols to evaluate")
    parser.add_argument("--train-start", type=str, default="2020-01-01", help="Training start date")
    parser.add_argument("--train-end", type=str, default="2025-12-31", help="Training end date")
    parser.add_argument("--eval-start", type=str, default="2026-01-01", help="Evaluation start date")
    parser.add_argument("--eval-end", type=str, default="2026-04-10", help="Evaluation end date")
    parser.add_argument("--output-dir", type=str, default="artifacts/strategy_backtest", help="Strategy artifact output directory")
    parser.add_argument("--forecast-output-dir", type=str, default="artifacts/backtest_forward_return", help="Existing forward-return forecast artifact directory")
    parser.add_argument("--horizons", type=str, default="3d,5d,20d", help="Comma-separated forward-return horizons")
    parser.add_argument("--algorithms", type=str, default="cart,xgboost,lightgbm,sarimax,ets", help="Comma-separated algorithms")
    parser.add_argument("--thresholds", type=str, default="0.0,0.005,0.01,0.02", help="Comma-separated long-only thresholds")
    parser.add_argument("--transaction-fee-bps", type=float, default=15.0, help="Per-side transaction fee in basis points")
    parser.add_argument("--slippage-bps", type=float, default=20.0, help="Per-side slippage in basis points")
    parser.add_argument("--sequence-length", type=int, default=20, help="Sequence length for sequence algorithms")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for sequence models")
    parser.add_argument("--num-layers", type=int, default=2, help="Layer count for sequence models")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for sequence models")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for sequence models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max-depth", type=int, default=4, help="Tree max depth")
    parser.add_argument("--min-samples-split", type=int, default=2, help="CART min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="CART min_samples_leaf")
    parser.add_argument("--criterion", type=str, default=None, help="CART criterion override")
    parser.add_argument("--disable-momentum-baseline", action="store_true", help="Disable the momentum continuation benchmark")
    parser.add_argument("--disable-forecast-rerun", action="store_true", help="Fail if forward-return artifacts are missing instead of regenerating them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyBacktestConfig(
        tickers=[ticker.upper().strip() for ticker in args.tickers if ticker.strip()],
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        output_dir=args.output_dir,
        forecast_output_dir=args.forecast_output_dir,
        horizons=[name.strip().lower() for name in args.horizons.split(",") if name.strip()],
        algorithms=[name.strip().lower() for name in args.algorithms.split(",") if name.strip()],
        thresholds=[float(value.strip()) for value in args.thresholds.split(",") if value.strip()],
        transaction_fee_bps=args.transaction_fee_bps,
        slippage_bps=args.slippage_bps,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        include_momentum_baseline=not args.disable_momentum_baseline,
        rerun_forecasts_if_missing=not args.disable_forecast_rerun,
    )
    result = StrategyBacktestRunner(config).run()

    print("Strategy backtest completed.")
    print(f"Forecast source: {result['forecast_output_dir']}")
    print("\nAvailable algorithms:")
    print(", ".join(result["available_algorithms"]))
    if result["skipped_algorithms"]:
        print("\nSkipped algorithms:")
        for item in result["skipped_algorithms"]:
            print(f"{item['algorithm']}: {item['reason']}")

    print("\nPer-horizon strategy artifacts:")
    for horizon, horizon_result in result["horizons"].items():
        print(f"\n[{horizon}]")
        for name, path in horizon_result["paths"].items():
            print(f"{name}: {path}")
        preview = horizon_result["portfolio_metrics"][
            [
                "model_name",
                "threshold",
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "number_of_trades",
                "beats_buy_and_hold",
            ]
        ].head(12)
        print(preview.round(6).to_string(index=False))

    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nOverall strategy ranking:")
    print(result["overall_ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
