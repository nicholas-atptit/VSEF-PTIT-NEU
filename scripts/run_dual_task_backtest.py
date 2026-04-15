"""Run dual-task backtesting for return regression and profit classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.dual_task import DualTaskBacktestConfig, DualTaskBacktestRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-task backtest on real vnstock data")
    parser.add_argument("--tickers", nargs="+", default=["DGC", "ACB", "MWG", "HPG"], help="Ticker symbols to fetch and evaluate")
    parser.add_argument("--train-start", type=str, default="2020-01-01", help="Inclusive training target-date start")
    parser.add_argument("--train-end", type=str, default="2025-12-31", help="Inclusive training target-date end")
    parser.add_argument("--eval-start", type=str, default="2026-01-01", help="Inclusive evaluation target-date start")
    parser.add_argument("--eval-end", type=str, default="2026-04-10", help="Inclusive evaluation target-date end")
    parser.add_argument("--output-dir", type=str, default="artifacts/dual_task", help="Dual-task artifact output directory")
    parser.add_argument("--horizons", type=str, default="3d,5d,20d", help="Comma-separated forward-return horizons")
    parser.add_argument("--algorithms", type=str, default="cart,xgboost,lightgbm,sarimax,ets", help="Comma-separated algorithms")
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
    parser.add_argument("--max-depth", type=int, default=4, help="Tree max depth for CART/XGBoost/LightGBM")
    parser.add_argument("--min-samples-split", type=int, default=2, help="CART min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="CART min_samples_leaf")
    parser.add_argument("--criterion", type=str, default=None, help="CART criterion override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DualTaskBacktestConfig(
        tickers=[ticker.upper().strip() for ticker in args.tickers if ticker.strip()],
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        output_dir=args.output_dir,
        horizons=[name.strip().lower() for name in args.horizons.split(",") if name.strip()],
        algorithms=[name.strip().lower() for name in args.algorithms.split(",") if name.strip()],
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
    )
    result = DualTaskBacktestRunner(config).run()

    print("Dual-task backtest completed.")
    print("\nAvailable algorithms:")
    print(", ".join(result["available_algorithms"]))
    if result["skipped_algorithms"]:
        print("\nSkipped algorithms:")
        for item in result["skipped_algorithms"]:
            print(f"{item['algorithm']}: {item['reason']}")

    print("\nPer-horizon regression and classification artifacts:")
    for horizon, horizon_result in result["horizons"].items():
        print(f"\n[{horizon}]")
        print("Regression:")
        for name, path in horizon_result["regression_paths"].items():
            print(f"{name}: {path}")
        regression_preview = horizon_result["regression_ranking"][
            ["model_name", "rmse", "mape", "directional_accuracy", "average_rank"]
        ].head(10)
        print(regression_preview.round(6).to_string(index=False))

        print("Classification:")
        for name, path in horizon_result["classification_paths"].items():
            print(f"{name}: {path}")
        classification_preview = horizon_result["classification_ranking"][
            ["model_name", "precision", "recall", "f1", "positive_class_precision", "average_rank"]
        ].head(10)
        print(classification_preview.round(6).to_string(index=False))

    print("\nSummary artifacts:")
    for name, path in result["summary_paths"].items():
        print(f"{name}: {path}")

    print("\nCross-task model ranking:")
    print(result["cross_task_model_ranking"].to_string(index=False))


if __name__ == "__main__":
    main()
