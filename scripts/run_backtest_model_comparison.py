"""Compare multiple model families on the fixed-window real-data backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.model_comparison import BacktestModelComparisonRunner, ModelComparisonConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-style model comparison on real vnstock data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["DGC", "ACB", "MWG", "HPG"],
        help="Ticker symbols to fetch and evaluate",
    )
    parser.add_argument("--train-start", type=str, default="2020-01-01", help="Inclusive training target start date")
    parser.add_argument("--train-end", type=str, default="2025-12-31", help="Inclusive training target end date")
    parser.add_argument("--eval-start", type=str, default="2026-01-01", help="Inclusive evaluation date start")
    parser.add_argument("--eval-end", type=str, default="2026-04-10", help="Inclusive evaluation date end")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/backtest_model_comparison",
        help="Artifact output directory",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="cart,xgboost,lightgbm,sarimax,ets",
        help="Comma-separated algorithms to compare",
    )
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
    config = ModelComparisonConfig(
        tickers=[ticker.upper().strip() for ticker in args.tickers if ticker.strip()],
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        output_dir=args.output_dir,
        algorithms=[name.strip().lower() for name in args.algorithms.split(",") if name.strip()],
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
    result = BacktestModelComparisonRunner(config).run()

    print("Model comparison completed.")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")

    print("\nAvailable algorithms:")
    print(", ".join(result["run_config"]["available_algorithms"]))
    if result["run_config"]["skipped_algorithms"]:
        print("\nSkipped algorithms:")
        for item in result["run_config"]["skipped_algorithms"]:
            print(f"{item['algorithm']}: {item['reason']}")

    print("\nPer-ticker model comparison preview:")
    preview_columns = [
        "ticker",
        "model_name",
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
        "beats_naive_baseline",
    ]
    print(result["model_comparison"][preview_columns].head(16).round(6).to_string(index=False))

    print("\nOverall ranking:")
    ranking_columns = [
        "model_name",
        "rmse",
        "mape",
        "directional_accuracy",
        "rank_rmse",
        "rank_mape",
        "rank_directional_accuracy",
        "average_rank",
        "tickers_beating_naive",
    ]
    print(result["ranking"][ranking_columns].round(6).to_string(index=False))

    print("\nFirst models beating the naive baseline by ticker:")
    beaters = result["model_comparison"][
        (result["model_comparison"]["model_name"] != "naive_previous_close")
        & (result["model_comparison"]["beats_naive_baseline"])
    ][["ticker", "model_name", "rmse", "mape", "directional_accuracy"]]
    if beaters.empty:
        print("No model beat the naive baseline under the configured rule.")
    else:
        print(beaters.sort_values(["ticker", "rmse"]).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
