"""Run a fixed-window real-data backtest using vnstock as the market data source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.real_data import FixedWindowBacktestConfig, RealDataBacktestRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-window OHLCV backtest on real vnstock data")
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
    parser.add_argument("--output-dir", type=str, default="artifacts/backtest", help="Artifact output directory")
    parser.add_argument("--algorithms", type=str, default="cart", help="Comma-separated algorithms to train")
    parser.add_argument("--primary-algorithm", type=str, default=None, help="Primary inference algorithm to evaluate")
    parser.add_argument("--sequence-length", type=int, default=20, help="Sequence length for LSTM/BiLSTM algorithms")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for sequence models")
    parser.add_argument("--num-layers", type=int, default=2, help="Layer count for sequence models")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for sequence models")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for sequence models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for sequence models")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max-depth", type=int, default=None, help="CART max_depth")
    parser.add_argument("--min-samples-split", type=int, default=2, help="CART min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="CART min_samples_leaf")
    parser.add_argument("--criterion", type=str, default=None, help="CART criterion override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    config = FixedWindowBacktestConfig(
        tickers=[ticker.upper().strip() for ticker in args.tickers if ticker.strip()],
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        output_dir=args.output_dir,
        algorithms=algorithms,
        primary_algorithm=args.primary_algorithm.lower() if args.primary_algorithm else None,
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
    result = RealDataBacktestRunner(config).run()

    print("Backtest completed.")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")

    print("\nFetched date coverage:")
    print(result["fetch_summary"].to_string(index=False))

    print("\nMetrics summary:")
    print(result["metrics"].round(6).to_string(index=False))

    print("\nBaseline comparison:")
    comparison_columns = [
        "ticker",
        "wins_vs_baseline",
        "beats_baseline_overall",
        "beats_baseline_mae",
        "beats_baseline_rmse",
        "beats_baseline_mape",
        "beats_baseline_directional_accuracy",
    ]
    print(result["metrics"][comparison_columns].to_string(index=False))

    print("\nPredicted vs actual preview:")
    preview_columns = [
        "date",
        "ticker",
        "actual_close",
        "predicted_close",
        "predicted_close_baseline",
        "absolute_error",
        "absolute_error_baseline",
        "pct_error",
    ]
    print(result["comparison"][preview_columns].head(10).round(6).to_string(index=False))

    print("\nCharts:")
    for ticker, files in sorted(result["chart_files"].items()):
        print(f"{ticker}: actual_vs_predicted={files['actual_vs_predicted']}, absolute_error={files['absolute_error']}")


if __name__ == "__main__":
    main()
