"""Benchmark the technical ML models on the same latest 5-year data window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ml_tickers import resolve_files
from src.ml.trainer import DualModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CART/LSTM/BiLSTM on the latest 5-year window")
    parser.add_argument("--daily", type=str, default="data/daily_market_split_data")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--report", type=str, default="reports/ml_benchmark.csv")
    parser.add_argument("--tickers", type=str, default="")
    parser.add_argument("--all", action="store_true", dest="train_all")
    parser.add_argument("--vn100", action="store_true")
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--algorithms", type=str, default="cart,lstm,bilstm")
    parser.add_argument("--primary-algorithm", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--criterion", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_files(args)
    trainer = DualModelTrainer(model_dir=args.output)
    algorithms = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]

    rows: list[dict] = []
    for csv_path in files:
        ticker = csv_path.stem.upper()
        daily_df = pd.read_csv(csv_path)
        result = trainer.train(
            ticker=ticker,
            df=daily_df,
            algorithms=algorithms,
            primary_algorithm=args.primary_algorithm,
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
        rows.extend(result["report_rows"])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(rows).sort_values(["ticker", "horizon", "algorithm"]).reset_index(drop=True)
    report_df.to_csv(report_path, index=False)

    if not report_df.empty:
        best = report_df.sort_values(["f1", "balanced_accuracy", "sharpe"], ascending=False).iloc[0]
        print(
            "Top model:",
            f"{best['ticker']} {best['horizon']} {best['algorithm']}",
            f"F1={best['f1']:.4f}",
            f"BalancedAcc={best['balanced_accuracy']:.4f}",
            f"Sharpe={best['sharpe']:.2f}",
        )
    print(f"Benchmark report written to {report_path}")


if __name__ == "__main__":
    main()
