"""Run the system benchmark across legacy and upgraded ML system modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ml_tickers import resolve_files
from src.ml.benchmark.system_benchmark import SystemBenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs risk/regime-aware ML system modes")
    parser.add_argument("--daily", type=str, default="data/daily_market_split_data")
    parser.add_argument("--output", type=str, default="models/system_benchmark")
    parser.add_argument("--report", type=str, default="reports/system_benchmark.csv")
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
    algorithms = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    runner = SystemBenchmarkRunner(model_root=args.output)
    result = runner.run(
        files=files,
        algorithms=algorithms,
        output_root=args.output,
        report_path=args.report,
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

    summary_df = result["summary"]
    if not summary_df.empty:
        best = summary_df.sort_values(["sharpe", "calmar", "directional_accuracy"], ascending=False).iloc[0]
        print(
            "Top benchmark mode:",
            f"{best['benchmark_mode']}",
            f"Sharpe={best['sharpe']:.2f}",
            f"Calmar={best['calmar']:.4f}",
            f"DirectionalAcc={best['directional_accuracy']:.4f}",
        )
    print(f"Benchmark detail report written to {result['detail_path']}")
    print(f"Benchmark markdown report written to {result['markdown_path']}")


if __name__ == "__main__":
    main()
