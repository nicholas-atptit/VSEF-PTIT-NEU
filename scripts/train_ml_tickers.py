"""Train technical ML models on the latest rolling 5-year data window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.universe import get_vn100_universe
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CART/LSTM/BiLSTM models on the latest 5-year window")
    parser.add_argument("--daily", type=str, default="data/daily_market_split_data", help="Per-ticker daily CSV directory")
    parser.add_argument("--output", type=str, default="models", help="Artifact output directory")
    parser.add_argument("--report", type=str, default="reports/ml_benchmark.csv", help="Benchmark CSV output path")
    parser.add_argument(
        "--prepared-output",
        type=str,
        default="data/processed/ml_5y",
        help="Directory for rebuilt feature datasets when --prepare-only is used",
    )
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated ticker list")
    parser.add_argument("--all", action="store_true", dest="train_all", help="Train every ticker found in the CSV directory")
    parser.add_argument("--vn100", action="store_true", help="Train the dynamic VN100 universe")
    parser.add_argument("--max-tickers", type=int, default=None, help="Optional cap for batch runs")
    parser.add_argument("--algorithms", type=str, default="cart", help="Comma-separated algorithms: cart,lstm,bilstm")
    parser.add_argument("--primary-algorithm", type=str, default=None, help="Primary inference algorithm to store in the manifest")
    parser.add_argument("--sequence-length", type=int, default=20, help="Sequence length for LSTM/BiLSTM")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for LSTM/BiLSTM")
    parser.add_argument("--num-layers", type=int, default=2, help="Layer count for LSTM/BiLSTM")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for LSTM/BiLSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for LSTM/BiLSTM")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for LSTM/BiLSTM")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for LSTM/BiLSTM")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience for LSTM/BiLSTM")
    parser.add_argument("--max-depth", type=int, default=None, help="CART max_depth")
    parser.add_argument("--min-samples-split", type=int, default=2, help="CART min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="CART min_samples_leaf")
    parser.add_argument("--criterion", type=str, default=None, help="CART criterion")
    parser.add_argument("--prepare-only", action="store_true", help="Rebuild the 5-year feature dataset without training models")
    parser.add_argument("--no-clean-output", action="store_true", help="Keep existing ticker artifacts instead of replacing them")
    return parser.parse_args()


def resolve_files(args: argparse.Namespace) -> list[Path]:
    daily_dir = Path(args.daily)
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily CSV directory not found: {daily_dir}")

    files = sorted(daily_dir.glob("*.csv"))
    if args.vn100:
        target_set = {ticker.upper() for ticker in get_vn100_universe(mode="current_plus_viettel")}
        files = [path for path in files if path.stem.upper() in target_set]
    elif args.tickers:
        selected = {ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()}
        files = [path for path in files if path.stem.upper() in selected]
    elif not args.train_all:
        raise ValueError("Specify --tickers, --vn100, or --all")

    if args.max_tickers is not None:
        files = files[: args.max_tickers]
    if not files:
        raise ValueError("No ticker CSV files matched the requested selection")
    return files


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for old debug scripts."""

    ticker = str(df.get("ticker", pd.Series(["UNKNOWN"])).iloc[0]).upper()
    trainer = DualModelTrainer()
    return trainer.compute_features_for_ticker(ticker, df)


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy placeholder kept only to avoid broken debug imports."""

    return df.copy()


def main() -> None:
    args = parse_args()
    files = resolve_files(args)
    algorithms = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    trainer = DualModelTrainer(model_dir=args.output)

    prepared_output_dir = Path(args.prepared_output)
    if args.prepare_only:
        prepared_output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows: list[dict] = []
    prepared_rows: list[dict] = []
    failures = 0

    for csv_path in files:
        ticker = csv_path.stem.upper()
        try:
            daily_df = pd.read_csv(csv_path)
            if args.prepare_only:
                prepared = trainer.prepare_ticker_data(
                    ticker=ticker,
                    df=daily_df,
                    max_sequence_length=args.sequence_length if any(a in {"lstm", "bilstm"} for a in algorithms) else 1,
                )
                output_path = prepared_output_dir / f"{ticker}.csv"
                prepared.feature_frame.to_csv(output_path, index=False)
                prepared_rows.append(
                    {
                        "ticker": ticker,
                        **prepared.raw_stats,
                        "feature_columns": len(prepared.feature_columns),
                        "prepared_path": str(output_path),
                    }
                )
                logger.info("prepared_dataset_written", ticker=ticker, path=str(output_path))
                continue

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
                clean=not args.no_clean_output,
            )
            benchmark_rows.extend(result["report_rows"])
        except Exception as exc:
            failures += 1
            logger.error("ticker_training_failed", ticker=ticker, error=str(exc))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        pd.DataFrame(prepared_rows).to_csv(report_path, index=False)
        print(f"Prepared {len(prepared_rows)} datasets. Report: {report_path}")
        if failures:
            print(f"Failures: {failures}")
        return

    benchmark_df = pd.DataFrame(benchmark_rows)
    if not benchmark_df.empty:
        benchmark_df = benchmark_df.sort_values(["ticker", "horizon", "algorithm"]).reset_index(drop=True)
    benchmark_df.to_csv(report_path, index=False)
    print(f"Trained {len(benchmark_df)} model bundles across {len(files) - failures} tickers. Report: {report_path}")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
