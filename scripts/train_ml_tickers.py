"""Train technical ML models on the latest rolling 5-year data window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.universe import get_vn100_universe
from src.ml.benchmark.final_report import write_full_system_report
from src.ml.benchmark.risk_tuning import RiskTuningRunner
from src.ml.benchmark.stress_test import StressTestRunner
from src.ml.benchmark.system_benchmark import SystemBenchmarkRunner
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
    # Phase 7 integration flags
    parser.add_argument("--tune-boosters", action="store_true", help="Enable Optuna tuning for XGBoost/LightGBM")
    parser.add_argument("--enable-stacking", action="store_true", help="Append stacking ensemble to algorithm list (requires >=2 compatible base learners)")
    parser.add_argument("--enable-risk", action="store_true", help="Attach Monte Carlo VaR/CVaR risk assessment to training artifacts")
    parser.add_argument("--enable-covar", action="store_true", help="Enable rolling CoVaR / Delta-CoVaR features and summaries")
    parser.add_argument("--enable-risk-engine", action="store_true", help="Enable the full rolling risk engine (VaR/CVaR/CoVaR/Drawdown)")
    parser.add_argument("--enable-regime", action="store_true", help="Enable market regime detection and regime summaries")
    parser.add_argument("--enable-regime-switching", action="store_true", help="Inject regime features into eligible models")
    parser.add_argument("--enable-allocation", action="store_true", help="Enable risk-aware allocation suggestions")
    parser.add_argument("--risk-simulations", type=int, default=10000, help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--risk-confidence-levels", type=str, default="0.95,0.99", help="Comma-separated confidence levels (default: 0.95,0.99)")
    parser.add_argument("--risk-seed", type=int, default=42, help="Random seed for reproducible risk simulations (default: 42)")
    parser.add_argument("--covar-quantile", type=float, default=0.05, help="Tail quantile for VaR/CoVaR metrics (default: 0.05)")
    parser.add_argument("--covar-window", type=int, default=60, help="Rolling window for CoVaR/risk features (default: 60)")
    parser.add_argument("--regime-method", type=str, default="threshold", help="Regime detection method (default: threshold)")
    parser.add_argument("--risk-penalty-strength", type=float, default=1.0, help="Penalty strength for risk-aware allocation")
    parser.add_argument("--enable-benchmark", action="store_true", help="Run the multi-mode system benchmark workflow")
    parser.add_argument("--enable-stress-test", action="store_true", help="Run deterministic crisis stress tests")
    parser.add_argument("--enable-risk-tuning", action="store_true", help="Run Optuna-based risk parameter tuning")
    parser.add_argument("--benchmark-report", type=str, default="reports/system_benchmark.csv", help="System benchmark detail CSV path")
    parser.add_argument("--stress-report", type=str, default="reports/stress_test.csv", help="Stress test detail CSV path")
    parser.add_argument("--tuning-report", type=str, default="reports/risk_tuning.csv", help="Risk tuning leaderboard CSV path")
    parser.add_argument("--risk-tuning-trials", type=int, default=10, help="Number of Optuna trials for risk tuning")
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

    # Phase 7B: safe stacking enablement
    if args.enable_stacking:
        STACKING_COMPATIBLE = {"cart", "xgboost", "lightgbm", "sarimax", "ets"}
        compatible_count = len(set(algorithms) & STACKING_COMPATIBLE)
        if compatible_count >= 2:
            if "stacking" not in algorithms:
                algorithms.append("stacking")
                logger.info("stacking_enabled", base_learner_count=compatible_count)
        else:
            logger.warning(
                "stacking_skipped",
                reason=f"Only {compatible_count} compatible base learner(s) found. Need >=2.",
                available=algorithms,
            )

    # Phase 7B: build risk config only if enabled
    risk_config: dict | None = None
    if any(
        [
            args.enable_risk,
            args.enable_covar,
            args.enable_risk_engine,
            args.enable_regime,
            args.enable_regime_switching,
            args.enable_allocation,
        ]
    ):
        confidence_levels = [float(x.strip()) for x in args.risk_confidence_levels.split(",") if x.strip()]
        risk_config = {
            "risk_enabled": args.enable_risk,
            "enable_covar": args.enable_covar,
            "enable_risk_engine": args.enable_risk_engine,
            "enable_regime_detection": args.enable_regime,
            "enable_regime_switching": args.enable_regime_switching,
            "enable_risk_allocation": args.enable_allocation,
            "covar_quantile": args.covar_quantile,
            "covar_window": args.covar_window,
            "regime_method": args.regime_method,
            "risk_penalty_strength": args.risk_penalty_strength,
            "high_vol_exposure_cut": 0.6,
            "crisis_exposure_cut": 0.25,
            "high_vol_threshold": 0.03,
            "crisis_drawdown_threshold": -0.12,
            "crisis_delta_covar_threshold": 0.015,
            "simulations": args.risk_simulations,
            "confidence_levels": confidence_levels,
            "random_seed": args.risk_seed,
        }

    advanced_workflow = args.enable_benchmark or args.enable_stress_test or args.enable_risk_tuning
    if advanced_workflow:
        if args.prepare_only:
            raise ValueError("--prepare-only cannot be combined with benchmark/stress/tuning workflows")
        benchmark_result = None
        stress_result = None
        tuning_result = None

        if args.enable_benchmark:
            benchmark_runner = SystemBenchmarkRunner(model_root=args.output)
            benchmark_result = benchmark_runner.run(
                files=files,
                algorithms=algorithms,
                output_root=args.output,
                report_path=args.benchmark_report,
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

        if args.enable_stress_test:
            stress_runner = StressTestRunner(model_root=args.output)
            stress_result = stress_runner.run(
                files=files,
                algorithms=algorithms,
                output_root=Path(args.output) / "stress",
                report_path=args.stress_report,
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

        if args.enable_risk_tuning:
            tuning_runner = RiskTuningRunner(model_root=args.output)
            tuning_result = tuning_runner.run(
                files=files,
                algorithms=algorithms,
                output_root=Path(args.output) / "tuning",
                report_path=args.tuning_report,
                max_trials=args.risk_tuning_trials,
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

        final_report_path = write_full_system_report(
            benchmark_summary=None if benchmark_result is None else benchmark_result["summary"],
            stress_summary=None if stress_result is None else stress_result["summary"],
            tuning_result=tuning_result,
        )
        print(f"Full system report: {final_report_path}")
        if benchmark_result is not None:
            print(f"Benchmark detail: {benchmark_result['detail_path']}")
        if stress_result is not None:
            print(f"Stress detail: {stress_result['detail_path']}")
        if tuning_result is not None:
            print(f"Tuning detail: {tuning_result['csv_path']}")
        return

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
                    risk_config=risk_config,
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
                tune_boosters=args.tune_boosters,
                risk_config=risk_config,
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
