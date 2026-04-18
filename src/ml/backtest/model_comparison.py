"""Research-style multi-model comparison on the fixed-window real-data backtest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.backtest.real_data import FixedWindowBacktestConfig, RealDataBacktestRunner
from src.ml.metrics import compare_prediction_metric_sets, compute_prediction_error_metrics
from src.ml.models.factory import create_model
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ModelComparisonConfig(FixedWindowBacktestConfig):
    output_dir: str = "artifacts/backtest_model_comparison"
    algorithms: list[str] = field(
        default_factory=lambda: ["cart", "xgboost", "lightgbm", "sarimax", "ets"]
    )
    beats_baseline_rule: str = "majority_of_metrics"


class BacktestModelComparisonRunner(RealDataBacktestRunner):
    """Train and compare multiple model families on the same real-data split."""

    def __init__(self, config: ModelComparisonConfig) -> None:
        super().__init__(config)
        self.config = config

    @staticmethod
    def _normalize_algorithm_list(values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip().lower() for value in values if str(value).strip()))

    def _resolve_available_algorithms(self) -> tuple[list[str], list[dict[str, str]]]:
        available: list[str] = []
        skipped: list[dict[str, str]] = []
        for algorithm in self._normalize_algorithm_list(self.config.algorithms):
            try:
                create_model(algorithm, task="classification")
                create_model(algorithm, task="regression")
                available.append(algorithm)
            except Exception as exc:
                skipped.append({"algorithm": algorithm, "reason": str(exc)})
                logger.warning("comparison_algorithm_unavailable", algorithm=algorithm, error=str(exc))
        if not available:
            raise ValueError("No requested algorithms are available in the current environment")
        return available, skipped

    def _algorithm_model_root(self, algorithm: str) -> Path:
        return self.output_dir / "models" / algorithm

    def _train_algorithm_ticker(
        self,
        trainer: DualModelTrainer,
        algorithm: str,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
    ) -> dict[str, Any]:
        train_history = history[history["date"] <= pd.Timestamp(self.config.train_end).normalize()].reset_index(drop=True)
        return trainer.train_explicit_split(
            ticker=ticker,
            df=train_history,
            train_start=self.config.train_start,
            train_end=self.config.train_end,
            algorithms=[algorithm],
            primary_algorithm=algorithm,
            horizon_name=self.config.horizon_name,
            horizon_days=self.config.horizon_days,
            sequence_length=self.config.sequence_length,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            patience=self.config.patience,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            criterion=self.config.criterion,
            clean=True,
            context_sources=context_sources,
            validation_fraction=self.config.validation_fraction,
            validation_min_rows=self.config.validation_min_rows,
            min_train_rows=self.config.min_train_rows,
        )

    def _evaluate_algorithm_ticker(
        self,
        trainer: DualModelTrainer,
        algorithm: str,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
    ) -> pd.DataFrame:
        manifest = trainer._manifests[ticker]
        horizon_info = manifest["horizons"][self.config.horizon_name]
        horizon_days = int(horizon_info["days"])
        feature_frame = trainer.prepare_ticker_data(
            ticker=ticker,
            df=history,
            max_sequence_length=self._required_sequence_length_for_trainer(trainer, ticker),
            context_sources=context_sources,
            risk_config=manifest.get("advanced_risk"),
            window_start=self.config.train_start,
            window_end=self.config.eval_end,
        ).feature_frame

        eval_rows = history[
            (history["date"] >= pd.Timestamp(self.config.eval_start).normalize())
            & (history["date"] <= pd.Timestamp(self.config.eval_end).normalize())
        ][["date", "close"]].copy()
        history_dates = pd.Index(pd.to_datetime(history["date"], errors="coerce").dt.normalize())

        comparison_rows: list[dict[str, Any]] = []
        for eval_row in eval_rows.itertuples(index=False):
            target_date = pd.Timestamp(eval_row.date).normalize()
            target_pos = int(history_dates.get_loc(target_date))
            prediction_pos = target_pos - horizon_days
            if prediction_pos < 0:
                raise ValueError(
                    f"{ticker} does not have enough trading history before target date {target_date.date()} "
                    f"for horizon_days={horizon_days}"
                )
            prediction_date = pd.Timestamp(history_dates[prediction_pos]).normalize()
            feature_slice = feature_frame[feature_frame["date"] <= prediction_date].reset_index(drop=True)
            if feature_slice.empty:
                raise ValueError(
                    f"{ticker} has no feature history available before evaluation date {target_date.date()}"
                )

            prediction = trainer.predict(
                ticker=ticker,
                features=feature_slice,
                horizon=self.config.horizon_name,
                algorithm=algorithm,
            )
            current_close = float(feature_slice["close"].iloc[-1])
            actual_close = float(eval_row.close)
            predicted_close = float(current_close * (1.0 + prediction["predicted_return"]))
            predicted_close_baseline = current_close
            absolute_error = abs(predicted_close - actual_close)
            absolute_error_baseline = abs(predicted_close_baseline - actual_close)
            pct_error = np.nan if actual_close == 0 else float((absolute_error / abs(actual_close)) * 100.0)
            pct_error_baseline = (
                np.nan if actual_close == 0 else float((absolute_error_baseline / abs(actual_close)) * 100.0)
            )
            comparison_rows.append(
                {
                    "date": str(target_date.date()),
                    "ticker": ticker,
                    "model_name": algorithm,
                    "prediction_date": str(prediction_date.date()),
                    "horizon": self.config.horizon_name,
                    "horizon_days": horizon_days,
                    "current_close": current_close,
                    "actual_close": actual_close,
                    "predicted_close": predicted_close,
                    "predicted_close_baseline": predicted_close_baseline,
                    "predicted_return": float(prediction["predicted_return"]),
                    "absolute_error": absolute_error,
                    "absolute_error_baseline": absolute_error_baseline,
                    "pct_error": pct_error,
                    "pct_error_baseline": pct_error_baseline,
                    "predicted_direction": int(
                        prediction.get("predicted_direction", int(predicted_close > current_close))
                    ),
                    "predicted_direction_baseline": 0,
                    "actual_direction": int(actual_close > current_close),
                }
            )
        return pd.DataFrame(comparison_rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    @staticmethod
    def _required_sequence_length_for_trainer(trainer: DualModelTrainer, ticker: str) -> int:
        manifest = trainer._manifests[ticker]
        sequence_length = 1
        for horizon_info in manifest.get("horizons", {}).values():
            for algorithm_info in horizon_info.get("algorithms", {}).values():
                sequence_length = max(sequence_length, int(algorithm_info.get("sequence_length") or 1))
        return sequence_length

    @staticmethod
    def _beats_baseline(
        model_row: dict[str, Any],
        *,
        rule: str,
    ) -> tuple[bool, dict[str, bool]]:
        return compare_prediction_metric_sets(
            {
                "mae": model_row["mae"],
                "rmse": model_row["rmse"],
                "mape": model_row["mape"],
                "directional_accuracy": model_row["directional_accuracy"],
            },
            {
                "mae": model_row["baseline_mae"],
                "rmse": model_row["baseline_rmse"],
                "mape": model_row["baseline_mape"],
                "directional_accuracy": model_row["baseline_directional_accuracy"],
            },
            rule=rule,
        )

    def _build_model_comparison(self, comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        ranking_rows: list[dict[str, Any]] = []

        for (ticker, model_name), group in comparison_df.groupby(["ticker", "model_name"], sort=True):
            model_metrics = compute_prediction_error_metrics(
                actual=group["actual_close"],
                predicted=group["predicted_close"],
                actual_direction=group["actual_direction"],
                predicted_direction=group["predicted_direction"],
                mape_denominator=group["actual_close"],
            )
            baseline_metrics = compute_prediction_error_metrics(
                actual=group["actual_close"],
                predicted=group["predicted_close_baseline"],
                actual_direction=group["actual_direction"],
                predicted_direction=group["predicted_direction_baseline"],
                mape_denominator=group["actual_close"],
            )
            row = {
                "ticker": str(ticker),
                "model_name": str(model_name),
                "observations": int(model_metrics["observations"]),
                "mae": float(model_metrics["mae"]),
                "rmse": float(model_metrics["rmse"]),
                "mape": model_metrics["mape"],
                "directional_accuracy": float(model_metrics["directional_accuracy"]),
                "baseline_mae": float(baseline_metrics["mae"]),
                "baseline_rmse": float(baseline_metrics["rmse"]),
                "baseline_mape": baseline_metrics["mape"],
                "baseline_directional_accuracy": float(baseline_metrics["directional_accuracy"]),
            }
            beats, metric_wins = self._beats_baseline(row, rule=self.config.beats_baseline_rule)
            row.update(
                {
                    "beats_naive_baseline": beats,
                    "beats_baseline_mae": metric_wins["mae"],
                    "beats_baseline_rmse": metric_wins["rmse"],
                    "beats_baseline_mape": metric_wins["mape"],
                    "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                }
            )
            rows.append(row)

        if not rows:
            raise ValueError("No model comparison rows were produced")

        model_comparison_df = pd.DataFrame(rows).sort_values(["ticker", "model_name"]).reset_index(drop=True)

        for model_name, group in comparison_df.groupby("model_name", sort=True):
            model_metrics = compute_prediction_error_metrics(
                actual=group["actual_close"],
                predicted=group["predicted_close"],
                actual_direction=group["actual_direction"],
                predicted_direction=group["predicted_direction"],
                mape_denominator=group["actual_close"],
            )
            ranking_rows.append(
                {
                    "model_name": str(model_name),
                    "observations": int(model_metrics["observations"]),
                    "mae": float(model_metrics["mae"]),
                    "rmse": float(model_metrics["rmse"]),
                    "mape": model_metrics["mape"],
                    "directional_accuracy": float(model_metrics["directional_accuracy"]),
                    "tickers_beating_naive": int(
                        model_comparison_df[
                            (model_comparison_df["model_name"] == str(model_name))
                            & (model_comparison_df["beats_naive_baseline"])
                        ]["ticker"].nunique()
                    ),
                }
            )

        baseline_rows: list[dict[str, Any]] = []
        for ticker, group in comparison_df.groupby("ticker", sort=True):
            baseline_metrics = compute_prediction_error_metrics(
                actual=group["actual_close"],
                predicted=group["predicted_close_baseline"],
                actual_direction=group["actual_direction"],
                predicted_direction=group["predicted_direction_baseline"],
                mape_denominator=group["actual_close"],
            )
            baseline_rows.append(
                {
                    "ticker": str(ticker),
                    "model_name": "naive_previous_close",
                    "observations": int(baseline_metrics["observations"]),
                    "mae": float(baseline_metrics["mae"]),
                    "rmse": float(baseline_metrics["rmse"]),
                    "mape": baseline_metrics["mape"],
                    "directional_accuracy": float(baseline_metrics["directional_accuracy"]),
                    "baseline_mae": np.nan,
                    "baseline_rmse": np.nan,
                    "baseline_mape": np.nan,
                    "baseline_directional_accuracy": np.nan,
                    "beats_naive_baseline": False,
                    "beats_baseline_mae": False,
                    "beats_baseline_rmse": False,
                    "beats_baseline_mape": False,
                    "beats_baseline_directional_accuracy": False,
                }
            )
        model_comparison_df = (
            pd.concat([model_comparison_df, pd.DataFrame(baseline_rows)], ignore_index=True)
            .sort_values(["ticker", "model_name"])
            .reset_index(drop=True)
        )

        baseline_overall_metrics = compute_prediction_error_metrics(
            actual=comparison_df["actual_close"],
            predicted=comparison_df["predicted_close_baseline"],
            actual_direction=comparison_df["actual_direction"],
            predicted_direction=comparison_df["predicted_direction_baseline"],
            mape_denominator=comparison_df["actual_close"],
        )
        ranking_rows.append(
            {
                "model_name": "naive_previous_close",
                "observations": int(baseline_overall_metrics["observations"]),
                "mae": float(baseline_overall_metrics["mae"]),
                "rmse": float(baseline_overall_metrics["rmse"]),
                "mape": baseline_overall_metrics["mape"],
                "directional_accuracy": float(baseline_overall_metrics["directional_accuracy"]),
                "tickers_beating_naive": 0,
            }
        )

        ranking_df = pd.DataFrame(ranking_rows).sort_values("model_name").reset_index(drop=True)
        ranking_df["rank_rmse"] = ranking_df["rmse"].rank(method="dense", ascending=True).astype(int)
        ranking_df["rank_mape"] = ranking_df["mape"].rank(method="dense", ascending=True).astype(int)
        ranking_df["rank_directional_accuracy"] = ranking_df["directional_accuracy"].rank(
            method="dense", ascending=False
        ).astype(int)
        ranking_df["average_rank"] = ranking_df[
            ["rank_rmse", "rank_mape", "rank_directional_accuracy"]
        ].mean(axis=1)
        ranking_df = ranking_df.sort_values(
            ["average_rank", "rank_rmse", "rank_mape", "rank_directional_accuracy", "model_name"]
        ).reset_index(drop=True)
        return model_comparison_df, ranking_df

    def run(self) -> dict[str, Any]:
        dates = self._normalize_dates(self.config)
        if dates["train_end"] >= dates["eval_start"]:
            raise ValueError(
                f"train_end must be strictly earlier than eval_start. Got {dates['train_end'].date()} and {dates['eval_start'].date()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fetch_start = self._fetch_start()
        histories, fetch_summary = self._fetch_histories(fetch_start, dates["eval_end"])
        context_sources = self._build_context_sources(fetch_start, dates["eval_end"])
        available_algorithms, skipped_algorithms = self._resolve_available_algorithms()

        comparison_frames: list[pd.DataFrame] = []
        training_rows: list[dict[str, Any]] = []
        for algorithm in available_algorithms:
            trainer = DualModelTrainer(model_dir=self._algorithm_model_root(algorithm))
            for ticker in sorted(histories):
                logger.info("comparison_train_start", ticker=ticker, algorithm=algorithm)
                train_result = self._train_algorithm_ticker(trainer, algorithm, ticker, histories[ticker], context_sources)
                for row in train_result["report_rows"]:
                    training_rows.append({"model_name": algorithm, **row})
                comparison_frames.append(
                    self._evaluate_algorithm_ticker(trainer, algorithm, ticker, histories[ticker], context_sources)
                )

        comparison_df = pd.concat(comparison_frames, ignore_index=True).sort_values(
            ["ticker", "model_name", "date"]
        ).reset_index(drop=True)
        model_comparison_df, ranking_df = self._build_model_comparison(comparison_df)
        training_df = pd.DataFrame(training_rows).sort_values(["model_name", "ticker"]).reset_index(drop=True)

        paths = {
            "predicted_vs_actual": self.output_dir / "predicted_vs_actual.csv",
            "model_comparison": self.output_dir / "model_comparison.csv",
            "overall_model_ranking": self.output_dir / "overall_model_ranking.csv",
            "run_config": self.output_dir / "run_config.json",
            "fetch_summary": self.output_dir / "fetch_summary.csv",
            "training_summary": self.output_dir / "training_summary.csv",
        }
        comparison_df.to_csv(paths["predicted_vs_actual"], index=False)
        model_comparison_df.to_csv(paths["model_comparison"], index=False)
        ranking_df.to_csv(paths["overall_model_ranking"], index=False)
        fetch_summary.to_csv(paths["fetch_summary"], index=False)
        training_df.to_csv(paths["training_summary"], index=False)

        run_config = asdict(self.config)
        run_config.update(
            {
                "source": "vnstock",
                "benchmark_basis": "shared_close_level_regression_task_with_directional_sign_check",
                "comparable_tasks_only": True,
                "evaluation_context": "fixed_window_holdout_target_date_model_family_comparison",
                "metric_semantics": {
                    "artifact_scope": "fixed_window_model_comparison",
                    "prediction_metrics": "close-level held-out backtest error metrics by model family",
                    "metric_basis": "evaluation window target dates",
                    "financial_performance_metrics_included": False,
                    "heuristic_scenario_risk_included": False,
                    "comparability_warning": "Models are ranked only on the shared close-level prediction task. Do not read these ranks as portfolio-performance or uncertainty-calibration rankings.",
                },
                "fetch_start": str(fetch_start.date()),
                "train_start": str(dates["train_start"].date()),
                "train_end": str(dates["train_end"].date()),
                "eval_start": str(dates["eval_start"].date()),
                "eval_end": str(dates["eval_end"].date()),
                "available_algorithms": available_algorithms,
                "skipped_algorithms": skipped_algorithms,
                "beats_baseline_rule": self.config.beats_baseline_rule,
                "output_files": {name: str(path) for name, path in paths.items()},
            }
        )
        paths["run_config"].write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        return {
            "comparison": comparison_df,
            "model_comparison": model_comparison_df,
            "ranking": ranking_df,
            "fetch_summary": fetch_summary,
            "training_summary": training_df,
            "paths": {name: str(path) for name, path in paths.items()},
            "run_config": run_config,
        }
