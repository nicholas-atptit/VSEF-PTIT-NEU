"""Multi-horizon forward-return backtesting on fixed-window vnstock data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ml.metrics import compare_prediction_metric_sets, compute_prediction_error_metrics
from src.ml.backtest.real_data import FixedWindowBacktestConfig, RealDataBacktestRunner
from src.ml.models.factory import create_model
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)

FORWARD_RETURN_HORIZONS = {"3d": 3, "5d": 5, "20d": 20}
NAIVE_BASELINE_NAME = "naive_flat_return"
MOMENTUM_BASELINE_NAME = "momentum_continuation"
BASELINE_MODEL_NAMES = {NAIVE_BASELINE_NAME, MOMENTUM_BASELINE_NAME}


@dataclass(slots=True)
class ForwardReturnBacktestConfig(FixedWindowBacktestConfig):
    output_dir: str = "artifacts/backtest_forward_return"
    algorithms: list[str] = field(
        default_factory=lambda: ["cart", "xgboost", "lightgbm", "sarimax", "ets"]
    )
    horizons: list[str] = field(default_factory=lambda: ["3d", "5d", "20d"])
    task_type: str = "regression"
    target_type: str = "forward_return"
    include_momentum_baseline: bool = True
    beats_baseline_rule: str = "majority_of_metrics"


def _normalize_horizon_list(values: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    requested = [str(value).strip().lower() for value in (values or list(FORWARD_RETURN_HORIZONS)) if str(value).strip()]
    if not requested:
        raise ValueError("At least one horizon must be specified")
    invalid = [value for value in requested if value not in FORWARD_RETURN_HORIZONS]
    if invalid:
        raise ValueError(
            f"Unsupported forward-return horizons: {invalid}. Available: {sorted(FORWARD_RETURN_HORIZONS)}"
        )
    return {name: FORWARD_RETURN_HORIZONS[name] for name in dict.fromkeys(requested)}


def _return_sign(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    sign = np.sign(numeric)
    sign[numeric.isna()] = np.nan
    return sign.astype("float")


def _compute_error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return compute_prediction_error_metrics(actual, predicted)


class ForwardReturnBacktestRunner(RealDataBacktestRunner):
    """Compare multi-horizon forward-return models on a fixed target-date window."""

    def __init__(self, config: ForwardReturnBacktestConfig) -> None:
        super().__init__(config)
        self.config = config
        self._resolved_horizons = _normalize_horizon_list(config.horizons)

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
                logger.warning("forward_return_algorithm_unavailable", algorithm=algorithm, error=str(exc))
        if not available:
            raise ValueError("No requested algorithms are available in the current environment")
        return available, skipped

    @staticmethod
    def _required_sequence_length_for_trainer(trainer: DualModelTrainer, ticker: str) -> int:
        manifest = trainer._manifests[ticker]
        sequence_length = 1
        for horizon_info in manifest.get("horizons", {}).values():
            for algorithm_info in horizon_info.get("algorithms", {}).values():
                sequence_length = max(sequence_length, int(algorithm_info.get("sequence_length") or 1))
        return sequence_length

    def _algorithm_model_root(self, horizon_name: str, algorithm: str) -> Path:
        return self.output_dir / horizon_name / "models" / algorithm

    def _train_algorithm_ticker(
        self,
        trainer: DualModelTrainer,
        algorithm: str,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
        *,
        horizon_name: str,
        horizon_days: int,
    ) -> dict[str, Any]:
        train_history = history[history["date"] <= pd.Timestamp(self.config.train_end).normalize()].reset_index(drop=True)
        return trainer.train_explicit_split(
            ticker=ticker,
            df=train_history,
            train_start=self.config.train_start,
            train_end=self.config.train_end,
            algorithms=[algorithm],
            primary_algorithm=algorithm,
            horizon_name=horizon_name,
            horizon_days=horizon_days,
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
        *,
        horizon_name: str,
        horizon_days: int,
    ) -> pd.DataFrame:
        manifest = trainer._manifests[ticker]
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
        if eval_rows.empty:
            raise ValueError(f"{ticker} has no evaluation target rows inside the requested window")
        history_dates = pd.Index(pd.to_datetime(history["date"], errors="coerce").dt.normalize())
        history_close = pd.to_numeric(history["close"], errors="coerce").reset_index(drop=True)

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
                    f"{ticker} has no feature history available before target date {target_date.date()}"
                )

            prediction = trainer.predict(
                ticker=ticker,
                features=feature_slice,
                horizon=horizon_name,
                algorithm=algorithm,
            )
            current_close = float(feature_slice["close"].iloc[-1])
            target_close = float(eval_row.close)
            actual_return = float((target_close / current_close) - 1.0)
            predicted_return = float(prediction["predicted_return"])
            naive_predicted_return = 0.0
            momentum_predicted_return = np.nan
            momentum_pos = prediction_pos - horizon_days
            if self.config.include_momentum_baseline and momentum_pos >= 0:
                momentum_close = float(history_close.iloc[momentum_pos])
                if momentum_close != 0:
                    momentum_predicted_return = float((current_close / momentum_close) - 1.0)

            absolute_error = abs(predicted_return - actual_return)
            absolute_error_naive = abs(naive_predicted_return - actual_return)
            pct_error = np.nan if actual_return == 0 else float((absolute_error / abs(actual_return)) * 100.0)
            pct_error_naive = (
                np.nan if actual_return == 0 else float((absolute_error_naive / abs(actual_return)) * 100.0)
            )

            row = {
                "date": str(target_date.date()),
                "target_date": str(target_date.date()),
                "ticker": ticker,
                "model_name": algorithm,
                "prediction_date": str(prediction_date.date()),
                "horizon": horizon_name,
                "horizon_days": horizon_days,
                "current_close": current_close,
                "target_close": target_close,
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "naive_predicted_return": naive_predicted_return,
                "absolute_error": absolute_error,
                "absolute_error_naive": absolute_error_naive,
                "pct_error": pct_error,
                "pct_error_naive": pct_error_naive,
                "predicted_direction": int(np.sign(predicted_return)),
                "predicted_direction_naive": 0,
                "actual_direction": int(np.sign(actual_return)),
            }
            if self.config.include_momentum_baseline:
                absolute_error_momentum = (
                    np.nan if np.isnan(momentum_predicted_return) else abs(momentum_predicted_return - actual_return)
                )
                pct_error_momentum = (
                    np.nan
                    if actual_return == 0 or np.isnan(momentum_predicted_return)
                    else float((absolute_error_momentum / abs(actual_return)) * 100.0)
                )
                row.update(
                    {
                        "momentum_predicted_return": momentum_predicted_return,
                        "absolute_error_momentum": absolute_error_momentum,
                        "pct_error_momentum": pct_error_momentum,
                        "predicted_direction_momentum": (
                            np.nan if np.isnan(momentum_predicted_return) else int(np.sign(momentum_predicted_return))
                        ),
                    }
                )
            comparison_rows.append(row)

        return pd.DataFrame(comparison_rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    @staticmethod
    def _beats_baseline(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any], *, rule: str) -> tuple[bool, dict[str, bool]]:
        return compare_prediction_metric_sets(model_metrics, baseline_metrics, rule=rule)

    def _build_metrics_summary(
        self,
        comparison_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_rows: list[dict[str, Any]] = []

        def baseline_metrics(group: pd.DataFrame, predicted_col: str) -> dict[str, float]:
            return _compute_error_metrics(group["actual_return"], group[predicted_col])

        for (ticker, model_name), group in comparison_df.groupby(["ticker", "model_name"], sort=True):
            model_metrics = _compute_error_metrics(group["actual_return"], group["predicted_return"])
            naive_metrics = baseline_metrics(group, "naive_predicted_return")
            momentum_metrics = (
                baseline_metrics(group, "momentum_predicted_return")
                if "momentum_predicted_return" in group.columns
                else {
                    "observations": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "directional_accuracy": np.nan,
                }
            )
            beats_naive, metric_wins = self._beats_baseline(
                model_metrics,
                naive_metrics,
                rule=self.config.beats_baseline_rule,
            )
            summary_rows.append(
                {
                    "ticker": str(ticker),
                    "model_name": str(model_name),
                    "observations": model_metrics["observations"],
                    "mae": model_metrics["mae"],
                    "rmse": model_metrics["rmse"],
                    "mape": model_metrics["mape"],
                    "directional_accuracy": model_metrics["directional_accuracy"],
                    "naive_mae": naive_metrics["mae"],
                    "naive_rmse": naive_metrics["rmse"],
                    "naive_mape": naive_metrics["mape"],
                    "naive_directional_accuracy": naive_metrics["directional_accuracy"],
                    "momentum_mae": momentum_metrics["mae"],
                    "momentum_rmse": momentum_metrics["rmse"],
                    "momentum_mape": momentum_metrics["mape"],
                    "momentum_directional_accuracy": momentum_metrics["directional_accuracy"],
                    "beats_naive_baseline": beats_naive,
                    "beats_baseline_mae": metric_wins["mae"],
                    "beats_baseline_rmse": metric_wins["rmse"],
                    "beats_baseline_mape": metric_wins["mape"],
                    "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                    "wins_vs_naive_baseline": int(sum(metric_wins.values())),
                }
            )

        if not summary_rows:
            raise ValueError("No forward-return model comparison rows were produced")

        summary_df = pd.DataFrame(summary_rows).sort_values(["ticker", "model_name"]).reset_index(drop=True)

        baseline_rows: list[dict[str, Any]] = []
        for ticker, group in comparison_df.groupby("ticker", sort=True):
            naive_metrics = baseline_metrics(group, "naive_predicted_return")
            baseline_rows.append(
                {
                    "ticker": str(ticker),
                    "model_name": NAIVE_BASELINE_NAME,
                    "observations": naive_metrics["observations"],
                    "mae": naive_metrics["mae"],
                    "rmse": naive_metrics["rmse"],
                    "mape": naive_metrics["mape"],
                    "directional_accuracy": naive_metrics["directional_accuracy"],
                    "naive_mae": np.nan,
                    "naive_rmse": np.nan,
                    "naive_mape": np.nan,
                    "naive_directional_accuracy": np.nan,
                    "momentum_mae": np.nan,
                    "momentum_rmse": np.nan,
                    "momentum_mape": np.nan,
                    "momentum_directional_accuracy": np.nan,
                    "beats_naive_baseline": False,
                    "beats_baseline_mae": False,
                    "beats_baseline_rmse": False,
                    "beats_baseline_mape": False,
                    "beats_baseline_directional_accuracy": False,
                    "wins_vs_naive_baseline": 0,
                }
            )
            if "momentum_predicted_return" in group.columns:
                momentum_metrics = baseline_metrics(group, "momentum_predicted_return")
                beats_naive, metric_wins = self._beats_baseline(
                    momentum_metrics,
                    naive_metrics,
                    rule=self.config.beats_baseline_rule,
                )
                baseline_rows.append(
                    {
                        "ticker": str(ticker),
                        "model_name": MOMENTUM_BASELINE_NAME,
                        "observations": momentum_metrics["observations"],
                        "mae": momentum_metrics["mae"],
                        "rmse": momentum_metrics["rmse"],
                        "mape": momentum_metrics["mape"],
                        "directional_accuracy": momentum_metrics["directional_accuracy"],
                        "naive_mae": naive_metrics["mae"],
                        "naive_rmse": naive_metrics["rmse"],
                        "naive_mape": naive_metrics["mape"],
                        "naive_directional_accuracy": naive_metrics["directional_accuracy"],
                        "momentum_mae": np.nan,
                        "momentum_rmse": np.nan,
                        "momentum_mape": np.nan,
                        "momentum_directional_accuracy": np.nan,
                        "beats_naive_baseline": beats_naive,
                        "beats_baseline_mae": metric_wins["mae"],
                        "beats_baseline_rmse": metric_wins["rmse"],
                        "beats_baseline_mape": metric_wins["mape"],
                        "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                        "wins_vs_naive_baseline": int(sum(metric_wins.values())),
                    }
                )

        summary_df = (
            pd.concat([summary_df, pd.DataFrame(baseline_rows)], ignore_index=True)
            .sort_values(["ticker", "model_name"])
            .reset_index(drop=True)
        )

        overall_rows: list[dict[str, Any]] = []
        for model_name, group in comparison_df.groupby("model_name", sort=True):
            model_metrics = _compute_error_metrics(group["actual_return"], group["predicted_return"])
            naive_metrics = baseline_metrics(group, "naive_predicted_return")
            momentum_metrics = (
                baseline_metrics(group, "momentum_predicted_return")
                if "momentum_predicted_return" in group.columns
                else {
                    "observations": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "directional_accuracy": np.nan,
                }
            )
            beats_naive, metric_wins = self._beats_baseline(
                model_metrics,
                naive_metrics,
                rule=self.config.beats_baseline_rule,
            )
            overall_rows.append(
                {
                    "ticker": "OVERALL",
                    "model_name": str(model_name),
                    "observations": model_metrics["observations"],
                    "mae": model_metrics["mae"],
                    "rmse": model_metrics["rmse"],
                    "mape": model_metrics["mape"],
                    "directional_accuracy": model_metrics["directional_accuracy"],
                    "naive_mae": naive_metrics["mae"],
                    "naive_rmse": naive_metrics["rmse"],
                    "naive_mape": naive_metrics["mape"],
                    "naive_directional_accuracy": naive_metrics["directional_accuracy"],
                    "momentum_mae": momentum_metrics["mae"],
                    "momentum_rmse": momentum_metrics["rmse"],
                    "momentum_mape": momentum_metrics["mape"],
                    "momentum_directional_accuracy": momentum_metrics["directional_accuracy"],
                    "beats_naive_baseline": beats_naive,
                    "beats_baseline_mae": metric_wins["mae"],
                    "beats_baseline_rmse": metric_wins["rmse"],
                    "beats_baseline_mape": metric_wins["mape"],
                    "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                    "wins_vs_naive_baseline": int(sum(metric_wins.values())),
                }
            )

        overall_comparison = comparison_df.copy()
        naive_metrics = baseline_metrics(overall_comparison, "naive_predicted_return")
        overall_rows.append(
            {
                "ticker": "OVERALL",
                "model_name": NAIVE_BASELINE_NAME,
                "observations": naive_metrics["observations"],
                "mae": naive_metrics["mae"],
                "rmse": naive_metrics["rmse"],
                "mape": naive_metrics["mape"],
                "directional_accuracy": naive_metrics["directional_accuracy"],
                "naive_mae": np.nan,
                "naive_rmse": np.nan,
                "naive_mape": np.nan,
                "naive_directional_accuracy": np.nan,
                "momentum_mae": np.nan,
                "momentum_rmse": np.nan,
                "momentum_mape": np.nan,
                "momentum_directional_accuracy": np.nan,
                "beats_naive_baseline": False,
                "beats_baseline_mae": False,
                "beats_baseline_rmse": False,
                "beats_baseline_mape": False,
                "beats_baseline_directional_accuracy": False,
                "wins_vs_naive_baseline": 0,
            }
        )
        if "momentum_predicted_return" in overall_comparison.columns:
            momentum_metrics = baseline_metrics(overall_comparison, "momentum_predicted_return")
            beats_naive, metric_wins = self._beats_baseline(
                momentum_metrics,
                naive_metrics,
                rule=self.config.beats_baseline_rule,
            )
            overall_rows.append(
                {
                    "ticker": "OVERALL",
                    "model_name": MOMENTUM_BASELINE_NAME,
                    "observations": momentum_metrics["observations"],
                    "mae": momentum_metrics["mae"],
                    "rmse": momentum_metrics["rmse"],
                    "mape": momentum_metrics["mape"],
                    "directional_accuracy": momentum_metrics["directional_accuracy"],
                    "naive_mae": naive_metrics["mae"],
                    "naive_rmse": naive_metrics["rmse"],
                    "naive_mape": naive_metrics["mape"],
                    "naive_directional_accuracy": naive_metrics["directional_accuracy"],
                    "momentum_mae": np.nan,
                    "momentum_rmse": np.nan,
                    "momentum_mape": np.nan,
                    "momentum_directional_accuracy": np.nan,
                    "beats_naive_baseline": beats_naive,
                    "beats_baseline_mae": metric_wins["mae"],
                    "beats_baseline_rmse": metric_wins["rmse"],
                    "beats_baseline_mape": metric_wins["mape"],
                    "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                    "wins_vs_naive_baseline": int(sum(metric_wins.values())),
                }
            )

        summary_df = (
            pd.concat([summary_df, pd.DataFrame(overall_rows)], ignore_index=True)
            .sort_values(["ticker", "model_name"])
            .reset_index(drop=True)
        )

        ranking_df = summary_df[summary_df["ticker"] == "OVERALL"].copy().reset_index(drop=True)
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
        return summary_df, ranking_df

    def _render_horizon_charts(
        self,
        comparison_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        *,
        charts_dir: Path,
    ) -> dict[str, dict[str, str]]:
        chart_paths: dict[str, dict[str, str]] = {}
        charts_dir.mkdir(parents=True, exist_ok=True)
        for existing_png in charts_dir.glob("*.png"):
            existing_png.unlink()

        per_ticker_metrics = metrics_df[
            (metrics_df["ticker"] != "OVERALL") & (~metrics_df["model_name"].isin(BASELINE_MODEL_NAMES))
        ].copy()
        if per_ticker_metrics.empty:
            return chart_paths

        best_models = (
            per_ticker_metrics.sort_values(["ticker", "rmse", "mape", "model_name"])
            .groupby("ticker", sort=True, as_index=False)
            .first()[["ticker", "model_name"]]
        )

        for row in best_models.itertuples(index=False):
            ticker = str(row.ticker)
            model_name = str(row.model_name)
            series = comparison_df[
                (comparison_df["ticker"] == ticker) & (comparison_df["model_name"] == model_name)
            ].copy()
            series["target_date"] = pd.to_datetime(series["target_date"], errors="coerce")
            series = series.sort_values("target_date").reset_index(drop=True)

            actual_chart_path = charts_dir / f"{ticker}_actual_vs_predicted_return.png"
            plt.figure(figsize=(10, 4.8))
            plt.plot(series["target_date"], series["actual_return"], label="Actual Forward Return", linewidth=2.0)
            plt.plot(series["target_date"], series["predicted_return"], label=f"{model_name} Predicted Return", linewidth=1.8)
            plt.plot(
                series["target_date"],
                series["naive_predicted_return"],
                label="Naive Flat Baseline",
                linewidth=1.4,
                linestyle="--",
            )
            if "momentum_predicted_return" in series.columns:
                plt.plot(
                    series["target_date"],
                    series["momentum_predicted_return"],
                    label="Momentum Baseline",
                    linewidth=1.2,
                    linestyle=":",
                )
            plt.title(f"{ticker} Actual vs Predicted Forward Return ({model_name})")
            plt.xlabel("Target Date")
            plt.ylabel("Forward Return")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(actual_chart_path, dpi=140)
            plt.close()

            error_chart_path = charts_dir / f"{ticker}_absolute_error.png"
            plt.figure(figsize=(10, 4.8))
            plt.plot(series["target_date"], series["absolute_error"], label=f"{model_name} Absolute Error", linewidth=1.8)
            plt.plot(
                series["target_date"],
                series["absolute_error_naive"],
                label="Naive Absolute Error",
                linewidth=1.4,
                linestyle="--",
            )
            if "absolute_error_momentum" in series.columns:
                plt.plot(
                    series["target_date"],
                    series["absolute_error_momentum"],
                    label="Momentum Absolute Error",
                    linewidth=1.2,
                    linestyle=":",
                )
            plt.title(f"{ticker} Forward-Return Error Trend ({model_name})")
            plt.xlabel("Target Date")
            plt.ylabel("Absolute Error")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(error_chart_path, dpi=140)
            plt.close()

            chart_paths[ticker] = {
                "best_model": model_name,
                "actual_vs_predicted_return": str(actual_chart_path),
                "absolute_error": str(error_chart_path),
            }
        return chart_paths

    def _write_horizon_artifacts(
        self,
        *,
        horizon_name: str,
        horizon_days: int,
        comparison_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        ranking_df: pd.DataFrame,
        fetch_summary: pd.DataFrame,
        training_df: pd.DataFrame,
        available_algorithms: list[str],
        skipped_algorithms: list[dict[str, str]],
    ) -> dict[str, Any]:
        horizon_dir = self.output_dir / horizon_name
        charts_dir = horizon_dir / "charts"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        chart_paths = self._render_horizon_charts(comparison_df, metrics_df, charts_dir=charts_dir)
        target_dates = pd.to_datetime(comparison_df["target_date"], errors="coerce").dt.normalize()
        prediction_dates = pd.to_datetime(comparison_df["prediction_date"], errors="coerce").dt.normalize()
        leakage_checks = {
            "train_end_before_eval_start": bool(
                pd.Timestamp(self.config.train_end).normalize() < pd.Timestamp(self.config.eval_start).normalize()
            ),
            "target_dates_only_in_eval_window": bool(
                target_dates.between(
                    pd.Timestamp(self.config.eval_start).normalize(),
                    pd.Timestamp(self.config.eval_end).normalize(),
                ).all()
            ),
            "prediction_dates_before_target_dates": bool((prediction_dates < target_dates).all()),
        }

        paths = {
            "predicted_vs_actual": horizon_dir / "predicted_vs_actual.csv",
            "metrics_summary": horizon_dir / "metrics_summary.csv",
            "model_comparison": horizon_dir / "model_comparison.csv",
            "overall_model_ranking": horizon_dir / "overall_model_ranking.csv",
            "run_config": horizon_dir / "run_config.json",
            "fetch_summary": horizon_dir / "fetch_summary.csv",
            "training_summary": horizon_dir / "training_summary.csv",
            "charts_dir": charts_dir,
        }
        comparison_df.to_csv(paths["predicted_vs_actual"], index=False)
        metrics_df.to_csv(paths["metrics_summary"], index=False)
        metrics_df.to_csv(paths["model_comparison"], index=False)
        ranking_df.to_csv(paths["overall_model_ranking"], index=False)
        fetch_summary.to_csv(paths["fetch_summary"], index=False)
        training_df.to_csv(paths["training_summary"], index=False)

        run_config = asdict(self.config)
        run_config.update(
            {
                "source": "vnstock",
                "task_type": self.config.task_type,
                "target_type": self.config.target_type,
                "benchmark_basis": "forward_return_regression_with_directional_sign_check_vs_naive_and_optional_momentum_baselines",
                "comparable_tasks_only": True,
                "evaluation_context": "fixed_window_target_date_forward_return_backtest",
                "metric_semantics": {
                    "artifact_scope": "forward_return_prediction_backtest",
                    "prediction_metrics": "forward-return error metrics; not portfolio PnL metrics",
                    "metric_basis": "evaluation window target dates",
                    "financial_performance_metrics_included": False,
                    "heuristic_scenario_risk_included": False,
                    "comparability_warning": "These rankings compare forecast error and sign accuracy on a shared forward-return task only. They do not compare strategy PnL or risk calibration.",
                },
                "horizon_name": horizon_name,
                "horizon": horizon_name,
                "horizon_days": horizon_days,
                "evaluation_window_is_applied_to": "target_date",
                "prediction_date_definition": f"target_date minus {horizon_days} trading days",
                "target_formula": f"close[target_date] / close[prediction_date] - 1 where prediction_date is {horizon_days} trading days earlier",
                "available_algorithms": available_algorithms,
                "skipped_algorithms": skipped_algorithms,
                "beats_baseline_rule": self.config.beats_baseline_rule,
                "include_momentum_baseline": self.config.include_momentum_baseline,
                "baseline_models": [NAIVE_BASELINE_NAME]
                + ([MOMENTUM_BASELINE_NAME] if self.config.include_momentum_baseline else []),
                "fetched_data_summary": fetch_summary.to_dict(orient="records"),
                "leakage_checks": leakage_checks,
                "output_files": {name: str(path) for name, path in paths.items()},
                "chart_files": chart_paths,
            }
        )
        paths["run_config"].write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        return {
            "comparison": comparison_df,
            "metrics": metrics_df,
            "ranking": ranking_df,
            "fetch_summary": fetch_summary,
            "training_summary": training_df,
            "paths": {name: str(path) for name, path in paths.items()},
            "run_config": run_config,
            "chart_files": chart_paths,
        }

    @staticmethod
    def _build_overall_horizon_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for horizon, group in summary_df.groupby("horizon", sort=True):
            overall_group = group[group["ticker"] == "OVERALL"].copy()
            model_rows = overall_group[~overall_group["model_name"].isin(BASELINE_MODEL_NAMES)].copy()
            if model_rows.empty:
                continue
            best_rmse = model_rows.sort_values(["rmse", "model_name"]).iloc[0]
            best_mape = model_rows.sort_values(["mape", "model_name"]).iloc[0]
            best_dir = model_rows.sort_values(["directional_accuracy", "model_name"], ascending=[False, True]).iloc[0]
            rows.append(
                {
                    "horizon": str(horizon),
                    "best_model_by_rmse": str(best_rmse["model_name"]),
                    "best_model_by_mape": str(best_mape["model_name"]),
                    "best_model_by_directional_accuracy": str(best_dir["model_name"]),
                    "any_model_beats_naive_overall": bool(model_rows["beats_naive_baseline"].any()),
                }
            )
        return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

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

        horizon_results: dict[str, dict[str, Any]] = {}
        summary_frames: list[pd.DataFrame] = []
        for horizon_name, horizon_days in self._resolved_horizons.items():
            comparison_frames: list[pd.DataFrame] = []
            training_rows: list[dict[str, Any]] = []
            for algorithm in available_algorithms:
                trainer = DualModelTrainer(model_dir=self._algorithm_model_root(horizon_name, algorithm))
                for ticker in sorted(histories):
                    logger.info(
                        "forward_return_train_start",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon_name,
                        horizon_days=horizon_days,
                    )
                    train_result = self._train_algorithm_ticker(
                        trainer,
                        algorithm,
                        ticker,
                        histories[ticker],
                        context_sources,
                        horizon_name=horizon_name,
                        horizon_days=horizon_days,
                    )
                    for row in train_result["report_rows"]:
                        training_rows.append({"model_name": algorithm, "target_type": self.config.target_type, **row})
                    comparison_frames.append(
                        self._evaluate_algorithm_ticker(
                            trainer,
                            algorithm,
                            ticker,
                            histories[ticker],
                            context_sources,
                            horizon_name=horizon_name,
                            horizon_days=horizon_days,
                        )
                    )

            comparison_df = pd.concat(comparison_frames, ignore_index=True).sort_values(
                ["ticker", "model_name", "target_date"]
            ).reset_index(drop=True)
            metrics_df, ranking_df = self._build_metrics_summary(comparison_df)
            training_df = pd.DataFrame(training_rows).sort_values(
                ["model_name", "ticker", "horizon"]
            ).reset_index(drop=True)
            horizon_result = self._write_horizon_artifacts(
                horizon_name=horizon_name,
                horizon_days=horizon_days,
                comparison_df=comparison_df,
                metrics_df=metrics_df,
                ranking_df=ranking_df,
                fetch_summary=fetch_summary,
                training_df=training_df,
                available_algorithms=available_algorithms,
                skipped_algorithms=skipped_algorithms,
            )
            horizon_results[horizon_name] = horizon_result
            summary_with_horizon = metrics_df.copy()
            summary_with_horizon.insert(0, "horizon", horizon_name)
            summary_frames.append(summary_with_horizon)

        overall_summary_df = pd.concat(summary_frames, ignore_index=True).sort_values(
            ["horizon", "ticker", "model_name"]
        ).reset_index(drop=True)
        overall_ranking_df = self._build_overall_horizon_ranking(overall_summary_df)

        overall_paths = {
            "overall_horizon_summary": self.output_dir / "overall_horizon_summary.csv",
            "overall_horizon_ranking": self.output_dir / "overall_horizon_ranking.csv",
        }
        overall_summary_df.to_csv(overall_paths["overall_horizon_summary"], index=False)
        overall_ranking_df.to_csv(overall_paths["overall_horizon_ranking"], index=False)

        return {
            "horizons": horizon_results,
            "overall_summary": overall_summary_df,
            "overall_ranking": overall_ranking_df,
            "overall_paths": {name: str(path) for name, path in overall_paths.items()},
            "available_algorithms": available_algorithms,
            "skipped_algorithms": skipped_algorithms,
        }
