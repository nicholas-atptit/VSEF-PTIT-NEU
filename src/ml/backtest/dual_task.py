"""Dual-task backtesting for forward-return regression and profit classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.backtest.forward_return import (
    ForwardReturnBacktestConfig,
    ForwardReturnBacktestRunner,
    _compute_error_metrics,
)
from src.ml.trainer import DualModelTrainer


@dataclass(slots=True)
class DualTaskBacktestConfig(ForwardReturnBacktestConfig):
    output_dir: str = "artifacts/dual_task"
    task_type: str = "dual_task"
    target_type: str = "forward_return_and_profit_label"
    transaction_fee_bps: float = 15.0
    slippage_bps: float = 20.0


def _compute_profit_classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probability: pd.Series | None = None,
) -> dict[str, Any]:
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    predicted_numeric = pd.to_numeric(predicted, errors="coerce")
    mask = actual_numeric.notna() & predicted_numeric.notna()
    if not mask.any():
        return {
            "observations": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "positive_class_precision": np.nan,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 0,
        }

    y_true = actual_numeric.loc[mask].astype(int).to_numpy()
    y_pred = predicted_numeric.loc[mask].astype(int).to_numpy()
    roc_auc = np.nan
    if probability is not None:
        y_prob = pd.to_numeric(probability, errors="coerce").loc[mask]
        if y_prob.notna().all() and len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob.to_numpy(dtype=float)))
            except ValueError:
                roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "observations": int(mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "positive_class_precision": precision,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


class DualTaskBacktestRunner(ForwardReturnBacktestRunner):
    """Evaluate parallel regression and profit/loss classification tasks."""

    def __init__(self, config: DualTaskBacktestConfig) -> None:
        super().__init__(config)
        self.config = config
        self.regression_root = self.output_dir / "regression"
        self.classification_root = self.output_dir / "classification"
        self.summary_root = self.output_dir / "summary"

    def _algorithm_model_root(self, horizon_name: str, algorithm: str) -> Path:
        return self.output_dir / "models" / horizon_name / algorithm

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
            transaction_fee_bps=self.config.transaction_fee_bps,
            slippage_bps=self.config.slippage_bps,
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
        history_open = pd.to_numeric(history["open"], errors="coerce").reset_index(drop=True)
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
            entry_pos = prediction_pos + 1
            if entry_pos >= len(history_dates):
                continue

            prediction_date = pd.Timestamp(history_dates[prediction_pos]).normalize()
            entry_date = pd.Timestamp(history_dates[entry_pos]).normalize()
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
            if "predicted_profit_label" not in prediction:
                raise ValueError(
                    f"Profit classifier output missing for {ticker} {algorithm} {horizon_name}. "
                    "Retrain with the dual-task trainer path."
                )

            current_close = float(feature_slice["close"].iloc[-1])
            target_close = float(eval_row.close)
            entry_open = float(history_open.iloc[entry_pos])
            actual_return = float((target_close / current_close) - 1.0)
            actual_net_trade_return = DualModelTrainer.calculate_net_trade_return(
                entry_open,
                target_close,
                transaction_fee_bps=self.config.transaction_fee_bps,
                slippage_bps=self.config.slippage_bps,
            )
            actual_profit_label = int(actual_net_trade_return > 0.0)
            predicted_return = float(prediction["predicted_return"])
            predicted_profit_label = int(prediction["predicted_profit_label"])
            predicted_profit_probability = float(prediction.get("predicted_profit_probability", np.nan))
            absolute_error = abs(predicted_return - actual_return)
            pct_error = np.nan if actual_return == 0 else float((absolute_error / abs(actual_return)) * 100.0)

            comparison_rows.append(
                {
                    "date": str(target_date.date()),
                    "target_date": str(target_date.date()),
                    "prediction_date": str(prediction_date.date()),
                    "entry_date": str(entry_date.date()),
                    "ticker": ticker,
                    "model_name": algorithm,
                    "horizon": horizon_name,
                    "horizon_days": horizon_days,
                    "current_close": current_close,
                    "entry_open": entry_open,
                    "target_close": target_close,
                    "actual_return": actual_return,
                    "predicted_return": predicted_return,
                    "absolute_error": absolute_error,
                    "pct_error": pct_error,
                    "actual_direction": int(np.sign(actual_return)),
                    "predicted_direction": int(prediction["predicted_direction"]),
                    "actual_net_trade_return": actual_net_trade_return,
                    "actual_profit_label": actual_profit_label,
                    "predicted_profit_label": predicted_profit_label,
                    "predicted_profit_probability": predicted_profit_probability,
                    "transaction_fee_bps": float(self.config.transaction_fee_bps),
                    "slippage_bps": float(self.config.slippage_bps),
                    "round_trip_cost_bps": float(2.0 * (self.config.transaction_fee_bps + self.config.slippage_bps)),
                }
            )

        return pd.DataFrame(comparison_rows).sort_values(["ticker", "model_name", "date"]).reset_index(drop=True)

    @staticmethod
    def _build_regression_summary(comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        for (ticker, model_name), group in comparison_df.groupby(["ticker", "model_name"], sort=True):
            metrics = _compute_error_metrics(group["actual_return"], group["predicted_return"])
            rows.append({"ticker": str(ticker), "model_name": str(model_name), **metrics})

        for model_name, group in comparison_df.groupby("model_name", sort=True):
            metrics = _compute_error_metrics(group["actual_return"], group["predicted_return"])
            rows.append({"ticker": "OVERALL", "model_name": str(model_name), **metrics})

        summary_df = pd.DataFrame(rows).sort_values(["ticker", "model_name"]).reset_index(drop=True)
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

    @staticmethod
    def _build_classification_summary(
        comparison_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        for (ticker, model_name), group in comparison_df.groupby(["ticker", "model_name"], sort=True):
            metrics = _compute_profit_classification_metrics(
                group["actual_profit_label"],
                group["predicted_profit_label"],
                group["predicted_profit_probability"],
            )
            row = {"ticker": str(ticker), "model_name": str(model_name), **metrics}
            rows.append(row)
            confusion_rows.append(
                {
                    "ticker": str(ticker),
                    "model_name": str(model_name),
                    "true_negative": metrics["true_negative"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                    "true_positive": metrics["true_positive"],
                }
            )

        for model_name, group in comparison_df.groupby("model_name", sort=True):
            metrics = _compute_profit_classification_metrics(
                group["actual_profit_label"],
                group["predicted_profit_label"],
                group["predicted_profit_probability"],
            )
            row = {"ticker": "OVERALL", "model_name": str(model_name), **metrics}
            rows.append(row)
            confusion_rows.append(
                {
                    "ticker": "OVERALL",
                    "model_name": str(model_name),
                    "true_negative": metrics["true_negative"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                    "true_positive": metrics["true_positive"],
                }
            )

        summary_df = pd.DataFrame(rows).sort_values(["ticker", "model_name"]).reset_index(drop=True)
        confusion_df = pd.DataFrame(confusion_rows).sort_values(["ticker", "model_name"]).reset_index(drop=True)
        ranking_df = summary_df[summary_df["ticker"] == "OVERALL"].copy().reset_index(drop=True)
        ranking_df["rank_f1"] = ranking_df["f1"].rank(method="dense", ascending=False).astype(int)
        ranking_df["rank_positive_precision"] = ranking_df["positive_class_precision"].rank(
            method="dense", ascending=False
        ).astype(int)
        ranking_df["rank_recall"] = ranking_df["recall"].rank(method="dense", ascending=False).astype(int)
        roc_auc_rank_values = ranking_df["roc_auc"].fillna(-1.0)
        ranking_df["rank_roc_auc"] = roc_auc_rank_values.rank(method="dense", ascending=False).astype(int)
        ranking_df["average_rank"] = ranking_df[
            ["rank_f1", "rank_positive_precision", "rank_recall", "rank_roc_auc"]
        ].mean(axis=1)
        ranking_df = ranking_df.sort_values(
            ["average_rank", "rank_f1", "rank_positive_precision", "rank_recall", "model_name"]
        ).reset_index(drop=True)
        return summary_df, ranking_df, confusion_df

    @staticmethod
    def _render_regression_charts(
        comparison_df: pd.DataFrame,
        ranking_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        charts_dir: Path,
    ) -> dict[str, str]:
        chart_paths: dict[str, str] = {}
        charts_dir.mkdir(parents=True, exist_ok=True)
        per_ticker_metrics = summary_df[summary_df["ticker"] != "OVERALL"].copy()
        if per_ticker_metrics.empty:
            return chart_paths
        best_models = (
            per_ticker_metrics.sort_values(["ticker", "rmse", "mape", "model_name"])
            .groupby("ticker", sort=True, as_index=False)
            .first()[["ticker", "model_name"]]
        )
        for row in best_models.itertuples(index=False):
            series = comparison_df[
                (comparison_df["ticker"] == row.ticker) & (comparison_df["model_name"] == row.model_name)
            ].copy()
            series["target_date"] = pd.to_datetime(series["target_date"], errors="coerce")
            series = series.sort_values("target_date")
            chart_path = charts_dir / f"{row.ticker}_actual_vs_predicted_return.png"
            plt.figure(figsize=(10, 4.8))
            plt.plot(series["target_date"], series["actual_return"], label="Actual Return", linewidth=2.0)
            plt.plot(series["target_date"], series["predicted_return"], label=f"{row.model_name} Predicted Return", linewidth=1.8)
            plt.title(f"{row.ticker} Actual vs Predicted Forward Return ({row.model_name})")
            plt.xlabel("Target Date")
            plt.ylabel("Forward Return")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=140)
            plt.close()
            chart_paths[str(row.ticker)] = str(chart_path)
        return chart_paths

    @staticmethod
    def _render_classification_chart(ranking_df: pd.DataFrame, charts_dir: Path) -> dict[str, str]:
        charts_dir.mkdir(parents=True, exist_ok=True)
        overall_rows = ranking_df.copy()
        if overall_rows.empty:
            return {}
        chart_path = charts_dir / "overall_precision_recall_f1.png"
        plt.figure(figsize=(10, 5))
        x = np.arange(len(overall_rows))
        width = 0.25
        plt.bar(x - width, overall_rows["precision"], width=width, label="Precision")
        plt.bar(x, overall_rows["recall"], width=width, label="Recall")
        plt.bar(x + width, overall_rows["f1"], width=width, label="F1")
        plt.xticks(x, overall_rows["model_name"], rotation=20)
        plt.ylabel("Score")
        plt.title("Overall Profit-Classification Precision / Recall / F1")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_path, dpi=140)
        plt.close()
        return {"overall_precision_recall_f1": str(chart_path)}

    def _write_horizon_artifacts(
        self,
        *,
        horizon_name: str,
        horizon_days: int,
        comparison_df: pd.DataFrame,
        regression_metrics_df: pd.DataFrame,
        regression_ranking_df: pd.DataFrame,
        classification_metrics_df: pd.DataFrame,
        classification_ranking_df: pd.DataFrame,
        confusion_df: pd.DataFrame,
        fetch_summary: pd.DataFrame,
        training_df: pd.DataFrame,
        available_algorithms: list[str],
        skipped_algorithms: list[dict[str, str]],
    ) -> dict[str, Any]:
        regression_dir = self.regression_root / horizon_name
        classification_dir = self.classification_root / horizon_name
        regression_charts_dir = regression_dir / "charts"
        classification_charts_dir = classification_dir / "charts"
        regression_dir.mkdir(parents=True, exist_ok=True)
        classification_dir.mkdir(parents=True, exist_ok=True)

        regression_chart_paths = self._render_regression_charts(
            comparison_df,
            regression_ranking_df,
            regression_metrics_df,
            regression_charts_dir,
        )
        classification_chart_paths = self._render_classification_chart(
            classification_ranking_df,
            classification_charts_dir,
        )

        regression_paths = {
            "predicted_vs_actual": regression_dir / "predicted_vs_actual.csv",
            "metrics_summary": regression_dir / "metrics_summary.csv",
            "model_comparison": regression_dir / "model_comparison.csv",
            "overall_model_ranking": regression_dir / "overall_model_ranking.csv",
            "run_config": regression_dir / "run_config.json",
        }
        classification_paths = {
            "predicted_vs_actual": classification_dir / "predicted_vs_actual.csv",
            "classification_metrics": classification_dir / "classification_metrics.csv",
            "model_comparison": classification_dir / "model_comparison.csv",
            "overall_model_ranking": classification_dir / "overall_model_ranking.csv",
            "confusion_matrix": classification_dir / "confusion_matrix.csv",
            "run_config": classification_dir / "run_config.json",
        }

        regression_view = comparison_df[
            [
                "date",
                "ticker",
                "model_name",
                "prediction_date",
                "target_date",
                "horizon",
                "horizon_days",
                "actual_return",
                "predicted_return",
                "absolute_error",
                "pct_error",
            ]
        ].copy()
        classification_view = comparison_df[
            [
                "date",
                "ticker",
                "model_name",
                "prediction_date",
                "entry_date",
                "target_date",
                "horizon",
                "horizon_days",
                "actual_net_trade_return",
                "actual_profit_label",
                "predicted_profit_label",
                "predicted_profit_probability",
            ]
        ].copy()

        regression_view.to_csv(regression_paths["predicted_vs_actual"], index=False)
        regression_metrics_df.to_csv(regression_paths["metrics_summary"], index=False)
        regression_metrics_df.to_csv(regression_paths["model_comparison"], index=False)
        regression_ranking_df.to_csv(regression_paths["overall_model_ranking"], index=False)

        classification_view.to_csv(classification_paths["predicted_vs_actual"], index=False)
        classification_metrics_df.to_csv(classification_paths["classification_metrics"], index=False)
        classification_metrics_df.to_csv(classification_paths["model_comparison"], index=False)
        classification_ranking_df.to_csv(classification_paths["overall_model_ranking"], index=False)
        confusion_df.to_csv(classification_paths["confusion_matrix"], index=False)

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
        base_run_config = asdict(self.config)
        base_run_config.update(
            {
                "source": "vnstock",
                "horizon": horizon_name,
                "horizon_days": horizon_days,
                "available_algorithms": available_algorithms,
                "skipped_algorithms": skipped_algorithms,
                "fetched_data_summary": fetch_summary.to_dict(orient="records"),
                "training_summary": training_df.to_dict(orient="records"),
                "profit_target_config": {
                    "transaction_fee_bps": float(self.config.transaction_fee_bps),
                    "slippage_bps": float(self.config.slippage_bps),
                    "entry_convention": "next_tradable_open",
                    "exit_convention": "target_date_close",
                },
                "evaluation_window_is_applied_to": "target_date",
                "leakage_checks": leakage_checks,
            }
        )
        regression_run_config = {
            **base_run_config,
            "task_type": "regression",
            "target_type": "forward_return",
            "output_files": {name: str(path) for name, path in regression_paths.items()},
            "chart_files": regression_chart_paths,
        }
        classification_run_config = {
            **base_run_config,
            "task_type": "classification",
            "target_type": "profit_label",
            "output_files": {name: str(path) for name, path in classification_paths.items()},
            "chart_files": classification_chart_paths,
        }
        regression_paths["run_config"].write_text(json.dumps(regression_run_config, indent=2), encoding="utf-8")
        classification_paths["run_config"].write_text(json.dumps(classification_run_config, indent=2), encoding="utf-8")

        return {
            "regression_paths": {name: str(path) for name, path in regression_paths.items()},
            "classification_paths": {name: str(path) for name, path in classification_paths.items()},
            "regression_chart_files": regression_chart_paths,
            "classification_chart_files": classification_chart_paths,
        }

    @staticmethod
    def _build_cross_task_ranking(dual_task_summary_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for horizon_name, group in dual_task_summary_df.groupby("horizon", sort=True):
            overall_group = group[group["ticker"] == "OVERALL"].copy()
            if overall_group.empty:
                continue
            best_regression = overall_group.sort_values(["regression_rmse", "regression_mape", "model_name"]).iloc[0]
            best_profit_f1 = overall_group.sort_values(
                ["classification_f1", "classification_positive_class_precision", "model_name"],
                ascending=[False, False, True],
            ).iloc[0]
            best_profit_precision = overall_group.sort_values(
                ["classification_positive_class_precision", "classification_f1", "model_name"],
                ascending=[False, False, True],
            ).iloc[0]
            actionability_score = float(
                (0.6 * best_profit_f1["classification_f1"])
                + (0.4 * best_profit_precision["classification_positive_class_precision"])
            )
            rows.append(
                {
                    "horizon": horizon_name,
                    "best_regression_model_by_rmse": str(best_regression["model_name"]),
                    "best_regression_rmse": float(best_regression["regression_rmse"]),
                    "best_classification_model_by_f1": str(best_profit_f1["model_name"]),
                    "best_classification_f1": float(best_profit_f1["classification_f1"]),
                    "best_classification_model_by_positive_precision": str(best_profit_precision["model_name"]),
                    "best_classification_positive_precision": float(best_profit_precision["classification_positive_class_precision"]),
                    "same_model_for_both_tasks": bool(best_regression["model_name"] == best_profit_f1["model_name"]),
                    "actionability_score": actionability_score,
                }
            )
        ranking_df = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        if not ranking_df.empty:
            ranking_df["actionability_rank"] = ranking_df["actionability_score"].rank(
                method="dense",
                ascending=False,
            ).astype(int)
        return ranking_df

    def _render_summary_charts(
        self,
        cross_task_ranking_df: pd.DataFrame,
    ) -> dict[str, str]:
        chart_paths: dict[str, str] = {}
        self.summary_root.mkdir(parents=True, exist_ok=True)
        if cross_task_ranking_df.empty:
            return chart_paths

        precision_chart = self.summary_root / "positive_precision_by_horizon.png"
        plt.figure(figsize=(8.5, 4.8))
        plt.bar(
            cross_task_ranking_df["horizon"],
            cross_task_ranking_df["best_classification_positive_precision"],
            color="#2563eb",
        )
        plt.title("Best Positive-Class Precision by Horizon")
        plt.xlabel("Horizon")
        plt.ylabel("Positive-Class Precision")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(precision_chart, dpi=140)
        plt.close()
        chart_paths["positive_precision_by_horizon"] = str(precision_chart)

        cross_task_chart = self.summary_root / "cross_task_model_comparison.png"
        plt.figure(figsize=(9.5, 5.0))
        x = np.arange(len(cross_task_ranking_df))
        width = 0.35
        plt.bar(
            x - width / 2,
            cross_task_ranking_df["best_regression_rmse"],
            width=width,
            label="Best Regression RMSE",
        )
        plt.bar(
            x + width / 2,
            cross_task_ranking_df["best_classification_f1"],
            width=width,
            label="Best Classification F1",
        )
        plt.xticks(x, cross_task_ranking_df["horizon"])
        plt.title("Cross-Task Best Model Comparison by Horizon")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(cross_task_chart, dpi=140)
        plt.close()
        chart_paths["cross_task_model_comparison"] = str(cross_task_chart)
        return chart_paths

    def run(self) -> dict[str, Any]:
        dates = self._normalize_dates(self.config)
        if dates["train_end"] >= dates["eval_start"]:
            raise ValueError(
                f"train_end must be strictly earlier than eval_start. Got {dates['train_end'].date()} and {dates['eval_start'].date()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regression_root.mkdir(parents=True, exist_ok=True)
        self.classification_root.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        fetch_start = self._fetch_start()
        histories, fetch_summary = self._fetch_histories(fetch_start, dates["eval_end"])
        context_sources = self._build_context_sources(fetch_start, dates["eval_end"])
        available_algorithms, skipped_algorithms = self._resolve_available_algorithms()

        horizon_results: dict[str, dict[str, Any]] = {}
        regression_summary_frames: list[pd.DataFrame] = []
        classification_summary_frames: list[pd.DataFrame] = []
        joined_frames: list[pd.DataFrame] = []

        for horizon_name, horizon_days in self._resolved_horizons.items():
            comparison_frames: list[pd.DataFrame] = []
            training_rows: list[dict[str, Any]] = []
            for algorithm in available_algorithms:
                trainer = DualModelTrainer(model_dir=self._algorithm_model_root(horizon_name, algorithm))
                for ticker in sorted(histories):
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
                        training_rows.append({"model_name": algorithm, **row})
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
            regression_metrics_df, regression_ranking_df = self._build_regression_summary(comparison_df)
            classification_metrics_df, classification_ranking_df, confusion_df = self._build_classification_summary(
                comparison_df
            )
            training_df = pd.DataFrame(training_rows).sort_values(
                ["model_name", "ticker", "horizon"]
            ).reset_index(drop=True)
            paths_info = self._write_horizon_artifacts(
                horizon_name=horizon_name,
                horizon_days=horizon_days,
                comparison_df=comparison_df,
                regression_metrics_df=regression_metrics_df,
                regression_ranking_df=regression_ranking_df,
                classification_metrics_df=classification_metrics_df,
                classification_ranking_df=classification_ranking_df,
                confusion_df=confusion_df,
                fetch_summary=fetch_summary,
                training_df=training_df,
                available_algorithms=available_algorithms,
                skipped_algorithms=skipped_algorithms,
            )
            horizon_results[horizon_name] = {
                "comparison": comparison_df,
                "regression_metrics": regression_metrics_df,
                "regression_ranking": regression_ranking_df,
                "classification_metrics": classification_metrics_df,
                "classification_ranking": classification_ranking_df,
                "confusion_matrix": confusion_df,
                **paths_info,
            }
            regression_with_horizon = regression_metrics_df.copy()
            regression_with_horizon.insert(0, "horizon", horizon_name)
            classification_with_horizon = classification_metrics_df.copy()
            classification_with_horizon.insert(0, "horizon", horizon_name)
            regression_summary_frames.append(regression_with_horizon)
            classification_summary_frames.append(classification_with_horizon)
            joined_frames.append(comparison_df.copy())

        all_regression_df = pd.concat(regression_summary_frames, ignore_index=True).sort_values(
            ["horizon", "ticker", "model_name"]
        ).reset_index(drop=True)
        all_classification_df = pd.concat(classification_summary_frames, ignore_index=True).sort_values(
            ["horizon", "ticker", "model_name"]
        ).reset_index(drop=True)
        joined_evaluation_df = pd.concat(joined_frames, ignore_index=True).sort_values(
            ["horizon", "ticker", "model_name", "date"]
        ).reset_index(drop=True)
        dual_task_summary_df = all_regression_df.merge(
            all_classification_df,
            on=["horizon", "ticker", "model_name"],
            how="inner",
            suffixes=("_regression", "_classification"),
        ).rename(
            columns={
                "observations_regression": "regression_observations",
                "mae": "regression_mae",
                "rmse": "regression_rmse",
                "mape": "regression_mape",
                "directional_accuracy": "regression_directional_accuracy",
                "observations_classification": "classification_observations",
                "accuracy": "classification_accuracy",
                "precision": "classification_precision",
                "recall": "classification_recall",
                "f1": "classification_f1",
                "roc_auc": "classification_roc_auc",
                "positive_class_precision": "classification_positive_class_precision",
                "true_negative": "classification_true_negative",
                "false_positive": "classification_false_positive",
                "false_negative": "classification_false_negative",
                "true_positive": "classification_true_positive",
            }
        )
        cross_task_ranking_df = self._build_cross_task_ranking(dual_task_summary_df)
        summary_chart_paths = self._render_summary_charts(cross_task_ranking_df)

        summary_paths = {
            "dual_task_summary": self.summary_root / "dual_task_summary.csv",
            "cross_task_model_ranking": self.summary_root / "cross_task_model_ranking.csv",
            "joined_evaluation": self.summary_root / "joined_regression_classification_evaluation.csv",
        }
        dual_task_summary_df.to_csv(summary_paths["dual_task_summary"], index=False)
        cross_task_ranking_df.to_csv(summary_paths["cross_task_model_ranking"], index=False)
        joined_evaluation_df[
            [
                "date",
                "ticker",
                "horizon",
                "model_name",
                "actual_return",
                "predicted_return",
                "actual_profit_label",
                "predicted_profit_label",
                "predicted_profit_probability",
            ]
        ].to_csv(summary_paths["joined_evaluation"], index=False)

        return {
            "horizons": horizon_results,
            "dual_task_summary": dual_task_summary_df,
            "cross_task_model_ranking": cross_task_ranking_df,
            "joined_evaluation": joined_evaluation_df,
            "summary_paths": {name: str(path) for name, path in summary_paths.items()},
            "summary_chart_files": summary_chart_paths,
            "available_algorithms": available_algorithms,
            "skipped_algorithms": skipped_algorithms,
        }
