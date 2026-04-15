"""Fixed-window real-data backtesting workflow for Vietnamese equities."""

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

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger
from src.validators.data_quality import DataQualityValidator

logger = get_logger(__name__)

STANDARD_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
MAX_TOLERATED_EVAL_TAIL_GAP_DAYS = 7


@dataclass(slots=True)
class FixedWindowBacktestConfig:
    tickers: list[str]
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    output_dir: str = "artifacts/backtest"
    algorithms: list[str] = field(default_factory=lambda: ["cart"])
    primary_algorithm: str | None = None
    interval: str = "1D"
    horizon_name: str = "daily"
    horizon_days: int = 1
    sequence_length: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    patience: int = 5
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    criterion: str | None = None
    validation_fraction: float = 0.15
    validation_min_rows: int = 20
    min_train_rows: int = 60
    clean_model_dir: bool = True


class RealDataBacktestRunner:
    """Train once on a fixed historical window, then evaluate daily on a holdout window."""

    def __init__(self, config: FixedWindowBacktestConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.model_root = self.output_dir / "models"
        self.charts_dir = self.output_dir / "charts"
        self.trainer = DualModelTrainer(model_dir=self.model_root)
        self.adapter = VnstockAdapter(symbol_list=[ticker.upper() for ticker in config.tickers])

    @staticmethod
    def _normalize_dates(config: FixedWindowBacktestConfig) -> dict[str, pd.Timestamp]:
        return {
            "train_start": pd.Timestamp(config.train_start).normalize(),
            "train_end": pd.Timestamp(config.train_end).normalize(),
            "eval_start": pd.Timestamp(config.eval_start).normalize(),
            "eval_end": pd.Timestamp(config.eval_end).normalize(),
        }

    def _fetch_start(self) -> pd.Timestamp:
        uses_sequence = any(algo.lower() in {"lstm", "bilstm"} for algo in self.config.algorithms)
        max_sequence = self.config.sequence_length if uses_sequence else 1
        return pd.Timestamp(self.config.train_start).normalize() - pd.Timedelta(
            days=self.trainer._warmup_buffer_days(max_sequence)
        )

    def _fetch_history(self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        history = self.adapter.get_ohlcv(
            ticker,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            interval=self.config.interval,
        )
        if history.empty:
            raise ValueError(f"No vnstock OHLCV data returned for {ticker}")

        standardized = self.trainer._normalize_ohlcv(history, ticker=ticker)[STANDARD_COLUMNS]
        DataQualityValidator(ticker=ticker).validate_ohlcv(standardized)
        return standardized

    def _build_context_sources(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> dict[str, pd.DataFrame | None]:
        context_sources = self.trainer._load_context_sources().copy()
        benchmark = self.adapter.get_index_ohlcv(
            "VNINDEX",
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            interval="1D",
        )
        if benchmark is not None and not benchmark.empty and "close" in benchmark.columns:
            benchmark = self.trainer._normalize_ohlcv(benchmark, ticker="VNINDEX")
            benchmark["m_ret"] = benchmark["close"].pct_change().fillna(0.0)
            context_sources["market_df"] = benchmark[["date", "m_ret"]].copy()
        return context_sources

    def _fetch_histories(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        histories: dict[str, pd.DataFrame] = {}
        summary_rows: list[dict[str, Any]] = []
        train_start_ts = pd.Timestamp(self.config.train_start).normalize()
        train_end_ts = pd.Timestamp(self.config.train_end).normalize()
        eval_start_ts = pd.Timestamp(self.config.eval_start).normalize()
        eval_end_ts = pd.Timestamp(self.config.eval_end).normalize()

        for raw_ticker in self.config.tickers:
            ticker = raw_ticker.upper().strip()
            history = self._fetch_history(ticker, start_ts, end_ts)
            fetched_min = pd.Timestamp(history["date"].min()).normalize()
            fetched_max = pd.Timestamp(history["date"].max()).normalize()
            if fetched_min > train_start_ts:
                raise ValueError(
                    f"{ticker} data starts at {fetched_min.date()}, later than requested train_start {train_start_ts.date()}"
                )

            train_rows = int(((history["date"] >= train_start_ts) & (history["date"] <= train_end_ts)).sum())
            eval_slice = history[
                (history["date"] >= eval_start_ts) & (history["date"] <= eval_end_ts)
            ].copy()
            eval_rows = int(len(eval_slice))
            if train_rows < self.config.min_train_rows:
                raise ValueError(f"{ticker} only has {train_rows} train-window rows; need at least {self.config.min_train_rows}")
            if eval_rows == 0:
                raise ValueError(f"{ticker} has no trading rows inside the evaluation window")
            eval_last_trading_date = pd.Timestamp(eval_slice["date"].max()).normalize()
            eval_tail_gap_days = int((eval_end_ts - eval_last_trading_date).days)
            if eval_tail_gap_days > MAX_TOLERATED_EVAL_TAIL_GAP_DAYS:
                raise ValueError(
                    f"{ticker} evaluation coverage ends at {eval_last_trading_date.date()}, too far before requested eval_end {eval_end_ts.date()}"
                )

            histories[ticker] = history
            summary_rows.append(
                {
                    "ticker": ticker,
                    "source": "vnstock",
                    "rows": int(len(history)),
                    "fetched_min_date": str(fetched_min.date()),
                    "fetched_max_date": str(fetched_max.date()),
                    "train_rows": train_rows,
                    "eval_rows": eval_rows,
                    "eval_last_trading_date": str(eval_last_trading_date.date()),
                    "eval_tail_gap_days": eval_tail_gap_days,
                }
            )

        return histories, pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True)

    def _required_sequence_length(self, ticker: str) -> int:
        manifest = self.trainer._manifests[ticker]
        sequence_length = 1
        for horizon_info in manifest.get("horizons", {}).values():
            for algorithm_info in horizon_info.get("algorithms", {}).values():
                sequence_length = max(sequence_length, int(algorithm_info.get("sequence_length") or 1))
        return sequence_length

    def _train_ticker(
        self,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
    ) -> dict[str, Any]:
        train_history = history[history["date"] <= pd.Timestamp(self.config.train_end).normalize()].reset_index(drop=True)
        return self.trainer.train_explicit_split(
            ticker=ticker,
            df=train_history,
            train_start=self.config.train_start,
            train_end=self.config.train_end,
            algorithms=self.config.algorithms,
            primary_algorithm=self.config.primary_algorithm,
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
            clean=self.config.clean_model_dir,
            context_sources=context_sources,
            validation_fraction=self.config.validation_fraction,
            validation_min_rows=self.config.validation_min_rows,
            min_train_rows=self.config.min_train_rows,
        )

    def _evaluate_ticker(
        self,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
    ) -> pd.DataFrame:
        manifest = self.trainer._manifests[ticker]
        horizon_info = manifest["horizons"][self.config.horizon_name]
        horizon_days = int(horizon_info["days"])
        feature_frame = self.trainer.prepare_ticker_data(
            ticker=ticker,
            df=history,
            max_sequence_length=self._required_sequence_length(ticker),
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
            raise ValueError(f"{ticker} has no evaluation rows inside the requested window")
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
                raise ValueError(f"{ticker} has no feature history available before evaluation date {target_date.date()}")
            prediction = self.trainer.predict(
                ticker=ticker,
                features=feature_slice,
                horizon=self.config.horizon_name,
                algorithm=self.config.primary_algorithm,
            )

            current_close = float(feature_slice["close"].iloc[-1])
            actual_close = float(eval_row.close)
            predicted_close = float(current_close * (1.0 + prediction["predicted_return"]))
            absolute_error = abs(predicted_close - actual_close)
            predicted_close_baseline = current_close
            absolute_error_baseline = abs(predicted_close_baseline - actual_close)
            pct_error = np.nan
            if actual_close != 0:
                pct_error = float((absolute_error / abs(actual_close)) * 100.0)
            pct_error_baseline = np.nan
            if actual_close != 0:
                pct_error_baseline = float((absolute_error_baseline / abs(actual_close)) * 100.0)
            comparison_rows.append(
                {
                    "date": str(target_date.date()),
                    "ticker": ticker,
                    "prediction_date": str(prediction_date.date()),
                    "horizon": self.config.horizon_name,
                    "horizon_days": horizon_days,
                    "current_close": current_close,
                    "actual_close": actual_close,
                    "predicted_close": predicted_close,
                    "predicted_close_baseline": predicted_close_baseline,
                    "predicted_return": float(prediction["predicted_return"]),
                    "predicted_return_baseline": 0.0,
                    "absolute_error": absolute_error,
                    "absolute_error_baseline": absolute_error_baseline,
                    "pct_error": pct_error,
                    "pct_error_baseline": pct_error_baseline,
                    "predicted_direction": int(prediction.get("predicted_direction", int(predicted_close > current_close))),
                    "predicted_direction_baseline": 0,
                    "actual_direction": int(actual_close > current_close),
                }
            )

        return pd.DataFrame(comparison_rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    @staticmethod
    def _metrics_rows(comparison_df: pd.DataFrame) -> pd.DataFrame:
        if comparison_df.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "observations",
                    "model_mae",
                    "baseline_mae",
                    "model_rmse",
                    "baseline_rmse",
                    "model_mape",
                    "baseline_mape",
                    "model_directional_accuracy",
                    "baseline_directional_accuracy",
                    "beats_baseline_overall",
                    "start_date",
                    "end_date",
                ]
            )

        rows: list[dict[str, Any]] = []
        groups: list[tuple[str, pd.DataFrame]] = [
            (str(ticker), group.copy())
            for ticker, group in comparison_df.groupby("ticker", sort=True, dropna=False)
        ]
        groups.append(("OVERALL", comparison_df.copy()))

        for label, group in groups:
            errors = pd.to_numeric(group["predicted_close"], errors="coerce") - pd.to_numeric(group["actual_close"], errors="coerce")
            baseline_errors = pd.to_numeric(group["predicted_close_baseline"], errors="coerce") - pd.to_numeric(group["actual_close"], errors="coerce")
            pct_errors = pd.to_numeric(group["pct_error"], errors="coerce")
            baseline_pct_errors = pd.to_numeric(group["pct_error_baseline"], errors="coerce")
            model_dir_acc = float(
                (group["predicted_direction"].astype(int) == group["actual_direction"].astype(int)).mean()
            )
            baseline_dir_acc = float(
                (group["predicted_direction_baseline"].astype(int) == group["actual_direction"].astype(int)).mean()
            )
            model_mae = float(pd.to_numeric(group["absolute_error"], errors="coerce").mean())
            baseline_mae = float(pd.to_numeric(group["absolute_error_baseline"], errors="coerce").mean())
            model_rmse = float(np.sqrt(np.mean(np.square(errors))))
            baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_errors))))
            model_mape = float(pct_errors.dropna().mean()) if not pct_errors.dropna().empty else np.nan
            baseline_mape = (
                float(baseline_pct_errors.dropna().mean()) if not baseline_pct_errors.dropna().empty else np.nan
            )
            metric_wins = {
                "mae": bool(model_mae < baseline_mae),
                "rmse": bool(model_rmse < baseline_rmse),
                "mape": bool(model_mape < baseline_mape) if not np.isnan(model_mape) and not np.isnan(baseline_mape) else False,
                "directional_accuracy": bool(model_dir_acc > baseline_dir_acc),
            }
            wins_vs_baseline = int(sum(metric_wins.values()))
            rows.append(
                {
                    "ticker": label,
                    "observations": int(len(group)),
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "mae_improvement_vs_baseline": baseline_mae - model_mae,
                    "model_rmse": model_rmse,
                    "baseline_rmse": baseline_rmse,
                    "rmse_improvement_vs_baseline": baseline_rmse - model_rmse,
                    "model_mape": model_mape,
                    "baseline_mape": baseline_mape,
                    "mape_improvement_vs_baseline": baseline_mape - model_mape if not np.isnan(model_mape) and not np.isnan(baseline_mape) else np.nan,
                    "model_directional_accuracy": model_dir_acc,
                    "baseline_directional_accuracy": baseline_dir_acc,
                    "directional_accuracy_improvement_vs_baseline": model_dir_acc - baseline_dir_acc,
                    "beats_baseline_mae": metric_wins["mae"],
                    "beats_baseline_rmse": metric_wins["rmse"],
                    "beats_baseline_mape": metric_wins["mape"],
                    "beats_baseline_directional_accuracy": metric_wins["directional_accuracy"],
                    "wins_vs_baseline": wins_vs_baseline,
                    "beats_baseline_overall": bool(wins_vs_baseline >= 3),
                    "start_date": str(group["date"].min()),
                    "end_date": str(group["date"].max()),
                }
            )
        return pd.DataFrame(rows)

    def _render_ticker_charts(self, comparison_df: pd.DataFrame) -> dict[str, dict[str, str]]:
        chart_paths: dict[str, dict[str, str]] = {}
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        for existing_png in self.charts_dir.glob("*.png"):
            existing_png.unlink()

        for ticker, group in comparison_df.groupby("ticker", sort=True):
            series = group.copy()
            series["date"] = pd.to_datetime(series["date"], errors="coerce")
            series = series.sort_values("date").reset_index(drop=True)

            close_chart_path = self.charts_dir / f"{ticker}_actual_vs_predicted.png"
            plt.figure(figsize=(10, 4.8))
            plt.plot(series["date"], series["actual_close"], label="Actual Close", linewidth=2.0)
            plt.plot(series["date"], series["predicted_close"], label="Model Predicted Close", linewidth=1.8)
            plt.plot(
                series["date"],
                series["predicted_close_baseline"],
                label="Naive Baseline Close",
                linewidth=1.4,
                linestyle="--",
            )
            plt.title(f"{ticker} Actual vs Predicted Close")
            plt.xlabel("Date")
            plt.ylabel("Close")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(close_chart_path, dpi=140)
            plt.close()

            error_chart_path = self.charts_dir / f"{ticker}_absolute_error.png"
            plt.figure(figsize=(10, 4.8))
            plt.plot(series["date"], series["absolute_error"], label="Model Absolute Error", linewidth=1.8)
            plt.plot(
                series["date"],
                series["absolute_error_baseline"],
                label="Naive Baseline Absolute Error",
                linewidth=1.4,
                linestyle="--",
            )
            plt.title(f"{ticker} Absolute Error Over Time")
            plt.xlabel("Date")
            plt.ylabel("Absolute Error")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(error_chart_path, dpi=140)
            plt.close()

            chart_paths[str(ticker)] = {
                "actual_vs_predicted": str(close_chart_path),
                "absolute_error": str(error_chart_path),
            }

        return chart_paths

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

        training_rows: list[dict[str, Any]] = []
        comparison_frames: list[pd.DataFrame] = []
        for ticker in sorted(histories):
            logger.info("backtest_train_start", ticker=ticker)
            train_result = self._train_ticker(ticker, histories[ticker], context_sources)
            training_rows.extend(train_result["report_rows"])
            comparison_frames.append(self._evaluate_ticker(ticker, histories[ticker], context_sources))

        comparison_df = pd.concat(comparison_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        metrics_df = self._metrics_rows(comparison_df)
        training_df = pd.DataFrame(training_rows).sort_values(["ticker", "algorithm"]).reset_index(drop=True)
        chart_paths = self._render_ticker_charts(comparison_df)

        eval_start_str = str(dates["eval_start"].date())
        eval_end_str = str(dates["eval_end"].date())
        comparison_date_values = pd.to_datetime(comparison_df["date"], errors="coerce").dt.normalize()
        leakage_checks = {
            "train_end_before_eval_start": bool(dates["train_end"] < dates["eval_start"]),
            "comparison_rows_only_in_eval_window": bool(
                comparison_date_values.between(dates["eval_start"], dates["eval_end"]).all()
            ),
            "prediction_dates_before_target_dates": bool(
                (
                    pd.to_datetime(comparison_df["prediction_date"], errors="coerce").dt.normalize()
                    < pd.to_datetime(comparison_df["date"], errors="coerce").dt.normalize()
                ).all()
            ),
        }

        paths = {
            "predicted_vs_actual": self.output_dir / "predicted_vs_actual.csv",
            "metrics_summary": self.output_dir / "metrics_summary.csv",
            "run_config": self.output_dir / "run_config.json",
            "fetch_summary": self.output_dir / "fetch_summary.csv",
            "training_summary": self.output_dir / "training_summary.csv",
            "charts_dir": self.charts_dir,
        }
        comparison_df.to_csv(paths["predicted_vs_actual"], index=False)
        metrics_df.to_csv(paths["metrics_summary"], index=False)
        fetch_summary.to_csv(paths["fetch_summary"], index=False)
        training_df.to_csv(paths["training_summary"], index=False)

        run_config = asdict(self.config)
        run_config.update(
            {
                "source": "vnstock",
                "model_root": str(self.model_root),
                "baseline_model": "naive_previous_close",
                "fetch_start": str(fetch_start.date()),
                "train_start": str(dates["train_start"].date()),
                "train_end": str(dates["train_end"].date()),
                "eval_start": eval_start_str,
                "eval_end": eval_end_str,
                "fetched_data_summary": fetch_summary.to_dict(orient="records"),
                "metrics_summary_rows": metrics_df.to_dict(orient="records"),
                "leakage_checks": leakage_checks,
                "output_files": {name: str(path) for name, path in paths.items()},
                "chart_files": chart_paths,
            }
        )
        paths["run_config"].write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        return {
            "comparison": comparison_df,
            "metrics": metrics_df,
            "fetch_summary": fetch_summary,
            "training_summary": training_df,
            "paths": {name: str(path) for name, path in paths.items()},
            "run_config": run_config,
            "chart_files": chart_paths,
        }
