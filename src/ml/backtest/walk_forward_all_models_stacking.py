"""Walk-forward forecasting experiment across all supported modern ML models.

This runner keeps the existing leakage-aware feature and training path from the
modern ML engine, then adds a prequential stacking layer over model outputs.
"""

from __future__ import annotations

import contextlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.data_loader import load_ohlcv_from_csv
from src.ml.backtest.linear_fold_diagnostics import (
    empty_coefficient_diagnostics_frame,
    fit_linear_fold_diagnostics,
    select_linear_diagnostic_features,
    summarize_coefficient_stability,
)
from src.ml.backtest.feature_importance_diagnostics import (
    SUPPORTED_IMPORTANCE_MODELS,
    compare_linear_and_importance_diagnostics,
    empty_feature_importance_diagnostics_frame,
    extract_feature_importance_rows,
    summarize_feature_importance_stability,
)
from src.ml.backtest.feature_governance_review import (
    build_feature_governance_review,
    empty_feature_governance_review_frame,
)
from src.ml.backtest.context_coverage_diagnostics import (
    build_context_coverage_rows,
    empty_context_coverage_diagnostics_frame,
    summarize_context_coverage,
)
from src.ml.backtest.foreign_flow_validation import validate_foreign_flow_artifact
from src.ml.metrics import (
    compute_brier_score,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_signal_turnover,
    compute_sortino_ratio,
    compute_win_rate,
)
from src.ml.models.factory import create_model, supported_algorithms
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger
from src.validators.data_quality import DataQualityValidator

logger = get_logger(__name__)

DEFAULT_EXPERIMENT_TICKERS = ["MSN", "MWG", "DGC", "SSI", "FPT", "ACB"]
DEFAULT_HORIZONS = {
    "short_5d": 5,
    "short_10d": 10,
    "short_20d": 20,
    "short_30d": 30,
    "long_3m": 63,
    "long_6m": 126,
}
SAFE_DIRECTION_PROBABILITY_ALGORITHMS = {"cart", "xgboost", "lightgbm", "lstm", "bilstm"}
FINAL_STACKING_MODEL_NAME = "stacking_final"
BUY_AND_HOLD_MODEL_NAME = "buy_and_hold"
MIN_META_SAMPLES = 20
STANDARD_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(slots=True)
class WalkForwardAllModelsStackingConfig:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_EXPERIMENT_TICKERS))
    history_start: str = "2018-01-01"
    history_end: str = "2026-03-31"
    initial_train_start: str = "2018-01-01"
    initial_train_end: str = "2024-12-31"
    forecast_start: str = "2025-01-01"
    forecast_end: str = "2026-03-31"
    output_dir: str = "artifacts/walk_forward_all_models_stacking_eval"
    horizons: list[str] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    step_sizes: list[int] = field(default_factory=lambda: [1, 2])
    algorithms: list[str] | None = None
    interval: str = "1D"
    sequence_length: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    patience: int = 5
    max_depth: int | None = 4
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    criterion: str | None = None
    validation_fraction: float = 0.15
    validation_min_rows: int = 20
    min_train_rows: int = 60
    meta_model_alpha: float = 1.0
    meta_min_samples: int = MIN_META_SAMPLES
    max_workers: int = 1
    enable_linear_coefficient_diagnostics: bool = True
    linear_diagnostic_models: list[str] = field(default_factory=lambda: ["linear", "ridge", "lasso"])
    enable_feature_importance_diagnostics: bool = True
    foreign_flow_path: str | None = None


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ("prediction_date", "feature_date", "target_date"):
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce").dt.normalize()
    return prepared


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isfinite(numeric):
        return numeric
    return float("nan")


def _smape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    predicted_numeric = pd.to_numeric(predicted, errors="coerce")
    mask = actual_numeric.notna() & predicted_numeric.notna()
    if not mask.any():
        return float("nan")
    actual_values = actual_numeric.loc[mask].astype(float)
    predicted_values = predicted_numeric.loc[mask].astype(float)
    denominator = actual_values.abs() + predicted_values.abs()
    valid = denominator > 0.0
    if not valid.any():
        return float("nan")
    return float((200.0 * (actual_values.loc[valid] - predicted_values.loc[valid]).abs() / denominator.loc[valid]).mean())


def _correlation(actual: pd.Series, predicted: pd.Series) -> float:
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    predicted_numeric = pd.to_numeric(predicted, errors="coerce")
    mask = actual_numeric.notna() & predicted_numeric.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    return float(actual_numeric.loc[mask].corr(predicted_numeric.loc[mask]))


def _cagr(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if pd.isna(start_date) or pd.isna(end_date):
        return float("nan")
    elapsed_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 1)
    years = elapsed_days / 365.25
    equity = max(1.0 + float(total_return), 1e-12)
    return float(equity ** (1.0 / years) - 1.0)


def _generate_daily_predictions_worker(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    with Path("NUL").open("w", encoding="utf-8", errors="ignore") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        config = WalkForwardAllModelsStackingConfig(**payload["config"])
        config.tickers = [str(payload["ticker"]).upper()]
        config.max_workers = 1
        runner = WalkForwardAllModelsStackingRunner(config)
        start_ts = pd.Timestamp(payload["history_start"]).normalize()
        end_ts = pd.Timestamp(payload["history_end"]).normalize()
        forecast_start = pd.Timestamp(payload["forecast_start"]).normalize()
        forecast_end = pd.Timestamp(payload["forecast_end"]).normalize()
        history = runner._fetch_history(str(payload["ticker"]).upper(), start_ts, end_ts)
        context_sources = runner._build_context_sources(start_ts, end_ts)
        base_predictions, linear_diagnostics, importance_diagnostics, context_coverage = runner._generate_daily_predictions_for_ticker(
            ticker=str(payload["ticker"]).upper(),
            history=history,
            context_sources=context_sources,
            algorithms=[str(value).lower() for value in payload["algorithms"]],
            horizons={str(name): int(days) for name, days in payload["horizons"].items()},
            step_size=int(payload.get("step_size", 1)),
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        return {
            "base_predictions": base_predictions,
            "linear_coefficient_diagnostics": linear_diagnostics,
            "feature_importance_diagnostics": importance_diagnostics,
            "context_coverage_diagnostics": context_coverage,
        }


class WalkForwardAllModelsStackingRunner:
    """Run rolling forecasts with all supported models and a safe stacking layer."""

    def __init__(self, config: WalkForwardAllModelsStackingConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.csv_dir = self.output_dir / "csv"
        self.charts_dir = self.output_dir / "charts"
        self.models_dir = self.output_dir / "tmp_models"
        self.report_path = self.output_dir / "report.md"
        self.adapter = VnstockAdapter(symbol_list=[ticker.upper() for ticker in config.tickers])
        self._foreign_flow_context_metadata: dict[str, Any] = {
            "foreign_flow_path": config.foreign_flow_path,
            "foreign_flow_path_explicit": bool(config.foreign_flow_path),
            "artifact_validation": None,
        }

    @staticmethod
    def _normalize_horizons(values: list[str] | tuple[str, ...] | None) -> dict[str, int]:
        requested = [str(value).strip().lower() for value in (values or list(DEFAULT_HORIZONS)) if str(value).strip()]
        if not requested:
            raise ValueError("At least one forecast horizon must be requested")
        invalid = [value for value in requested if value not in DEFAULT_HORIZONS]
        if invalid:
            raise ValueError(f"Unsupported horizons: {invalid}. Available: {sorted(DEFAULT_HORIZONS)}")
        return {name: DEFAULT_HORIZONS[name] for name in dict.fromkeys(requested)}

    def _resolve_step_sizes(self) -> list[int]:
        resolved = []
        for raw_value in self.config.step_sizes:
            value = int(raw_value)
            if value <= 0:
                raise ValueError("step_sizes must be positive integers")
            if value not in resolved:
                resolved.append(value)
        if not resolved:
            raise ValueError("At least one step size must be specified")
        return resolved

    def _resolve_available_algorithms(self) -> tuple[list[str], list[dict[str, str]]]:
        requested = self.config.algorithms or [name for name in supported_algorithms() if name != "stacking"]
        available: list[str] = []
        skipped: list[dict[str, str]] = []
        for algorithm in dict.fromkeys(str(name).strip().lower() for name in requested if str(name).strip()):
            if algorithm == "stacking":
                continue
            try:
                create_model(algorithm, task="classification")
                create_model(algorithm, task="regression")
                available.append(algorithm)
            except Exception as exc:
                skipped.append({"algorithm": algorithm, "reason": str(exc)})
                logger.warning("walk_forward_algorithm_unavailable", algorithm=algorithm, error=str(exc))
        if not available:
            raise ValueError("No supported forecasting algorithms are available in the current environment")
        return available, skipped

    def _fetch_history(self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        history = self.adapter.get_ohlcv(
            ticker,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            interval=self.config.interval,
        )
        source = f"vnstock_{str(history.attrs.get('vnstock_source', 'unknown')).lower()}" if not history.empty else "vnstock"
        if history.empty:
            history = load_ohlcv_from_csv(
                ticker,
                start_date=start_ts.to_pydatetime().date(),
                end_date=end_ts.to_pydatetime().date(),
            )
            source = "csv_fallback"
        if history.empty:
            raise ValueError(f"No OHLCV data returned for {ticker} from vnstock or local CSV fallback")
        trainer = DualModelTrainer(model_dir=self.models_dir / "_scratch")
        standardized = trainer._normalize_ohlcv(history, ticker=ticker)[STANDARD_COLUMNS]
        DataQualityValidator(ticker=ticker).validate_ohlcv(standardized)
        standardized.attrs["source"] = source
        return standardized

    def _build_context_sources(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> dict[str, Any]:
        trainer = DualModelTrainer(model_dir=self.models_dir / "_scratch")
        context_sources = trainer._load_context_sources(foreign_flow_path=self.config.foreign_flow_path).copy()
        foreign_flow_df = context_sources.get("foreign_flow_df")
        self._foreign_flow_context_metadata = {
            "foreign_flow_path": self.config.foreign_flow_path,
            "foreign_flow_path_explicit": bool(self.config.foreign_flow_path),
            "source_name": (
                foreign_flow_df.attrs.get("source_name")
                if isinstance(foreign_flow_df, pd.DataFrame)
                else None
            ),
            "source_provenance": (
                foreign_flow_df.attrs.get("source_provenance")
                if isinstance(foreign_flow_df, pd.DataFrame)
                else None
            ),
            "row_count": int(len(foreign_flow_df)) if isinstance(foreign_flow_df, pd.DataFrame) else 0,
            "artifact_validation": None,
        }
        if self.config.foreign_flow_path:
            validation_start = pd.Timestamp(self.config.forecast_start).normalize()
            validation_end = pd.Timestamp(self.config.forecast_end).normalize()
            self._foreign_flow_context_metadata["artifact_validation"] = validate_foreign_flow_artifact(
                foreign_flow_df if isinstance(foreign_flow_df, pd.DataFrame) else pd.DataFrame(),
                self.config.tickers,
                validation_start,
                validation_end,
            )
        benchmark = self.adapter.get_index_ohlcv(
            "VNINDEX",
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            interval="1D",
        )
        if benchmark is not None and not benchmark.empty and "close" in benchmark.columns:
            benchmark = trainer._normalize_ohlcv(benchmark, ticker="VNINDEX")
            benchmark["m_ret"] = benchmark["close"].pct_change().fillna(0.0)
            context_sources["market_df"] = benchmark[["date", "m_ret"]].copy()
        return context_sources

    def _fetch_histories(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        histories: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, Any]] = []
        for raw_ticker in self.config.tickers:
            ticker = raw_ticker.upper().strip()
            history = self._fetch_history(ticker, start_ts, end_ts)
            fetched_min = pd.Timestamp(history["date"].min()).normalize()
            fetched_max = pd.Timestamp(history["date"].max()).normalize()
            histories[ticker] = history
            rows.append(
                {
                    "ticker": ticker,
                    "source": str(history.attrs.get("source", "unknown")),
                    "rows": int(len(history)),
                    "fetched_min_date": str(fetched_min.date()),
                    "fetched_max_date": str(fetched_max.date()),
                }
            )
        return histories, pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    @staticmethod
    def _prediction_schedule(
        history: pd.DataFrame,
        *,
        step_size: int,
        forecast_start: pd.Timestamp,
        forecast_end: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        history_dates = pd.Index(pd.to_datetime(history["date"], errors="coerce").dt.normalize())
        forecast_positions = [
            pos
            for pos, date_value in enumerate(history_dates)
            if pd.Timestamp(date_value).normalize() >= forecast_start and pd.Timestamp(date_value).normalize() <= forecast_end
        ]
        if not forecast_positions:
            return []
        schedule: list[dict[str, Any]] = []
        for sequence_index, prediction_pos in enumerate(forecast_positions):
            if sequence_index % max(1, int(step_size)) != 0:
                continue
            prediction_date = pd.Timestamp(history_dates[prediction_pos]).normalize()
            schedule.append(
                {
                    "prediction_date": prediction_date,
                    "feature_date": prediction_date,
                    "prediction_pos": int(prediction_pos),
                    "forecast_sequence_index": int(sequence_index),
                }
            )
        return schedule

    # Step sizes are now fully integrated at the schedule execution phase.

    @staticmethod
    def _actual_outcome(history: pd.DataFrame, prediction_pos: int, horizon_days: int) -> dict[str, Any]:
        dates = pd.Index(pd.to_datetime(history["date"], errors="coerce").dt.normalize())
        closes = pd.to_numeric(history["close"], errors="coerce").reset_index(drop=True)
        target_pos = int(prediction_pos + horizon_days)
        if target_pos >= len(history):
            return {
                "target_date": pd.NaT,
                "actual_return": np.nan,
                "actual_direction": np.nan,
                "evaluation_eligible": False,
            }
        current_close = float(closes.iloc[prediction_pos])
        target_close = float(closes.iloc[target_pos])
        actual_return = float((target_close / current_close) - 1.0) if current_close != 0.0 else np.nan
        return {
            "target_date": pd.Timestamp(dates[target_pos]).normalize(),
            "actual_return": actual_return,
            "actual_direction": int(actual_return > 0.0) if not np.isnan(actual_return) else np.nan,
            "evaluation_eligible": bool(not np.isnan(actual_return)),
        }

    @staticmethod
    def _safe_positive_probability(algorithm: str, prediction: dict[str, Any]) -> float:
        if algorithm not in SAFE_DIRECTION_PROBABILITY_ALGORITHMS:
            return float("nan")
        trend_probabilities = prediction.get("trend_probabilities") or {}
        return _safe_float(trend_probabilities.get("up"))

    def _collect_feature_importance_diagnostics(
        self,
        *,
        trainer: DualModelTrainer,
        ticker: str,
        horizon_name: str,
        trained_algorithms: list[str],
        fold_id: str,
        step_size: int,
        forecast_sequence_index: int,
        prediction_date: pd.Timestamp,
        eval_end: Any,
        train_history: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.config.enable_feature_importance_diagnostics:
            return empty_feature_importance_diagnostics_frame()

        manifest = trainer._manifests.get(ticker.upper(), {})
        horizon_info = manifest.get("horizons", {}).get(horizon_name, {})
        algorithm_info_by_name = horizon_info.get("algorithms", {})
        if not algorithm_info_by_name:
            return empty_feature_importance_diagnostics_frame()

        if pd.isna(eval_end):
            horizon_days = int(horizon_info.get("days", 1))
            eval_end = prediction_date + pd.Timedelta(days=horizon_days)
        fallback_train_start = str(pd.Timestamp(train_history["date"].min()).date()) if not train_history.empty else ""
        fallback_train_end = str(pd.Timestamp(train_history["date"].max()).date()) if not train_history.empty else ""

        frames: list[pd.DataFrame] = []
        for algorithm in trained_algorithms:
            normalized_algorithm = str(algorithm).lower()
            if normalized_algorithm not in SUPPORTED_IMPORTANCE_MODELS:
                continue
            algorithm_info = algorithm_info_by_name.get(normalized_algorithm, {})
            feature_columns_by_task = algorithm_info.get("feature_columns_by_task", {})
            fallback_feature_columns = algorithm_info.get("feature_columns", [])
            train_window = algorithm_info.get("evaluation_metadata", {}).get("train_window", {})
            fold_base_context = {
                "fold_id": fold_id,
                "step_size": int(step_size),
                "forecast_sequence_index": int(forecast_sequence_index),
                "ticker": ticker,
                "prediction_date": str(pd.Timestamp(prediction_date).date()),
                "horizon": horizon_name,
                "train_start": str(train_window.get("start") or fallback_train_start),
                "train_end": str(train_window.get("end") or fallback_train_end),
                "eval_start": str(pd.Timestamp(prediction_date).date()),
                "eval_end": str(pd.Timestamp(eval_end).date()),
            }
            for task in ("trend", "profit", "return"):
                feature_columns = feature_columns_by_task.get(task) or fallback_feature_columns
                if not feature_columns:
                    continue
                try:
                    model = trainer._get_loaded_model(ticker, normalized_algorithm, horizon_name, task)
                except Exception as exc:
                    logger.debug(
                        "feature_importance_model_unavailable",
                        ticker=ticker,
                        algorithm=normalized_algorithm,
                        horizon=horizon_name,
                        task=task,
                        error=str(exc),
                    )
                    continue
                frame = extract_feature_importance_rows(
                    model_name=normalized_algorithm,
                    model=model,
                    feature_columns=feature_columns,
                    fold_context={**fold_base_context, "task": task},
                )
                if not frame.empty:
                    frames.append(frame)

        if not frames:
            return empty_feature_importance_diagnostics_frame()
        return pd.concat(frames, ignore_index=True, sort=False).reindex(
            columns=empty_feature_importance_diagnostics_frame().columns
        )

    def _train_and_predict(
        self,
        *,
        ticker: str,
        history: pd.DataFrame,
        feature_date: pd.Timestamp,
        prediction_date: pd.Timestamp,
        horizons: dict[str, int],
        algorithms: list[str],
        context_sources: dict[str, pd.DataFrame | None],
        model_dir: Path,
        fold_id: str,
        step_size: int,
        forecast_sequence_index: int,
        eval_end_by_horizon: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trainer = DualModelTrainer(model_dir=model_dir)
        train_history = history[
            (history["date"] >= pd.Timestamp(self.config.initial_train_start).normalize())
            & (history["date"] <= feature_date)
        ].reset_index(drop=True)
        result = trainer.train(
            ticker=ticker,
            df=train_history,
            algorithms=algorithms,
            primary_algorithm=algorithms[0],
            horizons=list(horizons),
            horizon_days_map=horizons,
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
        )
        trained_pairs = {
            (str(row["horizon"]).lower(), str(row["algorithm"]).lower())
            for row in result["report_rows"]
        }
        feature_frame = trainer.compute_features_for_ticker(
            ticker,
            train_history,
            window_start=self.config.history_start,
            window_end=str(feature_date.date()),
            context_sources=context_sources,
        )
        labeled_for_diagnostics = trainer._add_targets(feature_frame, horizons) if self.config.enable_linear_coefficient_diagnostics else pd.DataFrame()
        diagnostic_feature_columns = (
            select_linear_diagnostic_features(feature_frame)
            if self.config.enable_linear_coefficient_diagnostics
            else []
        )
        rows: list[dict[str, Any]] = []
        diagnostic_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        coverage_frames: list[pd.DataFrame] = []
        for horizon_name, horizon_days in horizons.items():
            trained_algorithms = [
                algorithm
                for algorithm in algorithms
                if (str(horizon_name).lower(), str(algorithm).lower()) in trained_pairs
            ]
            eval_end = eval_end_by_horizon.get(horizon_name)
            resolved_eval_end = eval_end
            if pd.isna(resolved_eval_end):
                resolved_eval_end = prediction_date + pd.Timedelta(days=int(horizon_days))
            coverage_context = {
                "ticker": ticker,
                "fold_id": fold_id,
                "step_size": int(step_size),
                "forecast_sequence_index": int(forecast_sequence_index),
                "prediction_date": str(pd.Timestamp(prediction_date).date()),
                "horizon": horizon_name,
                "train_start": str(pd.Timestamp(feature_frame["date"].min()).date()) if not feature_frame.empty else "",
                "train_end": str(pd.Timestamp(feature_frame["date"].max()).date()) if not feature_frame.empty else "",
                "eval_start": str(pd.Timestamp(prediction_date).date()),
                "eval_end": str(pd.Timestamp(resolved_eval_end).date()),
            }
            coverage_frames.append(
                build_context_coverage_rows(
                    feature_frame=feature_frame,
                    fold_context=coverage_context,
                )
            )
            importance_frame = self._collect_feature_importance_diagnostics(
                trainer=trainer,
                ticker=ticker,
                horizon_name=horizon_name,
                trained_algorithms=trained_algorithms,
                fold_id=fold_id,
                step_size=int(step_size),
                forecast_sequence_index=int(forecast_sequence_index),
                prediction_date=prediction_date,
                eval_end=eval_end,
                train_history=train_history,
            )
            if not importance_frame.empty:
                importance_frames.append(importance_frame)
            for algorithm in trained_algorithms:
                prediction = trainer.predict(
                    ticker=ticker,
                    features=feature_frame,
                    horizon=horizon_name,
                    algorithm=algorithm,
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "prediction_date": prediction_date,
                        "feature_date": feature_date,
                        "horizon": horizon_name,
                        "horizon_days": int(horizon_days),
                        "model_name": algorithm,
                        "predicted_return": _safe_float(prediction.get("predicted_return")),
                        "predicted_direction": int(prediction.get("predicted_direction", 0)),
                        "predicted_positive_probability": self._safe_positive_probability(algorithm, prediction),
                        "probability_semantics": (
                            "trend_classifier_positive_probability"
                            if algorithm in SAFE_DIRECTION_PROBABILITY_ALGORITHMS
                            else "withheld_non_native_or_heuristic_probability"
                        ),
                    }
                )
            if (
                self.config.enable_linear_coefficient_diagnostics
                and trained_algorithms
                and diagnostic_feature_columns
            ):
                target_column = f"target_return_{horizon_name}"
                diagnostic_problem = trainer._build_horizon_problem(
                    labeled_for_diagnostics,
                    diagnostic_feature_columns,
                    horizon_name,
                    sequence_length=1,
                    horizon_days=int(horizon_days),
                )
                if diagnostic_problem is not None:
                    split = diagnostic_problem["split"]
                    labeled = labeled_for_diagnostics.dropna(
                        subset=[
                            f"target_direction_{horizon_name}",
                            target_column,
                            f"target_profit_label_{horizon_name}",
                        ]
                    ).reset_index(drop=True)
                    train_frame = labeled.iloc[: split.train_stop].copy()
                    if not train_frame.empty:
                        if pd.isna(eval_end):
                            eval_end = prediction_date + pd.Timedelta(days=int(horizon_days))
                        fold_context = {
                            "fold_id": fold_id,
                            "step_size": int(step_size),
                            "forecast_sequence_index": int(forecast_sequence_index),
                            "ticker": ticker,
                            "prediction_date": str(pd.Timestamp(prediction_date).date()),
                            "horizon": horizon_name,
                            "horizon_days": int(horizon_days),
                            "task": "return",
                            "train_start": str(pd.Timestamp(train_frame["date"].min()).date()),
                            "train_end": str(pd.Timestamp(train_frame["date"].max()).date()),
                            "eval_start": str(pd.Timestamp(prediction_date).date()),
                            "eval_end": str(pd.Timestamp(eval_end).date()),
                        }
                        diagnostic_frames.append(
                            fit_linear_fold_diagnostics(
                                train_frame=train_frame,
                                feature_columns=diagnostic_feature_columns,
                                target_column=target_column,
                                fold_context=fold_context,
                                model_names=self.config.linear_diagnostic_models,
                            )
                        )
        if diagnostic_frames:
            diagnostics = pd.concat(diagnostic_frames, ignore_index=True, sort=False)
            diagnostics = diagnostics[~diagnostics["feature"].isna()].reset_index(drop=True)
        else:
            diagnostics = empty_coefficient_diagnostics_frame()
        if importance_frames:
            importance_diagnostics = pd.concat(importance_frames, ignore_index=True, sort=False)
            importance_diagnostics = importance_diagnostics[~importance_diagnostics["feature"].isna()].reset_index(drop=True)
        else:
            importance_diagnostics = empty_feature_importance_diagnostics_frame()
        context_coverage = (
            pd.concat(coverage_frames, ignore_index=True, sort=False)
            if coverage_frames
            else empty_context_coverage_diagnostics_frame()
        )
        return rows, diagnostics, importance_diagnostics, context_coverage

    def _generate_daily_predictions_for_ticker(
        self,
        *,
        ticker: str,
        history: pd.DataFrame,
        context_sources: dict[str, pd.DataFrame | None],
        algorithms: list[str],
        horizons: dict[str, int],
        step_size: int,
        forecast_start: pd.Timestamp,
        forecast_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        schedule = self._prediction_schedule(
            history,
            step_size=step_size,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        if not schedule:
            return (
                pd.DataFrame(),
                empty_coefficient_diagnostics_frame(),
                empty_feature_importance_diagnostics_frame(),
                empty_context_coverage_diagnostics_frame(),
            )
        model_dir = self.models_dir / "daily_full_walk_forward" / ticker
        rows: list[dict[str, Any]] = []
        diagnostic_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        coverage_frames: list[pd.DataFrame] = []
        for item in schedule:
            actual_by_horizon = {
                horizon_name: self._actual_outcome(history, int(item["prediction_pos"]), horizon_days)
                for horizon_name, horizon_days in horizons.items()
            }
            eval_end_by_horizon = {
                horizon_name: actual["target_date"]
                for horizon_name, actual in actual_by_horizon.items()
            }
            fold_id = (
                f"step{int(step_size)}_{ticker}_"
                f"seq{int(item['forecast_sequence_index']):04d}_"
                f"{pd.Timestamp(item['prediction_date']).strftime('%Y%m%d')}"
            )
            prediction_rows, coefficient_rows, importance_rows, coverage_rows = self._train_and_predict(
                ticker=ticker,
                history=history,
                feature_date=item["feature_date"],
                prediction_date=item["prediction_date"],
                horizons=horizons,
                algorithms=algorithms,
                context_sources=context_sources,
                model_dir=model_dir,
                fold_id=fold_id,
                step_size=int(step_size),
                forecast_sequence_index=int(item["forecast_sequence_index"]),
                eval_end_by_horizon=eval_end_by_horizon,
            )
            if not coefficient_rows.empty:
                diagnostic_frames.append(coefficient_rows)
            if not importance_rows.empty:
                importance_frames.append(importance_rows)
            if not coverage_rows.empty:
                coverage_frames.append(coverage_rows)
            for row in prediction_rows:
                actual = actual_by_horizon[str(row["horizon"])]
                predicted_return = _safe_float(row["predicted_return"])
                actual_return = _safe_float(actual["actual_return"])
                eligible = bool(actual["evaluation_eligible"])
                row.update(
                    {
                        "forecast_sequence_index": int(item["forecast_sequence_index"]),
                        "target_date": actual["target_date"],
                        "actual_return": actual_return,
                        "actual_direction": actual["actual_direction"],
                        "actual_realized_forward_return": actual_return,
                        "actual_realized_direction": actual["actual_direction"],
                        "absolute_error": (
                            abs(predicted_return - actual_return)
                            if eligible and not np.isnan(predicted_return) and not np.isnan(actual_return)
                            else np.nan
                        ),
                        "squared_error": (
                            float((predicted_return - actual_return) ** 2)
                            if eligible and not np.isnan(predicted_return) and not np.isnan(actual_return)
                            else np.nan
                        ),
                        "direction_correct": (
                            int(int(row["predicted_direction"]) == int(actual["actual_direction"]))
                            if eligible and not pd.isna(actual["actual_direction"])
                            else np.nan
                        ),
                        "evaluation_eligible": eligible,
                    }
                )
                rows.append(row)
        if not rows:
            return pd.DataFrame(), (
                pd.concat(diagnostic_frames, ignore_index=True, sort=False)
                if diagnostic_frames
                else empty_coefficient_diagnostics_frame()
            ), (
                pd.concat(importance_frames, ignore_index=True, sort=False)
                if importance_frames
                else empty_feature_importance_diagnostics_frame()
            ), (
                pd.concat(coverage_frames, ignore_index=True, sort=False)
                if coverage_frames
                else empty_context_coverage_diagnostics_frame()
            )
        result = pd.DataFrame(rows)
        diagnostics = (
            pd.concat(diagnostic_frames, ignore_index=True, sort=False)
            if diagnostic_frames
            else empty_coefficient_diagnostics_frame()
        )
        importance_diagnostics = (
            pd.concat(importance_frames, ignore_index=True, sort=False)
            if importance_frames
            else empty_feature_importance_diagnostics_frame()
        )
        context_coverage = (
            pd.concat(coverage_frames, ignore_index=True, sort=False)
            if coverage_frames
            else empty_context_coverage_diagnostics_frame()
        )
        return _normalize_dates(result).sort_values(
            ["prediction_date", "ticker", "horizon", "model_name"]
        ).reset_index(drop=True), diagnostics.reset_index(drop=True), importance_diagnostics.reset_index(drop=True), context_coverage.reset_index(drop=True)

    def _generate_base_predictions(
        self,
        *,
        histories: dict[str, pd.DataFrame],
        context_sources: dict[str, pd.DataFrame | None],
        algorithms: list[str],
        step_size: int,
        horizons: dict[str, int],
        forecast_start: pd.Timestamp,
        forecast_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        max_workers = max(1, int(self.config.max_workers))
        ticker_frames: list[pd.DataFrame] = []
        coefficient_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        coverage_frames: list[pd.DataFrame] = []
        if max_workers == 1 or len(histories) <= 1:
            for ticker, history in histories.items():
                ticker_frame, ticker_coefficients, ticker_importance, ticker_coverage = self._generate_daily_predictions_for_ticker(
                    ticker=ticker,
                    history=history,
                    context_sources=context_sources,
                    algorithms=algorithms,
                    horizons=horizons,
                    step_size=step_size,
                    forecast_start=forecast_start,
                    forecast_end=forecast_end,
                )
                if not ticker_frame.empty:
                    ticker_frames.append(ticker_frame)
                if not ticker_coefficients.empty:
                    coefficient_frames.append(ticker_coefficients)
                if not ticker_importance.empty:
                    importance_frames.append(ticker_importance)
                if not ticker_coverage.empty:
                    coverage_frames.append(ticker_coverage)
        else:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(histories))) as executor:
                futures = {
                    executor.submit(
                        _generate_daily_predictions_worker,
                        {
                            "config": asdict(self.config),
                            "ticker": ticker,
                            "algorithms": list(algorithms),
                            "horizons": {str(name): int(days) for name, days in horizons.items()},
                            "step_size": int(step_size),
                            "history_start": str(pd.Timestamp(self.config.history_start).date()),
                            "history_end": str(pd.Timestamp(self.config.history_end).date()),
                            "forecast_start": str(pd.Timestamp(forecast_start).date()),
                            "forecast_end": str(pd.Timestamp(forecast_end).date()),
                        },
                    ): ticker
                    for ticker in histories
                }
                for future in as_completed(futures):
                    payload = future.result()
                    ticker_frame = payload["base_predictions"]
                    ticker_coefficients = payload.get(
                        "linear_coefficient_diagnostics",
                        empty_coefficient_diagnostics_frame(),
                    )
                    ticker_importance = payload.get(
                        "feature_importance_diagnostics",
                        empty_feature_importance_diagnostics_frame(),
                    )
                    ticker_coverage = payload.get(
                        "context_coverage_diagnostics",
                        empty_context_coverage_diagnostics_frame(),
                    )
                    if not ticker_frame.empty:
                        ticker_frames.append(ticker_frame)
                    if not ticker_coefficients.empty:
                        coefficient_frames.append(ticker_coefficients)
                    if not ticker_importance.empty:
                        importance_frames.append(ticker_importance)
                    if not ticker_coverage.empty:
                        coverage_frames.append(ticker_coverage)
        if not ticker_frames:
            raise ValueError(f"Walk-forward run produced no base-model predictions for step_size={step_size}")
        result = pd.concat(ticker_frames, ignore_index=True, sort=False)
        result["step_size"] = int(step_size)
        coefficients = (
            pd.concat(coefficient_frames, ignore_index=True, sort=False)
            if coefficient_frames
            else empty_coefficient_diagnostics_frame()
        )
        importance_diagnostics = (
            pd.concat(importance_frames, ignore_index=True, sort=False)
            if importance_frames
            else empty_feature_importance_diagnostics_frame()
        )
        context_coverage = (
            pd.concat(coverage_frames, ignore_index=True, sort=False)
            if coverage_frames
            else empty_context_coverage_diagnostics_frame()
        )
        return (
            result.sort_values(["prediction_date", "ticker", "horizon", "model_name"]).reset_index(drop=True),
            coefficients.reset_index(drop=True),
            importance_diagnostics.reset_index(drop=True),
            context_coverage.reset_index(drop=True),
        )

    def _stacking_features(
        self,
        frame: pd.DataFrame,
        *,
        model_columns: list[str],
        fill_values: pd.Series | None = None,
    ) -> pd.DataFrame:
        prepared = frame.reindex(columns=model_columns).copy()
        if fill_values is None:
            fill_values = prepared.median(numeric_only=True)
        for column in model_columns:
            fallback = float(fill_values.get(column, 0.0)) if fill_values is not None else 0.0
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(fallback)
        return prepared

    def _build_stacking_predictions(self, base_df: pd.DataFrame) -> pd.DataFrame:
        if base_df.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        working = _normalize_dates(base_df)
        for (step_size, horizon_name), group in working.groupby(["step_size", "horizon"], sort=True):
            model_columns = sorted(group["model_name"].dropna().astype(str).unique())
            wide = (
                group.pivot_table(
                    index=["step_size", "prediction_date", "feature_date", "ticker", "horizon", "horizon_days"],
                    columns="model_name",
                    values="predicted_return",
                    aggfunc="first",
                )
                .reset_index()
                .sort_values(["prediction_date", "ticker"])
                .reset_index(drop=True)
            )
            actual_lookup = (
                group[
                    [
                        "step_size",
                        "prediction_date",
                        "feature_date",
                        "target_date",
                        "ticker",
                        "horizon",
                        "horizon_days",
                        "actual_return",
                        "actual_direction",
                        "evaluation_eligible",
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            wide = wide.merge(
                actual_lookup,
                on=["step_size", "prediction_date", "feature_date", "ticker", "horizon", "horizon_days"],
                how="left",
            )
            for prediction_date, current_rows in wide.groupby("prediction_date", sort=True):
                history_rows = wide[
                    (wide["prediction_date"] < prediction_date)
                    & (wide["target_date"].notna())
                    & (wide["target_date"] < prediction_date)
                    & (wide["evaluation_eligible"] == True)
                ].copy()
                fill_values = history_rows[model_columns].median(numeric_only=True)
                current_features = self._stacking_features(current_rows[model_columns], model_columns=model_columns, fill_values=fill_values)
                method_name = "mean_fallback"
                predicted_values = current_features.mean(axis=1).to_numpy(dtype=float)
                if len(history_rows) >= max(int(self.config.meta_min_samples), len(model_columns) + 1):
                    history_features = self._stacking_features(
                        history_rows[model_columns],
                        model_columns=model_columns,
                        fill_values=fill_values,
                    )
                    meta_model = Ridge(alpha=float(self.config.meta_model_alpha))
                    meta_model.fit(history_features.to_numpy(dtype=float), history_rows["actual_return"].to_numpy(dtype=float))
                    predicted_values = meta_model.predict(current_features.to_numpy(dtype=float))
                    method_name = "prequential_ridge_on_oos_base_predictions"
                for row, predicted_return in zip(current_rows.itertuples(index=False), predicted_values):
                    actual_return = _safe_float(row.actual_return)
                    eligible = bool(row.evaluation_eligible)
                    predicted_return = _safe_float(predicted_return)
                    rows.append(
                        {
                            "step_size": int(step_size),
                            "ticker": str(row.ticker),
                            "prediction_date": pd.Timestamp(row.prediction_date).normalize(),
                            "feature_date": pd.Timestamp(row.feature_date).normalize(),
                            "target_date": pd.Timestamp(row.target_date).normalize() if pd.notna(row.target_date) else pd.NaT,
                            "horizon": horizon_name,
                            "horizon_days": int(row.horizon_days),
                            "model_name": FINAL_STACKING_MODEL_NAME,
                            "stacking_method": method_name,
                            "training_meta_rows": int(len(history_rows)),
                            "final_predicted_return": predicted_return,
                            "final_predicted_direction": int(predicted_return > 0.0) if not np.isnan(predicted_return) else np.nan,
                            "final_positive_probability": np.nan,
                            "probability_semantics": "not_reported_for_regression_stack",
                            "actual_return": actual_return,
                            "actual_direction": row.actual_direction,
                            "actual_realized_forward_return": actual_return,
                            "actual_realized_direction": row.actual_direction,
                            "absolute_error": (
                                abs(predicted_return - actual_return)
                                if eligible and not np.isnan(predicted_return) and not np.isnan(actual_return)
                                else np.nan
                            ),
                            "squared_error": (
                                float((predicted_return - actual_return) ** 2)
                                if eligible and not np.isnan(predicted_return) and not np.isnan(actual_return)
                                else np.nan
                            ),
                            "direction_correct": (
                                int(int(predicted_return > 0.0) == int(row.actual_direction))
                                if eligible and not pd.isna(row.actual_direction)
                                else np.nan
                            ),
                            "evaluation_eligible": eligible,
                        }
                    )
        result = pd.DataFrame(rows)
        if result.empty:
            raise ValueError("Stacking layer produced no forecasts")
        return _normalize_dates(result).sort_values(
            ["step_size", "prediction_date", "ticker", "horizon"]
        ).reset_index(drop=True)

    @staticmethod
    def _evaluation_frame(
        frame: pd.DataFrame,
        *,
        predicted_return_col: str,
        predicted_direction_col: str,
        probability_col: str | None = None,
    ) -> pd.DataFrame:
        working = frame.copy()
        working = working[working["evaluation_eligible"] == True].copy()
        working[predicted_return_col] = pd.to_numeric(working[predicted_return_col], errors="coerce")
        working["actual_return"] = pd.to_numeric(working["actual_return"], errors="coerce")
        working[predicted_direction_col] = pd.to_numeric(working[predicted_direction_col], errors="coerce")
        working["actual_direction"] = pd.to_numeric(working["actual_direction"], errors="coerce")
        if probability_col is not None and probability_col in working.columns:
            working[probability_col] = pd.to_numeric(working[probability_col], errors="coerce")
        return working.dropna(subset=[predicted_return_col, "actual_return", predicted_direction_col, "actual_direction"]).copy()

    def _metric_row(
        self,
        frame: pd.DataFrame,
        *,
        predicted_return_col: str,
        predicted_direction_col: str,
        probability_col: str | None = None,
    ) -> dict[str, Any]:
        eval_frame = self._evaluation_frame(
            frame,
            predicted_return_col=predicted_return_col,
            predicted_direction_col=predicted_direction_col,
            probability_col=probability_col,
        )
        if eval_frame.empty:
            return {
                "observations": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "smape": np.nan,
                "correlation": np.nan,
                "directional_accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "brier_score": np.nan,
            }
        actual = eval_frame["actual_return"].astype(float)
        predicted = eval_frame[predicted_return_col].astype(float)
        actual_direction = eval_frame["actual_direction"].astype(int)
        predicted_direction = eval_frame[predicted_direction_col].astype(int)
        mae = float((predicted - actual).abs().mean())
        rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
        denominator = actual.abs()
        valid_mape = denominator > 0.0
        mape = float((((predicted - actual).abs() / denominator)[valid_mape]).mean() * 100.0) if valid_mape.any() else np.nan
        precision = float(precision_score(actual_direction, predicted_direction, zero_division=0))
        recall = float(recall_score(actual_direction, predicted_direction, zero_division=0))
        f1 = float(f1_score(actual_direction, predicted_direction, zero_division=0))
        brier = np.nan
        if probability_col is not None and probability_col in eval_frame.columns and eval_frame[probability_col].notna().any():
            brier = compute_brier_score(actual_direction, eval_frame[probability_col].clip(lower=0.0, upper=1.0))
        return {
            "observations": int(len(eval_frame)),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "smape": _smape(actual, predicted),
            "correlation": _correlation(actual, predicted),
            "directional_accuracy": float((actual_direction == predicted_direction).mean()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier_score": brier,
        }

    def _build_summary_tables(
        self,
        base_df: pd.DataFrame,
        stack_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_rows: list[dict[str, Any]] = []
        merged = pd.concat(
            [
                base_df,
                stack_df.rename(
                    columns={
                        "final_predicted_return": "predicted_return",
                        "final_predicted_direction": "predicted_direction",
                        "final_positive_probability": "predicted_positive_probability",
                    }
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        for (step_size, horizon, model_name), group in merged.groupby(["step_size", "horizon", "model_name"], sort=True):
            summary_rows.append(
                {
                    "step_size": int(step_size),
                    "horizon": str(horizon),
                    "model_name": str(model_name),
                    "ticker": "OVERALL",
                    **self._metric_row(
                        group,
                        predicted_return_col="predicted_return",
                        predicted_direction_col="predicted_direction",
                        probability_col="predicted_positive_probability",
                    ),
                }
            )
        summary_by_horizon = pd.DataFrame(summary_rows).sort_values(["step_size", "horizon", "model_name"]).reset_index(drop=True)

        ticker_rows: list[dict[str, Any]] = []
        for (step_size, ticker, horizon, model_name), group in merged.groupby(
            ["step_size", "ticker", "horizon", "model_name"], sort=True
        ):
            ticker_rows.append(
                {
                    "step_size": int(step_size),
                    "ticker": str(ticker),
                    "horizon": str(horizon),
                    "model_name": str(model_name),
                    **self._metric_row(
                        group,
                        predicted_return_col="predicted_return",
                        predicted_direction_col="predicted_direction",
                        probability_col="predicted_positive_probability",
                    ),
                }
            )
        summary_by_ticker = pd.DataFrame(ticker_rows).sort_values(
            ["step_size", "ticker", "horizon", "model_name"]
        ).reset_index(drop=True)

        model_rows: list[dict[str, Any]] = []
        for (step_size, model_name), group in merged.groupby(["step_size", "model_name"], sort=True):
            model_rows.append(
                {
                    "step_size": int(step_size),
                    "model_name": str(model_name),
                    **self._metric_row(
                        group,
                        predicted_return_col="predicted_return",
                        predicted_direction_col="predicted_direction",
                        probability_col="predicted_positive_probability",
                    ),
                }
        )
        summary_by_model = pd.DataFrame(model_rows).sort_values(["step_size", "model_name"]).reset_index(drop=True)
        return summary_by_horizon, summary_by_ticker, summary_by_model

    def _build_actual_comparison_summary(
        self,
        base_df: pd.DataFrame,
        stack_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        merged = pd.concat(
            [
                base_df[
                    ["step_size", "ticker", "horizon", "model_name", "prediction_date", "target_date", "evaluation_eligible"]
                ],
                stack_df[
                    ["step_size", "ticker", "horizon", "model_name", "prediction_date", "target_date", "evaluation_eligible"]
                ],
            ],
            ignore_index=True,
        )
        comparison_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        for (step_size, ticker, horizon, model_name), group in merged.groupby(
            ["step_size", "ticker", "horizon", "model_name"], sort=True
        ):
            eligible = group[group["evaluation_eligible"] == True]
            coverage_rows.append(
                {
                    "step_size": int(step_size),
                    "ticker": str(ticker),
                    "horizon": str(horizon),
                    "model_name": str(model_name),
                    "total_predictions": int(len(group)),
                    "evaluation_eligible_predictions": int(len(eligible)),
                    "evaluation_ineligible_predictions": int(len(group) - len(eligible)),
                    "coverage_ratio": float(len(eligible) / len(group)) if len(group) else np.nan,
                    "first_prediction_date": str(pd.to_datetime(group["prediction_date"]).min().date()),
                    "last_prediction_date": str(pd.to_datetime(group["prediction_date"]).max().date()),
                    "last_eligible_prediction_date": (
                        str(pd.to_datetime(eligible["prediction_date"]).max().date()) if not eligible.empty else ""
                    ),
                }
            )
        for (step_size, horizon), group in merged.groupby(["step_size", "horizon"], sort=True):
            comparison_rows.append(
                {
                    "step_size": int(step_size),
                    "horizon": str(horizon),
                    "total_predictions": int(len(group)),
                    "evaluation_eligible_predictions": int((group["evaluation_eligible"] == True).sum()),
                    "evaluation_ineligible_predictions": int((group["evaluation_eligible"] != True).sum()),
                    "coverage_ratio": float((group["evaluation_eligible"] == True).mean()),
                }
            )
        return (
            pd.DataFrame(comparison_rows).sort_values(["step_size", "horizon"]).reset_index(drop=True),
            pd.DataFrame(coverage_rows).sort_values(["step_size", "ticker", "horizon", "model_name"]).reset_index(drop=True),
        )

    def _backtest_metrics(
        self,
        frame: pd.DataFrame,
        *,
        predicted_direction_col: str,
        actual_return_col: str,
        step_size: int,
    ) -> dict[str, Any]:
        eval_frame = frame[frame["evaluation_eligible"] == True].copy()
        if eval_frame.empty:
            return {
                "total_return": np.nan,
                "cagr": np.nan,
                "sharpe": np.nan,
                "sortino": np.nan,
                "max_drawdown": np.nan,
                "turnover": np.nan,
                "win_rate": np.nan,
                "observations": 0,
            }
        eval_frame = eval_frame.sort_values("prediction_date").reset_index(drop=True)
        positions = pd.to_numeric(eval_frame[predicted_direction_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        realized = pd.to_numeric(eval_frame[actual_return_col], errors="coerce").fillna(0.0)
        strategy_returns = positions * realized
        equity = (1.0 + strategy_returns).cumprod()
        total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
        annualization = TRADING_DAYS_PER_YEAR / max(int(step_size), 1)
        start_date = pd.to_datetime(eval_frame["prediction_date"], errors="coerce").min()
        end_date = pd.to_datetime(eval_frame["target_date"], errors="coerce").max()
        return {
            "total_return": total_return,
            "cagr": _cagr(total_return, start_date, end_date),
            "sharpe": compute_sharpe_ratio(strategy_returns, annualization_factor=annualization),
            "sortino": compute_sortino_ratio(strategy_returns, annualization_factor=annualization),
            "max_drawdown": compute_max_drawdown(equity),
            "turnover": compute_signal_turnover(positions),
            "win_rate": compute_win_rate(strategy_returns, ignore_zero_returns=True),
            "observations": int(len(eval_frame)),
        }

    def _build_backtest_tables(
        self,
        base_df: pd.DataFrame,
        stack_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.concat(
            [
                base_df[
                    [
                        "step_size",
                        "ticker",
                        "horizon",
                        "model_name",
                        "prediction_date",
                        "target_date",
                        "actual_return",
                        "predicted_direction",
                        "evaluation_eligible",
                    ]
                ].rename(columns={"predicted_direction": "signal_direction"}),
                stack_df[
                    [
                        "step_size",
                        "ticker",
                        "horizon",
                        "model_name",
                        "prediction_date",
                        "target_date",
                        "actual_return",
                        "final_predicted_direction",
                        "evaluation_eligible",
                    ]
                ].rename(columns={"final_predicted_direction": "signal_direction"}),
            ],
            ignore_index=True,
        )
        backtest_rows: list[dict[str, Any]] = []
        buy_hold_rows: list[dict[str, Any]] = []
        for (step_size, ticker, horizon, model_name), group in combined.groupby(
            ["step_size", "ticker", "horizon", "model_name"], sort=True
        ):
            metrics = self._backtest_metrics(
                group,
                predicted_direction_col="signal_direction",
                actual_return_col="actual_return",
                step_size=int(step_size),
            )
            backtest_rows.append(
                {
                    "step_size": int(step_size),
                    "ticker": str(ticker),
                    "horizon": str(horizon),
                    "model_name": str(model_name),
                    **metrics,
                }
            )
        for (step_size, ticker, horizon), group in combined.groupby(["step_size", "ticker", "horizon"], sort=True):
            buy_hold_group = group.copy()
            buy_hold_group["signal_direction"] = 1
            metrics = self._backtest_metrics(
                buy_hold_group,
                predicted_direction_col="signal_direction",
                actual_return_col="actual_return",
                step_size=int(step_size),
            )
            buy_hold_rows.append(
                {
                    "step_size": int(step_size),
                    "ticker": str(ticker),
                    "horizon": str(horizon),
                    "model_name": BUY_AND_HOLD_MODEL_NAME,
                    **metrics,
                }
            )
        backtest_summary = pd.DataFrame(backtest_rows).sort_values(
            ["step_size", "ticker", "horizon", "model_name"]
        ).reset_index(drop=True)
        buy_hold = pd.DataFrame(buy_hold_rows).sort_values(["step_size", "ticker", "horizon"]).reset_index(drop=True)
        comparison = backtest_summary.merge(
            buy_hold,
            on=["step_size", "ticker", "horizon"],
            suffixes=("", "_buy_and_hold"),
            how="left",
        )
        comparison["beats_buy_and_hold_total_return"] = comparison["total_return"] > comparison["total_return_buy_and_hold"]
        return backtest_summary, comparison

    @staticmethod
    def _build_stacking_vs_models(
        summary_by_ticker: pd.DataFrame,
        summary_by_horizon: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        stack_ticker = summary_by_ticker[summary_by_ticker["model_name"] == FINAL_STACKING_MODEL_NAME].copy()
        base_ticker = summary_by_ticker[summary_by_ticker["model_name"] != FINAL_STACKING_MODEL_NAME].copy()
        for stack_row in stack_ticker.itertuples(index=False):
            competitors = base_ticker[
                (base_ticker["step_size"] == stack_row.step_size)
                & (base_ticker["ticker"] == stack_row.ticker)
                & (base_ticker["horizon"] == stack_row.horizon)
            ]
            for competitor in competitors.itertuples(index=False):
                rows.append(
                    {
                        "scope": "ticker",
                        "step_size": int(stack_row.step_size),
                        "ticker": str(stack_row.ticker),
                        "horizon": str(stack_row.horizon),
                        "competitor_model": str(competitor.model_name),
                        "stacking_mae_minus_model": float(stack_row.mae - competitor.mae),
                        "stacking_rmse_minus_model": float(stack_row.rmse - competitor.rmse),
                        "stacking_directional_accuracy_minus_model": float(
                            stack_row.directional_accuracy - competitor.directional_accuracy
                        ),
                        "stacking_better_mae": bool(stack_row.mae < competitor.mae),
                        "stacking_better_rmse": bool(stack_row.rmse < competitor.rmse),
                        "stacking_better_directional_accuracy": bool(
                            stack_row.directional_accuracy > competitor.directional_accuracy
                        ),
                    }
                )
        stack_horizon = summary_by_horizon[summary_by_horizon["model_name"] == FINAL_STACKING_MODEL_NAME].copy()
        base_horizon = summary_by_horizon[summary_by_horizon["model_name"] != FINAL_STACKING_MODEL_NAME].copy()
        for stack_row in stack_horizon.itertuples(index=False):
            competitors = base_horizon[
                (base_horizon["step_size"] == stack_row.step_size)
                & (base_horizon["horizon"] == stack_row.horizon)
                & (base_horizon["ticker"] == "OVERALL")
            ]
            for competitor in competitors.itertuples(index=False):
                rows.append(
                    {
                        "scope": "overall_horizon",
                        "step_size": int(stack_row.step_size),
                        "ticker": "OVERALL",
                        "horizon": str(stack_row.horizon),
                        "competitor_model": str(competitor.model_name),
                        "stacking_mae_minus_model": float(stack_row.mae - competitor.mae),
                        "stacking_rmse_minus_model": float(stack_row.rmse - competitor.rmse),
                        "stacking_directional_accuracy_minus_model": float(
                            stack_row.directional_accuracy - competitor.directional_accuracy
                        ),
                        "stacking_better_mae": bool(stack_row.mae < competitor.mae),
                        "stacking_better_rmse": bool(stack_row.rmse < competitor.rmse),
                        "stacking_better_directional_accuracy": bool(
                            stack_row.directional_accuracy > competitor.directional_accuracy
                        ),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "scope",
                    "step_size",
                    "ticker",
                    "horizon",
                    "competitor_model",
                    "stacking_mae_minus_model",
                    "stacking_rmse_minus_model",
                    "stacking_directional_accuracy_minus_model",
                    "stacking_better_mae",
                    "stacking_better_rmse",
                    "stacking_better_directional_accuracy",
                ]
            )
        return pd.DataFrame(rows).sort_values(["scope", "step_size", "ticker", "horizon", "competitor_model"]).reset_index(drop=True)

    def _write_csv_outputs(
        self,
        *,
        base_df: pd.DataFrame,
        stack_df: pd.DataFrame,
        actual_comparison_summary: pd.DataFrame,
        summary_by_horizon: pd.DataFrame,
        summary_by_ticker: pd.DataFrame,
        summary_by_model: pd.DataFrame,
        stacking_vs_all_models: pd.DataFrame,
        backtest_summary: pd.DataFrame,
        buy_and_hold_comparison: pd.DataFrame,
        forecast_coverage_summary: pd.DataFrame,
        linear_coefficient_diagnostics: pd.DataFrame,
        linear_coefficient_stability_summary: pd.DataFrame,
        feature_importance_diagnostics: pd.DataFrame,
        feature_importance_stability_summary: pd.DataFrame,
        linear_vs_importance_feature_comparison: pd.DataFrame,
        feature_governance_review: pd.DataFrame,
        context_coverage_diagnostics: pd.DataFrame,
        context_coverage_summary: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "predictions_detailed": self.csv_dir / "predictions_detailed.csv",
            "stacking_predictions_detailed": self.csv_dir / "stacking_predictions_detailed.csv",
            "actual_comparison_summary": self.csv_dir / "actual_comparison_summary.csv",
            "summary_by_horizon": self.csv_dir / "summary_by_horizon.csv",
            "summary_by_ticker": self.csv_dir / "summary_by_ticker.csv",
            "summary_by_model": self.csv_dir / "summary_by_model.csv",
            "stacking_vs_all_models": self.csv_dir / "stacking_vs_all_models.csv",
            "backtest_summary": self.csv_dir / "backtest_summary.csv",
            "buy_and_hold_comparison": self.csv_dir / "buy_and_hold_comparison.csv",
            "forecast_coverage_summary": self.csv_dir / "forecast_coverage_summary.csv",
            "linear_coefficient_diagnostics": self.csv_dir / "linear_coefficient_diagnostics.csv",
            "linear_coefficient_stability_summary": self.csv_dir / "linear_coefficient_stability_summary.csv",
            "feature_importance_diagnostics": self.csv_dir / "feature_importance_diagnostics.csv",
            "feature_importance_stability_summary": self.csv_dir / "feature_importance_stability_summary.csv",
            "linear_vs_importance_feature_comparison": self.csv_dir / "linear_vs_importance_feature_comparison.csv",
            "feature_governance_review": self.csv_dir / "feature_governance_review.csv",
            "context_coverage_diagnostics": self.csv_dir / "context_coverage_diagnostics.csv",
            "context_coverage_summary": self.csv_dir / "context_coverage_summary.csv",
            "run_metadata": self.csv_dir / "run_metadata.json",
        }
        base_df.to_csv(outputs["predictions_detailed"], index=False)
        stack_df.to_csv(outputs["stacking_predictions_detailed"], index=False)
        actual_comparison_summary.to_csv(outputs["actual_comparison_summary"], index=False)
        summary_by_horizon.to_csv(outputs["summary_by_horizon"], index=False)
        summary_by_ticker.to_csv(outputs["summary_by_ticker"], index=False)
        summary_by_model.to_csv(outputs["summary_by_model"], index=False)
        stacking_vs_all_models.to_csv(outputs["stacking_vs_all_models"], index=False)
        backtest_summary.to_csv(outputs["backtest_summary"], index=False)
        buy_and_hold_comparison.to_csv(outputs["buy_and_hold_comparison"], index=False)
        forecast_coverage_summary.to_csv(outputs["forecast_coverage_summary"], index=False)
        linear_coefficient_diagnostics.to_csv(outputs["linear_coefficient_diagnostics"], index=False)
        linear_coefficient_stability_summary.to_csv(outputs["linear_coefficient_stability_summary"], index=False)
        feature_importance_diagnostics.to_csv(outputs["feature_importance_diagnostics"], index=False)
        feature_importance_stability_summary.to_csv(outputs["feature_importance_stability_summary"], index=False)
        linear_vs_importance_feature_comparison.to_csv(outputs["linear_vs_importance_feature_comparison"], index=False)
        feature_governance_review.to_csv(outputs["feature_governance_review"], index=False)
        context_coverage_diagnostics.to_csv(outputs["context_coverage_diagnostics"], index=False)
        context_coverage_summary.to_csv(outputs["context_coverage_summary"], index=False)
        outputs["run_metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {name: str(path) for name, path in outputs.items()}

    def _render_charts(
        self,
        *,
        base_df: pd.DataFrame,
        stack_df: pd.DataFrame,
        summary_by_horizon: pd.DataFrame,
        summary_by_ticker: pd.DataFrame,
        buy_and_hold_comparison: pd.DataFrame,
        forecast_coverage_summary: pd.DataFrame,
    ) -> dict[str, str]:
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: dict[str, str] = {}

        combined = pd.concat(
            [
                base_df.assign(output_kind="base").rename(
                    columns={
                        "predicted_return": "plot_predicted_return",
                        "predicted_direction": "plot_predicted_direction",
                    }
                ),
                stack_df.assign(output_kind="stacking").rename(
                    columns={
                        "final_predicted_return": "plot_predicted_return",
                        "final_predicted_direction": "plot_predicted_direction",
                    }
                ),
            ],
            ignore_index=True,
            sort=False,
        )

        for (step_size, horizon, ticker, model_name), group in combined.groupby(["step_size", "horizon", "ticker", "model_name"], sort=True):
            eligible = group[group["evaluation_eligible"] == True].copy()
            if eligible.empty:
                continue
            
            eligible = eligible.sort_values("target_date").dropna(subset=["target_date"]).reset_index(drop=True)
            if eligible.empty:
                continue
                
            realizations = eligible[["target_date", "actual_return"]].copy().dropna()
            realizations = realizations.sort_values("target_date").rename(
                columns={"target_date": "target_date_match", "actual_return": "momentum_prediction"}
            )
            
            eligible_sorted = eligible.sort_values("prediction_date")
            eval_df = pd.merge_asof(
                eligible_sorted, 
                realizations.rename(columns={"target_date_match": "prediction_date"}), 
                on="prediction_date", 
                direction="backward"
            )
            eval_df["momentum_prediction"] = eval_df["momentum_prediction"].fillna(0.0)
            eval_df["naive_prediction"] = 0.0
            
            eval_df["model_error"] = (eval_df["plot_predicted_return"] - eval_df["actual_return"]).abs()
            eval_df["momentum_error"] = (eval_df["momentum_prediction"] - eval_df["actual_return"]).abs()
            eval_df["naive_error"] = (eval_df["naive_prediction"] - eval_df["actual_return"]).abs()
            
            eval_df = eval_df.sort_values("target_date")
            x_dates = eval_df["target_date"]
            
            horizon_dir = self.charts_dir / str(horizon)
            actual_dir = horizon_dir / "actual_vs_predicted"
            error_dir = horizon_dir / "error_trend"
            actual_dir.mkdir(parents=True, exist_ok=True)
            error_dir.mkdir(parents=True, exist_ok=True)
            
            prefix = f"step{step_size}_{ticker}_{model_name}"
            display_model = str(model_name).upper()
            if display_model == FINAL_STACKING_MODEL_NAME.upper():
                display_model = "STACKING"
            
            avp_path = actual_dir / f"{prefix}_actual_vs_predicted.png"
            plt.figure(figsize=(12, 6))
            plt.plot(x_dates, eval_df["actual_return"], label="Actual Forward Return", color="black", linewidth=2.0)
            plt.plot(x_dates, eval_df["plot_predicted_return"], label=f"{display_model} Predicted Return", color="#2563eb", linewidth=1.5)
            plt.axhline(0.0, label="Naive Flat Baseline (0)", color="gray", linestyle="--", linewidth=1.5)
            plt.plot(x_dates, eval_df["momentum_prediction"], label="Momentum Baseline (Prev)", color="#d97706", linestyle=":", linewidth=1.5)
            plt.title(f"{ticker} Actual vs Predicted Forward Return ({display_model}) | {horizon}")
            plt.xlabel("Target Date")
            plt.ylabel("Forward Return")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(avp_path, dpi=140)
            plt.close()
            chart_paths[f"{prefix}_{horizon}_actual_vs_predicted"] = str(avp_path)
            
            error_path = error_dir / f"{prefix}_error_trend.png"
            plt.figure(figsize=(12, 6))
            plt.plot(x_dates, eval_df["model_error"], label=f"{display_model} Absolute Error", color="#dc2626", linewidth=1.5)
            plt.plot(x_dates, eval_df["naive_error"], label="Naive Absolute Error", color="gray", linestyle="--", linewidth=1.5)
            plt.plot(x_dates, eval_df["momentum_error"], label="Momentum Absolute Error", color="#d97706", linestyle=":", linewidth=1.5)
            plt.title(f"{ticker} Forward-Return Error Trend ({display_model}) | {horizon}")
            plt.xlabel("Target Date")
            plt.ylabel("Absolute Error")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(error_path, dpi=140)
            plt.close()
            chart_paths[f"{prefix}_{horizon}_error_trend"] = str(error_path)

        for (step_size, horizon), group in summary_by_horizon.groupby(["step_size", "horizon"], sort=True):
            plot_frame = group[group["ticker"] == "OVERALL"].copy()
            if plot_frame.empty:
                continue
            plot_frame = plot_frame.sort_values(["rmse", "model_name"]).reset_index(drop=True)
            comparison_path = self.charts_dir / f"step{step_size}_{horizon}_model_comparison.png"
            x = np.arange(len(plot_frame))
            width = 0.25
            plt.figure(figsize=(11, 5.5))
            plt.bar(x - width, plot_frame["mae"], width=width, label="MAE")
            plt.bar(x, plot_frame["rmse"], width=width, label="RMSE")
            plt.bar(x + width, plot_frame["directional_accuracy"], width=width, label="Directional Accuracy")
            plt.xticks(x, plot_frame["model_name"], rotation=25)
            plt.title(f"Model Comparison by Horizon: {horizon} step={step_size}")
            plt.grid(axis="y", alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=140)
            plt.close()
            chart_paths[f"model_comparison_step{step_size}_{horizon}"] = str(comparison_path)

            stack_compare_path = self.charts_dir / f"step{step_size}_{horizon}_stacking_vs_all.png"
            compare_frame = plot_frame.set_index("model_name")
            if FINAL_STACKING_MODEL_NAME in compare_frame.index:
                stacking_row = compare_frame.loc[FINAL_STACKING_MODEL_NAME]
                others = compare_frame.drop(index=FINAL_STACKING_MODEL_NAME, errors="ignore").reset_index()
                if not others.empty:
                    plt.figure(figsize=(11, 5))
                    delta_rmse = stacking_row["rmse"] - others["rmse"]
                    plt.bar(others["model_name"], delta_rmse, color=["#16a34a" if value < 0 else "#dc2626" for value in delta_rmse])
                    plt.axhline(0.0, color="black", linewidth=1.0)
                    plt.title(f"Stacking vs Individual Models (RMSE delta): {horizon} step={step_size}")
                    plt.ylabel("Stacking RMSE - Base Model RMSE")
                    plt.xticks(rotation=25)
                    plt.grid(axis="y", alpha=0.25)
                    plt.tight_layout()
                    plt.savefig(stack_compare_path, dpi=140)
                    plt.close()
                    chart_paths[f"stacking_vs_all_step{step_size}_{horizon}"] = str(stack_compare_path)

        for (step_size, horizon), group in combined.groupby(["step_size", "horizon"], sort=True):
            eligible = group[group["evaluation_eligible"] == True].copy()
            if eligible.empty:
                continue
            error_path = self.charts_dir / f"step{step_size}_{horizon}_error_distribution.png"
            plt.figure(figsize=(11, 5))
            plot_frame = eligible[["model_name", "absolute_error"]].dropna().copy()
            ordered_models = sorted(plot_frame["model_name"].unique())
            data = [plot_frame.loc[plot_frame["model_name"] == model_name, "absolute_error"].to_numpy(dtype=float) for model_name in ordered_models]
            if data:
                plt.boxplot(data, tick_labels=ordered_models, showfliers=False)
                plt.title(f"Absolute Error Distribution: {horizon} step={step_size}")
                plt.ylabel("Absolute Error")
                plt.xticks(rotation=25)
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                plt.savefig(error_path, dpi=140)
                plt.close()
                chart_paths[f"error_distribution_step{step_size}_{horizon}"] = str(error_path)

        for (step_size, ticker, horizon), group in buy_and_hold_comparison.groupby(["step_size", "ticker", "horizon"], sort=True):
            stack_row = group[group["model_name"] == FINAL_STACKING_MODEL_NAME]
            if stack_row.empty:
                continue
            curve_path = self.charts_dir / f"step{step_size}_{ticker}_{horizon}_cumulative_vs_buy_hold.png"
            model_total = float(stack_row.iloc[0]["total_return"])
            buy_hold_total = float(stack_row.iloc[0]["total_return_buy_and_hold"])
            plt.figure(figsize=(8.5, 4.8))
            plt.bar(["Stacking", "Buy and Hold"], [model_total, buy_hold_total], color=["#2563eb", "#6b7280"])
            plt.title(f"Cumulative Performance vs Buy and Hold: {ticker} {horizon} step={step_size}")
            plt.ylabel("Total Return")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(curve_path, dpi=140)
            plt.close()
            chart_paths[f"cumulative_vs_buy_hold_step{step_size}_{ticker}_{horizon}"] = str(curve_path)

        if not forecast_coverage_summary.empty:
            coverage_plot = (
                forecast_coverage_summary.groupby(["step_size", "horizon"], as_index=False)["coverage_ratio"].mean().sort_values(["step_size", "horizon"])
            )
            coverage_path = self.charts_dir / "forecast_coverage_summary.png"
            plt.figure(figsize=(11, 5))
            labels = [f"step={row.step_size} {row.horizon}" for row in coverage_plot.itertuples(index=False)]
            plt.bar(labels, coverage_plot["coverage_ratio"], color="#0f766e")
            plt.title("Forecast Evaluation Coverage by Horizon")
            plt.ylabel("Eligible Prediction Ratio")
            plt.xticks(rotation=30)
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(coverage_path, dpi=140)
            plt.close()
            chart_paths["forecast_coverage_summary"] = str(coverage_path)

        return chart_paths

    def _write_report(
        self,
        *,
        fetch_summary: pd.DataFrame,
        algorithms: list[str],
        skipped_algorithms: list[dict[str, str]],
        summary_by_horizon: pd.DataFrame,
        summary_by_model: pd.DataFrame,
        stacking_vs_all_models: pd.DataFrame,
        stack_df: pd.DataFrame,
    ) -> str:
        overall_stacking = summary_by_model[summary_by_model["model_name"] == FINAL_STACKING_MODEL_NAME].copy()
        best_horizon_rows = summary_by_horizon[summary_by_horizon["ticker"] == "OVERALL"].copy()
        best_horizon_rows = (
            best_horizon_rows.sort_values(["step_size", "horizon", "rmse", "mae", "model_name"])
            .groupby(["step_size", "horizon"], as_index=False)
            .first()
        )
        worst_divergence = stack_df[stack_df["evaluation_eligible"] == True].copy().sort_values("absolute_error", ascending=False).head(10)
        stacking_wins = stacking_vs_all_models.groupby(["scope", "step_size", "horizon"], as_index=False)[
            ["stacking_better_rmse", "stacking_better_directional_accuracy"]
        ].mean()

        lines = [
            "# Walk-Forward Forecasting Report",
            "",
            "## Experiment Setup",
            f"- Ticker universe: {', '.join(str(ticker).upper() for ticker in self.config.tickers)}",
            f"- Historical input window: {self.config.history_start} through {self.config.history_end}",
            f"- Historical training window baseline: {self.config.initial_train_start} through {self.config.initial_train_end}",
            f"- Rolling forecast window: {self.config.forecast_start} through {self.config.forecast_end}",
            f"- Horizons: {', '.join(self._normalize_horizons(self.config.horizons).keys())}",
            f"- Step sizes completed: {', '.join(str(value) for value in self._resolve_step_sizes())}",
            f"- Models actually run: {', '.join(algorithms)}",
            "- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, pooled by horizon and step size, with mean fallback before enough realized rows exist.",
            "- Actual data source used: vnstock daily OHLCV with KBS fallback ahead of local CSV fallback when the repo-local CSV history was too short.",
            "- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.",
            "- Prediction-date semantics: the forecast anchor is the latest feature row date, so `prediction_date` equals the date whose close is used in the forward-return denominator.",
            "- Evaluation-eligible rows: only predictions whose realized `target_date` close existed inside the fetched history were scored; later rows were kept in output tables but excluded from aggregate metrics.",
            "",
            "## Model Coverage",
        ]
        if not fetch_summary.empty:
            lines.append("- Fetched history coverage by ticker:")
            for row in fetch_summary.itertuples(index=False):
                lines.append(
                    f"  - {row.ticker}: source={row.source}, rows={row.rows}, available_range={row.fetched_min_date} through {row.fetched_max_date}"
                )
        if skipped_algorithms:
            for item in skipped_algorithms:
                lines.append(f"- Skipped `{item['algorithm']}`: {item['reason']}")
        else:
            lines.append("- No algorithms were skipped after environment capability checks.")
        lines.extend(["", "## Best Performers"])
        if not overall_stacking.empty:
            for row in overall_stacking.itertuples(index=False):
                lines.append(
                    f"- step_size={row.step_size}: stacking overall RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, directional_accuracy={row.directional_accuracy:.4f}"
                )
        for row in best_horizon_rows.itertuples(index=False):
            lines.append(
                f"- step_size={row.step_size}, horizon={row.horizon}: best RMSE model was `{row.model_name}` with RMSE={row.rmse:.6f}"
            )
        lines.extend(["", "## Stacking vs Individual Models"])
        if stacking_wins.empty:
            lines.append("- No stacking comparison rows were available.")
        else:
            for row in stacking_wins.itertuples(index=False):
                lines.append(
                    f"- scope={row.scope}, step_size={row.step_size}, horizon={row.horizon}: stacking beat the field on RMSE in {row.stacking_better_rmse:.2%} of pairwise comparisons and on directional accuracy in {row.stacking_better_directional_accuracy:.2%}."
                )
        lines.extend(["", "## Largest Divergences"])
        if worst_divergence.empty:
            lines.append("- No evaluation-eligible stacking rows were available for divergence analysis.")
        else:
            for row in worst_divergence.itertuples(index=False):
                lines.append(
                    f"- {row.ticker} {row.horizon} step={row.step_size} prediction_date={row.prediction_date.date()}: predicted={row.final_predicted_return:.6f}, actual={row.actual_return:.6f}, absolute_error={row.absolute_error:.6f}"
                )
        lines.extend(
            [
                "",
                "## Limitations",
                "- The repo-local daily CSV cache starts on 2020-12-21 for the requested tickers, so this experiment depends on the live vnstock KBS history path for pre-2020 backfill.",
                "- The requested calendar start date was 2018-01-01, but the first tradable session returned for all six tickers was 2018-01-02; January 1 was not a market session.",
                "- The current modern trainer path retrains on each rolling prediction date using only data available up to that prediction-date close.",
                "- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.",
                "- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.",
                "- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.",
                "",
                "## Output Paths",
                f"- csv/: `{self.csv_dir}`",
                f"- charts/: `{self.charts_dir}`",
                f"- report.md: `{self.report_path}`",
            ]
        )
        report_text = "\n".join(lines) + "\n"
        self.report_path.write_text(report_text, encoding="utf-8")
        return report_text

    def run(self) -> dict[str, Any]:
        horizons = self._normalize_horizons(self.config.horizons)
        step_sizes = self._resolve_step_sizes()
        algorithms, skipped_algorithms = self._resolve_available_algorithms()

        dates = {
            "history_start": pd.Timestamp(self.config.history_start).normalize(),
            "history_end": pd.Timestamp(self.config.history_end).normalize(),
            "forecast_start": pd.Timestamp(self.config.forecast_start).normalize(),
            "forecast_end": pd.Timestamp(self.config.forecast_end).normalize(),
        }
        if pd.Timestamp(self.config.initial_train_end).normalize() >= dates["forecast_start"]:
            raise ValueError("initial_train_end must be earlier than forecast_start")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        histories, fetch_summary = self._fetch_histories(dates["history_start"], dates["history_end"])
        context_sources = self._build_context_sources(dates["history_start"], dates["history_end"])

        base_frames: list[pd.DataFrame] = []
        coefficient_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []
        context_coverage_frames: list[pd.DataFrame] = []
        logger.info("walk_forward_base_generation_start", step_sizes=step_sizes, max_workers=self.config.max_workers)
        
        for step_size in step_sizes:
            logger.info("running_step_size", step_size=step_size)
            step_base_df, step_coefficient_df, step_importance_df, step_context_coverage_df = self._generate_base_predictions(
                histories=histories,
                context_sources=context_sources,
                algorithms=algorithms,
                step_size=int(step_size),
                horizons=horizons,
                forecast_start=dates["forecast_start"],
                forecast_end=dates["forecast_end"],
            )
            base_frames.append(step_base_df)
            if not step_coefficient_df.empty:
                coefficient_frames.append(step_coefficient_df)
            if not step_importance_df.empty:
                importance_frames.append(step_importance_df)
            if not step_context_coverage_df.empty:
                context_coverage_frames.append(step_context_coverage_df)
            
        base_df = pd.concat(base_frames, ignore_index=True).sort_values(["step_size", "prediction_date", "ticker", "horizon", "model_name"]).reset_index(drop=True)
        linear_coefficient_diagnostics = (
            pd.concat(coefficient_frames, ignore_index=True, sort=False)
            if coefficient_frames
            else empty_coefficient_diagnostics_frame()
        )
        linear_coefficient_stability_summary = summarize_coefficient_stability(linear_coefficient_diagnostics)
        feature_importance_diagnostics = (
            pd.concat(importance_frames, ignore_index=True, sort=False)
            if importance_frames
            else empty_feature_importance_diagnostics_frame()
        )
        feature_importance_stability_summary = summarize_feature_importance_stability(feature_importance_diagnostics)
        linear_vs_importance_feature_comparison = compare_linear_and_importance_diagnostics(
            linear_summary=linear_coefficient_stability_summary,
            importance_summary=feature_importance_stability_summary,
        )
        feature_governance_review = build_feature_governance_review(
            linear_summary=linear_coefficient_stability_summary,
            importance_summary=feature_importance_stability_summary,
            comparison=linear_vs_importance_feature_comparison,
        )
        if feature_governance_review.empty:
            feature_governance_review = empty_feature_governance_review_frame()
        context_coverage_diagnostics = (
            pd.concat(context_coverage_frames, ignore_index=True, sort=False)
            if context_coverage_frames
            else empty_context_coverage_diagnostics_frame()
        )
        context_coverage_summary = summarize_context_coverage(context_coverage_diagnostics)

        stack_df = self._build_stacking_predictions(base_df)
        actual_comparison_summary, coverage_summary = self._build_actual_comparison_summary(base_df, stack_df)
        summary_by_horizon, summary_by_ticker, summary_by_model = self._build_summary_tables(base_df, stack_df)
        stacking_vs_all_models = self._build_stacking_vs_models(summary_by_ticker, summary_by_horizon)
        backtest_summary, buy_and_hold_comparison = self._build_backtest_tables(base_df, stack_df)

        metadata = {
            "config": asdict(self.config),
            "resolved_horizons": horizons,
            "step_sizes": step_sizes,
            "available_algorithms": algorithms,
            "skipped_algorithms": skipped_algorithms,
            "fetch_summary": fetch_summary.to_dict(orient="records"),
            "foreign_flow_context": self._foreign_flow_context_metadata,
        }
        csv_paths = self._write_csv_outputs(
            base_df=base_df,
            stack_df=stack_df,
            actual_comparison_summary=actual_comparison_summary,
            summary_by_horizon=summary_by_horizon,
            summary_by_ticker=summary_by_ticker,
            summary_by_model=summary_by_model,
            stacking_vs_all_models=stacking_vs_all_models,
            backtest_summary=backtest_summary,
            buy_and_hold_comparison=buy_and_hold_comparison,
            forecast_coverage_summary=coverage_summary,
            linear_coefficient_diagnostics=linear_coefficient_diagnostics,
            linear_coefficient_stability_summary=linear_coefficient_stability_summary,
            feature_importance_diagnostics=feature_importance_diagnostics,
            feature_importance_stability_summary=feature_importance_stability_summary,
            linear_vs_importance_feature_comparison=linear_vs_importance_feature_comparison,
            feature_governance_review=feature_governance_review,
            context_coverage_diagnostics=context_coverage_diagnostics,
            context_coverage_summary=context_coverage_summary,
            metadata=metadata,
        )
        chart_paths = self._render_charts(
            base_df=base_df,
            stack_df=stack_df,
            summary_by_horizon=summary_by_horizon,
            summary_by_ticker=summary_by_ticker,
            buy_and_hold_comparison=buy_and_hold_comparison,
            forecast_coverage_summary=coverage_summary,
        )
        report_text = self._write_report(
            fetch_summary=fetch_summary,
            algorithms=algorithms,
            skipped_algorithms=skipped_algorithms,
            summary_by_horizon=summary_by_horizon,
            summary_by_model=summary_by_model,
            stacking_vs_all_models=stacking_vs_all_models,
            stack_df=stack_df,
        )

        return {
            "available_algorithms": algorithms,
            "skipped_algorithms": skipped_algorithms,
            "csv_paths": csv_paths,
            "chart_paths": chart_paths,
            "csv_dir": str(self.csv_dir),
            "charts_dir": str(self.charts_dir),
            "report_path": str(self.report_path),
            "report_text": report_text,
            "fetch_summary": fetch_summary,
            "base_df": base_df,
            "stack_df": stack_df,
            "actual_comparison_summary": actual_comparison_summary,
            "forecast_coverage_summary": coverage_summary,
            "linear_coefficient_diagnostics": linear_coefficient_diagnostics,
            "linear_coefficient_stability_summary": linear_coefficient_stability_summary,
            "feature_importance_diagnostics": feature_importance_diagnostics,
            "feature_importance_stability_summary": feature_importance_stability_summary,
            "linear_vs_importance_feature_comparison": linear_vs_importance_feature_comparison,
            "feature_governance_review": feature_governance_review,
            "context_coverage_diagnostics": context_coverage_diagnostics,
            "context_coverage_summary": context_coverage_summary,
            "backtest_summary": backtest_summary,
            "buy_and_hold_comparison": buy_and_hold_comparison,
            "summary_by_model": summary_by_model,
            "summary_by_horizon": summary_by_horizon,
            "summary_by_ticker": summary_by_ticker,
            "stacking_vs_all_models": stacking_vs_all_models,
        }
