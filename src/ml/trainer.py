"""Unified ML training and inference facade for technical stock models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from config.settings import get_settings
from src.ml.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    artifact_path,
    cleanup_ticker_dir,
    ensure_ticker_dir,
    load_manifest,
    write_manifest,
)
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.data_loader import (
    apply_context_features,
    load_market_proxy,
    load_sector_proxies,
    load_sentiment,
    load_ticker_sectors,
)
from src.ml.feature_engineering import FeatureEngineer
from src.ml.models.factory import create_model, load_model
from src.ml.sequence_dataset import (
    build_latest_sequence,
    create_sequence_dataset,
    select_sequence_range,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

HORIZON_DAYS = {
    "short": 5,
    "mid": 20,
    "long": 120,
}
SEQUENCE_ALGORITHMS = {"lstm", "bilstm"}
CONTEXT_COLUMNS = {"m_ret", "m_ret_5d", "rel_to_market", "s_ret", "s_ret_5d", "rel_to_sector"}


@dataclass(frozen=True)
class PreparedTickerData:
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    raw_stats: dict[str, Any]
    data_start: str
    data_end: str


@dataclass(frozen=True)
class SplitDefinition:
    train_stop: int
    val_start: int
    val_stop: int
    test_start: int
    gap: int


class DualModelTrainer:
    """Manifest-driven trainer and inference loader for ML artifacts."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self._model_dir = Path(model_dir or settings.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._feature_engineer = FeatureEngineer()
        self._metrics_evaluator = MetricsEvaluator()
        self._context_cache: dict[str, pd.DataFrame | None] | None = None

        # Compatibility cache: callers still inspect _models[ticker]["feature_cols"].
        self._models: dict[str, dict[str, Any]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._loaded_models: dict[tuple[str, str, str, str], Any] = {}

    # ------------------------------------------------------------------
    # Context and feature preparation
    # ------------------------------------------------------------------
    def _load_context_sources(self) -> dict[str, pd.DataFrame | None]:
        if self._context_cache is None:
            self._context_cache = {
                "market_df": load_market_proxy(),
                "sector_df": load_sector_proxies(),
                "ticker_sectors": load_ticker_sectors(),
                "sentiment_df": load_sentiment(),
            }
        return self._context_cache

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        renamed = df.copy()
        rename_map = {}
        for col in renamed.columns:
            lowered = str(col).lower().strip()
            if lowered in {"time", "datetime"}:
                rename_map[col] = "date"
            elif lowered in {"open", "high", "low", "close", "volume", "ticker", "date"}:
                rename_map[col] = lowered
        renamed = renamed.rename(columns=rename_map)
        if "date" not in renamed.columns:
            raise ValueError("Input data must contain a date/time column")

        renamed["date"] = pd.to_datetime(renamed["date"]).dt.normalize()
        for col in ("open", "high", "low", "close"):
            if col not in renamed.columns:
                raise ValueError(f"Missing OHLCV column '{col}'")
            renamed[col] = renamed[col].astype(float)
        if "volume" not in renamed.columns:
            renamed["volume"] = 0
        renamed["volume"] = renamed["volume"].fillna(0).astype(float)
        if "ticker" not in renamed.columns:
            renamed["ticker"] = ticker.upper()
        else:
            renamed["ticker"] = renamed["ticker"].fillna(ticker.upper()).astype(str).str.upper()

        return renamed.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    @staticmethod
    def _warmup_buffer_days(max_sequence_length: int) -> int:
        return max(180, max_sequence_length * 5)

    @staticmethod
    def _latest_five_year_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        end_ts = pd.Timestamp(df["date"].max()).normalize()
        start_target = end_ts - pd.DateOffset(years=5)
        return start_target.normalize(), end_ts

    def _filter_sentiment(
        self,
        sentiment_df: pd.DataFrame | None,
        ticker: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame | None:
        if sentiment_df is None or sentiment_df.empty:
            return None
        filtered = sentiment_df.copy()
        if "ticker" in filtered.columns:
            filtered = filtered[filtered["ticker"].astype(str).str.upper() == ticker.upper()]
        if filtered.empty:
            return None
        if "date" in filtered.columns:
            filtered["date"] = pd.to_datetime(filtered["date"]).dt.normalize()
            filtered = filtered[(filtered["date"] >= start_ts) & (filtered["date"] <= end_ts)]
        return filtered if not filtered.empty else None

    def _ensure_context_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        context_sources: dict[str, pd.DataFrame | None],
    ) -> pd.DataFrame:
        if CONTEXT_COLUMNS & set(df.columns):
            return df.copy()
        return apply_context_features(
            df,
            ticker,
            market_df=context_sources.get("market_df"),
            sector_df=context_sources.get("sector_df"),
            ticker_sectors=context_sources.get("ticker_sectors"),
        )

    def prepare_ticker_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        max_sequence_length: int = 20,
        context_sources: dict[str, pd.DataFrame | None] | None = None,
    ) -> PreparedTickerData:
        context_sources = context_sources or self._load_context_sources()
        normalized = self._normalize_ohlcv(df, ticker=ticker)
        if normalized.empty:
            raise ValueError(f"No rows available for {ticker}")

        start_target, end_ts = self._latest_five_year_bounds(normalized)
        warmup_start = start_target - pd.Timedelta(days=self._warmup_buffer_days(max_sequence_length))

        raw_scope = normalized[(normalized["date"] >= start_target) & (normalized["date"] <= end_ts)].reset_index(drop=True)
        if raw_scope.empty:
            raise ValueError(f"No rows remain for {ticker} after the 5-year window filter")

        buffer_scope = normalized[(normalized["date"] >= warmup_start) & (normalized["date"] <= end_ts)].reset_index(drop=True)
        context_buffer = self._ensure_context_features(buffer_scope, ticker, context_sources)
        ticker_sentiment = self._filter_sentiment(
            context_sources.get("sentiment_df"),
            ticker=ticker,
            start_ts=warmup_start,
            end_ts=end_ts,
        )
        if ticker_sentiment is not None and set(ticker_sentiment.columns) - {"date", "ticker"} <= set(context_buffer.columns):
            ticker_sentiment = None

        feature_buffer = self._feature_engineer.transform(
            context_buffer,
            sentiment_df=ticker_sentiment,
            drop_na=True,
        )
        feature_scope = feature_buffer[feature_buffer["date"] >= start_target].reset_index(drop=True)
        if feature_scope.empty:
            raise ValueError(f"Feature engineering produced no usable rows for {ticker}")

        # Stats-only pass on the strict 5-year scope to make warmup loss explicit.
        stats_scope = self._ensure_context_features(raw_scope, ticker, context_sources)
        stats_sentiment = self._filter_sentiment(
            context_sources.get("sentiment_df"),
            ticker=ticker,
            start_ts=start_target,
            end_ts=end_ts,
        )
        if stats_sentiment is not None and set(stats_sentiment.columns) - {"date", "ticker"} <= set(stats_scope.columns):
            stats_sentiment = None
        strict_features = self._feature_engineer.transform(
            stats_scope,
            sentiment_df=stats_sentiment,
            drop_na=True,
        )
        indicator_rows_lost = max(len(raw_scope) - len(strict_features), 0)
        feature_columns = self._feature_engineer.get_feature_columns(feature_scope)

        data_start = str(raw_scope["date"].min().date())
        data_end = str(raw_scope["date"].max().date())
        stats = {
            "data_start": data_start,
            "data_end": data_end,
            "raw_rows": int(len(raw_scope)),
            "indicator_warmup_rows": int(indicator_rows_lost),
            "feature_rows": int(len(feature_scope)),
            "warmup_buffer_start": str(buffer_scope["date"].min().date()),
        }
        logger.info(
            "prepared_ticker_data",
            ticker=ticker,
            data_start=data_start,
            data_end=data_end,
            raw_rows=stats["raw_rows"],
            indicator_warmup_rows=stats["indicator_warmup_rows"],
            feature_rows=stats["feature_rows"],
        )
        return PreparedTickerData(
            feature_frame=feature_scope,
            feature_columns=feature_columns,
            raw_stats=stats,
            data_start=data_start,
            data_end=data_end,
        )

    def compute_features_for_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rebuild features on the latest 5-year window for inference."""

        required_sequence_length = 20
        try:
            self._ensure_models_loaded(ticker)
            manifest = self._manifests[ticker.upper()]
            for horizon_info in manifest.get("horizons", {}).values():
                for algorithm_info in horizon_info.get("algorithms", {}).values():
                    seq_len = int(algorithm_info.get("sequence_length") or 1)
                    required_sequence_length = max(required_sequence_length, seq_len)
        except FileNotFoundError:
            pass

        prepared = self.prepare_ticker_data(
            ticker=ticker,
            df=df,
            max_sequence_length=required_sequence_length,
        )
        return prepared.feature_frame

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------
    @staticmethod
    def _add_targets(feature_frame: pd.DataFrame) -> pd.DataFrame:
        dataset = feature_frame.copy()
        for horizon, days in HORIZON_DAYS.items():
            future_return = dataset["close"].shift(-days) / dataset["close"] - 1.0
            dataset[f"target_return_{horizon}"] = future_return
            direction = pd.Series(np.nan, index=dataset.index, dtype=float)
            valid_mask = future_return.notna()
            direction.loc[valid_mask] = (future_return.loc[valid_mask] > 0.0).astype(int)
            dataset[f"target_direction_{horizon}"] = direction
        return dataset

    @staticmethod
    def _build_split_definition(n_rows: int, horizon_days: int) -> SplitDefinition:
        train_cut = int(n_rows * 0.70)
        val_cut = int(n_rows * 0.85)
        gap = horizon_days
        train_stop = max(train_cut - gap, 0)
        val_start = train_cut
        val_stop = max(val_cut - gap, val_start)
        test_start = val_cut
        return SplitDefinition(
            train_stop=train_stop,
            val_start=val_start,
            val_stop=val_stop,
            test_start=test_start,
            gap=gap,
        )

    def _build_horizon_problem(
        self,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        horizon: str,
        sequence_length: int,
    ) -> dict[str, Any] | None:
        direction_col = f"target_direction_{horizon}"
        return_col = f"target_return_{horizon}"
        labeled = dataset.dropna(subset=[direction_col, return_col]).reset_index(drop=True)
        if labeled.empty:
            return None

        split = self._build_split_definition(len(labeled), HORIZON_DAYS[horizon])
        if split.train_stop < 60 or split.test_start >= len(labeled):
            return None
        if len(labeled) - split.test_start < 10:
            return None

        X_all = labeled[feature_columns].to_numpy(dtype=float)
        y_direction = labeled[direction_col].astype(int).to_numpy()
        y_return = labeled[return_col].astype(float).to_numpy()
        closes = labeled["close"].astype(float).to_numpy()

        tabular = {
            "X_train": X_all[: split.train_stop],
            "X_val": X_all[split.val_start : split.val_stop],
            "X_test": X_all[split.test_start :],
            "y_train_direction": y_direction[: split.train_stop],
            "y_val_direction": y_direction[split.val_start : split.val_stop],
            "y_test_direction": y_direction[split.test_start :],
            "y_train_return": y_return[: split.train_stop],
            "y_val_return": y_return[split.val_start : split.val_stop],
            "y_test_return": y_return[split.test_start :],
            "test_closes": closes[split.test_start :],
        }

        direction_sequences = create_sequence_dataset(
            labeled[feature_columns],
            y_direction,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
        return_sequences = create_sequence_dataset(
            labeled[feature_columns],
            y_return,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )

        seq_train_direction = select_sequence_range(direction_sequences, stop_index=split.train_stop)
        seq_val_direction = select_sequence_range(
            direction_sequences,
            start_index=split.val_start,
            stop_index=split.val_stop,
        )
        seq_test_direction = select_sequence_range(direction_sequences, start_index=split.test_start)
        seq_train_return = select_sequence_range(return_sequences, stop_index=split.train_stop)
        seq_val_return = select_sequence_range(
            return_sequences,
            start_index=split.val_start,
            stop_index=split.val_stop,
        )
        seq_test_return = select_sequence_range(return_sequences, start_index=split.test_start)

        sequence = {
            "X_train": seq_train_direction.X,
            "X_val": seq_val_direction.X,
            "X_test": seq_test_direction.X,
            "y_train_direction": seq_train_direction.y,
            "y_val_direction": seq_val_direction.y,
            "y_test_direction": seq_test_direction.y,
            "y_train_return": seq_train_return.y,
            "y_val_return": seq_val_return.y,
            "y_test_return": seq_test_return.y,
            "test_closes": closes[seq_test_return.target_indices],
            "rows_lost": direction_sequences.rows_lost,
        }
        if len(sequence["X_train"]) == 0 or len(sequence["X_test"]) == 0:
            return None

        return {
            "labeled_rows": int(len(labeled)),
            "target_rows_lost": int(HORIZON_DAYS[horizon]),
            "split": split,
            "tabular": tabular,
            "sequence": sequence,
        }

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_algorithms(algorithms: list[str] | tuple[str, ...] | None) -> list[str]:
        values = [algo.strip().lower() for algo in (algorithms or ["cart"]) if algo.strip()]
        if not values:
            raise ValueError("At least one algorithm must be specified")
        return list(dict.fromkeys(values))

    @staticmethod
    def _normalize_horizons(horizons: list[str] | tuple[str, ...] | None) -> list[str]:
        values = [h.strip().lower() for h in (horizons or list(HORIZON_DAYS)) if h.strip()]
        invalid = [h for h in values if h not in HORIZON_DAYS]
        if invalid:
            raise ValueError(f"Unsupported horizons: {invalid}. Available: {sorted(HORIZON_DAYS)}")
        return list(dict.fromkeys(values))

    @staticmethod
    def _artifact_type(algorithm: str) -> str:
        return "torch" if algorithm in SEQUENCE_ALGORITHMS else "joblib"

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
        }

    def _trading_metrics(
        self,
        predicted_direction: np.ndarray,
        realized_future_returns: np.ndarray,
        horizon_days: int,
    ) -> dict[str, float]:
        signal = np.asarray(predicted_direction).astype(int)
        clipped_returns = np.clip(np.asarray(realized_future_returns, dtype=float), -0.999999, None)
        dailyized_returns = np.power(1.0 + clipped_returns, 1.0 / max(horizon_days, 1)) - 1.0
        evaluation = self._metrics_evaluator.evaluate_strategy(signal, dailyized_returns)
        return {
            "cagr": float(evaluation["metrics"]["cagr"]),
            "sharpe": float(evaluation["metrics"]["sharpe"]),
            "sortino": float(evaluation["metrics"]["sortino"]),
            "max_drawdown": float(evaluation["metrics"]["max_drawdown"]),
        }

    @staticmethod
    def _build_calibration(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        if X_val is None or y_val is None or len(X_val) == 0:
            return {"q10": -0.02, "q50": 0.0, "q90": 0.02}
        residuals = np.asarray(y_val, dtype=float) - np.asarray(model.predict(X_val), dtype=float)
        return {
            "q10": float(np.quantile(residuals, 0.10)),
            "q50": float(np.quantile(residuals, 0.50)),
            "q90": float(np.quantile(residuals, 0.90)),
        }

    def _model_params(
        self,
        *,
        algorithm: str,
        task: str,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        patience: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        criterion: str | None,
    ) -> dict[str, Any]:
        if algorithm in SEQUENCE_ALGORITHMS:
            return {
                "task": task,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
            }
        return {
            "task": task,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
        }

    def train(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        algorithms: list[str] | tuple[str, ...] | None = None,
        primary_algorithm: str | None = None,
        horizons: list[str] | tuple[str, ...] | None = None,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 30,
        patience: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str | None = None,
        clean: bool = True,
        context_sources: dict[str, pd.DataFrame | None] | None = None,
    ) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        algorithms = self._normalize_algorithms(algorithms)
        horizons = self._normalize_horizons(horizons)
        primary_algorithm = (primary_algorithm or algorithms[0]).lower()
        if primary_algorithm not in algorithms:
            raise ValueError("primary_algorithm must be one of the requested algorithms")

        max_sequence = sequence_length if any(algo in SEQUENCE_ALGORITHMS for algo in algorithms) else 1
        prepared = self.prepare_ticker_data(
            ticker=ticker,
            df=df,
            max_sequence_length=max_sequence,
            context_sources=context_sources,
        )
        labeled_dataset = self._add_targets(prepared.feature_frame)

        if clean:
            cleanup_ticker_dir(self._model_dir, ticker)
        ensure_ticker_dir(self._model_dir, ticker)

        report_rows: list[dict[str, Any]] = []
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "feature_columns": prepared.feature_columns,
            "data_window": {
                "start": prepared.data_start,
                "end": prepared.data_end,
            },
            "raw_stats": prepared.raw_stats,
            "horizons": {},
        }

        for horizon in horizons:
            problem = self._build_horizon_problem(
                labeled_dataset,
                prepared.feature_columns,
                horizon,
                sequence_length,
            )
            if problem is None:
                logger.warning("skipping_horizon", ticker=ticker, horizon=horizon, reason="insufficient_rows")
                continue

            manifest["horizons"][horizon] = {
                "days": HORIZON_DAYS[horizon],
                "target_rows_lost": problem["target_rows_lost"],
                "labeled_rows": problem["labeled_rows"],
                "algorithms": {},
            }
            for algorithm in algorithms:
                use_sequence = algorithm in SEQUENCE_ALGORITHMS
                inputs = problem["sequence" if use_sequence else "tabular"]
                rows_lost_to_sequence = int(inputs.get("rows_lost", 0))
                if len(inputs["X_train"]) == 0 or len(inputs["X_test"]) == 0:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="insufficient_split_rows",
                    )
                    continue
                if len(np.unique(inputs["y_train_direction"])) < 2:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="one_class_training_target",
                    )
                    continue

                trend_model = create_model(
                    algorithm,
                    **self._model_params(
                        algorithm=algorithm,
                        task="classification",
                        sequence_length=sequence_length,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        criterion=criterion,
                    ),
                )
                return_model = create_model(
                    algorithm,
                    **self._model_params(
                        algorithm=algorithm,
                        task="regression",
                        sequence_length=sequence_length,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        criterion=criterion,
                    ),
                )

                train_start = time.perf_counter()
                trend_model.fit(
                    inputs["X_train"],
                    inputs["y_train_direction"],
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_direction"] if len(inputs["X_val"]) else None,
                )
                return_model.fit(
                    inputs["X_train"],
                    inputs["y_train_return"],
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_return"] if len(inputs["X_val"]) else None,
                )
                train_seconds = float(time.perf_counter() - train_start)

                test_pred_direction = np.asarray(trend_model.predict(inputs["X_test"]))
                test_pred_return = np.asarray(return_model.predict(inputs["X_test"]), dtype=float)
                classification = self._classification_metrics(inputs["y_test_direction"], test_pred_direction)
                regression = self._regression_metrics(inputs["y_test_return"], test_pred_return)
                trading = self._trading_metrics(
                    test_pred_direction,
                    inputs["y_test_return"],
                    HORIZON_DAYS[horizon],
                )

                latency_start = time.perf_counter()
                trend_model.predict(inputs["X_test"])
                return_model.predict(inputs["X_test"])
                inference_latency_ms = float(
                    ((time.perf_counter() - latency_start) * 1000.0) / max(len(inputs["X_test"]), 1)
                )

                calibration = self._build_calibration(
                    return_model,
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_return"] if len(inputs["X_val"]) else None,
                )

                trend_path = artifact_path(
                    self._model_dir,
                    ticker=ticker,
                    task="trend",
                    algorithm=algorithm,
                    horizon=horizon,
                )
                return_path = artifact_path(
                    self._model_dir,
                    ticker=ticker,
                    task="return",
                    algorithm=algorithm,
                    horizon=horizon,
                )
                trend_model.save(trend_path)
                return_model.save(return_path)

                algorithm_manifest = {
                    "artifact_type": self._artifact_type(algorithm),
                    "sequence_length": sequence_length if use_sequence else None,
                    "trend_model_file": trend_path.name,
                    "return_model_file": return_path.name,
                    "calibration": calibration,
                    "metrics": {
                        **classification,
                        **regression,
                        **trading,
                        "train_seconds": train_seconds,
                        "inference_latency_ms": inference_latency_ms,
                    },
                }
                manifest["horizons"][horizon]["algorithms"][algorithm] = algorithm_manifest
                report_rows.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "horizon_days": HORIZON_DAYS[horizon],
                        "algorithm": algorithm,
                        "artifact_type": self._artifact_type(algorithm),
                        "sequence_length": sequence_length if use_sequence else "",
                        "data_start": prepared.data_start,
                        "data_end": prepared.data_end,
                        "raw_rows": prepared.raw_stats["raw_rows"],
                        "indicator_warmup_rows": prepared.raw_stats["indicator_warmup_rows"],
                        "target_rows_lost": problem["target_rows_lost"],
                        "sequence_rows_lost": rows_lost_to_sequence,
                        "final_usable_rows": int(len(inputs["X_train"]) + len(inputs["X_val"]) + len(inputs["X_test"])),
                        **algorithm_manifest["metrics"],
                    }
                )
                logger.info(
                    "trained_model_bundle",
                    ticker=ticker,
                    algorithm=algorithm,
                    horizon=horizon,
                    accuracy=classification["accuracy"],
                    f1=classification["f1"],
                    mae=regression["mae"],
                    train_seconds=train_seconds,
                )

        if not report_rows:
            raise ValueError(f"No model bundles were trained for {ticker}")

        write_manifest(self._model_dir, ticker, manifest)
        self._manifests[ticker] = manifest
        self._models[ticker] = {
            "feature_cols": prepared.feature_columns,
            "primary_algorithm": primary_algorithm,
            "manifest": manifest,
        }
        self._loaded_models = {
            key: model for key, model in self._loaded_models.items() if key[0] != ticker
        }

        return {
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "algorithms": algorithms,
            "data_start": prepared.data_start,
            "data_end": prepared.data_end,
            "feature_count": len(prepared.feature_columns),
            "report_rows": report_rows,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _ensure_models_loaded(self, ticker: str) -> None:
        ticker_key = ticker.upper()
        if ticker_key in self._manifests:
            return
        manifest = load_manifest(self._model_dir, ticker_key)
        self._manifests[ticker_key] = manifest
        self._models[ticker_key] = {
            "feature_cols": manifest.get("feature_columns", []),
            "primary_algorithm": manifest.get("primary_algorithm"),
            "manifest": manifest,
        }

    def _get_loaded_model(
        self,
        ticker: str,
        algorithm: str,
        horizon: str,
        task: str,
    ) -> Any:
        ticker_key = ticker.upper()
        cache_key = (ticker_key, algorithm, horizon, task)
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        self._ensure_models_loaded(ticker_key)
        manifest = self._manifests[ticker_key]
        horizon_info = manifest.get("horizons", {}).get(horizon)
        if horizon_info is None:
            raise FileNotFoundError(f"No artifacts found for {ticker_key} horizon '{horizon}'")
        algorithm_info = horizon_info.get("algorithms", {}).get(algorithm)
        if algorithm_info is None:
            raise FileNotFoundError(
                f"No artifacts found for {ticker_key} algorithm '{algorithm}' horizon '{horizon}'"
            )
        file_key = "trend_model_file" if task == "trend" else "return_model_file"
        model_path = self._model_dir / ticker_key / algorithm_info[file_key]
        model = load_model(algorithm, model_path)
        self._loaded_models[cache_key] = model
        return model

    @staticmethod
    def _trend_probabilities_from_binary(probs: np.ndarray) -> dict[str, float]:
        down = float(probs[0])
        up = float(probs[1]) if len(probs) > 1 else 0.0
        sideways = min(up, down) * 0.5
        total = up + down + sideways
        return {
            "up": round(up / total, 4),
            "sideways": round(sideways / total, 4),
            "down": round(down / total, 4),
        }

    @staticmethod
    def _expected_range(
        current_close: float,
        predicted_return: float,
        calibration: dict[str, float],
    ) -> dict[str, float]:
        bottom = current_close * (1.0 + predicted_return + calibration.get("q10", -0.02))
        median = current_close * (1.0 + predicted_return + calibration.get("q50", 0.0))
        top = current_close * (1.0 + predicted_return + calibration.get("q90", 0.02))
        ordered = sorted([bottom, median, top])
        return {
            "bottom_10th": round(float(ordered[0]), 2),
            "median_50th": round(float(ordered[1]), 2),
            "ceiling_90th": round(float(ordered[2]), 2),
        }

    def predict(
        self,
        ticker: str,
        features: np.ndarray | pd.Series | pd.DataFrame,
        horizon: str = "short",
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        horizon = horizon.lower()
        if horizon in {"1w", "5d"}:
            horizon = "short"
        elif horizon in {"1m", "20d"}:
            horizon = "mid"
        elif horizon in {"6m", "120d"}:
            horizon = "long"
        if horizon not in HORIZON_DAYS:
            raise ValueError(f"Unsupported horizon '{horizon}'")

        self._ensure_models_loaded(ticker)
        manifest = self._manifests[ticker]
        algorithm = (algorithm or manifest.get("primary_algorithm")).lower()
        horizon_info = manifest.get("horizons", {}).get(horizon)
        if horizon_info is None:
            raise FileNotFoundError(f"No trained horizon '{horizon}' found for {ticker}")
        algorithm_info = horizon_info.get("algorithms", {}).get(algorithm)
        if algorithm_info is None:
            raise FileNotFoundError(f"No trained algorithm '{algorithm}' found for {ticker} horizon '{horizon}'")

        feature_columns = manifest.get("feature_columns", [])
        if isinstance(features, pd.Series):
            feature_frame = pd.DataFrame([features.to_dict()])
        elif isinstance(features, pd.DataFrame):
            feature_frame = features.copy()
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                feature_frame = pd.DataFrame([features], columns=feature_columns)
            else:
                feature_frame = pd.DataFrame(features, columns=feature_columns)
        else:
            raise TypeError("features must be a numpy array, Series, or DataFrame")

        missing = [column for column in feature_columns if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"Missing mandatory features for {ticker}: {missing}")
        if "close" not in feature_frame.columns:
            raise ValueError("Inference requires the latest close price in the feature frame")
        current_close = float(feature_frame["close"].iloc[-1])

        if algorithm in SEQUENCE_ALGORITHMS:
            sequence_length = int(algorithm_info.get("sequence_length") or 0)
            model_input = build_latest_sequence(
                feature_frame,
                feature_columns=feature_columns,
                sequence_length=sequence_length,
            )
        else:
            model_input = feature_frame[feature_columns].iloc[[-1]].to_numpy(dtype=float)

        trend_model = self._get_loaded_model(ticker, algorithm, horizon, "trend")
        return_model = self._get_loaded_model(ticker, algorithm, horizon, "return")
        trend_probs = self._trend_probabilities_from_binary(trend_model.predict_proba(model_input)[0])
        predicted_return = float(np.asarray(return_model.predict(model_input)).reshape(-1)[0])
        expected_range = self._expected_range(current_close, predicted_return, algorithm_info.get("calibration", {}))

        return {
            "algorithm": algorithm,
            "artifact_type": algorithm_info.get("artifact_type"),
            "horizon": horizon,
            "sequence_length": algorithm_info.get("sequence_length"),
            "predicted_return": predicted_return,
            "trend_probabilities": trend_probs,
            "expected_range": expected_range,
            "feature_set_version": f"ml_schema_v{ARTIFACT_SCHEMA_VERSION}",
        }
