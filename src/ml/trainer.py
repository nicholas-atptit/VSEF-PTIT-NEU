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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
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
from src.ml.portfolio.allocation import RiskAwareAllocator
from src.ml.regime.regime_detector import REGIME_TO_CODE, RegimeDetector
from src.ml.risk import RiskEngine
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
BOOSTER_ALGORITHMS = {"xgboost", "lightgbm"}
CONTEXT_COLUMNS = {"m_ret", "m_ret_5d", "rel_to_market", "s_ret", "s_ret_5d", "rel_to_sector"}
RISK_FEATURE_COLUMNS = ["var_q", "cvar_q", "covar_q", "delta_covar", "rolling_drawdown"]
REGIME_FEATURE_COLUMNS = ["regime_label", "regime_probability"]


@dataclass(frozen=True)
class PreparedTickerData:
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    base_feature_columns: list[str]
    raw_stats: dict[str, Any]
    data_start: str
    data_end: str
    risk_summary: dict[str, Any] | None = None
    regime_distribution: dict[str, float] | None = None
    advanced_config: dict[str, Any] | None = None


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

    @staticmethod
    def _resolve_horizon_day_map(
        horizons: list[str] | tuple[str, ...] | None = None,
        horizon_days_map: dict[str, int] | None = None,
    ) -> dict[str, int]:
        base_map = {str(name).lower(): int(days) for name, days in HORIZON_DAYS.items()}
        if horizon_days_map:
            for raw_name, raw_days in horizon_days_map.items():
                name = str(raw_name).strip().lower()
                if not name:
                    continue
                day_count = int(raw_days)
                if day_count <= 0:
                    raise ValueError(f"Horizon '{name}' must use a positive number of days")
                base_map[name] = day_count

        values = [str(h).strip().lower() for h in (horizons or list(HORIZON_DAYS)) if str(h).strip()]
        invalid = [h for h in values if h not in base_map]
        if invalid:
            raise ValueError(f"Unsupported horizons: {invalid}. Available: {sorted(base_map)}")
        return {name: base_map[name] for name in dict.fromkeys(values)}

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

    @staticmethod
    def _normalize_risk_config(risk_config: dict[str, Any] | None) -> dict[str, Any]:
        settings = get_settings()
        incoming = risk_config.copy() if risk_config else {}
        confidence_levels = incoming.get("confidence_levels", [0.95, 0.99])
        return {
            "risk_enabled": bool(incoming.get("risk_enabled", bool(risk_config))),
            "enable_covar": bool(incoming.get("enable_covar", settings.enable_covar)),
            "enable_risk_engine": bool(incoming.get("enable_risk_engine", settings.enable_risk_engine)),
            "enable_regime_detection": bool(incoming.get("enable_regime_detection", settings.enable_regime_detection)),
            "enable_regime_switching": bool(incoming.get("enable_regime_switching", settings.enable_regime_switching)),
            "enable_risk_allocation": bool(incoming.get("enable_risk_allocation", settings.enable_risk_allocation)),
            "covar_quantile": float(incoming.get("covar_quantile", settings.covar_quantile)),
            "covar_window": int(incoming.get("covar_window", settings.covar_window)),
            "regime_method": str(incoming.get("regime_method", settings.regime_method)),
            "risk_penalty_strength": float(
                incoming.get("risk_penalty_strength", settings.risk_penalty_strength)
            ),
            "high_vol_exposure_cut": float(
                incoming.get("high_vol_exposure_cut", settings.high_vol_exposure_cut)
            ),
            "crisis_exposure_cut": float(
                incoming.get("crisis_exposure_cut", settings.crisis_exposure_cut)
            ),
            "high_vol_threshold": float(
                incoming.get("high_vol_threshold", settings.high_vol_threshold)
            ),
            "crisis_drawdown_threshold": float(
                incoming.get("crisis_drawdown_threshold", settings.crisis_drawdown_threshold)
            ),
            "crisis_delta_covar_threshold": float(
                incoming.get("crisis_delta_covar_threshold", settings.crisis_delta_covar_threshold)
            ),
            "simulations": int(incoming.get("simulations", 10000)),
            "confidence_levels": list(confidence_levels),
            "random_seed": int(incoming.get("random_seed", 42)),
        }

    @staticmethod
    def _advanced_features_enabled(config: dict[str, Any]) -> bool:
        return bool(
            config.get("enable_covar")
            or config.get("enable_risk_engine")
            or config.get("enable_regime_detection")
            or config.get("enable_regime_switching")
            or config.get("enable_risk_allocation")
        )

    def _apply_advanced_risk_features(
        self,
        ticker: str,
        feature_frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, float] | None]:
        augmented = feature_frame.copy()
        risk_summary: dict[str, Any] | None = None
        regime_distribution: dict[str, float] | None = None

        if not self._advanced_features_enabled(config):
            return augmented, risk_summary, regime_distribution

        asset_returns = pd.to_numeric(augmented.get("pct_return"), errors="coerce")
        if asset_returns is None or asset_returns.empty:
            asset_returns = pd.to_numeric(augmented["close"].pct_change(), errors="coerce")
        market_returns = pd.to_numeric(augmented.get("m_ret"), errors="coerce") if "m_ret" in augmented.columns else None
        include_mc = bool(config.get("risk_enabled"))

        if config.get("enable_risk_engine") or config.get("enable_covar") or config.get("enable_risk_allocation"):
            risk_engine = RiskEngine(
                window=int(config["covar_window"]),
                quantile=float(config["covar_quantile"]),
                simulations=int(config["simulations"]),
                random_seed=int(config["random_seed"]),
                include_monte_carlo=include_mc,
            )
            evaluated = risk_engine.evaluate(asset_returns.rename(ticker), market_returns=market_returns)
            risk_frame = evaluated["per_asset_frames"][ticker]
            for column in RISK_FEATURE_COLUMNS:
                if column in risk_frame.columns:
                    augmented[column] = risk_frame[column].reindex(augmented.index)
            risk_summary = evaluated["risk_summary"]

        if config.get("enable_regime_detection") or config.get("enable_regime_switching"):
            detector = RegimeDetector(
                method=str(config["regime_method"]),
                high_vol_threshold=float(config.get("high_vol_threshold", 0.03)),
                crisis_drawdown_threshold=float(config.get("crisis_drawdown_threshold", -0.12)),
                crisis_delta_covar_threshold=float(config.get("crisis_delta_covar_threshold", 0.015)),
            )
            regime_result = detector.detect_from_frame(augmented)
            augmented["regime_label"] = regime_result.encoded_labels.reindex(augmented.index)
            assigned_prob = regime_result.probabilities.max(axis=1).rename("regime_probability")
            augmented["regime_probability"] = assigned_prob.reindex(augmented.index)
            counts = regime_result.labels.value_counts(dropna=False).to_dict()
            total = max(int(len(regime_result.labels)), 1)
            regime_distribution = {str(k): float(v) / total for k, v in counts.items()}

        return augmented, risk_summary, regime_distribution

    @staticmethod
    def _select_algorithm_feature_columns(
        prepared: PreparedTickerData,
        algorithm: str,
        advanced_config: dict[str, Any],
    ) -> list[str]:
        selected = list(prepared.base_feature_columns)
        if algorithm in BOOSTER_ALGORITHMS and (
            advanced_config.get("enable_risk_engine")
            or advanced_config.get("enable_covar")
            or advanced_config.get("enable_risk_allocation")
        ):
            for column in RISK_FEATURE_COLUMNS:
                if column in prepared.feature_frame.columns and column not in selected:
                    selected.append(column)
        if algorithm in BOOSTER_ALGORITHMS and advanced_config.get("enable_regime_switching"):
            for column in REGIME_FEATURE_COLUMNS:
                if column in prepared.feature_frame.columns and column not in selected:
                    selected.append(column)
        return selected

    def prepare_ticker_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        max_sequence_length: int = 20,
        context_sources: dict[str, pd.DataFrame | None] | None = None,
        risk_config: dict[str, Any] | None = None,
        window_start: Any | None = None,
        window_end: Any | None = None,
    ) -> PreparedTickerData:
        context_sources = context_sources or self._load_context_sources()
        advanced_config = self._normalize_risk_config(risk_config)
        normalized = self._normalize_ohlcv(df, ticker=ticker)
        if normalized.empty:
            raise ValueError(f"No rows available for {ticker}")

        if window_end is not None:
            normalized = normalized[normalized["date"] <= pd.Timestamp(window_end).normalize()].reset_index(drop=True)
            if normalized.empty:
                raise ValueError(f"No rows available for {ticker} on or before {window_end}")

        if window_start is None:
            start_target, end_ts = self._latest_five_year_bounds(normalized)
        else:
            end_ts = pd.Timestamp(normalized["date"].max()).normalize()
            start_target = pd.Timestamp(window_start).normalize()
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
        base_feature_columns = self._feature_engineer.get_feature_columns(feature_scope)
        feature_scope, risk_summary, regime_distribution = self._apply_advanced_risk_features(
            ticker,
            feature_scope,
            advanced_config,
        )

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
            base_feature_columns=base_feature_columns,
            raw_stats=stats,
            data_start=data_start,
            data_end=data_end,
            risk_summary=risk_summary,
            regime_distribution=regime_distribution,
            advanced_config=advanced_config,
        )

    def compute_features_for_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        window_start: Any | None = None,
        window_end: Any | None = None,
    ) -> pd.DataFrame:
        """Rebuild features on the latest 5-year window for inference."""

        required_sequence_length = 20
        advanced_config: dict[str, Any] | None = None
        try:
            self._ensure_models_loaded(ticker)
            manifest = self._manifests[ticker.upper()]
            advanced_config = manifest.get("advanced_risk")
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
            risk_config=advanced_config,
            window_start=window_start,
            window_end=window_end,
        )
        return prepared.feature_frame

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------
    @staticmethod
    def _side_cost_rate(
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> float:
        return (float(transaction_fee_bps) + float(slippage_bps)) / 10000.0

    @classmethod
    def calculate_net_trade_return(
        cls,
        entry_open: float | int,
        exit_close: float | int,
        *,
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> float:
        side_cost_rate = cls._side_cost_rate(
            transaction_fee_bps=transaction_fee_bps,
            slippage_bps=slippage_bps,
        )
        entry = float(entry_open)
        exit_price = float(exit_close)
        if entry <= 0.0 or exit_price <= 0.0:
            return float("nan")
        effective_entry = entry * (1.0 + side_cost_rate)
        effective_exit = exit_price * (1.0 - side_cost_rate)
        return float((effective_exit / effective_entry) - 1.0)

    @classmethod
    def calculate_net_trade_return_series(
        cls,
        entry_open: pd.Series,
        exit_close: pd.Series,
        *,
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> pd.Series:
        side_cost_rate = cls._side_cost_rate(
            transaction_fee_bps=transaction_fee_bps,
            slippage_bps=slippage_bps,
        )
        entry = pd.to_numeric(entry_open, errors="coerce")
        exit_price = pd.to_numeric(exit_close, errors="coerce")
        effective_entry = entry * (1.0 + side_cost_rate)
        effective_exit = exit_price * (1.0 - side_cost_rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            net_return = (effective_exit / effective_entry) - 1.0
        invalid_mask = effective_entry.le(0.0) | effective_exit.le(0.0)
        net_return[invalid_mask] = np.nan
        return net_return.astype(float)

    @staticmethod
    def _add_targets(
        feature_frame: pd.DataFrame,
        horizon_days_map: dict[str, int] | None = None,
        *,
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> pd.DataFrame:
        dataset = feature_frame.copy()
        resolved_horizons = horizon_days_map or HORIZON_DAYS
        for horizon, days in resolved_horizons.items():
            future_return = dataset["close"].shift(-days) / dataset["close"] - 1.0
            dataset[f"target_return_{horizon}"] = future_return
            direction = pd.Series(np.nan, index=dataset.index, dtype=float)
            valid_mask = future_return.notna()
            direction.loc[valid_mask] = (future_return.loc[valid_mask] > 0.0).astype(int)
            dataset[f"target_direction_{horizon}"] = direction
            dataset[f"target_date_{horizon}"] = dataset["date"].shift(-days)
            dataset[f"entry_date_{horizon}"] = dataset["date"].shift(-1)
            entry_open = dataset["open"].shift(-1) if "open" in dataset.columns else dataset["close"].shift(-1)
            dataset[f"entry_open_{horizon}"] = entry_open
            target_close = dataset["close"].shift(-days)
            net_trade_return = DualModelTrainer.calculate_net_trade_return_series(
                entry_open,
                target_close,
                transaction_fee_bps=transaction_fee_bps,
                slippage_bps=slippage_bps,
            )
            dataset[f"target_net_return_{horizon}"] = net_trade_return
            dataset[f"net_trade_return_{horizon}"] = net_trade_return
            profit_label = pd.Series(np.nan, index=dataset.index, dtype=float)
            valid_profit = net_trade_return.notna()
            profit_label.loc[valid_profit] = (net_trade_return.loc[valid_profit] > 0.0).astype(int)
            dataset[f"target_profit_label_{horizon}"] = profit_label
            dataset[f"profit_label_{horizon}"] = profit_label
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
        *,
        horizon_days: int | None = None,
    ) -> dict[str, Any] | None:
        direction_col = f"target_direction_{horizon}"
        return_col = f"target_return_{horizon}"
        profit_col = f"target_profit_label_{horizon}"
        labeled = dataset.dropna(subset=[direction_col, return_col, profit_col]).reset_index(drop=True)
        if labeled.empty:
            return None

        resolved_horizon_days = int(horizon_days or HORIZON_DAYS[horizon])
        split = self._build_split_definition(len(labeled), resolved_horizon_days)
        if split.train_stop < 60 or split.test_start >= len(labeled):
            return None
        if len(labeled) - split.test_start < 10:
            return None

        X_all = labeled[feature_columns].to_numpy(dtype=float)
        y_direction = labeled[direction_col].astype(int).to_numpy()
        y_return = labeled[return_col].astype(float).to_numpy()
        y_profit = labeled[profit_col].astype(int).to_numpy()
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
            "y_train_profit": y_profit[: split.train_stop],
            "y_val_profit": y_profit[split.val_start : split.val_stop],
            "y_test_profit": y_profit[split.test_start :],
            "val_feature_frame": labeled.iloc[split.val_start : split.val_stop].reset_index(drop=True),
            "val_indices": np.arange(split.val_start, split.val_stop),
            "test_closes": closes[split.test_start :],
            "test_feature_frame": labeled.iloc[split.test_start :].reset_index(drop=True),
            "test_indices": np.arange(split.test_start, len(labeled)),
        }

        direction_sequences = create_sequence_dataset(
            labeled[feature_columns],
            y_direction,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
        profit_sequences = create_sequence_dataset(
            labeled[feature_columns],
            y_profit,
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
        seq_train_profit = select_sequence_range(profit_sequences, stop_index=split.train_stop)
        seq_val_profit = select_sequence_range(
            profit_sequences,
            start_index=split.val_start,
            stop_index=split.val_stop,
        )
        seq_test_profit = select_sequence_range(profit_sequences, start_index=split.test_start)
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
            "y_train_profit": seq_train_profit.y.astype(int),
            "y_val_profit": seq_val_profit.y.astype(int),
            "y_test_profit": seq_test_profit.y.astype(int),
            "y_train_return": seq_train_return.y,
            "y_val_return": seq_val_return.y,
            "y_test_return": seq_test_return.y,
            "val_feature_frame": labeled.iloc[seq_val_return.target_indices].reset_index(drop=True),
            "val_indices": seq_val_return.target_indices,
            "test_closes": closes[seq_test_return.target_indices],
            "test_feature_frame": labeled.iloc[seq_test_return.target_indices].reset_index(drop=True),
            "test_indices": seq_test_return.target_indices,
            "rows_lost": direction_sequences.rows_lost,
        }
        if len(sequence["X_train"]) == 0 or len(sequence["X_test"]) == 0:
            return None

        return {
            "labeled_rows": int(len(labeled)),
            "target_rows_lost": int(resolved_horizon_days),
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
    def normalize_strategy_returns(
        realized_future_returns: np.ndarray | pd.Series,
        horizon_days: int,
    ) -> np.ndarray:
        clipped_returns = np.clip(np.asarray(realized_future_returns, dtype=float), -0.999999, None)
        return np.power(1.0 + clipped_returns, 1.0 / max(horizon_days, 1)) - 1.0

    @classmethod
    def evaluate_strategy_for_horizon(
        cls,
        signal: np.ndarray | pd.Series,
        realized_future_returns: np.ndarray | pd.Series,
        horizon_days: int,
        *,
        evaluator: MetricsEvaluator | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        strategy_returns = cls.normalize_strategy_returns(realized_future_returns, horizon_days)
        metric_engine = evaluator or MetricsEvaluator()
        return metric_engine.evaluate_strategy(signal, strategy_returns, config)

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return DualModelTrainer._binary_classification_metrics(y_true, y_pred)

    @staticmethod
    def _binary_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        true_array = np.asarray(y_true, dtype=int)
        pred_array = np.asarray(y_pred, dtype=int)
        metrics = {
            "accuracy": float(accuracy_score(true_array, pred_array)),
            "balanced_accuracy": float(balanced_accuracy_score(true_array, pred_array)),
            "precision": float(precision_score(true_array, pred_array, zero_division=0)),
            "recall": float(recall_score(true_array, pred_array, zero_division=0)),
            "f1": float(f1_score(true_array, pred_array, zero_division=0)),
            "positive_class_precision": float(precision_score(true_array, pred_array, zero_division=0)),
        }
        roc_auc = np.nan
        if y_prob is not None:
            probability_array = np.asarray(y_prob, dtype=float).reshape(-1)
            if len(np.unique(true_array)) > 1:
                try:
                    roc_auc = float(roc_auc_score(true_array, probability_array))
                except ValueError:
                    roc_auc = np.nan
        metrics["roc_auc"] = roc_auc
        tn, fp, fn, tp = confusion_matrix(true_array, pred_array, labels=[0, 1]).ravel()
        metrics.update(
            {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
        return metrics

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        residuals = y_true - y_pred
        residual_std = float(np.std(residuals))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "residual_std": residual_std,
            "volatility_proxy_source": "test_residuals_std" if len(y_true) > 1 else "validation_rmse"
        }

    def _trading_metrics(
        self,
        predicted_direction: np.ndarray,
        realized_future_returns: np.ndarray,
        horizon_days: int,
    ) -> dict[str, float]:
        signal = np.asarray(predicted_direction).astype(int)
        evaluation = self.evaluate_strategy_for_horizon(
            signal,
            realized_future_returns,
            horizon_days,
            evaluator=self._metrics_evaluator,
        )
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
        tune_boosters: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"task": task}
        if tune_boosters and algorithm in {"xgboost", "lightgbm"}:
            params["tuned"] = True
            
        if algorithm in SEQUENCE_ALGORITHMS:
            return {
                **params,
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
            **params,
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
        horizon_days_map: dict[str, int] | None = None,
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
        tune_boosters: bool = False,
        risk_config: dict[str, Any] | None = None,
        context_sources: dict[str, pd.DataFrame | None] | None = None,
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        normalized_risk_config = self._normalize_risk_config(risk_config)
        algorithms = self._normalize_algorithms(algorithms)
        resolved_horizons = self._resolve_horizon_day_map(horizons, horizon_days_map)
        horizons = list(resolved_horizons)
        primary_algorithm = (primary_algorithm or algorithms[0]).lower()
        if primary_algorithm not in algorithms:
            raise ValueError("primary_algorithm must be one of the requested algorithms")

        max_sequence = sequence_length if any(algo in SEQUENCE_ALGORITHMS for algo in algorithms) else 1
        prepared = self.prepare_ticker_data(
            ticker=ticker,
            df=df,
            max_sequence_length=max_sequence,
            context_sources=context_sources,
            risk_config=normalized_risk_config,
        )
        labeled_dataset = self._add_targets(
            prepared.feature_frame,
            resolved_horizons,
            transaction_fee_bps=transaction_fee_bps,
            slippage_bps=slippage_bps,
        )

        if clean:
            cleanup_ticker_dir(self._model_dir, ticker)
        ensure_ticker_dir(self._model_dir, ticker)

        report_rows: list[dict[str, Any]] = []
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "target_type": "forward_return",
            "feature_columns": prepared.feature_columns,
            "base_feature_columns": prepared.base_feature_columns,
            "data_window": {
                "start": prepared.data_start,
                "end": prepared.data_end,
            },
            "raw_stats": prepared.raw_stats,
            "advanced_risk": prepared.advanced_config,
            "risk_summary": prepared.risk_summary or {},
            "regime_distribution": prepared.regime_distribution or {},
            "covar_config": {
                "enabled": bool(
                    normalized_risk_config.get("enable_covar")
                    or normalized_risk_config.get("enable_risk_engine")
                ),
                "quantile": normalized_risk_config.get("covar_quantile"),
                "window": normalized_risk_config.get("covar_window"),
            },
            "profit_target_config": {
                "transaction_fee_bps": float(transaction_fee_bps),
                "slippage_bps": float(slippage_bps),
                "entry_convention": "next_tradable_open",
                "exit_convention": "target_date_close",
                "label_definition": "1 if net trade return after costs > 0 else 0",
            },
            "horizons": {},
        }

        for horizon in horizons:
            horizon_problem = self._build_horizon_problem(
                labeled_dataset,
                prepared.base_feature_columns,
                horizon,
                sequence_length,
                horizon_days=resolved_horizons[horizon],
            )
            if horizon_problem is None:
                logger.warning("skipping_horizon", ticker=ticker, horizon=horizon, reason="insufficient_rows")
                continue

            manifest["horizons"][horizon] = {
                "days": resolved_horizons[horizon],
                "target_rows_lost": horizon_problem["target_rows_lost"],
                "labeled_rows": horizon_problem["labeled_rows"],
                "algorithms": {},
            }
            for algorithm in algorithms:
                algorithm_feature_columns = self._select_algorithm_feature_columns(
                    prepared,
                    algorithm,
                    normalized_risk_config,
                )
                problem = self._build_horizon_problem(
                    labeled_dataset,
                    algorithm_feature_columns,
                    horizon,
                    sequence_length,
                    horizon_days=resolved_horizons[horizon],
                )
                if problem is None:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="insufficient_rows_for_feature_set",
                    )
                    continue
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
                if len(np.unique(inputs["y_train_profit"])) < 2:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="one_class_profit_target",
                    )
                    continue

                trend_params = self._model_params(
                    algorithm,
                    "classification",
                    sequence_length,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    batch_size,
                    epochs,
                    patience,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    criterion,
                    tune_boosters,
                )
                profit_params = self._model_params(
                    algorithm,
                    "classification",
                    sequence_length,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    batch_size,
                    epochs,
                    patience,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    criterion,
                    tune_boosters,
                )
                return_params = self._model_params(
                    algorithm,
                    "regression",
                    sequence_length,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    batch_size,
                    epochs,
                    patience,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    criterion,
                    tune_boosters,
                )
                trend_model = create_model(algorithm, **trend_params)
                profit_model = create_model(algorithm, **profit_params)
                return_model = create_model(algorithm, **return_params)

                train_start = time.perf_counter()
                trend_model.fit(
                    inputs["X_train"],
                    inputs["y_train_direction"],
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_direction"] if len(inputs["X_val"]) else None,
                )
                profit_model.fit(
                    inputs["X_train"],
                    inputs["y_train_profit"],
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_profit"] if len(inputs["X_val"]) else None,
                )
                return_model.fit(
                    inputs["X_train"],
                    inputs["y_train_return"],
                    inputs["X_val"] if len(inputs["X_val"]) else None,
                    inputs["y_val_return"] if len(inputs["X_val"]) else None,
                )
                train_seconds = float(time.perf_counter() - train_start)

                test_pred_direction = np.asarray(trend_model.predict(inputs["X_test"]))
                test_pred_profit = np.asarray(profit_model.predict(inputs["X_test"]), dtype=int)
                test_profit_probs = np.asarray(profit_model.predict_proba(inputs["X_test"]), dtype=float)
                if test_profit_probs.ndim == 2 and test_profit_probs.shape[1] > 1:
                    test_profit_positive_prob = test_profit_probs[:, 1]
                else:
                    test_profit_positive_prob = np.asarray(test_profit_probs).reshape(-1)
                test_pred_return = np.asarray(return_model.predict(inputs["X_test"]), dtype=float)
                classification = self._classification_metrics(inputs["y_test_direction"], test_pred_direction)
                profit_classification = self._binary_classification_metrics(
                    inputs["y_test_profit"],
                    test_pred_profit,
                    test_profit_positive_prob,
                )
                regression = self._regression_metrics(inputs["y_test_return"], test_pred_return)
                trading = self._trading_metrics(
                    test_pred_direction,
                    inputs["y_test_return"],
                    resolved_horizons[horizon],
                )

                latency_start = time.perf_counter()
                trend_model.predict(inputs["X_test"])
                profit_model.predict(inputs["X_test"])
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
                profit_path = artifact_path(
                    self._model_dir,
                    ticker=ticker,
                    task="profit",
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
                profit_model.save(profit_path)
                return_model.save(return_path)

                algorithm_manifest = {
                    "artifact_type": self._artifact_type(algorithm),
                    "sequence_length": sequence_length if use_sequence else None,
                    "feature_columns": algorithm_feature_columns,
                    "trend_model_file": trend_path.name,
                    "profit_model_file": profit_path.name,
                    "return_model_file": return_path.name,
                    "calibration": calibration,
                    "metrics": {
                        **classification,
                        **{f"profit_{key}": value for key, value in profit_classification.items()},
                        **regression,
                        **trading,
                        "train_seconds": train_seconds,
                        "inference_latency_ms": inference_latency_ms,
                    },
                }
                
                # Forward static risk configs into manifest defaults if risk is enabled at training time
                if normalized_risk_config.get("risk_enabled") or self._advanced_features_enabled(normalized_risk_config):
                    # Choose residual_std if test set is large enough, else fallback to validation rmse/rmse.
                    vol_val = regression.get("residual_std", regression.get("rmse", 0.05))
                    vol_src = regression.get("volatility_proxy_source", "validation_rmse")
                    
                    algorithm_manifest["risk_config"] = {
                        "risk_enabled": bool(normalized_risk_config.get("risk_enabled", False)),
                        "risk_engine_enabled": bool(normalized_risk_config.get("enable_risk_engine", False)),
                        "covar_enabled": bool(normalized_risk_config.get("enable_covar", False)),
                        "regime_detection_enabled": bool(normalized_risk_config.get("enable_regime_detection", False)),
                        "regime_switching_enabled": bool(normalized_risk_config.get("enable_regime_switching", False)),
                        "risk_allocation_enabled": bool(normalized_risk_config.get("enable_risk_allocation", False)),
                        "covar_quantile": float(normalized_risk_config.get("covar_quantile", 0.05)),
                        "covar_window": int(normalized_risk_config.get("covar_window", 60)),
                        "regime_method": normalized_risk_config.get("regime_method", "threshold"),
                        "risk_penalty_strength": float(normalized_risk_config.get("risk_penalty_strength", 1.0)),
                        "high_vol_exposure_cut": float(normalized_risk_config.get("high_vol_exposure_cut", 0.6)),
                        "crisis_exposure_cut": float(normalized_risk_config.get("crisis_exposure_cut", 0.25)),
                        "high_vol_threshold": float(normalized_risk_config.get("high_vol_threshold", 0.03)),
                        "crisis_drawdown_threshold": float(normalized_risk_config.get("crisis_drawdown_threshold", -0.12)),
                        "crisis_delta_covar_threshold": float(normalized_risk_config.get("crisis_delta_covar_threshold", 0.015)),
                        "risk_simulations": normalized_risk_config.get("simulations", 10000),
                        "risk_confidence_levels": normalized_risk_config.get("confidence_levels", [0.95, 0.99]),
                        "risk_seed": normalized_risk_config.get("random_seed", 42),
                        "volatility_proxy": float(vol_val),
                        "volatility_proxy_source": vol_src,
                        "risk_assumptions": "Normal distribution of residuals around forecast mean",
                    }
                else:
                    algorithm_manifest["risk_config"] = {"risk_enabled": False}

                manifest["horizons"][horizon]["algorithms"][algorithm] = algorithm_manifest
                report_rows.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "horizon_days": resolved_horizons[horizon],
                        "algorithm": algorithm,
                        "artifact_type": self._artifact_type(algorithm),
                        "sequence_length": sequence_length if use_sequence else "",
                        "feature_columns": len(algorithm_feature_columns),
                        "data_start": prepared.data_start,
                        "data_end": prepared.data_end,
                        "raw_rows": prepared.raw_stats["raw_rows"],
                        "indicator_warmup_rows": prepared.raw_stats["indicator_warmup_rows"],
                        "target_rows_lost": problem["target_rows_lost"],
                        "sequence_rows_lost": rows_lost_to_sequence,
                        "final_usable_rows": int(len(inputs["X_train"]) + len(inputs["X_val"]) + len(inputs["X_test"])),
                        "risk_engine_enabled": bool(normalized_risk_config.get("enable_risk_engine", False)),
                        "regime_switching_enabled": bool(normalized_risk_config.get("enable_regime_switching", False)),
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
                    profit_f1=profit_classification["f1"],
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

    @staticmethod
    def _contiguous_index_window(mask: pd.Series | np.ndarray) -> tuple[int, int]:
        positions = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(positions) == 0:
            raise ValueError("The requested date split produced no usable labeled rows")
        expected = np.arange(positions[0], positions[-1] + 1)
        if not np.array_equal(positions, expected):
            raise ValueError("The requested date split must map to a contiguous time window")
        return int(positions[0]), int(positions[-1] + 1)

    @staticmethod
    def _validation_start_index(
        start_index: int,
        stop_index: int,
        validation_fraction: float,
        validation_min_rows: int,
        min_train_rows: int,
    ) -> int:
        candidate_rows = stop_index - start_index
        if candidate_rows < (min_train_rows + validation_min_rows):
            raise ValueError(
                "Insufficient labeled rows for explicit training split: "
                f"need at least {min_train_rows + validation_min_rows}, got {candidate_rows}"
            )
        proposed_val = max(int(np.ceil(candidate_rows * validation_fraction)), validation_min_rows)
        proposed_val = min(proposed_val, candidate_rows - min_train_rows)
        return int(stop_index - proposed_val)

    def train_explicit_split(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        train_start: Any,
        train_end: Any,
        algorithms: list[str] | tuple[str, ...] | None = None,
        primary_algorithm: str | None = None,
        horizon_name: str = "daily",
        horizon_days: int = 1,
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
        tune_boosters: bool = False,
        risk_config: dict[str, Any] | None = None,
        context_sources: dict[str, pd.DataFrame | None] | None = None,
        validation_fraction: float = 0.15,
        validation_min_rows: int = 20,
        min_train_rows: int = 60,
        transaction_fee_bps: float = 15.0,
        slippage_bps: float = 20.0,
    ) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        normalized_risk_config = self._normalize_risk_config(risk_config)
        algorithms = self._normalize_algorithms(algorithms)
        primary_algorithm = (primary_algorithm or algorithms[0]).lower()
        if primary_algorithm not in algorithms:
            raise ValueError("primary_algorithm must be one of the requested algorithms")

        horizon_key = str(horizon_name).strip().lower()
        resolved_horizons = self._resolve_horizon_day_map([horizon_key], {horizon_key: int(horizon_days)})
        max_sequence = sequence_length if any(algo in SEQUENCE_ALGORITHMS for algo in algorithms) else 1

        prepared = self.prepare_ticker_data(
            ticker=ticker,
            df=df,
            max_sequence_length=max_sequence,
            context_sources=context_sources,
            risk_config=normalized_risk_config,
            window_start=train_start,
            window_end=train_end,
        )
        labeled_dataset = self._add_targets(
            prepared.feature_frame,
            resolved_horizons,
            transaction_fee_bps=transaction_fee_bps,
            slippage_bps=slippage_bps,
        )

        target_date_col = f"target_date_{horizon_key}"
        train_start_ts = pd.Timestamp(train_start).normalize()
        train_end_ts = pd.Timestamp(train_end).normalize()

        if clean:
            cleanup_ticker_dir(self._model_dir, ticker)
        ensure_ticker_dir(self._model_dir, ticker)

        feature_frame = labeled_dataset.dropna(
            subset=[
                f"target_direction_{horizon_key}",
                f"target_return_{horizon_key}",
                f"target_profit_label_{horizon_key}",
                target_date_col,
            ]
        ).reset_index(drop=True)
        target_dates = pd.to_datetime(feature_frame[target_date_col], errors="coerce").dt.normalize()
        candidate_mask = (target_dates >= train_start_ts) & (target_dates <= train_end_ts)
        candidate_start, candidate_stop = self._contiguous_index_window(candidate_mask)
        validation_start = self._validation_start_index(
            candidate_start,
            candidate_stop,
            validation_fraction=validation_fraction,
            validation_min_rows=validation_min_rows,
            min_train_rows=min_train_rows,
        )
        all_dates = pd.to_datetime(feature_frame["date"]).dt.normalize()

        report_rows: list[dict[str, Any]] = []
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "target_type": "forward_return",
            "feature_columns": prepared.feature_columns,
            "base_feature_columns": prepared.base_feature_columns,
            "data_window": {
                "start": prepared.data_start,
                "end": prepared.data_end,
            },
            "raw_stats": prepared.raw_stats,
            "advanced_risk": prepared.advanced_config,
            "risk_summary": prepared.risk_summary or {},
            "regime_distribution": prepared.regime_distribution or {},
            "covar_config": {
                "enabled": bool(
                    normalized_risk_config.get("enable_covar")
                    or normalized_risk_config.get("enable_risk_engine")
                ),
                "quantile": normalized_risk_config.get("covar_quantile"),
                "window": normalized_risk_config.get("covar_window"),
            },
            "profit_target_config": {
                "transaction_fee_bps": float(transaction_fee_bps),
                "slippage_bps": float(slippage_bps),
                "entry_convention": "next_tradable_open",
                "exit_convention": "target_date_close",
                "label_definition": "1 if net trade return after costs > 0 else 0",
            },
            "split_config": {
                "mode": "explicit_date_window",
                "train_start": str(train_start_ts.date()),
                "train_end": str(train_end_ts.date()),
                "validation_fraction": float(validation_fraction),
                "validation_min_rows": int(validation_min_rows),
            },
            "horizons": {
                horizon_key: {
                    "days": resolved_horizons[horizon_key],
                    "target_rows_lost": resolved_horizons[horizon_key],
                    "labeled_rows": int(candidate_stop - candidate_start),
                    "algorithms": {},
                }
            },
        }
        for algorithm in algorithms:
            algorithm_feature_columns = self._select_algorithm_feature_columns(
                prepared,
                algorithm,
                normalized_risk_config,
            )
            tabular_features = feature_frame[algorithm_feature_columns].copy()
            X_all = tabular_features.to_numpy(dtype=float)
            y_direction = feature_frame[f"target_direction_{horizon_key}"].astype(float).to_numpy()
            y_return = feature_frame[f"target_return_{horizon_key}"].astype(float).to_numpy()
            y_profit = feature_frame[f"target_profit_label_{horizon_key}"].astype(float).to_numpy()

            train_direction = y_direction[candidate_start:validation_start]
            if len(np.unique(train_direction.astype(int))) < 2:
                logger.warning(
                    "skipping_algorithm",
                    ticker=ticker,
                    algorithm=algorithm,
                    horizon=horizon_key,
                    reason="one_class_training_target",
                )
                continue
            train_profit = y_profit[candidate_start:validation_start]
            if len(np.unique(train_profit.astype(int))) < 2:
                logger.warning(
                    "skipping_algorithm",
                    ticker=ticker,
                    algorithm=algorithm,
                    horizon=horizon_key,
                    reason="one_class_profit_target",
                )
                continue

            use_sequence = algorithm in SEQUENCE_ALGORITHMS
            if use_sequence:
                direction_sequences = create_sequence_dataset(
                    feature_frame[algorithm_feature_columns],
                    y_direction.astype(int),
                    sequence_length=sequence_length,
                    feature_columns=algorithm_feature_columns,
                )
                profit_sequences = create_sequence_dataset(
                    feature_frame[algorithm_feature_columns],
                    y_profit.astype(int),
                    sequence_length=sequence_length,
                    feature_columns=algorithm_feature_columns,
                )
                return_sequences = create_sequence_dataset(
                    feature_frame[algorithm_feature_columns],
                    y_return,
                    sequence_length=sequence_length,
                    feature_columns=algorithm_feature_columns,
                )
                seq_train_direction = select_sequence_range(
                    direction_sequences,
                    start_index=candidate_start,
                    stop_index=validation_start,
                )
                seq_val_direction = select_sequence_range(
                    direction_sequences,
                    start_index=validation_start,
                    stop_index=candidate_stop,
                )
                seq_train_profit = select_sequence_range(
                    profit_sequences,
                    start_index=candidate_start,
                    stop_index=validation_start,
                )
                seq_val_profit = select_sequence_range(
                    profit_sequences,
                    start_index=validation_start,
                    stop_index=candidate_stop,
                )
                seq_train_return = select_sequence_range(
                    return_sequences,
                    start_index=candidate_start,
                    stop_index=validation_start,
                )
                seq_val_return = select_sequence_range(
                    return_sequences,
                    start_index=validation_start,
                    stop_index=candidate_stop,
                )
                inputs = {
                    "X_train": seq_train_direction.X,
                    "X_val": seq_val_direction.X,
                    "y_train_direction": seq_train_direction.y.astype(int),
                    "y_val_direction": seq_val_direction.y.astype(int),
                    "y_train_profit": seq_train_profit.y.astype(int),
                    "y_val_profit": seq_val_profit.y.astype(int),
                    "y_train_return": seq_train_return.y.astype(float),
                    "y_val_return": seq_val_return.y.astype(float),
                    "rows_lost": int(direction_sequences.rows_lost),
                }
            else:
                x_train = X_all[candidate_start:validation_start]
                x_val = X_all[validation_start:candidate_stop]
                if algorithm in BOOSTER_ALGORITHMS:
                    x_train = tabular_features.iloc[candidate_start:validation_start].copy()
                    x_val = tabular_features.iloc[validation_start:candidate_stop].copy()
                inputs = {
                    "X_train": x_train,
                    "X_val": x_val,
                    "y_train_direction": y_direction[candidate_start:validation_start].astype(int),
                    "y_val_direction": y_direction[validation_start:candidate_stop].astype(int),
                    "y_train_profit": y_profit[candidate_start:validation_start].astype(int),
                    "y_val_profit": y_profit[validation_start:candidate_stop].astype(int),
                    "y_train_return": y_return[candidate_start:validation_start].astype(float),
                    "y_val_return": y_return[validation_start:candidate_stop].astype(float),
                    "rows_lost": 0,
                }

            if len(inputs["X_train"]) == 0 or len(inputs["X_val"]) == 0:
                logger.warning(
                    "skipping_algorithm",
                    ticker=ticker,
                    algorithm=algorithm,
                    horizon=horizon_key,
                    reason="insufficient_explicit_split_rows",
                )
                continue

            trend_params = self._model_params(
                algorithm,
                "classification",
                sequence_length,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                batch_size,
                epochs,
                patience,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                criterion,
                tune_boosters,
            )
            profit_params = self._model_params(
                algorithm,
                "classification",
                sequence_length,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                batch_size,
                epochs,
                patience,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                criterion,
                tune_boosters,
            )
            return_params = self._model_params(
                algorithm,
                "regression",
                sequence_length,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                batch_size,
                epochs,
                patience,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                criterion,
                tune_boosters,
            )
            trend_model = create_model(algorithm, **trend_params)
            profit_model = create_model(algorithm, **profit_params)
            return_model = create_model(algorithm, **return_params)

            train_clock = time.perf_counter()
            trend_model.fit(
                inputs["X_train"],
                inputs["y_train_direction"],
                inputs["X_val"],
                inputs["y_val_direction"],
            )
            profit_model.fit(
                inputs["X_train"],
                inputs["y_train_profit"],
                inputs["X_val"],
                inputs["y_val_profit"],
            )
            return_model.fit(
                inputs["X_train"],
                inputs["y_train_return"],
                inputs["X_val"],
                inputs["y_val_return"],
            )
            train_seconds = float(time.perf_counter() - train_clock)

            val_pred_direction = np.asarray(trend_model.predict(inputs["X_val"]), dtype=int)
            val_pred_profit = np.asarray(profit_model.predict(inputs["X_val"]), dtype=int)
            val_profit_probs = np.asarray(profit_model.predict_proba(inputs["X_val"]), dtype=float)
            if val_profit_probs.ndim == 2 and val_profit_probs.shape[1] > 1:
                val_profit_positive_prob = val_profit_probs[:, 1]
            else:
                val_profit_positive_prob = np.asarray(val_profit_probs).reshape(-1)
            val_pred_return = np.asarray(return_model.predict(inputs["X_val"]), dtype=float).reshape(-1)
            classification = self._classification_metrics(inputs["y_val_direction"], val_pred_direction)
            profit_classification = self._binary_classification_metrics(
                inputs["y_val_profit"],
                val_pred_profit,
                val_profit_positive_prob,
            )
            regression = self._regression_metrics(inputs["y_val_return"], val_pred_return)
            trading = self._trading_metrics(
                val_pred_direction,
                inputs["y_val_return"],
                resolved_horizons[horizon_key],
            )
            latency_start = time.perf_counter()
            trend_model.predict(inputs["X_val"])
            profit_model.predict(inputs["X_val"])
            return_model.predict(inputs["X_val"])
            inference_latency_ms = float(
                ((time.perf_counter() - latency_start) * 1000.0) / max(len(inputs["X_val"]), 1)
            )
            calibration = self._build_calibration(
                return_model,
                inputs["X_val"],
                inputs["y_val_return"],
            )

            trend_path = artifact_path(
                self._model_dir,
                ticker=ticker,
                task="trend",
                algorithm=algorithm,
                horizon=horizon_key,
            )
            profit_path = artifact_path(
                self._model_dir,
                ticker=ticker,
                task="profit",
                algorithm=algorithm,
                horizon=horizon_key,
            )
            return_path = artifact_path(
                self._model_dir,
                ticker=ticker,
                task="return",
                algorithm=algorithm,
                horizon=horizon_key,
            )
            trend_model.save(trend_path)
            profit_model.save(profit_path)
            return_model.save(return_path)

            algorithm_manifest = {
                "artifact_type": self._artifact_type(algorithm),
                "sequence_length": sequence_length if use_sequence else None,
                "feature_columns": algorithm_feature_columns,
                "trend_model_file": trend_path.name,
                "profit_model_file": profit_path.name,
                "return_model_file": return_path.name,
                "calibration": calibration,
                "metrics": {
                    **classification,
                    **{f"profit_{key}": value for key, value in profit_classification.items()},
                    **regression,
                    **trading,
                    "train_seconds": train_seconds,
                    "inference_latency_ms": inference_latency_ms,
                },
            }
            if normalized_risk_config.get("risk_enabled") or self._advanced_features_enabled(normalized_risk_config):
                vol_val = regression.get("residual_std", regression.get("rmse", 0.05))
                vol_src = regression.get("volatility_proxy_source", "validation_rmse")
                algorithm_manifest["risk_config"] = {
                    "risk_enabled": bool(normalized_risk_config.get("risk_enabled", False)),
                    "risk_engine_enabled": bool(normalized_risk_config.get("enable_risk_engine", False)),
                    "covar_enabled": bool(normalized_risk_config.get("enable_covar", False)),
                    "regime_detection_enabled": bool(normalized_risk_config.get("enable_regime_detection", False)),
                    "regime_switching_enabled": bool(normalized_risk_config.get("enable_regime_switching", False)),
                    "risk_allocation_enabled": bool(normalized_risk_config.get("enable_risk_allocation", False)),
                    "covar_quantile": float(normalized_risk_config.get("covar_quantile", 0.05)),
                    "covar_window": int(normalized_risk_config.get("covar_window", 60)),
                    "regime_method": normalized_risk_config.get("regime_method", "threshold"),
                    "risk_penalty_strength": float(normalized_risk_config.get("risk_penalty_strength", 1.0)),
                    "high_vol_exposure_cut": float(normalized_risk_config.get("high_vol_exposure_cut", 0.6)),
                    "crisis_exposure_cut": float(normalized_risk_config.get("crisis_exposure_cut", 0.25)),
                    "high_vol_threshold": float(normalized_risk_config.get("high_vol_threshold", 0.03)),
                    "crisis_drawdown_threshold": float(normalized_risk_config.get("crisis_drawdown_threshold", -0.12)),
                    "crisis_delta_covar_threshold": float(normalized_risk_config.get("crisis_delta_covar_threshold", 0.015)),
                    "risk_simulations": normalized_risk_config.get("simulations", 10000),
                    "risk_confidence_levels": normalized_risk_config.get("confidence_levels", [0.95, 0.99]),
                    "risk_seed": normalized_risk_config.get("random_seed", 42),
                    "volatility_proxy": float(vol_val),
                    "volatility_proxy_source": vol_src,
                    "risk_assumptions": "Normal distribution of residuals around forecast mean",
                }
            else:
                algorithm_manifest["risk_config"] = {"risk_enabled": False}

            manifest["horizons"][horizon_key]["algorithms"][algorithm] = algorithm_manifest
            report_rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon_key,
                    "horizon_days": resolved_horizons[horizon_key],
                    "algorithm": algorithm,
                    "artifact_type": self._artifact_type(algorithm),
                    "sequence_length": sequence_length if use_sequence else "",
                    "feature_columns": len(algorithm_feature_columns),
                    "data_start": prepared.data_start,
                    "data_end": prepared.data_end,
                    "raw_rows": prepared.raw_stats["raw_rows"],
                    "indicator_warmup_rows": prepared.raw_stats["indicator_warmup_rows"],
                    "target_rows_lost": resolved_horizons[horizon_key],
                    "sequence_rows_lost": int(inputs.get("rows_lost", 0)),
                    "final_usable_rows": int(len(inputs["X_train"]) + len(inputs["X_val"])),
                    "train_window_start": str(pd.Timestamp(all_dates.iloc[candidate_start]).date()),
                    "train_window_end": str(pd.Timestamp(all_dates.iloc[validation_start - 1]).date()),
                    "validation_window_start": str(pd.Timestamp(all_dates.iloc[validation_start]).date()),
                    "validation_window_end": str(pd.Timestamp(all_dates.iloc[candidate_stop - 1]).date()),
                    "risk_engine_enabled": bool(normalized_risk_config.get("enable_risk_engine", False)),
                    "regime_switching_enabled": bool(normalized_risk_config.get("enable_regime_switching", False)),
                    **algorithm_manifest["metrics"],
                }
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
            "horizon_name": horizon_key,
            "horizon_days": resolved_horizons[horizon_key],
            "split_config": manifest["split_config"],
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
        file_key_map = {
            "trend": "trend_model_file",
            "return": "return_model_file",
            "profit": "profit_model_file",
        }
        if task not in file_key_map:
            raise ValueError(f"Unsupported model task '{task}'")
        file_key = file_key_map[task]
        if file_key not in algorithm_info:
            raise FileNotFoundError(
                f"No artifact file registered for task '{task}' on {ticker_key} algorithm '{algorithm}' horizon '{horizon}'"
            )
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
    def _binary_probabilities(probs: np.ndarray, *, negative_label: str, positive_label: str) -> dict[str, float]:
        negative = float(probs[0]) if len(probs) > 0 else 0.0
        positive = float(probs[1]) if len(probs) > 1 else 0.0
        total = negative + positive
        if total <= 0.0:
            return {negative_label: 0.0, positive_label: 0.0}
        return {
            negative_label: round(negative / total, 4),
            positive_label: round(positive / total, 4),
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
        risk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        horizon = horizon.lower()
        self._ensure_models_loaded(ticker)
        manifest = self._manifests[ticker]
        available_horizons = manifest.get("horizons", {})
        if horizon not in available_horizons:
            if horizon in {"1d"} and "daily" in available_horizons:
                horizon = "daily"
            elif horizon in {"1w", "5d"} and "short" in available_horizons:
                horizon = "short"
            elif horizon in {"1m", "20d"} and "mid" in available_horizons:
                horizon = "mid"
            elif horizon in {"6m", "120d"} and "long" in available_horizons:
                horizon = "long"
        if horizon not in manifest.get("horizons", {}) and horizon not in HORIZON_DAYS:
            raise ValueError(f"Unsupported horizon '{horizon}'")
        algorithm = (algorithm or manifest.get("primary_algorithm")).lower()
        horizon_info = manifest.get("horizons", {}).get(horizon)
        if horizon_info is None:
            raise FileNotFoundError(f"No trained horizon '{horizon}' found for {ticker}")
        horizon_days = int(horizon_info.get("days", HORIZON_DAYS.get(horizon, 1)))
        algorithm_info = horizon_info.get("algorithms", {}).get(algorithm)
        if algorithm_info is None:
            raise FileNotFoundError(f"No trained algorithm '{algorithm}' found for {ticker} horizon '{horizon}'")

        feature_columns = algorithm_info.get("feature_columns", manifest.get("feature_columns", []))
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
        elif algorithm in BOOSTER_ALGORITHMS:
            model_input = feature_frame[feature_columns].iloc[[-1]].copy()
        else:
            model_input = feature_frame[feature_columns].iloc[[-1]].to_numpy(dtype=float)

        trend_model = self._get_loaded_model(ticker, algorithm, horizon, "trend")
        return_model = self._get_loaded_model(ticker, algorithm, horizon, "return")
        trend_probs = self._trend_probabilities_from_binary(trend_model.predict_proba(model_input)[0])
        predicted_return = float(np.asarray(return_model.predict(model_input)).reshape(-1)[0])
        predicted_direction = int(trend_probs["up"] >= trend_probs["down"])
        expected_range = self._expected_range(current_close, predicted_return, algorithm_info.get("calibration", {}))

        output = {
            "algorithm": algorithm,
            "artifact_type": algorithm_info.get("artifact_type"),
            "horizon": horizon,
            "horizon_days": horizon_days,
            "sequence_length": algorithm_info.get("sequence_length"),
            "predicted_return": predicted_return,
            "predicted_direction": predicted_direction,
            "trend_probabilities": trend_probs,
            "expected_range": expected_range,
            "feature_set_version": f"ml_schema_v{ARTIFACT_SCHEMA_VERSION}",
        }

        if "profit_model_file" in algorithm_info:
            profit_model = self._get_loaded_model(ticker, algorithm, horizon, "profit")
            raw_profit_probs = np.asarray(profit_model.predict_proba(model_input)[0], dtype=float)
            profit_probabilities = self._binary_probabilities(
                raw_profit_probs,
                negative_label="loss_or_flat",
                positive_label="profit",
            )
            predicted_profit_probability = float(raw_profit_probs[1]) if len(raw_profit_probs) > 1 else 0.0
            predicted_profit_label = int(
                np.asarray(profit_model.predict(model_input), dtype=int).reshape(-1)[0]
            )
            output.update(
                {
                    "predicted_profit_label": predicted_profit_label,
                    "predicted_profit_probability": predicted_profit_probability,
                    "profit_probabilities": profit_probabilities,
                }
            )

        manifest_risk_config = algorithm_info.get("risk_config", {})
        override_risk_config = self._normalize_risk_config(risk_config) if risk_config else {}
        advanced_risk = manifest.get("advanced_risk", {})

        if advanced_risk.get("enable_risk_engine") or advanced_risk.get("enable_covar"):
            latest_risk = {}
            for column in RISK_FEATURE_COLUMNS:
                if column in feature_frame.columns:
                    non_na = pd.to_numeric(feature_frame[column], errors="coerce").dropna()
                    latest_risk[column] = None if non_na.empty else float(non_na.iloc[-1])
            output["risk_summary"] = {
                "asset": latest_risk,
                "system": manifest.get("risk_summary", {}).get("system", {}),
            }

        if advanced_risk.get("enable_regime_detection") or advanced_risk.get("enable_regime_switching"):
            regime_series = (
                pd.to_numeric(feature_frame["regime_label"], errors="coerce").dropna()
                if "regime_label" in feature_frame.columns
                else pd.Series(dtype=float)
            )
            probability_series = (
                pd.to_numeric(feature_frame["regime_probability"], errors="coerce").dropna()
                if "regime_probability" in feature_frame.columns
                else pd.Series(dtype=float)
            )
            regime_value = regime_series.iloc[-1] if not regime_series.empty else np.nan
            probability_value = probability_series.iloc[-1] if not probability_series.empty else np.nan
            regime_name = None
            if pd.notna(regime_value):
                reverse_map = {value: key for key, value in REGIME_TO_CODE.items()}
                regime_name = reverse_map.get(int(regime_value))
            output["regime"] = {
                "label": regime_name,
                "encoded": None if pd.isna(regime_value) else int(regime_value),
                "probability": None if pd.isna(probability_value) else float(probability_value),
            }

        if advanced_risk.get("enable_risk_allocation"):
            risk_snapshot = pd.DataFrame(
                [
                    {
                        column: float(pd.to_numeric(feature_frame[column], errors="coerce").dropna().iloc[-1])
                        if column in feature_frame.columns and not pd.to_numeric(feature_frame[column], errors="coerce").dropna().empty
                        else 0.0
                        for column in RISK_FEATURE_COLUMNS
                    }
                ],
                index=[ticker],
            )
            regime_labels = None
            if "regime" in output and output["regime"]["label"] is not None:
                regime_labels = pd.Series({ticker: output["regime"]["label"]})
            allocator = RiskAwareAllocator(
                risk_penalty_strength=float(advanced_risk.get("risk_penalty_strength", 1.0)),
                high_vol_exposure_cut=float(advanced_risk.get("high_vol_exposure_cut", 0.6)),
                crisis_exposure_cut=float(advanced_risk.get("crisis_exposure_cut", 0.25)),
            )
            output["allocation"] = allocator.allocate(
                risk_frame=risk_snapshot,
                regime_labels=regime_labels,
                base_weights=pd.Series({ticker: 1.0}),
            ).to_dict()
        
        # If execution overrides risk config, merge over the manifest default
        has_risk = bool(override_risk_config.get("risk_enabled")) or manifest_risk_config.get("risk_enabled", False)
        if has_risk:
            # Overrides take precedence
            eval_config = manifest_risk_config.copy()
            if risk_config:
                eval_config.update(override_risk_config)
                
            from src.ml.risk import MonteCarloRiskSimulator
            
            simulator = MonteCarloRiskSimulator(
                simulations=eval_config.get("risk_simulations", eval_config.get("simulations", 10000)),
                random_seed=eval_config.get("risk_seed", eval_config.get("random_seed", 42))
            )
            
            volatility_proxy = eval_config.get("volatility_proxy", 0.05)
            if volatility_proxy <= 0:
                volatility_proxy = 0.05

            risk_assessment = simulator.simulate_risk(
                forecast_mean=predicted_return,
                volatility_proxy=float(volatility_proxy),
                horizon=horizon,
                confidence_levels=eval_config.get("risk_confidence_levels", [0.95, 0.99])
            )
            # Annotate with the source
            risk_assessment["metadata"]["volatility_proxy_source"] = eval_config.get("volatility_proxy_source", "validation_rmse")
            
            output["risk_assessment"] = risk_assessment
            
        return output
