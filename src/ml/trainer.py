"""Unified ML training and inference facade for technical stock models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.ml.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_CREATED_BY,
    MANIFEST_COMPATIBILITY_VERSION,
    artifact_path,
    cleanup_ticker_dir,
    ensure_ticker_dir,
    load_manifest,
    write_manifest,
)
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.data_loader import (
    apply_context_features,
    load_foreign_flow,
    load_macro_context,
    load_market_breadth,
    load_market_proxy,
    load_sector_proxies,
    load_sentiment,
    load_ticker_sectors,
)
from src.ml.feature_engineering import FeatureEngineer
from src.ml.features.registry import (
    FEATURE_REGISTRY_PATH,
    approved_feature_sets,
    feature_selection_evidence,
    final_task_feature_sets,
    price_reference_semantics,
    resolve_feature_set,
    resolve_task_feature_set,
    sentiment_policy,
)
from src.ml.metrics import (
    compute_binary_classification_metrics,
    compute_prediction_error_metrics,
    summarize_binary_probability_calibration,
    summarize_regression_residual_diagnostics,
)
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

FOREIGN_FLOW_DISABLED_REASON = "foreign-flow context intentionally disabled"


def _disabled_foreign_flow_frame() -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["foreign_flow_context_mode"] = "disabled"
    frame.attrs["foreign_flow_coverage_status"] = "disabled"
    frame.attrs["source_name"] = "disabled"
    frame.attrs["source_provenance"] = "disabled"
    frame.attrs["disabled_reason"] = FOREIGN_FLOW_DISABLED_REASON
    return frame

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
SCENARIO_RISK_OUTPUT_FIELD = "heuristic_scenario_risk"
SCENARIO_RISK_LEGACY_ALIAS = "risk_assessment"
SCENARIO_RISK_MODEL_TYPE = "residual_normal_scenario_simulation"
SCENARIO_RISK_ASSUMPTION = (
    "Residual-based normal scenarios around the point forecast; not calibrated forecast confidence."
)
SCENARIO_RISK_UNCERTAINTY_METHODOLOGY = (
    "residual_based_normal_scenario_simulation_using_point_forecast_and_validation_error_scale"
)
SCENARIO_RISK_INTERPRETATION_WARNING = (
    "Scenario VaR/CVaR values are heuristic tail summaries from residual-based simulated returns. "
    "They are not calibrated confidence intervals or guaranteed loss bounds."
)


@dataclass(frozen=True)
class PreparedTickerData:
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    base_feature_columns: list[str]
    feature_build_mode: str
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
    def _load_context_sources(
        self,
        *,
        include_sentiment: bool = False,
        require_validated_sentiment: bool = True,
        foreign_flow_path: str | Path | None = None,
        foreign_flow_mode: str = "auto",
    ) -> dict[str, Any]:
        mode = str(foreign_flow_mode or "auto").strip().lower()
        if mode not in {"auto", "path", "disabled"}:
            raise ValueError(f"Unsupported foreign_flow_mode: {foreign_flow_mode}")
        explicit_foreign_flow_path = Path(foreign_flow_path) if foreign_flow_path is not None else None
        if mode == "path" and explicit_foreign_flow_path is None:
            raise ValueError("foreign_flow_mode='path' requires foreign_flow_path")
        if mode == "disabled" and explicit_foreign_flow_path is not None:
            raise ValueError("foreign_flow_mode='disabled' cannot be combined with foreign_flow_path")
        if explicit_foreign_flow_path is not None and not explicit_foreign_flow_path.exists():
            raise FileNotFoundError(f"Configured foreign_flow_path does not exist: {explicit_foreign_flow_path}")

        if self._context_cache is None or explicit_foreign_flow_path is not None or mode == "disabled":
            context_cache = {
                "market_df": load_market_proxy(),
                "sector_df": load_sector_proxies(),
                "ticker_sectors": load_ticker_sectors(),
                "breadth_df": load_market_breadth(),
                "macro_df": load_macro_context(),
                "foreign_flow_df": (
                    _disabled_foreign_flow_frame()
                    if mode == "disabled"
                    else (
                        load_foreign_flow(path=explicit_foreign_flow_path)
                        if explicit_foreign_flow_path is not None
                        else load_foreign_flow()
                    )
                ),
            }
            if explicit_foreign_flow_path is None and mode != "disabled":
                self._context_cache = context_cache
        else:
            context_cache = self._context_cache
        context_sources = dict(context_cache)
        context_sources["_foreign_flow_path_explicit"] = explicit_foreign_flow_path is not None
        context_sources["_foreign_flow_path"] = str(explicit_foreign_flow_path) if explicit_foreign_flow_path is not None else None
        context_sources["_foreign_flow_mode"] = mode
        context_sources["sentiment_df"] = load_sentiment(
            enabled=include_sentiment,
            require_validated_source=require_validated_sentiment,
        )
        return context_sources

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
        context_sources: dict[str, Any],
    ) -> pd.DataFrame:
        foreign_flow_df = context_sources.get("foreign_flow_df")
        foreign_flow_mode = str(context_sources.get("_foreign_flow_mode", "auto")).strip().lower()
        explicit_foreign_flow_path = bool(context_sources.get("_foreign_flow_path_explicit", False))
        if foreign_flow_mode == "disabled":
            foreign_flow_df = foreign_flow_df if isinstance(foreign_flow_df, pd.DataFrame) else _disabled_foreign_flow_frame()
        elif (foreign_flow_df is None or foreign_flow_df.empty) and not explicit_foreign_flow_path:
            foreign_flow_df = load_foreign_flow(
                tickers=[ticker],
                start_date=pd.to_datetime(df["date"]).min().date() if "date" in df.columns and not df.empty else None,
                end_date=pd.to_datetime(df["date"]).max().date() if "date" in df.columns and not df.empty else None,
            )
        elif foreign_flow_df is None:
            foreign_flow_df = pd.DataFrame()
        return apply_context_features(
            df,
            ticker,
            market_df=context_sources.get("market_df"),
            sector_df=context_sources.get("sector_df"),
            ticker_sectors=context_sources.get("ticker_sectors"),
            breadth_df=context_sources.get("breadth_df"),
            foreign_flow_df=foreign_flow_df,
            macro_df=context_sources.get("macro_df"),
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

    @staticmethod
    def _window_bounds(target_dates: pd.Series, start_index: int, stop_index: int) -> dict[str, str] | None:
        if stop_index <= start_index:
            return None
        window = pd.to_datetime(target_dates.iloc[start_index:stop_index], errors="coerce").dropna()
        if window.empty:
            return None
        return {
            "start": str(pd.Timestamp(window.iloc[0]).date()),
            "end": str(pd.Timestamp(window.iloc[-1]).date()),
        }

    @staticmethod
    def _evaluation_metadata(
        *,
        evaluation_split_name: str,
        metric_source: str,
        validation_method: str,
        train_window: dict[str, str] | None = None,
        validation_window: dict[str, str] | None = None,
        test_window: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "evaluation_split_name": evaluation_split_name,
            "metric_source": metric_source,
            "validation_method": validation_method,
        }
        if train_window:
            metadata["train_window"] = train_window
        if validation_window:
            metadata["validation_window"] = validation_window
        if test_window:
            metadata["test_window"] = test_window
        return metadata

    @staticmethod
    def _target_definition_manifest(resolved_horizons: dict[str, int]) -> dict[str, Any]:
        return {
            "task_bundle": "trend_classification_profit_classification_forward_return_regression",
            "forward_return_formula": "close[target_date] / close[prediction_date] - 1",
            "trend_label_definition": "1 if forward return > 0 else 0",
            "profit_label_definition": "1 if net trade return after costs > 0 else 0",
            "prediction_date_definition": "target_date minus horizon trading days",
            "horizons": {str(name): int(days) for name, days in resolved_horizons.items()},
        }

    def _feature_generation_manifest(self, prepared: PreparedTickerData) -> dict[str, Any]:
        return {
            "engine": "src.ml.feature_engineering.FeatureEngineer",
            "feature_build_mode": prepared.feature_build_mode,
            "feature_build_modes": self._feature_engineer.build_mode_manifest(),
            "feature_columns": list(prepared.feature_columns),
            "base_feature_columns": list(prepared.base_feature_columns),
            "feature_registry_path": str(FEATURE_REGISTRY_PATH),
            "approved_feature_sets": approved_feature_sets(),
            "final_task_feature_sets": final_task_feature_sets(),
            "feature_selection_evidence": feature_selection_evidence(),
            "sentiment_policy": sentiment_policy(),
            "price_reference_semantics": price_reference_semantics(),
            "regime_definitions": self._feature_engineer.regime_definition_manifest(),
            "corporate_action_diagnostics": {
                **self._feature_engineer.corporate_action_diagnostic_manifest(),
                **dict(prepared.raw_stats.get("corporate_action_diagnostics", {})),
            },
            "shared_training_feature_policy": (
                "base_feature_columns remain the backward-compatible governed union. "
                "Active training resolves task-specific final_task_feature_sets "
                "for classification and regression inputs when available."
            ),
            "technical_indicator_dependency_behavior": "local_deterministic_numpy_pandas_computation",
            "technical_indicator_backend": "src.ml.feature_engineering",
            "advanced_risk_features_enabled": bool(
                prepared.advanced_config and DualModelTrainer._advanced_features_enabled(prepared.advanced_config)
            ),
        }

    @staticmethod
    def _training_backend_manifest(*, tune_boosters: bool) -> dict[str, Any]:
        return {
            "canonical_training_path": "src.ml.trainer.DualModelTrainer",
            "model_registry": "src.ml.models.factory",
            "artifact_writer": "src.ml.artifacts.write_manifest",
            "tuning_backend": "model_factory_booster_tuning" if tune_boosters else "model_factory_defaults_only",
            "authoritative_dependency_manifest": "pyproject.toml",
            "requirements_file_role": "core_runtime_compatibility",
        }

    @staticmethod
    def _prediction_output_semantics() -> dict[str, Any]:
        return {
            "risk_output_field": SCENARIO_RISK_OUTPUT_FIELD,
            "risk_output_aliases": [SCENARIO_RISK_LEGACY_ALIAS],
            "risk_semantics": "heuristic_scenario_risk_not_calibrated_confidence",
            "uncertainty_methodology": SCENARIO_RISK_UNCERTAINTY_METHODOLOGY,
            "calibration_status": "heuristic_not_calibrated",
            "interpretation_warning": SCENARIO_RISK_INTERPRETATION_WARNING,
            "deprecated_output_aliases": {
                SCENARIO_RISK_LEGACY_ALIAS: {
                    "alias_for": SCENARIO_RISK_OUTPUT_FIELD,
                    "status": "deprecated_backward_compat_alias",
                }
            },
        }

    @staticmethod
    def _scenario_risk_manifest(
        normalized_risk_config: dict[str, Any],
        regression: dict[str, float],
    ) -> dict[str, Any]:
        vol_val = regression.get("residual_std", regression.get("rmse", 0.05))
        vol_src = regression.get("volatility_proxy_source", "validation_rmse")
        return {
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
            "scenario_confidence_levels": normalized_risk_config.get("confidence_levels", [0.95, 0.99]),
            "risk_seed": normalized_risk_config.get("random_seed", 42),
            "volatility_proxy": float(vol_val),
            "volatility_proxy_source": vol_src,
            "risk_model_type": SCENARIO_RISK_MODEL_TYPE,
            "risk_output_field": SCENARIO_RISK_OUTPUT_FIELD,
            "calibration_status": "heuristic_not_calibrated",
            "risk_assumptions": SCENARIO_RISK_ASSUMPTION,
            "uncertainty_methodology": SCENARIO_RISK_UNCERTAINTY_METHODOLOGY,
            "interpretation_warning": SCENARIO_RISK_INTERPRETATION_WARNING,
            "deprecated_output_aliases": {
                SCENARIO_RISK_LEGACY_ALIAS: {
                    "alias_for": SCENARIO_RISK_OUTPUT_FIELD,
                    "status": "deprecated_backward_compat_alias",
                }
            },
        }

    @staticmethod
    def _predict_positive_class_probability(model: Any, X: np.ndarray | pd.DataFrame | None) -> np.ndarray | None:
        if X is None or len(X) == 0:
            return None
        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            return None
        try:
            probabilities = np.asarray(predict_proba(X), dtype=float)
        except Exception:
            return None
        if probabilities.size == 0:
            return None
        if probabilities.ndim == 1:
            return probabilities.reshape(-1)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]

    @staticmethod
    def _direction_probability_diagnostics(
        y_true: np.ndarray,
        y_prob: np.ndarray | None,
    ) -> dict[str, Any]:
        if y_prob is None:
            return {
                "available": False,
                "reason": "predict_proba_not_available_for_direction_model",
                "metric_scope": "direction_probability_calibration",
                "interpretation": "No directional probability calibration summary because the selected model does not expose predict_proba.",
            }
        diagnostics = summarize_binary_probability_calibration(y_true, y_prob)
        diagnostics["metric_scope"] = "direction_probability_calibration"
        return diagnostics

    @staticmethod
    def _regression_error_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        diagnostics = summarize_regression_residual_diagnostics(y_true, y_pred)
        diagnostics["metric_scope"] = "regression_error_distribution"
        return diagnostics

    @staticmethod
    def _normalize_inference_risk_override(risk_config: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize inference-time scenario-risk overrides without changing enablement implicitly."""
        if not risk_config:
            return {}

        incoming = risk_config.copy()
        normalized: dict[str, Any] = {}

        if "risk_enabled" in incoming:
            normalized["risk_enabled"] = bool(incoming["risk_enabled"])
        if "simulations" in incoming:
            normalized["risk_simulations"] = int(incoming["simulations"])
        if "risk_simulations" in incoming:
            normalized["risk_simulations"] = int(incoming["risk_simulations"])
        if "random_seed" in incoming:
            normalized["risk_seed"] = int(incoming["random_seed"])
        if "risk_seed" in incoming:
            normalized["risk_seed"] = int(incoming["risk_seed"])
        if "confidence_levels" in incoming:
            normalized["scenario_confidence_levels"] = list(incoming["confidence_levels"])
        if "risk_confidence_levels" in incoming:
            normalized["scenario_confidence_levels"] = list(incoming["risk_confidence_levels"])
        if "scenario_confidence_levels" in incoming:
            normalized["scenario_confidence_levels"] = list(incoming["scenario_confidence_levels"])

        for key in (
            "volatility_proxy",
            "volatility_proxy_source",
            "risk_model_type",
            "risk_output_field",
            "calibration_status",
        ):
            if key in incoming:
                normalized[key] = incoming[key]

        return normalized

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
        task_name: str | None = None,
    ) -> list[str]:
        task_fallbacks = {
            "regression_forecasting": "forecast_core_features",
            "directional_classification": "classification_signal_features",
            "regime_detection": "regime_features",
            "risk_layer": "risk_features",
        }
        if task_name:
            selected = resolve_task_feature_set(
                task_name,
                available_columns=prepared.feature_frame.columns,
            )
            if not selected:
                selected = resolve_feature_set(
                    task_fallbacks.get(task_name, "forecast_core_features"),
                    available_columns=prepared.feature_frame.columns,
                )
            if not selected:
                selected = list(prepared.base_feature_columns)
        else:
            selected = list(prepared.base_feature_columns)
        if algorithm in BOOSTER_ALGORITHMS and (
            advanced_config.get("enable_risk_engine")
            or advanced_config.get("enable_covar")
            or advanced_config.get("enable_risk_allocation")
        ):
            for column in resolve_feature_set(
                "risk_features",
                available_columns=prepared.feature_frame.columns,
            ):
                if column in prepared.feature_frame.columns and column not in selected:
                    selected.append(column)
        if algorithm in BOOSTER_ALGORITHMS and advanced_config.get("enable_regime_switching"):
            for column in resolve_feature_set(
                "regime_features",
                available_columns=prepared.feature_frame.columns,
            ):
                if column in prepared.feature_frame.columns and column not in selected:
                    selected.append(column)
        for column in RISK_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS:
            if column in prepared.feature_frame.columns and column not in selected:
                selected.append(column)
        return selected

    def prepare_ticker_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        *,
        max_sequence_length: int = 20,
        feature_build_mode: str = "full_research_mode",
        context_sources: dict[str, pd.DataFrame | None] | None = None,
        include_sentiment: bool = False,
        risk_config: dict[str, Any] | None = None,
        window_start: Any | None = None,
        window_end: Any | None = None,
    ) -> PreparedTickerData:
        if context_sources is None:
            context_sources = self._load_context_sources(
                include_sentiment=include_sentiment,
                require_validated_sentiment=True,
            )
        else:
            context_sources = dict(context_sources)
            if not include_sentiment:
                context_sources["sentiment_df"] = load_sentiment(
                    enabled=False,
                    require_validated_source=True,
                )
            elif "sentiment_df" not in context_sources:
                context_sources["sentiment_df"] = load_sentiment(
                    enabled=True,
                    require_validated_source=True,
                )
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
            build_mode=feature_build_mode,
        )
        feature_scope = feature_buffer[feature_buffer["date"] >= start_target].reset_index(drop=True)
        if feature_scope.empty:
            raise ValueError(f"Feature engineering produced no usable rows for {ticker}")
        all_feature_columns = self._feature_engineer.get_feature_columns(feature_scope)
        base_feature_columns = resolve_feature_set(
            ["forecast_core_features", "classification_signal_features"],
            available_columns=all_feature_columns,
        )
        if not base_feature_columns:
            base_feature_columns = list(all_feature_columns)
        feature_scope, risk_summary, regime_distribution = self._apply_advanced_risk_features(
            ticker,
            feature_scope,
            advanced_config,
        )

        # Reuse the already-contextualized buffer for warmup-loss accounting.
        stats_scope = context_buffer[context_buffer["date"] >= start_target].reset_index(drop=True)
        stats_sentiment = None
        if ticker_sentiment is not None:
            stats_sentiment = ticker_sentiment[ticker_sentiment["date"] >= start_target].reset_index(drop=True)
            if stats_sentiment.empty:
                stats_sentiment = None
        if stats_sentiment is not None and set(stats_sentiment.columns) - {"date", "ticker"} <= set(stats_scope.columns):
            stats_sentiment = None
        strict_features = self._feature_engineer.transform(
            stats_scope,
            sentiment_df=stats_sentiment,
            drop_na=True,
            build_mode=feature_build_mode,
        )
        indicator_rows_lost = max(len(raw_scope) - len(strict_features), 0)
        feature_columns = self._feature_engineer.get_feature_columns(feature_scope)

        flagged_event_dates: list[str] = []
        if "potential_corporate_action_flag" in feature_scope.columns:
            flagged = feature_scope.loc[
                pd.to_numeric(feature_scope["potential_corporate_action_flag"], errors="coerce") > 0,
                "date",
            ]
            flagged_event_dates = [str(pd.Timestamp(value).date()) for value in pd.to_datetime(flagged, errors="coerce").dropna()]

        data_start = str(raw_scope["date"].min().date())
        data_end = str(raw_scope["date"].max().date())
        stats = {
            "data_start": data_start,
            "data_end": data_end,
            "raw_rows": int(len(raw_scope)),
            "indicator_warmup_rows": int(indicator_rows_lost),
            "feature_rows": int(len(feature_scope)),
            "warmup_buffer_start": str(buffer_scope["date"].min().date()),
            "sentiment": {
                "enabled": bool(include_sentiment),
                "source_provenance": (
                    context_sources.get("sentiment_df").attrs.get("source_provenance")
                    if isinstance(context_sources.get("sentiment_df"), pd.DataFrame)
                    else None
                ),
                "integration_status": (
                    context_sources.get("sentiment_df").attrs.get("sentiment_integration_status")
                    if isinstance(context_sources.get("sentiment_df"), pd.DataFrame)
                    else None
                ),
                "main_pipeline_recommendation": (
                    context_sources.get("sentiment_df").attrs.get("sentiment_main_pipeline_recommendation")
                    if isinstance(context_sources.get("sentiment_df"), pd.DataFrame)
                    else None
                ),
            },
            "corporate_action_diagnostics": {
                "series_adjustment_status": str(feature_scope.get("price_adjustment_status", pd.Series(dtype=object)).dropna().iloc[-1])
                if "price_adjustment_status" in feature_scope.columns and feature_scope["price_adjustment_status"].dropna().any()
                else "unknown",
                "flagged_event_count": int(len(flagged_event_dates)),
                "flagged_event_dates": flagged_event_dates[-25:],
                "note": (
                    "Potential corporate-action dates are diagnostic gap-based flags only. "
                    "Live vnstock_data adjusted closes remain unavailable in the active quote path."
                ),
            },
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
            feature_build_mode=self._feature_engineer.normalize_build_mode(feature_build_mode),
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
        context_sources: dict[str, pd.DataFrame | None] | None = None,
    ) -> pd.DataFrame:
        """Rebuild features on the latest 5-year window for inference."""

        required_sequence_length = 20
        advanced_config: dict[str, Any] | None = None
        feature_build_mode = "full_research_mode"
        try:
            self._ensure_models_loaded(ticker)
            manifest = self._manifests[ticker.upper()]
            advanced_config = manifest.get("advanced_risk")
            feature_build_mode = (
                manifest.get("feature_generation", {}).get("feature_build_mode")
                or feature_build_mode
            )
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
            feature_build_mode=feature_build_mode,
            risk_config=advanced_config,
            window_start=window_start,
            window_end=window_end,
            context_sources=context_sources,
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
        target_date_col = f"target_date_{horizon}"
        labeled = dataset.dropna(subset=[direction_col, return_col, profit_col]).reset_index(drop=True)
        if labeled.empty:
            return None

        resolved_horizon_days = int(horizon_days or HORIZON_DAYS[horizon])
        split = self._build_split_definition(len(labeled), resolved_horizon_days)
        if split.train_stop < 24 or split.test_start >= len(labeled):
            return None
        if len(labeled) - split.test_start < 8:
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
            "target_dates": pd.to_datetime(labeled[target_date_col], errors="coerce").dt.normalize(),
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
        return compute_binary_classification_metrics(y_true, y_pred, y_prob)

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        metrics = compute_prediction_error_metrics(
            y_true,
            y_pred,
            include_residual_std=True,
            residual_std_source="test_residuals_std" if len(y_true) > 1 else "validation_rmse",
        )
        return {
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "residual_std": float(metrics["residual_std"]),
            "volatility_proxy_source": str(metrics["volatility_proxy_source"]),
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
    def _build_calibration(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        if X_val is None or y_val is None or len(X_val) == 0:
            return {
                "q10": -0.02,
                "q50": 0.0,
                "q90": 0.02,
                "uncertainty_methodology": "validation_residual_quantiles_not_probability_calibration",
                "calibration_status": "not_probability_calibration",
                "interpretation_warning": (
                    "This legacy calibration payload stores residual-error quantiles only; "
                    "it is not calibrated probability confidence."
                ),
                "deprecated_field_name": "calibration",
            }
        residuals = np.asarray(y_val, dtype=float) - np.asarray(model.predict(X_val), dtype=float)
        return {
            "q10": float(np.quantile(residuals, 0.10)),
            "q50": float(np.quantile(residuals, 0.50)),
            "q90": float(np.quantile(residuals, 0.90)),
            "uncertainty_methodology": "validation_residual_quantiles_not_probability_calibration",
            "calibration_status": "not_probability_calibration",
            "interpretation_warning": (
                "This legacy calibration payload stores residual-error quantiles only; "
                "it is not calibrated probability confidence."
            ),
            "deprecated_field_name": "calibration",
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
        feature_build_mode: str = "full_research_mode",
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
            feature_build_mode=feature_build_mode,
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
            "manifest_schema_version": ARTIFACT_SCHEMA_VERSION,
            "compatibility_version": MANIFEST_COMPATIBILITY_VERSION,
            "artifact_created_by": ARTIFACT_CREATED_BY,
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "target_type": "forward_return",
            "task_type": "multi_task_forecast",
            "feature_columns": prepared.feature_columns,
            "base_feature_columns": prepared.base_feature_columns,
            "feature_generation": self._feature_generation_manifest(prepared),
            "feature_governance": {
                "registry_path": str(FEATURE_REGISTRY_PATH),
                "approved_feature_sets": approved_feature_sets(),
                "final_task_feature_sets": final_task_feature_sets(),
                "feature_selection_evidence": feature_selection_evidence(),
                "sentiment_policy": sentiment_policy(),
                "price_reference_semantics": price_reference_semantics(),
                "feature_build_mode": prepared.feature_build_mode,
            },
            "target_definition": self._target_definition_manifest(resolved_horizons),
            "training_backend": self._training_backend_manifest(tune_boosters=tune_boosters),
            "prediction_output_semantics": self._prediction_output_semantics(),
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
                "mode": "chronological_holdout_with_gap",
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "gap_rule": "gap_days_equals_forecast_horizon",
                "validation_method": "single_validation_plus_held_out_test",
                "evaluation_split_name": "test",
                "metric_source": "held_out_test",
            },
            "evaluation_semantics": {
                "evaluation_split_name": "test",
                "metric_source": "held_out_test",
                "validation_method": "single_validation_plus_held_out_test",
                "portfolio_metric_basis": "held_out_split_prediction_metrics_only",
                "heuristic_scenario_risk_included_in_ranking": False,
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
                classification_feature_columns = self._select_algorithm_feature_columns(
                    prepared,
                    algorithm,
                    normalized_risk_config,
                    task_name="directional_classification",
                )
                return_feature_columns = self._select_algorithm_feature_columns(
                    prepared,
                    algorithm,
                    normalized_risk_config,
                    task_name="regression_forecasting",
                )
                feature_columns_by_task = {
                    "trend": classification_feature_columns,
                    "profit": classification_feature_columns,
                    "return": return_feature_columns,
                }
                algorithm_feature_columns = list(
                    dict.fromkeys(
                        classification_feature_columns + return_feature_columns
                    )
                )
                classification_problem = self._build_horizon_problem(
                    labeled_dataset,
                    classification_feature_columns,
                    horizon,
                    sequence_length,
                    horizon_days=resolved_horizons[horizon],
                )
                return_problem = self._build_horizon_problem(
                    labeled_dataset,
                    return_feature_columns,
                    horizon,
                    sequence_length,
                    horizon_days=resolved_horizons[horizon],
                )
                if classification_problem is None or return_problem is None:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="insufficient_rows_for_feature_set",
                    )
                    continue
                target_dates = classification_problem["target_dates"]
                split = classification_problem["split"]
                train_window = self._window_bounds(target_dates, 0, split.train_stop)
                validation_window = self._window_bounds(target_dates, split.val_start, split.val_stop)
                test_window = self._window_bounds(target_dates, split.test_start, len(target_dates))
                evaluation_metadata = self._evaluation_metadata(
                    evaluation_split_name="test",
                    metric_source="held_out_test",
                    validation_method="single_validation_plus_held_out_test",
                    train_window=train_window,
                    validation_window=validation_window,
                    test_window=test_window,
                )
                use_sequence = algorithm in SEQUENCE_ALGORITHMS
                classification_inputs = classification_problem["sequence" if use_sequence else "tabular"]
                return_inputs = return_problem["sequence" if use_sequence else "tabular"]
                rows_lost_to_sequence = int(
                    max(
                        classification_inputs.get("rows_lost", 0),
                        return_inputs.get("rows_lost", 0),
                    )
                )
                if (
                    len(classification_inputs["X_train"]) == 0
                    or len(classification_inputs["X_test"]) == 0
                    or len(return_inputs["X_train"]) == 0
                    or len(return_inputs["X_test"]) == 0
                ):
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="insufficient_split_rows",
                    )
                    continue
                if len(np.unique(classification_inputs["y_train_direction"])) < 2:
                    logger.warning(
                        "skipping_algorithm",
                        ticker=ticker,
                        algorithm=algorithm,
                        horizon=horizon,
                        reason="one_class_training_target",
                    )
                    continue
                if len(np.unique(classification_inputs["y_train_profit"])) < 2:
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
                    classification_inputs["X_train"],
                    classification_inputs["y_train_direction"],
                    classification_inputs["X_val"] if len(classification_inputs["X_val"]) else None,
                    classification_inputs["y_val_direction"] if len(classification_inputs["X_val"]) else None,
                )
                profit_model.fit(
                    classification_inputs["X_train"],
                    classification_inputs["y_train_profit"],
                    classification_inputs["X_val"] if len(classification_inputs["X_val"]) else None,
                    classification_inputs["y_val_profit"] if len(classification_inputs["X_val"]) else None,
                )
                return_model.fit(
                    return_inputs["X_train"],
                    return_inputs["y_train_return"],
                    return_inputs["X_val"] if len(return_inputs["X_val"]) else None,
                    return_inputs["y_val_return"] if len(return_inputs["X_val"]) else None,
                )
                train_seconds = float(time.perf_counter() - train_start)

                test_pred_direction = np.asarray(trend_model.predict(classification_inputs["X_test"]))
                test_pred_profit = np.asarray(profit_model.predict(classification_inputs["X_test"]), dtype=int)
                test_profit_probs = np.asarray(profit_model.predict_proba(classification_inputs["X_test"]), dtype=float)
                if test_profit_probs.ndim == 2 and test_profit_probs.shape[1] > 1:
                    test_profit_positive_prob = test_profit_probs[:, 1]
                else:
                    test_profit_positive_prob = np.asarray(test_profit_probs).reshape(-1)
                test_pred_return = np.asarray(return_model.predict(return_inputs["X_test"]), dtype=float)
                test_trend_positive_prob = self._predict_positive_class_probability(
                    trend_model,
                    classification_inputs["X_test"],
                )
                classification = self._classification_metrics(
                    classification_inputs["y_test_direction"],
                    test_pred_direction,
                )
                profit_classification = self._binary_classification_metrics(
                    classification_inputs["y_test_profit"],
                    test_pred_profit,
                    test_profit_positive_prob,
                )
                regression = self._regression_metrics(return_inputs["y_test_return"], test_pred_return)
                direction_calibration = self._direction_probability_diagnostics(
                    classification_inputs["y_test_direction"],
                    test_trend_positive_prob,
                )
                regression_diagnostics = self._regression_error_diagnostics(
                    return_inputs["y_test_return"],
                    test_pred_return,
                )
                trading = self._trading_metrics(
                    test_pred_direction,
                    classification_inputs["y_test_return"],
                    resolved_horizons[horizon],
                )

                latency_start = time.perf_counter()
                trend_model.predict(classification_inputs["X_test"])
                profit_model.predict(classification_inputs["X_test"])
                return_model.predict(return_inputs["X_test"])
                inference_latency_ms = float(
                    ((time.perf_counter() - latency_start) * 1000.0)
                    / max(max(len(classification_inputs["X_test"]), len(return_inputs["X_test"])), 1)
                )

                calibration = self._build_calibration(
                    return_model,
                    return_inputs["X_val"] if len(return_inputs["X_val"]) else None,
                    return_inputs["y_val_return"] if len(return_inputs["X_val"]) else None,
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
                    "model_type": algorithm,
                    "task_bundle": "trend_profit_return",
                    "artifact_type": self._artifact_type(algorithm),
                    "sequence_length": sequence_length if use_sequence else None,
                    "feature_columns": algorithm_feature_columns,
                    "feature_columns_by_task": feature_columns_by_task,
                    "trend_model_file": trend_path.name,
                    "profit_model_file": profit_path.name,
                    "return_model_file": return_path.name,
                    "calibration": calibration,
                    "calibration_diagnostics": {
                        "direction_probability": direction_calibration,
                        "regression_residuals": regression_diagnostics,
                        "scenario_risk": {
                            "available": bool(normalized_risk_config.get("risk_enabled")),
                            "calibration_status": "heuristic_not_calibrated",
                            "interpretation": SCENARIO_RISK_INTERPRETATION_WARNING,
                        },
                    },
                    "evaluation_metadata": evaluation_metadata,
                    "training_backend": self._training_backend_manifest(tune_boosters=tune_boosters),
                    "prediction_output_semantics": self._prediction_output_semantics(),
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
                    algorithm_manifest["risk_config"] = self._scenario_risk_manifest(
                        normalized_risk_config,
                        regression,
                    )
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
                        "trend_feature_columns": len(classification_feature_columns),
                        "return_feature_columns": len(return_feature_columns),
                        "data_start": prepared.data_start,
                        "data_end": prepared.data_end,
                        "raw_rows": prepared.raw_stats["raw_rows"],
                        "indicator_warmup_rows": prepared.raw_stats["indicator_warmup_rows"],
                        "target_rows_lost": classification_problem["target_rows_lost"],
                        "sequence_rows_lost": rows_lost_to_sequence,
                        "final_usable_rows": int(
                            len(classification_inputs["X_train"])
                            + len(classification_inputs["X_val"])
                            + len(classification_inputs["X_test"])
                        ),
                        "evaluation_split_name": evaluation_metadata["evaluation_split_name"],
                        "metric_source": evaluation_metadata["metric_source"],
                        "validation_method": evaluation_metadata["validation_method"],
                        "train_window_start": (train_window or {}).get("start"),
                        "train_window_end": (train_window or {}).get("end"),
                        "validation_window_start": (validation_window or {}).get("start"),
                        "validation_window_end": (validation_window or {}).get("end"),
                        "test_window_start": (test_window or {}).get("start"),
                        "test_window_end": (test_window or {}).get("end"),
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
        feature_build_mode: str = "full_research_mode",
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
            feature_build_mode=feature_build_mode,
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
            "manifest_schema_version": ARTIFACT_SCHEMA_VERSION,
            "compatibility_version": MANIFEST_COMPATIBILITY_VERSION,
            "artifact_created_by": ARTIFACT_CREATED_BY,
            "ticker": ticker,
            "primary_algorithm": primary_algorithm,
            "target_type": "forward_return",
            "task_type": "multi_task_forecast",
            "feature_columns": prepared.feature_columns,
            "base_feature_columns": prepared.base_feature_columns,
            "feature_generation": self._feature_generation_manifest(prepared),
            "feature_governance": {
                "registry_path": str(FEATURE_REGISTRY_PATH),
                "approved_feature_sets": approved_feature_sets(),
                "final_task_feature_sets": final_task_feature_sets(),
                "feature_selection_evidence": feature_selection_evidence(),
                "sentiment_policy": sentiment_policy(),
                "price_reference_semantics": price_reference_semantics(),
                "feature_build_mode": prepared.feature_build_mode,
            },
            "target_definition": self._target_definition_manifest(resolved_horizons),
            "training_backend": self._training_backend_manifest(tune_boosters=tune_boosters),
            "prediction_output_semantics": self._prediction_output_semantics(),
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
                "validation_method": "explicit_date_window_with_validation_tail",
                "evaluation_split_name": "validation",
                "metric_source": "validation_window",
            },
            "evaluation_semantics": {
                "evaluation_split_name": "validation",
                "metric_source": "validation_window",
                "validation_method": "explicit_date_window_with_validation_tail",
                "portfolio_metric_basis": "validation_window_prediction_metrics_only",
                "heuristic_scenario_risk_included_in_ranking": False,
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
            classification_feature_columns = self._select_algorithm_feature_columns(
                prepared,
                algorithm,
                normalized_risk_config,
                task_name="directional_classification",
            )
            return_feature_columns = self._select_algorithm_feature_columns(
                prepared,
                algorithm,
                normalized_risk_config,
                task_name="regression_forecasting",
            )
            feature_columns_by_task = {
                "trend": classification_feature_columns,
                "profit": classification_feature_columns,
                "return": return_feature_columns,
            }
            algorithm_feature_columns = list(
                dict.fromkeys(classification_feature_columns + return_feature_columns)
            )
            classification_features = feature_frame[classification_feature_columns].copy()
            return_features = feature_frame[return_feature_columns].copy()
            X_direction = classification_features.to_numpy(dtype=float)
            X_return = return_features.to_numpy(dtype=float)
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
            train_window = self._window_bounds(all_dates, candidate_start, validation_start)
            validation_window = self._window_bounds(all_dates, validation_start, candidate_stop)
            evaluation_metadata = self._evaluation_metadata(
                evaluation_split_name="validation",
                metric_source="validation_window",
                validation_method="explicit_date_window_with_validation_tail",
                train_window=train_window,
                validation_window=validation_window,
            )

            use_sequence = algorithm in SEQUENCE_ALGORITHMS
            if use_sequence:
                direction_sequences = create_sequence_dataset(
                    classification_features,
                    y_direction.astype(int),
                    sequence_length=sequence_length,
                    feature_columns=classification_feature_columns,
                )
                profit_sequences = create_sequence_dataset(
                    classification_features,
                    y_profit.astype(int),
                    sequence_length=sequence_length,
                    feature_columns=classification_feature_columns,
                )
                return_sequences = create_sequence_dataset(
                    return_features,
                    y_return,
                    sequence_length=sequence_length,
                    feature_columns=return_feature_columns,
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
                classification_inputs = {
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
                return_inputs = {
                    "X_train": seq_train_return.X,
                    "X_val": seq_val_return.X,
                    "y_train_return": seq_train_return.y.astype(float),
                    "y_val_return": seq_val_return.y.astype(float),
                    "rows_lost": int(return_sequences.rows_lost),
                }
            else:
                x_train_direction = X_direction[candidate_start:validation_start]
                x_val_direction = X_direction[validation_start:candidate_stop]
                x_train_return = X_return[candidate_start:validation_start]
                x_val_return = X_return[validation_start:candidate_stop]
                if algorithm in BOOSTER_ALGORITHMS:
                    x_train_direction = classification_features.iloc[candidate_start:validation_start].copy()
                    x_val_direction = classification_features.iloc[validation_start:candidate_stop].copy()
                    x_train_return = return_features.iloc[candidate_start:validation_start].copy()
                    x_val_return = return_features.iloc[validation_start:candidate_stop].copy()
                classification_inputs = {
                    "X_train": x_train_direction,
                    "X_val": x_val_direction,
                    "y_train_direction": y_direction[candidate_start:validation_start].astype(int),
                    "y_val_direction": y_direction[validation_start:candidate_stop].astype(int),
                    "y_train_profit": y_profit[candidate_start:validation_start].astype(int),
                    "y_val_profit": y_profit[validation_start:candidate_stop].astype(int),
                    "y_train_return": y_return[candidate_start:validation_start].astype(float),
                    "y_val_return": y_return[validation_start:candidate_stop].astype(float),
                    "rows_lost": 0,
                }
                return_inputs = {
                    "X_train": x_train_return,
                    "X_val": x_val_return,
                    "y_train_return": y_return[candidate_start:validation_start].astype(float),
                    "y_val_return": y_return[validation_start:candidate_stop].astype(float),
                    "rows_lost": 0,
                }

            if (
                len(classification_inputs["X_train"]) == 0
                or len(classification_inputs["X_val"]) == 0
                or len(return_inputs["X_train"]) == 0
                or len(return_inputs["X_val"]) == 0
            ):
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
                classification_inputs["X_train"],
                classification_inputs["y_train_direction"],
                classification_inputs["X_val"],
                classification_inputs["y_val_direction"],
            )
            profit_model.fit(
                classification_inputs["X_train"],
                classification_inputs["y_train_profit"],
                classification_inputs["X_val"],
                classification_inputs["y_val_profit"],
            )
            return_model.fit(
                return_inputs["X_train"],
                return_inputs["y_train_return"],
                return_inputs["X_val"],
                return_inputs["y_val_return"],
            )
            train_seconds = float(time.perf_counter() - train_clock)

            val_pred_direction = np.asarray(trend_model.predict(classification_inputs["X_val"]), dtype=int)
            val_pred_profit = np.asarray(profit_model.predict(classification_inputs["X_val"]), dtype=int)
            val_profit_probs = np.asarray(profit_model.predict_proba(classification_inputs["X_val"]), dtype=float)
            if val_profit_probs.ndim == 2 and val_profit_probs.shape[1] > 1:
                val_profit_positive_prob = val_profit_probs[:, 1]
            else:
                val_profit_positive_prob = np.asarray(val_profit_probs).reshape(-1)
            val_pred_return = np.asarray(return_model.predict(return_inputs["X_val"]), dtype=float).reshape(-1)
            val_trend_positive_prob = self._predict_positive_class_probability(
                trend_model,
                classification_inputs["X_val"],
            )
            classification = self._classification_metrics(
                classification_inputs["y_val_direction"],
                val_pred_direction,
            )
            profit_classification = self._binary_classification_metrics(
                classification_inputs["y_val_profit"],
                val_pred_profit,
                val_profit_positive_prob,
            )
            regression = self._regression_metrics(return_inputs["y_val_return"], val_pred_return)
            direction_calibration = self._direction_probability_diagnostics(
                classification_inputs["y_val_direction"],
                val_trend_positive_prob,
            )
            regression_diagnostics = self._regression_error_diagnostics(
                return_inputs["y_val_return"],
                val_pred_return,
            )
            trading = self._trading_metrics(
                val_pred_direction,
                classification_inputs["y_val_return"],
                resolved_horizons[horizon_key],
            )
            latency_start = time.perf_counter()
            trend_model.predict(classification_inputs["X_val"])
            profit_model.predict(classification_inputs["X_val"])
            return_model.predict(return_inputs["X_val"])
            inference_latency_ms = float(
                ((time.perf_counter() - latency_start) * 1000.0)
                / max(max(len(classification_inputs["X_val"]), len(return_inputs["X_val"])), 1)
            )
            calibration = self._build_calibration(
                return_model,
                return_inputs["X_val"],
                return_inputs["y_val_return"],
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
                "model_type": algorithm,
                "task_bundle": "trend_profit_return",
                "artifact_type": self._artifact_type(algorithm),
                "sequence_length": sequence_length if use_sequence else None,
                "feature_columns": algorithm_feature_columns,
                "feature_columns_by_task": feature_columns_by_task,
                "trend_model_file": trend_path.name,
                "profit_model_file": profit_path.name,
                "return_model_file": return_path.name,
                "calibration": calibration,
                "calibration_diagnostics": {
                    "direction_probability": direction_calibration,
                    "regression_residuals": regression_diagnostics,
                    "scenario_risk": {
                        "available": bool(normalized_risk_config.get("risk_enabled")),
                        "calibration_status": "heuristic_not_calibrated",
                        "interpretation": SCENARIO_RISK_INTERPRETATION_WARNING,
                    },
                },
                "evaluation_metadata": evaluation_metadata,
                "training_backend": self._training_backend_manifest(tune_boosters=tune_boosters),
                "prediction_output_semantics": self._prediction_output_semantics(),
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
                algorithm_manifest["risk_config"] = self._scenario_risk_manifest(
                    normalized_risk_config,
                    regression,
                )
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
                    "trend_feature_columns": len(classification_feature_columns),
                    "return_feature_columns": len(return_feature_columns),
                    "data_start": prepared.data_start,
                    "data_end": prepared.data_end,
                    "raw_rows": prepared.raw_stats["raw_rows"],
                    "indicator_warmup_rows": prepared.raw_stats["indicator_warmup_rows"],
                    "target_rows_lost": resolved_horizons[horizon_key],
                    "sequence_rows_lost": int(
                        max(classification_inputs.get("rows_lost", 0), return_inputs.get("rows_lost", 0))
                    ),
                    "final_usable_rows": int(len(classification_inputs["X_train"]) + len(classification_inputs["X_val"])),
                    "evaluation_split_name": evaluation_metadata["evaluation_split_name"],
                    "metric_source": evaluation_metadata["metric_source"],
                    "validation_method": evaluation_metadata["validation_method"],
                    "train_window_start": (train_window or {}).get("start"),
                    "train_window_end": (train_window or {}).get("end"),
                    "validation_window_start": (validation_window or {}).get("start"),
                    "validation_window_end": (validation_window or {}).get("end"),
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

        feature_columns_by_task = algorithm_info.get("feature_columns_by_task", {})
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

        def _task_model_input(task_key: str) -> Any:
            task_columns = feature_columns_by_task.get(task_key, feature_columns)
            missing_task = [column for column in task_columns if column not in feature_frame.columns]
            if missing_task:
                raise ValueError(f"Missing mandatory features for {ticker} task '{task_key}': {missing_task}")
            if algorithm in SEQUENCE_ALGORITHMS:
                sequence_length = int(algorithm_info.get("sequence_length") or 0)
                return build_latest_sequence(
                    feature_frame,
                    feature_columns=task_columns,
                    sequence_length=sequence_length,
                )
            if algorithm in BOOSTER_ALGORITHMS:
                return feature_frame[task_columns].iloc[[-1]].copy()
            return feature_frame[task_columns].iloc[[-1]].to_numpy(dtype=float)

        trend_input = _task_model_input("trend")
        profit_input = _task_model_input("profit")
        return_input = _task_model_input("return")

        status = "success"
        error_code = None
        error_msg = None
        fallback_used = False
        stacking_fallback_policy = "none"
        failed_tasks = []

        trend_probs = {"up": None, "sideways": None, "down": None}
        predicted_direction = None
        try:
            trend_model = self._get_loaded_model(ticker, algorithm, horizon, "trend")
            probs_raw = trend_model.predict_proba(trend_input)
            if np.isnan(probs_raw).any() or np.isinf(probs_raw).any():
                raise ValueError("[invalid_numeric_prediction] NaN/Inf in trend probability")
            trend_probs = self._trend_probabilities_from_binary(probs_raw[0])
            predicted_direction = int(trend_probs["up"] >= trend_probs["down"])
            # Check stacking fallback
            if getattr(trend_model, "last_fallback_policy", "none") not in ["none", ""]:
                fallback_used = True
                stacking_fallback_policy = trend_model.last_fallback_policy
        except Exception as e:
            failed_tasks.append("trend")
            error_code = error_code or "model_prediction_failed"
            error_msg = error_msg or f"Trend model failed: {str(e)}"

        predicted_return = None
        expected_range = {"bottom_10th": None, "median_50th": None, "ceiling_90th": None}
        try:
            return_model = self._get_loaded_model(ticker, algorithm, horizon, "return")
            ret_raw = return_model.predict(return_input)
            if np.isnan(ret_raw).any() or np.isinf(ret_raw).any():
                raise ValueError("[invalid_numeric_prediction] NaN/Inf in return prediction")
            predicted_return = float(np.asarray(ret_raw).reshape(-1)[0])
            expected_range = self._expected_range(current_close, predicted_return, algorithm_info.get("calibration", {}))
            if getattr(return_model, "last_fallback_policy", "none") not in ["none", ""]:
                fallback_used = True
                stacking_fallback_policy = getattr(return_model, "last_fallback_policy", stacking_fallback_policy)
        except Exception as e:
            failed_tasks.append("return")
            error_code = error_code or "model_prediction_failed"
            error_msg = error_msg or f"Return model failed: {str(e)}"

        if "profit_model_file" in algorithm_info:
            predicted_profit_label = None
            predicted_profit_probability = None
            profit_probabilities = {"loss_or_flat": None, "profit": None}
            try:
                profit_model = self._get_loaded_model(ticker, algorithm, horizon, "profit")
                probs_raw = profit_model.predict_proba(profit_input)
                if np.isnan(probs_raw).any() or np.isinf(probs_raw).any():
                    raise ValueError("[invalid_numeric_prediction] NaN/Inf in profit probability")
                profit_probabilities = self._binary_probabilities(
                    probs_raw[0], negative_label="loss_or_flat", positive_label="profit"
                )
                predicted_profit_probability = float(probs_raw[0][1]) if len(probs_raw[0]) > 1 else 0.0
                
                label_raw = profit_model.predict(profit_input)
                predicted_profit_label = int(np.asarray(label_raw).reshape(-1)[0])
                if getattr(profit_model, "last_fallback_policy", "none") not in ["none", ""]:
                    fallback_used = True
                    stacking_fallback_policy = getattr(profit_model, "last_fallback_policy", stacking_fallback_policy)
            except Exception as e:
                failed_tasks.append("profit")
                error_code = error_code or "model_prediction_failed"
                error_msg = error_msg or f"Profit model failed: {str(e)}"

        # Status Logic
        if "trend" in failed_tasks and "return" in failed_tasks:
            status = "failed"
            error_code = "all_models_failed"
        elif failed_tasks:
            status = "partial_success"
        elif fallback_used:
            status = "degraded"

        output = {
            "status": status,
            "error_code": error_code,
            "error_msg": error_msg,
            "fallback_used": fallback_used,
            "stacking_fallback_policy": stacking_fallback_policy,
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
            "manifest_schema_version": manifest.get("manifest_schema_version", manifest.get("schema_version")),
            "compatibility_version": manifest.get("compatibility_version"),
            "artifact_created_by": manifest.get("artifact_created_by"),
            "evaluation_metadata": algorithm_info.get("evaluation_metadata", {}),
            "prediction_semantics": algorithm_info.get(
                "prediction_output_semantics",
                manifest.get("prediction_output_semantics", {}),
            ),
        }

        if "profit_model_file" in algorithm_info:
            output.update(
                {
                    "predicted_profit_label": predicted_profit_label,
                    "predicted_profit_probability": predicted_profit_probability,
                    "profit_probabilities": profit_probabilities,
                }
            )

        manifest_risk_config = algorithm_info.get("risk_config", {})
        override_risk_config = self._normalize_inference_risk_override(risk_config)
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
        
        # Inference overrides should only change enablement when risk_enabled is explicitly provided.
        has_risk = (
            bool(override_risk_config["risk_enabled"])
            if "risk_enabled" in override_risk_config
            else bool(manifest_risk_config.get("risk_enabled", False))
        )
        if has_risk:
            eval_config = manifest_risk_config.copy()
            if override_risk_config:
                eval_config.update(override_risk_config)
            if "scenario_confidence_levels" not in eval_config and "risk_confidence_levels" in eval_config:
                eval_config["scenario_confidence_levels"] = list(eval_config["risk_confidence_levels"])
                
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
                confidence_levels=eval_config.get(
                    "scenario_confidence_levels",
                    eval_config.get("risk_confidence_levels", [0.95, 0.99]),
                ),
            )
            # Annotate with the source
            risk_assessment["metadata"]["volatility_proxy_source"] = eval_config.get("volatility_proxy_source", "validation_rmse")
            risk_assessment["metadata"]["risk_model_type"] = eval_config.get("risk_model_type", SCENARIO_RISK_MODEL_TYPE)
            risk_assessment["metadata"]["calibration_status"] = eval_config.get(
                "calibration_status",
                "heuristic_not_calibrated",
            )
            risk_assessment["metadata"]["uncertainty_methodology"] = eval_config.get(
                "uncertainty_methodology",
                SCENARIO_RISK_UNCERTAINTY_METHODOLOGY,
            )
            risk_assessment["metadata"]["interpretation_warning"] = eval_config.get(
                "interpretation_warning",
                SCENARIO_RISK_INTERPRETATION_WARNING,
            )
            risk_assessment["metadata"]["output_field"] = eval_config.get(
                "risk_output_field",
                SCENARIO_RISK_OUTPUT_FIELD,
            )
            risk_assessment["metadata"]["deprecated_aliases"] = [SCENARIO_RISK_LEGACY_ALIAS]
            
            output[SCENARIO_RISK_OUTPUT_FIELD] = risk_assessment
            # Temporary backward-compat alias for older inference consumers.
            output[SCENARIO_RISK_LEGACY_ALIAS] = risk_assessment
            
        return output
