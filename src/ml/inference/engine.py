"""Manifest-driven inference engine for full-history ticker predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Facade over ``DualModelTrainer`` for manifest-driven batch inference."""

    def __init__(self, model_root: str | Path | None = None) -> None:
        """Initialize the engine with the model artifact root."""
        self.trainer = DualModelTrainer(model_dir=model_root)
        self.model_root = Path(self.trainer._model_dir)

    @staticmethod
    def _ticker_column(history_df: pd.DataFrame) -> str:
        if "ticker" in history_df.columns:
            return "ticker"
        if "symbol" in history_df.columns:
            return "symbol"
        raise ValueError("History input must include a 'ticker' or 'symbol' column")

    def required_history_start(self, ticker: str, as_of: Any | None = None) -> str:
        """Return the start date needed to rebuild features for a ticker."""
        ticker_key = ticker.upper().strip()
        self.trainer._ensure_models_loaded(ticker_key)
        manifest = self.trainer._manifests[ticker_key]

        max_sequence_length = 1
        for horizon_info in manifest.get("horizons", {}).values():
            for algorithm_info in horizon_info.get("algorithms", {}).values():
                max_sequence_length = max(max_sequence_length, int(algorithm_info.get("sequence_length") or 1))

        end_ts = pd.Timestamp(as_of or pd.Timestamp.now()).normalize()
        warmup_days = self.trainer._warmup_buffer_days(max_sequence_length)
        start_ts = (end_ts - pd.DateOffset(years=5) - pd.Timedelta(days=warmup_days)).normalize()
        return start_ts.strftime("%Y-%m-%d")

    def _coerce_history_batches(
        self,
        histories: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        if isinstance(histories, dict):
            batches: dict[str, pd.DataFrame] = {}
            for ticker, history_df in histories.items():
                if history_df is None or history_df.empty:
                    raise ValueError(f"No history rows supplied for {ticker}")
                batches[str(ticker).upper().strip()] = history_df.copy()
            return batches

        if not isinstance(histories, pd.DataFrame):
            raise TypeError("histories must be a pandas DataFrame or dict[str, DataFrame]")
        if histories.empty:
            return {}

        ticker_col = self._ticker_column(histories)
        batches = {}
        for ticker, history_df in histories.groupby(ticker_col, sort=False):
            ticker_key = str(ticker).upper().strip()
            batches[ticker_key] = history_df.copy()
        return batches

    def predict_ticker(
        self,
        ticker: str,
        history_df: pd.DataFrame,
        *,
        horizon: str = "short",
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        """Predict one ticker from its full OHLCV history."""
        ticker_key = ticker.upper().strip()
        as_of_date = None
        if hasattr(history_df, "columns") and not history_df.empty:
            if "date" in history_df.columns:
                as_of_date = pd.Timestamp(history_df["date"].iloc[-1]).date().isoformat()
            elif "time" in history_df.columns:
                as_of_date = pd.Timestamp(history_df["time"].iloc[-1]).date().isoformat()
            elif "datetime" in history_df.columns:
                as_of_date = pd.Timestamp(history_df["datetime"].iloc[-1]).date().isoformat()
            
        base_schema = {
            "ticker": ticker_key,
            "symbol": ticker_key,
            "as_of_date": as_of_date,
            "status": "failed",
            "error_code": "unknown_error",
            "error_msg": None,
            "fallback_used": False,
            "stacking_fallback_policy": "none",
            "algorithm": algorithm,
            "horizon": horizon,
            "predicted_return": None,
            "predicted_direction": None,
            "trend_probabilities": {"up": None, "sideways": None, "down": None},
            "expected_range": {"bottom_10th": None, "median_50th": None, "ceiling_90th": None},
            "predicted_profit_label": None,
            "predicted_profit_probability": None,
            "profit_probabilities": {"loss_or_flat": None, "profit": None},
        }

        if history_df is None or history_df.empty:
            base_schema.update({"error_code": "invalid_ohlcv_input", "error_msg": f"No history rows supplied for {ticker_key}"})
            return base_schema

        try:
            from src.ml.data_loader import validate_ohlcv
            history_df = self.trainer._normalize_ohlcv(history_df, ticker=ticker_key)
            history_df = validate_ohlcv(history_df, ticker_key, min_rows=60)
            
            features = self.trainer.compute_features_for_ticker(ticker_key, history_df)
            prediction = self.trainer.predict(
                ticker=ticker_key,
                features=features,
                horizon=horizon,
                algorithm=algorithm,
            )
        except Exception as exc:
            logger.error("inference_ticker_failed", ticker=ticker_key, error=str(exc))
            err_code = "model_prediction_failed"
            exc_str = str(exc)
            if "invalid_ohlcv_input" in exc_str: err_code = "invalid_ohlcv_input"
            elif "insufficient_history" in exc_str: err_code = "insufficient_history"
            elif "feature_validation_failed" in exc_str: err_code = "feature_validation_failed"
            elif "Missing manifest" in exc_str or "No trained" in exc_str or "No artifact" in exc_str or isinstance(exc, FileNotFoundError): 
                err_code = "artifact_missing"
            
            base_schema.update({"error_code": err_code, "error_msg": exc_str})
            return base_schema

        manifest = self.trainer._manifests[ticker_key]
        horizon_info = manifest.get("horizons", {}).get(str(prediction["horizon"]).lower(), {})
        algorithm_info = horizon_info.get("algorithms", {}).get(str(prediction["algorithm"]).lower(), {})
        evaluation_metadata = algorithm_info.get("evaluation_metadata", {})
        prediction_semantics = algorithm_info.get(
            "prediction_output_semantics",
            manifest.get("prediction_output_semantics", {}),
        )
        feature_generation = manifest.get("feature_generation", {})
        scenario_risk = prediction.get("heuristic_scenario_risk", {})
        scenario_metadata = scenario_risk.get("metadata", {}) if isinstance(scenario_risk, dict) else {}

        if "date" in features.columns and not features.empty:
            as_of_date = pd.Timestamp(features["date"].iloc[-1]).date().isoformat()

        base_schema.update({
            "ticker": ticker_key,
            "symbol": ticker_key,
            "history_rows": int(len(history_df)),
            "feature_rows": int(len(features)),
            "as_of_date": as_of_date,
            "manifest_schema_version": manifest.get("manifest_schema_version", manifest.get("schema_version")),
            "compatibility_version": manifest.get("compatibility_version"),
            "artifact_created_by": manifest.get("artifact_created_by"),
            "evaluation_split_name": evaluation_metadata.get("evaluation_split_name"),
            "metric_source": evaluation_metadata.get("metric_source"),
            "validation_method": evaluation_metadata.get("validation_method"),
            "risk_semantics": prediction_semantics.get("risk_semantics"),
            "uncertainty_methodology": prediction_semantics.get("uncertainty_methodology"),
            "risk_calibration_status": scenario_metadata.get("calibration_status"),
            "risk_interpretation_warning": scenario_metadata.get("interpretation_warning"),
            "feature_dependency_behavior": feature_generation.get("technical_indicator_dependency_behavior"),
            **prediction,
        })
        return base_schema

    def predict_batch(
        self,
        histories: pd.DataFrame | dict[str, pd.DataFrame],
        *,
        horizon: str = "short",
        algorithm: str | None = None,
    ) -> pd.DataFrame:
        """Generate manifest-driven predictions from full ticker histories."""
        batches = self._coerce_history_batches(histories)
        if not batches:
            return pd.DataFrame()

        logger.info("running_batch_inference", ticker_count=len(batches))

        results: list[dict[str, Any]] = []

        for ticker, history_df in batches.items():
            results.append(
                self.predict_ticker(
                    ticker=ticker,
                    history_df=history_df,
                    horizon=horizon,
                    algorithm=algorithm,
                )
            )

        return pd.DataFrame(results)
