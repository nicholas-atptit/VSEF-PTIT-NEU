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
        if history_df is None or history_df.empty:
            raise ValueError(f"No history rows supplied for {ticker_key}")

        try:
            features = self.trainer.compute_features_for_ticker(ticker_key, history_df)
            prediction = self.trainer.predict(
                ticker=ticker_key,
                features=features,
                horizon=horizon,
                algorithm=algorithm,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing manifest or model artifacts for {ticker_key} under {self.model_root / ticker_key}: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{ticker_key}: {exc}") from exc

        as_of_date = None
        if "date" in features.columns and not features.empty:
            as_of_date = pd.Timestamp(features["date"].iloc[-1]).date().isoformat()

        return {
            "ticker": ticker_key,
            "symbol": ticker_key,
            "history_rows": int(len(history_df)),
            "feature_rows": int(len(features)),
            "as_of_date": as_of_date,
            **prediction,
        }

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
        errors: list[str] = []

        for ticker, history_df in batches.items():
            try:
                results.append(
                    self.predict_ticker(
                        ticker=ticker,
                        history_df=history_df,
                        horizon=horizon,
                        algorithm=algorithm,
                    )
                )
            except Exception as exc:
                logger.error("inference_error_ticker", ticker=ticker, error=str(exc))
                errors.append(str(exc))

        if errors:
            raise RuntimeError("Batch inference failed:\n" + "\n".join(errors))
        return pd.DataFrame(results)
