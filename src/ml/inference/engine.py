"""Inference engine for daily predictions.

Facade over DualModelTrainer that provides a stateless, batch-oriented interface
for inference on features across multiple tickers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Manifest-driven inference engine backed by DualModelTrainer.
    
    This is a facade/adapter that provides a simpler interface for batch
    inference over the structured training/artifact system.
    """

    def __init__(self, model_root: str | Path | None = None) -> None:
        """Initialize with the path to the model/artifact directory.
        
        If model_root is None, uses the default from settings.
        """
        self.trainer = DualModelTrainer(model_dir=model_root)

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for a batch of tickers.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Expected columns: 'ticker', 'symbol' (or similar identifier),
            plus all feature columns required by the loaded models.
        
        Returns
        -------
        pd.DataFrame
            Results with columns: ticker, symbol, horizon, algorithm, 
            predicted_return, trend_probabilities, expected_range, etc.
        """
        if features_df.empty:
            return pd.DataFrame()

        logger.info("running_batch_inference", row_count=len(features_df))

        # Group by ticker to handle predictions per symbol
        results = []
        ticker_col = "ticker" if "ticker" in features_df.columns else "symbol"
        if ticker_col not in features_df.columns:
            logger.error("inference_error", error="features_df must have 'ticker' or 'symbol' column")
            return pd.DataFrame()

        for ticker in features_df[ticker_col].unique():
            try:
                ticker_features = features_df[features_df[ticker_col] == ticker].copy()
                # Try to make a prediction for this ticker on the 'short' horizon
                prediction = self.trainer.predict(
                    ticker=ticker,
                    features=ticker_features,
                    horizon="short",
                )
                # Flatten the prediction result and add to batch results
                pred_row = {
                    "ticker": ticker,
                    "symbol": ticker,
                    "horizon": "short",
                    "algorithm": prediction.get("algorithm", ""),
                    "predicted_return": prediction.get("predicted_return"),
                    "trend_probabilities": prediction.get("trend_probabilities", {}),
                    "expected_range": prediction.get("expected_range", {}),
                    "sequence_length": prediction.get("sequence_length"),
                }
                results.append(pred_row)
            except Exception as exc:
                logger.error("inference_error_ticker", ticker=ticker, error=str(exc))
                # Continue with next ticker instead of failing the batch
                continue

        if not results:
            logger.warning("batch_inference_produced_no_results")
            return pd.DataFrame()

        return pd.DataFrame(results)
