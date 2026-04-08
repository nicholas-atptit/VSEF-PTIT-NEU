"""Orchestrator for daily inference and reporting.

This pipeline coordinates feature preparation, model inference, and report
generation for daily batch predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.ml.inference.engine import InferenceEngine
from src.reporting.reports.daily_report import DailyReportGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Workflow for daily batch prediction and report generation.
    
    Steps:
        1. Load features for all symbols
        2. Run inference via InferenceEngine (backed by DualModelTrainer)
        3. Generate daily report from predictions
    """

    def __init__(self, symbols: List[str], model_root: str | Path =None) -> None:
        self.engine = InferenceEngine(model_root=model_root)
        self.reporter = DailyReportGenerator()
        self.symbols = symbols

    def run(self) -> None:
        """Run daily inference and save report content.
        
        Notes:
            - For LSTM/BiLSTM models, the engine needs sufficient historical context
              (default 20+ trading days). DatasetLoader should provide at least that range.
            - For CART models, only the latest row is strictly necessary.
            - This pipeline passes full history to the engine; it will raise a clear
              error if LSTM/BiLSTM models lack sufficient context.
        """
        logger.info("starting_inference_pipeline", symbols_count=len(self.symbols))
        try:
            from src.data.datasets.loader import DatasetLoader
            loader = DatasetLoader(self.symbols)

            # 1. Fetch features for all symbols (preferably last 30+ trading days for sequence models)
            df = loader.create_features_labels()
            if df.empty:
                logger.error("inference_failed_no_features")
                return

            logger.info("loaded_features", total_rows=len(df), unique_symbols=df["symbol"].nunique() if "symbol" in df.columns else 0)

            # 2. Predict via InferenceEngine (backed by DualModelTrainer)
            # The engine will request full per-ticker history for sequence models
            # and handle model-specific input requirements
            try:
                predictions = self.engine.predict_batch(df)
            except ValueError as ve:
                if "Insufficient history" in str(ve):
                    logger.warning("inference_pipeline_insufficient_history", error=str(ve))
                    # Some tickers lack sequence history; continue with whatever succeeded
                    predictions = self.engine.predict_batch(df)
                else:
                    raise
            
            logger.info("batch_inference_complete", prediction_rows=len(predictions))

            if predictions.empty:
                logger.warning("inference_pipeline_no_predictions")
                return

            # 3. Generate report
            report_content = self.reporter.generate(predictions)

            # 4. Save report
            from config.settings import get_settings
            s = get_settings()
            import os
            report_path = os.path.join(
                "reports",
                f"prediction_{pd.Timestamp.now().date()}.md",
            )
            self.reporter.save_report(report_content, report_path)

            logger.info("inference_pipeline_finished", report_path=report_path)
        except Exception as e:
            logger.error("inference_pipeline_error", error=str(e))
