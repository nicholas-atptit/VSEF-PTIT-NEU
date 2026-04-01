"""Orchestrator for daily inference and reporting."""

from __future__ import annotations

from typing import List
from src.ml.inference.engine import InferenceEngine
from src.reporting.reports.daily_report import DailyReportGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)

class InferencePipeline:
    """Workflow for daily batch prediction and report generation."""

    def __init__(self, symbols: List[str], model_path: str) -> None:
        self.engine = InferenceEngine(model_path)
        self.reporter = DailyReportGenerator()
        self.symbols = symbols

    def run(self) -> None:
        """Run daily inference and save report content.
        
        Steps: Fetch Latest Features -> Predict -> reporting.
        """
        logger.info("starting_inference_pipeline")
        try:
            from src.data.datasets.loader import DatasetLoader
            loader = DatasetLoader(self.symbols)
            
            # 1. Fetch latest features for all symbols
            df = loader.create_features_labels()
            if df.empty:
                logger.error("inference_failed_no_features")
                return
                
            # Take only the latest sample for each symbol
            latest_features = df.sort_values('timestamp').groupby('symbol').tail(1)
            
            # 2. Predict
            predictions = self.engine.predict_batch(latest_features)
            
            # 3. Generate report
            report_content = self.reporter.generate(predictions)
            
            # 4. Save report
            from config.settings import get_settings
            s = get_settings()
            import os
            report_path = os.path.join("reports", f"prediction_{pd.Timestamp.now().date()}.md")
            self.reporter.save_report(report_content, report_path)
            
            logger.info("inference_pipeline_finished", report=report_path)
        except Exception as e:
            logger.error("inference_pipeline_error", error=str(e))
