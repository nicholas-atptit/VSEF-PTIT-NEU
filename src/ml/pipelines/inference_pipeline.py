"""Orchestrator for manifest-driven daily inference and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.ml.inference.engine import InferenceEngine
from src.reporting.reports.daily_report import DailyReportGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Workflow for full-history batch prediction and report generation."""

    def __init__(self, symbols: List[str], model_root: str | Path | None = None) -> None:
        self.engine = InferenceEngine(model_root=model_root)
        self.reporter = DailyReportGenerator()
        self.symbols = [symbol.upper().strip() for symbol in symbols if symbol.strip()]

    def _load_histories(self) -> dict[str, pd.DataFrame]:
        from src.data.adapters.vnstock_adapter import VnstockAdapter

        adapter = VnstockAdapter(symbol_list=self.symbols)
        end_date = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        histories: dict[str, pd.DataFrame] = {}

        for symbol in self.symbols:
            start_date = self.engine.required_history_start(symbol, as_of=end_date)
            history_df = adapter.get_ohlcv(symbol, start_date=start_date, end_date=end_date)
            if history_df.empty:
                raise ValueError(f"No OHLCV history returned for {symbol} between {start_date} and {end_date}")
            history_df = history_df.copy()
            history_df["ticker"] = symbol
            histories[symbol] = history_df

        return histories

    def run(self) -> None:
        """Run daily inference from full ticker histories and save the report."""
        logger.info("starting_inference_pipeline", symbols_count=len(self.symbols))
        try:
            histories = self._load_histories()
            if not histories:
                logger.error("inference_failed_no_history")
                return

            total_rows = sum(len(history_df) for history_df in histories.values())
            logger.info("loaded_histories", total_rows=total_rows, unique_symbols=len(histories))

            predictions = self.engine.predict_batch(histories)
            logger.info("batch_inference_complete", prediction_rows=len(predictions))

            if predictions.empty:
                logger.warning("inference_pipeline_no_predictions")
                return

            report_content = self.reporter.generate(predictions)

            report_path = Path("reports") / f"prediction_{pd.Timestamp.now().date()}.md"
            self.reporter.save_report(report_content, report_path)

            logger.info("inference_pipeline_finished", report_path=str(report_path))
        except Exception as exc:
            logger.error("inference_pipeline_error", error=str(exc))
            raise
