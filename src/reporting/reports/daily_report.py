"""Daily stock prediction reporting module."""

from __future__ import annotations

import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DailyReportGenerator:
    """Generates summary reports from predictions."""

    def generate(self, predictions_df: pd.DataFrame) -> str:
        """Create a Markdown summary of daily predictions."""
        logger.info("generating_daily_report")
        if predictions_df.empty:
            return "No predictions available for today."
            
        # Sort by predicted return
        top_gainers = predictions_df.sort_values('predicted_return', ascending=False).head(10)
        
        report = f"# VN100 Stock Prediction Report - {pd.Timestamp.now().date()}\n\n"
        report += "## Top 10 Predicted Gainers (Next Day)\n\n"
        report += "| Symbol | Predicted Return (%) |\n"
        report += "| :--- | :--- |\n"
        
        for _, row in top_gainers.iterrows():
            report += f"| {row['symbol']} | {row['predicted_return']*100:.2f}% |\n"
            
        return report

    def save_report(self, content: str, path: str) -> None:
        """Save report to a file."""
        logger.info("saving_report", path=path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
