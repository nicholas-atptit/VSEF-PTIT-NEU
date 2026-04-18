"""Daily stock prediction reporting module."""

from __future__ import annotations

import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DailyReportGenerator:
    """Generates summary reports from predictions."""

    @staticmethod
    def _format_heuristic_risk(value: object) -> str:
        if not isinstance(value, dict):
            return "n/a"
        var_map = value.get("var", {})
        cvar_map = value.get("cvar", {})
        var_95 = var_map.get("95.0")
        cvar_95 = cvar_map.get("95.0")
        if var_95 is None and cvar_95 is None:
            return "scenario-based"
        pieces = []
        if var_95 is not None:
            pieces.append(f"Scenario VaR95 {float(var_95) * 100:.2f}%")
        if cvar_95 is not None:
            pieces.append(f"Scenario CVaR95 {float(cvar_95) * 100:.2f}%")
        return ", ".join(pieces)

    def generate(self, predictions_df: pd.DataFrame) -> str:
        """Create a Markdown summary of daily predictions."""
        logger.info("generating_daily_report")
        if predictions_df.empty:
            return "No predictions available for today."
            
        # Sort by predicted return
        top_gainers = predictions_df.sort_values('predicted_return', ascending=False).head(10)
        include_profit = "predicted_profit_probability" in top_gainers.columns
        include_risk = "heuristic_scenario_risk" in top_gainers.columns
        include_eval = {"evaluation_split_name", "metric_source"} <= set(top_gainers.columns)
        
        report = f"# VN100 Stock Prediction Report - {pd.Timestamp.now().date()}\n\n"
        report += "## Top 10 Predicted Gainers\n\n"
        headers = ["Symbol", "Predicted Return (%)"]
        if include_profit:
            headers.append("Profit Probability (%)")
        if include_risk:
            headers.append("Heuristic Scenario Risk")
        if include_eval:
            headers.append("Evaluation Basis")
        report += "| " + " | ".join(headers) + " |\n"
        report += "| " + " | ".join([":---"] * len(headers)) + " |\n"
        
        for _, row in top_gainers.iterrows():
            line = [
                str(row["symbol"]),
                f"{row['predicted_return'] * 100:.2f}%",
            ]
            if include_profit:
                probability = pd.to_numeric(pd.Series([row.get("predicted_profit_probability")]), errors="coerce").iloc[0]
                line.append("n/a" if pd.isna(probability) else f"{float(probability) * 100:.1f}%")
            if include_risk:
                line.append(self._format_heuristic_risk(row.get("heuristic_scenario_risk")))
            if include_eval:
                line.append(
                    f"{row.get('metric_source', 'unknown')} / {row.get('evaluation_split_name', 'unknown')}"
                )
            report += "| " + " | ".join(line) + " |\n"

        report += "\n## Semantic Notes\n\n"
        report += "- Any scenario risk shown here is residual-based heuristic scenario risk, not calibrated forecast confidence.\n"
        report += "- Scenario VaR/CVaR values summarize simulated residual tails around the point forecast; they are not guaranteed loss bounds.\n"
        if include_eval:
            report += "- Evaluation basis reports which stored split the training metrics came from; do not compare validation and held-out test scores as if they were the same.\n"
        report += "- This daily brief is a per-name inference summary, not a portfolio-level performance report.\n"
            
        return report

    def save_report(self, content: str, path: str) -> None:
        """Save report to a file."""
        logger.info("saving_report", path=path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
