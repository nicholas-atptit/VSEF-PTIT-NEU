"""Ranked Prediction Generator Module.

Consumes batch inference JSON outputs and generates ranked tables for:
- top_long_1d (Proxy: short horizon)
- top_long_5d (Proxy: short horizon)
- top_expected_return (Predicted Median vs Current Price)
- high_risk_volatility_names (Volatility Regressor output)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

class RankedPredictionGenerator:
    """Generates ranked tables from batch prediction results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: Dict[str, Any] = {}

    def load_from_file(self, file_path: str) -> None:
        """Load predictions from a JSON file."""
        logger.info("loading_predictions", path=file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both raw latest_predictions.json and batch_inference_*.json formats
            if "predictions" in data:
                self.predictions = data["predictions"]
            else:
                self.predictions = data
        logger.info("predictions_loaded", count=len(self.predictions))

    def flatten_predictions(self) -> pd.DataFrame:
        """Convert nested JSON predictions into a flat DataFrame for ranking."""
        rows = []
        for ticker, payload in self.predictions.items():
            try:
                # Extract technical data (v5.1 format)
                tech = payload.get("technical", {})
                current_price = tech.get("current_price")
                
                # We prioritize the first horizon in 'horizons' list (usually short)
                horizons = tech.get("horizons", [])
                if not horizons:
                    continue
                
                h_data = horizons[0] # Default to first horizon
                
                # Check multi_horizon for specific 1d/5d/etc if requested in future
                # For now, we take the primary horizon
                
                row = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "horizon": h_data.get("horizon"),
                    "prob_up": h_data.get("trend_probs", {}).get("up", 0.0),
                    "prob_down": h_data.get("trend_probs", {}).get("down", 0.0),
                    "median_target": h_data.get("expected_range", {}).get("median_50th"),
                    "volatility": h_data.get("volatility_score"),
                    "confidence": h_data.get("confidence", 0.0),
                    "action": payload.get("fusion", {}).get("action", "WAIT")
                }
                
                # Calculate expected return
                if row["current_price"] and row["median_target"]:
                    row["expected_return"] = (row["median_target"] / row["current_price"]) - 1
                else:
                    row["expected_return"] = 0.0
                    
                rows.append(row)
            except Exception as e:
                logger.warning("flattening_failed", ticker=ticker, error=str(e))
                
        return pd.DataFrame(rows)

    def generate_ranks(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Generate all required ranked tables."""
        df = self.flatten_predictions()
        if df.empty:
            logger.warning("no_data_to_rank")
            return {}

        ranks = {}
        
        # 1. Top Long 1D (Highest prob_up in short horizon)
        ranks["top_long_1d"] = df.sort_values("prob_up", ascending=False).head(top_n).copy()
        
        # 2. Top Long 5D (Same for now, or if multiple horizons exist)
        ranks["top_long_5d"] = df.sort_values("prob_up", ascending=False).head(top_n).copy()
        
        # 3. Top Expected Return
        ranks["top_expected_return"] = df.sort_values("expected_return", ascending=False).head(top_n).copy()
        
        # 4. High Risk Volatility Names
        ranks["high_risk_volatility_names"] = df.sort_values("volatility", ascending=False).head(top_n).copy()
        
        return ranks

    def _render_markdown_table(self, df: pd.DataFrame) -> str:
        """Render markdown tables without requiring the optional tabulate package."""
        if df.empty:
            return "_No rows available._"
        try:
            return df.to_markdown(index=False)
        except ImportError:
            header = "| " + " | ".join(map(str, df.columns)) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = [
                "| " + " | ".join(map(str, row)) + " |"
                for row in df.itertuples(index=False, name=None)
            ]
            return "\n".join([header, separator, *rows])

    def export_reports(self, top_n: int = 10) -> List[str]:
        """Export all ranks to CSV and return a consolidated Markdown summary."""
        ranks = self.generate_ranks(top_n)
        if not ranks:
            return []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        # CSV Export
        for name, rdf in ranks.items():
            csv_path = self.output_dir / f"{name}_{timestamp}.csv"
            rdf.to_csv(csv_path, index=False)
            saved_files.append(str(csv_path))
            
        # Markdown Generation
        md_report = f"# Ranked Predictions Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for name, rdf in ranks.items():
            title = name.replace("_", " ").title()
            md_report += f"## {title}\n\n"
            md_report += self._render_markdown_table(
                rdf[["ticker", "current_price", "prob_up", "expected_return", "volatility"]]
            )
            md_report += "\n\n"
            
        md_report_path = self.output_dir / f"ranked_report_{timestamp}.md"
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        saved_files.append(str(md_report_path))
        
        logger.info("reports_exported", files=saved_files)
        return saved_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ranked Prediction Table Generator")
    parser.add_argument("--file", type=str, default="data/latest_predictions.json", 
                        help="Path to predictions JSON file")
    parser.add_argument("--top", type=int, default=10, help="Number of items per rank")
    parser.add_argument("--latest", action="store_true", help="Use latest batch inference file")
    
    args = parser.parse_args()
    
    gen = RankedPredictionGenerator()
    
    target_file = args.file
    if args.latest:
        batch_dir = Path("data/processed")
        batch_files = sorted(batch_dir.glob("batch_inference_*.json"))
        if batch_files:
            target_file = str(batch_files[-1])
            print(f"Using latest batch file: {target_file}")
    
    if os.path.exists(target_file):
        gen.load_from_file(target_file)
        exports = gen.export_reports(top_n=args.top)
        print(f"Successfully generated {len(exports)} report files in reports/")
    else:
        print(f"Error: File not found: {target_file}")
