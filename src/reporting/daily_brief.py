"""Daily Brief Report Generator.

Consolidated daily summary for VN100 predictions including:
- universe statistics
- top bullish names
- top expected return names
- high risk volatility names
- data quality alerts
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

from config.settings import get_settings
from src.utils.time_utils import now_vn

logger = structlog.get_logger(__name__)

class DailyBriefGenerator:
    """Generates consolidated daily reports from batch inference results."""

    def __init__(self, output_dir: str = "reports", data_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
        self.df: pd.DataFrame = pd.DataFrame()

    def load_latest_batch(self, file_path: Optional[str] = None) -> bool:
        """Find and load the newest batch inference JSON file."""
        if file_path:
            target = Path(file_path)
        else:
            # Fallback to newest batch file
            batch_files = sorted(self.data_dir.glob("batch_inference_*.json"))
            if not batch_files:
                # Try generic cache if batch not found
                target = Path("data/latest_predictions.json")
            else:
                target = batch_files[-1]

        if not target.exists():
            logger.error("input_file_not_found", path=str(target))
            return False

        logger.info("loading_daily_brief_source", path=str(target))
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Metadata extraction
            self.metadata = {
                "source": target.name,
                "timestamp": data.get("timestamp", now_vn().isoformat()),
                "total_tickers": data.get("total_tickers", 0),
                "success_count": data.get("success_count", 0),
                "elapsed_sec": data.get("elapsed_sec", 0.0)
            }
            
            # Predictions extraction
            if "predictions" in data:
                self.predictions = data["predictions"]
            else:
                self.predictions = data
                if not self.metadata["total_tickers"]:
                    self.metadata["total_tickers"] = len(self.predictions)
                    self.metadata["success_count"] = len(self.predictions)

        self._flatten_to_df()
        return not self.df.empty

    def _flatten_to_df(self) -> None:
        """Process nested JSON into a clean DataFrame for reporting."""
        rows = []
        for ticker, payload in self.predictions.items():
            try:
                # payload is from SignalGenerator
                # It contains 'multi_horizon' with '1w', '1m', '6m'
                # and top-level fields for the primary (short) horizon
                
                row = {
                    "ticker": ticker,
                    "current_price": payload.get("current_price"),
                    "prob_up": payload.get("trend_probs", {}).get("up", 0.0),
                    "prob_down": payload.get("trend_probs", {}).get("down", 0.0),
                    "median_target": payload.get("expected_range", {}).get("median_50th"),
                    "volatility": payload.get("volatility_score", 0.0),
                    "confidence": payload.get("confidence", 0.0),
                    "action": payload.get("fusion", {}).get("action", "WAIT")
                }
                
                # Calculate expected return
                if row["current_price"] and row["median_target"]:
                    row["expected_return"] = (row["median_target"] / row["current_price"]) - 1
                else:
                    row["expected_return"] = 0.0
                    
                rows.append(row)
            except Exception as e:
                logger.warning("ticker_flatten_failed", ticker=ticker, error=str(e))
                
        self.df = pd.DataFrame(rows)

    def get_top_n(self, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Get top N rows based on a metric."""
        if self.df.empty: return pd.DataFrame()
        return self.df.sort_values(metric, ascending=ascending).head(n)

    def generate_brief(self) -> str:
        """Generate full Markdown brief content."""
        settings = get_settings()
        vn_now = now_vn().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Executive Summary
        success_rate = (self.metadata['success_count'] / self.metadata['total_tickers'] * 100) if self.metadata['total_tickers'] > 0 else 0
        
        md = f"# VN100 Daily Brief Report - {vn_now}\n\n"
        md += "## Executive Summary\n"
        md += f"- **Run Timestamp**: {self.metadata['timestamp']}\n"
        md += f"- **Universe Size**: {self.metadata['total_tickers']} tickers\n"
        md += f"- **Success Rate**: {self.metadata['success_count']}/{self.metadata['total_tickers']} ({success_rate:.1f}%)\n"
        md += f"- **Processing Time**: {self.metadata['elapsed_sec']:.1f}s\n"
        md += f"- **Model Mode**: Default (DualModelTrainer)\n"
        md += f"- **Trend Threshold**: ±{settings.trend_threshold_pct}%\n\n"

        # 2. Daily Ranks
        md += "## Today's Top Picks\n\n"
        
        # Bullish
        md += "### 🚀 Top 10 Bullish (Highest Up Probability)\n"
        top_bull = self.get_top_n("prob_up", 10)
        md += top_bull[["ticker", "current_price", "prob_up", "action"]].to_markdown(index=False)
        md += "\n\n"
        
        # Expected Return
        md += "### 💰 Top 10 Expected Return (Short Horizon)\n"
        top_ret = self.get_top_n("expected_return", 10)
        # Format percentage
        top_ret_fmt = top_ret.copy()
        top_ret_fmt["expected_return"] = (top_ret_fmt["expected_return"] * 100).map("{:.2f}%".format)
        md += top_ret_fmt[["ticker", "current_price", "expected_return", "median_target"]].to_markdown(index=False)
        md += "\n\n"
        
        # Volatility
        md += "### ⚠️ High Volatility / Risk Names\n"
        top_vol = self.get_top_n("volatility", 10)
        md += top_vol[["ticker", "current_price", "volatility", "confidence"]].to_markdown(index=False)
        md += "\n\n"

        # 3. Data Quality Warnings
        md += "## Data Quality & System Alerts\n"
        if self.metadata['success_count'] < self.metadata['total_tickers']:
            missing = self.metadata['total_tickers'] - self.metadata['success_count']
            md += f"- ⚠️ **WARNING**: {missing} tickers skipped due to data gaps or model errors.\n"
        else:
            md += "- ✅ All target tickers processed successfully.\n"
            
        low_conf = self.df[self.df["confidence"] < 0.5] if not self.df.empty else pd.DataFrame()
        if not low_conf.empty:
            md += f"- ⚠️ **LOW CONFIDENCE**: {len(low_conf)} tickers have confidence < 50%.\n"
            
        return md

    def export_all(self, base_name: Optional[str] = None) -> List[str]:
        """Export to MD, CSV, and HTML."""
        if not base_name:
            base_name = f"daily_brief_{now_vn().strftime('%Y%m%d_%H%M%S')}"
            
        md_content = self.generate_brief()
        files = []
        
        # Markdown
        md_path = self.output_dir / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        files.append(str(md_path))
        
        # CSV
        csv_path = self.output_dir / f"{base_name}.csv"
        self.df.to_csv(csv_path, index=False)
        files.append(str(csv_path))
        
        # HTML (Lightweight)
        html_path = self.output_dir / f"{base_name}.html"
        html_template = f"<html><head><style>body{{font-family:sans-serif;padding:20px;}} table{{border-collapse:collapse;width:100%;margin-bottom:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background-color:#f2f2f2;}}</style></head><body>{md_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('\n', '<br>')}</body></html>"
        # Note: pd.to_html() is better for tables but we have mixed content. 
        # For simplicity we use a basic string replace for the header and then we could embed tables properly if needed.
        # But per requirements "lightweight and consistent", a basic wrapper is fine.
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        files.append(str(html_path))
        
        logger.info("daily_brief_exported", count=len(files), files=files)
        return files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily Brief Generator")
    parser.add_argument("--file", type=str, help="Path to specific batch inference JSON")
    parser.add_argument("--latest", action="store_true", help="Use latest file in data/processed")
    args = parser.parse_args()
    
    gen = DailyBriefGenerator()
    if gen.load_latest_batch(args.file):
        exported = gen.export_all()
        print(f"Generated {len(exported)} files:")
        for f in exported:
            print(f" - {f}")
    else:
        print("Failed to load batch data.")
