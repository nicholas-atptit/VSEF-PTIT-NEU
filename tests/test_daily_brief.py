"""Smoke Test for Daily Brief Reporting."""

import json
import os
import unittest
from pathlib import Path
import pandas as pd
from src.reporting.daily_brief import DailyBriefGenerator

class TestDailyBriefSmoke(unittest.TestCase):
    """Smoke tests for the DailyBriefGenerator."""

    def setUp(self):
        # Create a mock data directory and dummy JSON
        self.test_dir = Path("tmp_test_reports")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.test_dir / "data"
        self.reports_dir = self.test_dir / "reports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.mock_json = self.data_dir / "batch_inference_MOCK.json"
        
        mock_data = {
            "timestamp": "2026-04-02T03:00:00",
            "total_tickers": 2,
            "success_count": 2,
            "elapsed_sec": 1.5,
            "predictions": {
                "AAA": {
                    "current_price": 10000,
                    "trend_probs": {"up": 0.8, "down": 0.1},
                    "expected_range": {"median_50th": 11000},
                    "volatility_score": 0.25,
                    "confidence": 0.9,
                    "fusion": {"action": "BUY"}
                },
                "VGI": {
                    "current_price": 45000,
                    "trend_probs": {"up": 0.6, "down": 0.2},
                    "expected_range": {"median_50th": 47000},
                    "volatility_score": 0.35,
                    "confidence": 0.8,
                    "fusion": {"action": "HOLD"}
                }
            }
        }
        
        with open(self.mock_json, "w", encoding="utf-8") as f:
            json.dump(mock_data, f)
            
        self.gen = DailyBriefGenerator(output_dir=str(self.reports_dir), data_dir=str(self.data_dir))

    def tearDown(self):
        # Cleanup
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_and_export(self):
        """Test loading mock JSON and exporting results."""
        success = self.gen.load_latest_batch(str(self.mock_json))
        self.assertTrue(success)
        self.assertEqual(len(self.gen.df), 2)
        
        files = self.gen.export_all("smoke_test_brief")
        self.assertEqual(len(files), 3)
        
        # Verify MD, CSV, HTML files exist
        for f in files:
            self.assertTrue(os.path.exists(f))
            
        # Check CSV content
        df_loaded = pd.read_csv(files[1])
        self.assertIn("ticker", df_loaded.columns)
        self.assertEqual(len(df_loaded), 2)
        self.assertEqual(df_loaded.iloc[0]["ticker"], "AAA")

if __name__ == "__main__":
    unittest.main()
