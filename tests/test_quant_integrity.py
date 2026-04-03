import unittest
import pandas as pd
import numpy as np
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.data.database.decision_card_schema import DecisionCard, AuditMetadata
from src.ml.feature_engineering import FeatureEngineer

class TestQuantIntegrity(unittest.TestCase):
    def setUp(self):
        self.evaluator = MetricsEvaluator()
        self.fe = FeatureEngineer()

    def test_benchmark_not_random(self):
        """Mission B: Verify metrics are deterministic based on inputs."""
        d1 = DecisionCard(
            meta=AuditMetadata(ticker="SSI", provider="test"),
            tech_summary={"close": 100.0},
            news_summary={}, bull_thesis="", bear_thesis="",
            risk_veto=False, risk_reason="", action="BUY", target_weight=1.0, rationale=""
        )
        d2 = DecisionCard(
            meta=AuditMetadata(ticker="SSI", provider="test"),
            tech_summary={"close": 110.0}, # 10% gain
            news_summary={}, bull_thesis="", bear_thesis="",
            risk_veto=False, risk_reason="", action="BUY", target_weight=1.0, rationale=""
        )
        
        res1 = self.evaluator.evaluate_financial_metrics([d1, d2])
        res2 = self.evaluator.evaluate_financial_metrics([d1, d2])
        
        # Verify deterministic
        self.assertEqual(res1["cagr"], res2["cagr"])
        self.assertEqual(res1["sharpe"], res2["sharpe"])
        
        # Verify math: 10% gain over 2 days (approximate CAGR)
        self.assertGreater(res1["cagr"], 0)
        self.assertEqual(res1["total_return_pct"], 10.0)

    def test_leakage_prevention(self):
        """Mission D: Verify bfill doesn't pull future data."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "open": [9.8, 10.8, 11.8],
            "close": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "volume": [100, 110, 120]
        })
        sent_df = pd.DataFrame({
            "date": ["2023-01-03"], # Only last day has sentiment
            "sentiment_score": [1.0]
        })
        
        # With our fix, day 1 and 2 should NOT have sentiment 1.0 (leakage)
        feat_df = self.fe.transform(df, sentiment_df=sent_df, drop_na=False)
        
        self.assertEqual(feat_df.iloc[0]["sentiment_score"], 0)
        self.assertEqual(feat_df.iloc[1]["sentiment_score"], 0)
        self.assertEqual(feat_df.iloc[2]["sentiment_score"], 1.0)
        
    def test_close_raw_preservation(self):
        """Mission D: Verify close_raw is preserved."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "open": [9.8, 10.8],
            "close": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "volume": [100, 110]
        })
        feat_df = self.fe.transform(df, drop_na=False)
        self.assertIn("close_raw", feat_df.columns)
        self.assertEqual(feat_df.iloc[0]["close_raw"], 10.0)

if __name__ == "__main__":
    unittest.main()
