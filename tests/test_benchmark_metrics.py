import unittest
import numpy as np
import pandas as pd
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.benchmark.baselines import BaselineStrategies

class TestBenchmarkMetrics(unittest.TestCase):
    def setUp(self):
        self.evaluator = MetricsEvaluator()
        self.baselines = BaselineStrategies()

    def test_constant_positive_returns(self):
        """Constant 1% daily return should have high Sharpe and no drawdown."""
        n = 100
        returns = np.array([0.01] * n)
        signal = np.ones(n) # Always long
        
        # Using 0 fees/slippage for simpler verification
        config = {"fee": 0.0, "slippage": 0.0}
        res = self.evaluator.evaluate_strategy(signal, returns, config)
        m = res["metrics"]
        
        self.assertGreater(m["cagr"], 0)
        self.assertGreater(m["sharpe"], 10) # Very high since std is 0 (or near 0)
        self.assertEqual(m["max_drawdown"], 0)
        self.assertEqual(m["win_rate"], 1.0)

    def test_zero_returns(self):
        """Zero returns should yield zero metrics."""
        n = 50
        returns = np.zeros(n)
        signal = np.random.choice([0, 1], size=n)
        
        res = self.evaluator.evaluate_strategy(signal, returns, {"fee": 0, "slippage": 0})
        m = res["metrics"]
        
        self.assertEqual(m["total_return"], 0.0)
        self.assertEqual(m["cagr"], 0.0)
        self.assertEqual(m["sharpe"], 0.0)

    def test_drawdown_calculation(self):
        """Test exact drawdown: 1.0 -> 1.1 -> 0.88 (20% drop from peak)."""
        # Returns to achieve: +10%, -20%
        returns = np.array([0.0, 0.1, -0.2])
        signal = np.array([1, 1, 1])
        
        res = self.evaluator.evaluate_strategy(signal, returns, {"fee": 0, "slippage": 0})
        # Equity: [1.0, 1.1, 0.88]
        # Peak: 1.1. Drawdown at end: (0.88 - 1.1) / 1.1 = -0.2
        self.assertAlmostEqual(res["metrics"]["max_drawdown"], -0.2)

    def test_transaction_costs(self):
        """Enter and Exit should reduce total return."""
        returns = np.array([0.0, 0.0, 0.0]) # Flat market
        signal = np.array([0, 1, 0]) # Enter at t=1, Exit at t=2
        
        # Fee 0.1% + Slippage 0.1% = 0.2% per leg
        # Total cost = 0.2% (enter) + 0.2% (exit) = 0.4%
        config = {"fee": 0.001, "slippage": 0.001}
        res = self.evaluator.evaluate_strategy(signal, returns, config)
        
        # Expect ~ -0.4% total return
        self.assertLess(res["metrics"]["total_return"], -0.0039)
        self.assertGreater(res["metrics"]["total_return"], -0.0041)

    def test_baselines(self):
        """Verify baselines execute without error."""
        prices = np.array([100, 101, 102, 101, 100, 105])
        res = self.baselines.evaluate_all_baselines(prices)
        
        self.assertIn("buy_and_hold", res)
        self.assertIn("ma_crossover", res)
        self.assertIn("momentum", res)

if __name__ == "__main__":
    unittest.main()
