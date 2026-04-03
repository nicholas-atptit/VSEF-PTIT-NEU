import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .evaluator import MetricsEvaluator

class BaselineStrategies:
    """
    Standard benchmark strategies to compare against ML models.
    """
    def __init__(self):
        self.evaluator = MetricsEvaluator()

    def buy_and_hold(self, returns: np.ndarray | pd.Series) -> Dict[str, Any]:
        """Always long (signal = 1)."""
        returns = np.array(returns)
        signal = np.ones(len(returns))
        return self.evaluator.evaluate_strategy(signal, returns)

    def ma_crossover(self, 
                     prices: np.ndarray | pd.Series, 
                     fast: int = 5, 
                     slow: int = 20) -> Dict[str, Any]:
        """MA(5) vs MA(20) crossover."""
        prices = pd.Series(prices)
        ma_fast = prices.rolling(fast).mean()
        ma_slow = prices.rolling(slow).mean()
        
        # Signal: 1 if fast > slow, else 0
        signal = (ma_fast > ma_slow).astype(int).values
        returns = prices.pct_change().fillna(0).values
        
        return self.evaluator.evaluate_strategy(signal, returns)

    def simple_momentum(self, 
                        prices: np.ndarray | pd.Series, 
                        window: int = 20) -> Dict[str, Any]:
        """Long if n-day return > 0."""
        prices = pd.Series(prices)
        roll_ret = prices.pct_change(window)
        
        # Signal: 1 if momentum > 0, else 0
        signal = (roll_ret > 0).astype(int).values
        returns = prices.pct_change().fillna(0).values
        
        return self.evaluator.evaluate_strategy(signal, returns)

    def evaluate_all_baselines(self, prices: np.ndarray | pd.Series) -> Dict[str, Any]:
        """Helper to run all benchmarks at once."""
        prices = np.array(prices)
        returns = pd.Series(prices).pct_change().fillna(0).values
        
        return {
            "buy_and_hold": self.buy_and_hold(returns),
            "ma_crossover": self.ma_crossover(prices),
            "momentum": self.simple_momentum(prices)
        }
