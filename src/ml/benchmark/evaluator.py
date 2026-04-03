import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.data.database.decision_card_schema import DecisionCard

class MetricsEvaluator:
    """
    Quantitative Finance Engine for evaluating trading strategies and model performance.
    All calculations are deterministic and vectorized.
    """
    def __init__(self):
        pass

    def build_equity_curve(self, 
                           signal: np.ndarray | pd.Series, 
                           returns: np.ndarray | pd.Series, 
                           fee: float = 0.0015, 
                           slippage: float = 0.002) -> np.ndarray:
        """
        Construct an equity curve from signals and returns, accounting for costs.
        
        Logic:
        - Daily return: signal[t-1] * returns[t]
        - Cost (Fee + Slippage): abs(signal[t] - signal[t-1]) * (fee + slippage)
        - Initial equity: 1.0
        """
        signal = np.array(signal)
        returns = np.array(returns)
        
        # Shift signal to align with future returns (signal at t-1 acts on return at t)
        strat_returns = signal[:-1] * returns[1:]
        
        # Calculate turnover costs on position changes
        # abs diff of signal[t] - signal[t-1]
        turnover = np.abs(np.diff(signal))
        costs = turnover * (fee + slippage)
        
        # Net returns
        net_returns = strat_returns - costs
        
        # Cumulative equity (starts at 1.0)
        equity_curve = np.cumprod(1 + net_returns)
        # Prepend initial 1.0
        return np.insert(equity_curve, 0, 1.0)

    def evaluate_strategy(self, 
                          signal: np.ndarray | pd.Series, 
                          returns: np.ndarray | pd.Series, 
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a trading strategy.
        """
        fee = config.get("fee", 0.0015) if config else 0.0015
        slippage = config.get("slippage", 0.002) if config else 0.002
        
        # 1. Build Equity Curve
        equity = self.build_equity_curve(signal, returns, fee, slippage)
        
        # 2. Derive daily net returns for metric calculation
        # Re-derive from equity to be consistent
        net_returns = np.diff(equity) / equity[:-1]
        
        metrics = {
            "total_return": round(float(equity[-1] - 1.0), 4),
            "cagr": round(self._calculate_cagr(equity), 4),
            "sharpe": round(self._calculate_sharpe(net_returns), 2),
            "sortino": round(self._calculate_sortino(net_returns), 2),
            "max_drawdown": round(self._calculate_max_drawdown(equity), 4),
            "win_rate": round(self._calculate_win_rate(net_returns), 3),
            "profit_factor": round(self._calculate_profit_factor(net_returns), 2)
        }
        
        # 3. Trade stats
        trade_signals = np.diff(np.insert((np.array(signal) != 0).astype(int), 0, 0))
        num_trades = int(np.sum(np.abs(trade_signals)) / 2) # Enter + Exit = 1 trade
        
        return {
            "equity_curve": equity.tolist(),
            "metrics": metrics,
            "trade_stats": {
                "num_trades": num_trades,
                "exposure_time": round(float(np.mean(np.array(signal) != 0)), 3)
            }
        }

    def _calculate_cagr(self, equity: np.ndarray) -> float:
        """CAGR = (Final/Initial) ^ (252/n) - 1"""
        n = len(equity)
        if n < 2 or equity[-1] <= 0: return 0.0
        return float((equity[-1] / equity[0])**(252 / n) - 1)

    def _calculate_sharpe(self, returns: np.ndarray, rfr: float = 0.0) -> float:
        """Annualized Sharpe = (Mean/Std) * sqrt(252)"""
        std = np.std(returns)
        if std == 0: return 0.0
        return float((np.mean(returns) - rfr) / std * np.sqrt(252))

    def _calculate_sortino(self, returns: np.ndarray, rfr: float = 0.0) -> float:
        """Annualized Sortino = (Mean/DownsideStd) * sqrt(252)"""
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        if downside_std == 0: return 0.0
        return float((np.mean(returns) - rfr) / downside_std * np.sqrt(252))

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Max Drawdown = Peak-to-Trough percentage drop."""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-9)
        return float(np.min(drawdown))

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Win Rate = count_positive / count_nonzero."""
        active_returns = returns[returns != 0]
        if len(active_returns) == 0: return 0.0
        return float(np.sum(active_returns > 0) / len(active_returns))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Profit Factor = Sum(Gains) / |Sum(Losses)|."""
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) == 0: return 10.0 if len(gains) > 0 else 0.0 # Cap at 10
        return float(np.sum(gains) / np.abs(np.sum(losses)))

    def compare_vs_baseline(self, model_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Comparison helper for reports."""
        return {
            "alpha_cagr": round(model_metrics["cagr"] - baseline_metrics["cagr"], 4),
            "alpha_sharpe": round(model_metrics["sharpe"] - baseline_metrics["sharpe"], 2),
            "outperformance": model_metrics["total_return"] > baseline_metrics["total_return"]
        }
    
    # ── Legacy Compatibility ─────────────────────────────────────
    def evaluate_financial_metrics(self, decisions: List[DecisionCard]) -> Dict[str, Any]:
        """Legacy hook used by existing simulators."""
        if not decisions or len(decisions) < 2:
            return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            
        prices = [float(d.tech_summary.get("close", 1.0)) for d in decisions]
        weights = [float(d.target_weight) for d in decisions]
        
        # Mock-to-Real Bridge
        returns = pd.Series(prices).pct_change().fillna(0).values
        signal = np.array(weights)
        
        eval_res = self.evaluate_strategy(signal, returns)
        return eval_res["metrics"]
