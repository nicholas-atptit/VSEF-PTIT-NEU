import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.data.database.decision_card_schema import DecisionCard
from src.ml.metrics import (
    compute_annualized_volatility,
    compute_average_drawdown,
    compute_calmar_ratio,
    compute_drawdown_series,
    compute_exposure_ratio,
    compute_max_drawdown,
    compute_prediction_error_metrics,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_signal_turnover,
    compute_sortino_ratio,
    compute_tail_loss,
    compute_win_rate,
)

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

    @staticmethod
    def drawdown_series(equity: np.ndarray) -> np.ndarray:
        return compute_drawdown_series(equity).to_numpy(dtype=float)

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
        drawdown = self.drawdown_series(equity)
        annualized_volatility = self._calculate_annualized_volatility(net_returns)
        max_drawdown = self._calculate_max_drawdown(equity)
        avg_drawdown = self._calculate_average_drawdown(drawdown)
        calmar = self._calculate_calmar(self._calculate_cagr(equity), max_drawdown)
        tail_loss = self._calculate_tail_loss(net_returns)
        turnover = self._calculate_turnover(signal)
        exposure = self._calculate_exposure(signal)
        
        metrics = {
            "total_return": round(float(equity[-1] - 1.0), 4),
            "cumulative_return": round(float(equity[-1] - 1.0), 4),
            "cagr": round(self._calculate_cagr(equity), 4),
            "volatility": round(annualized_volatility, 4),
            "sharpe": round(self._calculate_sharpe(net_returns), 2),
            "sortino": round(self._calculate_sortino(net_returns), 2),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_drawdown, 4),
            "avg_drawdown": round(avg_drawdown, 4),
            "tail_loss": round(tail_loss, 4),
            "win_rate": round(self._calculate_win_rate(net_returns), 3),
            "profit_factor": round(self._calculate_profit_factor(net_returns), 2)
        }
        
        # 3. Trade stats
        trade_signals = np.diff(np.insert((np.array(signal) != 0).astype(int), 0, 0))
        num_trades = int(np.sum(np.abs(trade_signals)) / 2) # Enter + Exit = 1 trade
        
        return {
            "equity_curve": equity.tolist(),
            "net_returns": net_returns.tolist(),
            "drawdown_curve": drawdown.tolist(),
            "metrics": metrics,
            "trade_stats": {
                "num_trades": num_trades,
                "trade_count": num_trades,
                "turnover": round(turnover, 4),
                "exposure": round(exposure, 4),
                "exposure_time": round(float(np.mean(np.array(signal) != 0)), 3)
            }
        }

    def _calculate_cagr(self, equity: np.ndarray) -> float:
        """CAGR = (Final/Initial) ^ (252/n) - 1"""
        n = len(equity)
        if n < 2 or equity[-1] <= 0: return 0.0
        return float((equity[-1] / equity[0])**(252 / n) - 1)

    def _calculate_annualized_volatility(self, returns: np.ndarray) -> float:
        return compute_annualized_volatility(returns)

    def _calculate_sharpe(self, returns: np.ndarray, rfr: float = 0.0) -> float:
        """Annualized Sharpe = (Mean/Std) * sqrt(252)"""
        if rfr != 0.0:
            returns = np.asarray(returns, dtype=float) - float(rfr)
        return compute_sharpe_ratio(returns)

    def _calculate_sortino(self, returns: np.ndarray, rfr: float = 0.0) -> float:
        """Annualized Sortino = (Mean/DownsideStd) * sqrt(252)"""
        if rfr != 0.0:
            returns = np.asarray(returns, dtype=float) - float(rfr)
        return compute_sortino_ratio(returns)

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Max Drawdown = Peak-to-Trough percentage drop."""
        return compute_max_drawdown(equity)

    def _calculate_average_drawdown(self, drawdown: np.ndarray) -> float:
        return compute_average_drawdown(drawdown)

    def _calculate_calmar(self, cagr: float, max_drawdown: float) -> float:
        return compute_calmar_ratio(cagr, max_drawdown)

    def _calculate_tail_loss(self, returns: np.ndarray, quantile: float = 0.05) -> float:
        return compute_tail_loss(returns, quantile=quantile)

    def _calculate_turnover(self, signal: np.ndarray | pd.Series) -> float:
        return compute_signal_turnover(signal)

    def _calculate_exposure(self, signal: np.ndarray | pd.Series) -> float:
        return compute_exposure_ratio(np.abs(np.asarray(signal, dtype=float)))

    def evaluate_prediction_quality(
        self,
        predicted_returns: np.ndarray | pd.Series,
        realized_returns: np.ndarray | pd.Series,
        predicted_direction: np.ndarray | pd.Series,
        realized_direction: np.ndarray | pd.Series,
    ) -> Dict[str, float]:
        metrics = compute_prediction_error_metrics(
            actual=realized_returns,
            predicted=predicted_returns,
            actual_direction=realized_direction,
            predicted_direction=predicted_direction,
        )
        if metrics["observations"] == 0:
            return {"rmse": 0.0, "mae": 0.0, "directional_accuracy": 0.0}
        return {
            "rmse": round(float(metrics["rmse"]), 6),
            "mae": round(float(metrics["mae"]), 6),
            "directional_accuracy": round(float(metrics["directional_accuracy"]), 6),
        }

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Win Rate = count_positive / count_nonzero."""
        return compute_win_rate(returns, ignore_zero_returns=True)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Profit Factor = Sum(Gains) / |Sum(Losses)|."""
        return compute_profit_factor(returns)

    def compare_vs_baseline(self, model_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Comparison helper for reports."""
        return {
            "alpha_cagr": round(model_metrics["cagr"] - baseline_metrics["cagr"], 4),
            "alpha_sharpe": round(model_metrics["sharpe"] - baseline_metrics["sharpe"], 2),
            "outperformance": model_metrics["total_return"] > baseline_metrics["total_return"]
        }
    
    # ── Legacy Compatibility ─────────────────────────────────────
    def evaluate_financial_metrics(self, decisions: List[DecisionCard]) -> Dict[str, Any]:
        """Legacy hook used by older simulator code paths.

        The canonical research and demo path should use the manifest-driven
        trainer and backtest layers instead of decision-card replay.
        """
        if not decisions or len(decisions) < 2:
            return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            
        prices = [float(d.tech_summary.get("close", 1.0)) for d in decisions]
        weights = [float(d.target_weight) for d in decisions]
        
        # Mock-to-Real Bridge
        returns = pd.Series(prices).pct_change().fillna(0).values
        signal = np.array(weights)
        
        eval_res = self.evaluate_strategy(signal, returns)
        metrics = dict(eval_res["metrics"])
        if prices:
            metrics["total_return_pct"] = round(((prices[-1] / prices[0]) - 1.0) * 100.0, 4)
        else:
            metrics["total_return_pct"] = 0.0
        return metrics
