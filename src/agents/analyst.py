from __future__ import annotations

from .contracts import AnalystDecision, MarketSignal


class AnalystAgent:
    """Rule-based analyst agent.

    Purpose:
    - consume normalized quantitative signals
    - produce a stable BUY/SELL/HOLD decision
    - expose reasons that can later feed an LLM explainer
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        min_abs_pred_return: float = 0.005,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_abs_pred_return = min_abs_pred_return

    def decide(self, signal: MarketSignal) -> AnalystDecision:
        score = 0.0
        reasons: list[str] = []

        if signal.confidence >= self.min_confidence:
            score += 1.0
            reasons.append(f"model confidence {signal.confidence:.2f} >= {self.min_confidence:.2f}")
        else:
            score -= 1.0
            reasons.append(f"model confidence {signal.confidence:.2f} < {self.min_confidence:.2f}")

        if signal.pred_return >= self.min_abs_pred_return:
            score += 1.5
            reasons.append(f"predicted return {signal.pred_return:.2%} is positive enough")
        elif signal.pred_return <= -self.min_abs_pred_return:
            score -= 1.5
            reasons.append(f"predicted return {signal.pred_return:.2%} is negative enough")
        else:
            reasons.append("predicted return is too small to be directional")

        if signal.trend_up_prob > 0.60:
            score += 1.0
            reasons.append(f"uptrend probability {signal.trend_up_prob:.2f} > 0.60")
        if signal.trend_down_prob > 0.60:
            score -= 1.0
            reasons.append(f"downtrend probability {signal.trend_down_prob:.2f} > 0.60")

        if signal.sentiment_score > 0.15:
            score += 0.5
            reasons.append("positive sentiment support")
        elif signal.sentiment_score < -0.15:
            score -= 0.5
            reasons.append("negative sentiment drag")

        if signal.rsi_14 is not None:
            if signal.rsi_14 >= 75:
                score -= 0.5
                reasons.append(f"RSI {signal.rsi_14:.1f} indicates overbought risk")
            elif signal.rsi_14 <= 30:
                score += 0.5
                reasons.append(f"RSI {signal.rsi_14:.1f} indicates oversold rebound potential")

        if signal.sma_20 and signal.sma_50 and signal.current_price > signal.sma_20 > signal.sma_50:
            score += 0.5
            reasons.append("price > SMA20 > SMA50 confirms short-term trend")

        action = "HOLD"
        if score >= 1.5:
            action = "BUY"
        elif score <= -1.5:
            action = "SELL"

        confidence = min(0.95, max(0.05, 0.50 + abs(score) * 0.10))
        return AnalystDecision(
            ticker=signal.ticker,
            action=action,
            confidence=confidence,
            score=score,
            reasons=reasons,
        )
