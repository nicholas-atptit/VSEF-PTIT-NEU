from __future__ import annotations

from typing import Any

from src.agents.contracts import MarketSignal


def build_market_signal(
    ticker: str,
    current_price: float,
    model_output: dict[str, Any],
    feature_snapshot: dict[str, Any] | None = None,
    sentiment_payload: dict[str, Any] | None = None,
) -> MarketSignal:
    """Normalize current repo outputs into a stable contract.

    This adapter is the key seam between the existing ML pipeline and the new agent layer.
    """
    feature_snapshot = feature_snapshot or {}
    sentiment_payload = sentiment_payload or {}
    trend_probs = model_output.get("trend_probabilities", {}) or {}

    pred_return = float(model_output.get("pred_return", 0.0))
    if pred_return == 0.0 and "expected_range" in model_output:
        median = float(model_output.get("expected_range", {}).get("median_50th", current_price))
        if current_price:
            pred_return = (median - current_price) / current_price

    regime = "unclear"
    up = float(trend_probs.get("up", 0.0))
    down = float(trend_probs.get("down", 0.0))
    side = float(trend_probs.get("sideways", 1.0 if not trend_probs else 0.0))
    if max(up, down) >= 0.60:
        regime = "trend"
    elif side >= 0.50:
        regime = "range"

    return MarketSignal(
        ticker=ticker.upper(),
        current_price=float(current_price),
        pred_return=pred_return,
        confidence=float(model_output.get("confidence", max(up, down, side))),
        volatility=float(model_output.get("volatility", feature_snapshot.get("volatility", 0.02))),
        sentiment_score=float(sentiment_payload.get("sentiment_score", 0.0)),
        trend_up_prob=up,
        trend_down_prob=down,
        trend_sideways_prob=side,
        rsi_14=_safe_float(feature_snapshot.get("rsi_14")),
        sma_20=_safe_float(feature_snapshot.get("sma_20")),
        sma_50=_safe_float(feature_snapshot.get("sma_50")),
        sma_200=_safe_float(feature_snapshot.get("sma_200")),
        atr_14=_safe_float(feature_snapshot.get("atr_14")),
        regime=regime,
        source="normalized_signal_builder",
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
