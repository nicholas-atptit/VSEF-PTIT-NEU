"""Module 3: Quantitative Logic Engine — Signal Generator.

Maps outputs from Model A (trend classifier) and Model B (range regressor)
into actionable trading plans with enforced system constraints.

System Constraints:
    - max_risk_tolerance capped at 0.70 (70%) regardless of input.
    - confidence_metrics.stock_quantitative_data = 0.95
    - confidence_metrics.general_market_context = 0.70
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Action constants ──────────────────────────────────────────
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_RANGE_TRADE = "RANGE_TRADE"
ACTION_STAND_ASIDE = "STAND_ASIDE"

# ── Probability threshold for directional conviction ─────────
DIRECTIONAL_THRESHOLD = 0.60


class SignalGenerator:
    """Generates trading signals from dual-model predictions.

    Usage::

        sg = SignalGenerator()
        signal = sg.generate(
            ticker="SSI",
            current_close=35.0,
            model_output={
                "trend_probabilities": {"up": 0.65, "sideways": 0.15, "down": 0.20},
                "expected_range": {"bottom_10th": 34.5, "median_50th": 35.8, "ceiling_90th": 36.9},
            },
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def generate(
        self,
        ticker: str,
        current_close: float,
        model_output: dict[str, Any],
        risk_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Generate full prediction payload matching the JSON API contract.

        Args:
            ticker: Stock symbol.
            current_close: Latest closing price.
            model_output: Output from DualModelTrainer.predict().
            risk_tolerance: Requested risk tolerance (will be capped at 0.70).

        Returns:
            Complete JSON-serializable payload for the API response.
        """
        trend_probs = model_output["trend_probabilities"]
        expected_range = model_output["expected_range"]

        # Generate action plan based on dominant trend
        action_plan = self._generate_action_plan(
            trend_probs=trend_probs,
            expected_range=expected_range,
            current_close=current_close,
        )

        # Enforce system constraints
        system_params = self._build_system_parameters(risk_tolerance)

        payload = {
            "ticker": ticker.upper(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "quantitative_signals": {
                "trend_probabilities": trend_probs,
                "expected_range": expected_range,
                "action_plan": action_plan,
            },
            "system_parameters": system_params,
        }

        logger.info(
            "signal_generated",
            ticker=ticker,
            action=action_plan["recommendation"],
            p_up=trend_probs["up"],
        )

        return payload

    # ── Action Plan Logic ─────────────────────────────────────

    def _generate_action_plan(
        self,
        trend_probs: dict[str, float],
        expected_range: dict[str, float],
        current_close: float,
    ) -> dict[str, Any]:
        """Map trend probabilities + range to a specific trading action.

        Rules:
            P_Up > 60%      → BUY
            P_Down > 60%    → SELL / STAND ASIDE
            Otherwise       → RANGE_TRADE (ping-pong)
        """
        p_up = trend_probs["up"]
        p_down = trend_probs["down"]

        q10 = expected_range["bottom_10th"]
        q50 = expected_range["median_50th"]
        q90 = expected_range["ceiling_90th"]

        if p_up > DIRECTIONAL_THRESHOLD:
            # ── UPTREND: BUY ──
            stop_loss = round(q10 * (1 - 0.015), 2)  # Q10 - 1.5%
            return {
                "recommendation": ACTION_BUY,
                "entry_zone": [q10, current_close],
                "stop_loss": stop_loss,
                "take_profit": q90,
            }

        elif p_down > DIRECTIONAL_THRESHOLD:
            # ── DOWNTREND: SELL / STAND ASIDE ──
            return {
                "recommendation": ACTION_SELL,
                "entry_zone": [q90, q90],  # Wait for bounce to Q90
                "stop_loss": round(q90 * 1.015, 2),  # Above Q90
                "take_profit": q10,
            }

        else:
            # ── SIDEWAYS: RANGE TRADE ──
            return {
                "recommendation": ACTION_RANGE_TRADE,
                "entry_zone": [q10, q90],
                "stop_loss": round(q10 * (1 - 0.015), 2),
                "take_profit": q90,
            }

    # ── System Constraints ────────────────────────────────────

    def _build_system_parameters(
        self,
        risk_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Build system parameters with enforced constraints.

        Core Rule — Risk Cap Override:
            max_risk_tolerance is ALWAYS capped at 0.70 (70%)
            even if the client requests a higher value.

        Confidence Routing:
            stock_quantitative_data  → 0.95 (model-derived)
            general_market_context   → 0.70 (default for context/metadata)
        """
        max_risk = self._settings.max_risk_tolerance

        if risk_tolerance is not None:
            # Clamp to ceiling — never exceed 0.70
            max_risk = min(float(risk_tolerance), max_risk)

        return {
            "max_risk_tolerance": max_risk,
            "confidence_metrics": {
                "stock_quantitative_data": self._settings.confidence_stock_quantitative,
                "general_market_context": self._settings.confidence_general_context,
            },
        }
