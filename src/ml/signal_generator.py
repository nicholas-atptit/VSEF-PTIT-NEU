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

from src.api.schemas_v2 import TerminalPayload, SentimentForecast
from src.engine.matrix import evaluate_decision_matrix
from src.engine.risk import apply_risk_constraints
from config.settings import get_settings
from src.utils.logging import get_logger
from src.llm.news_intel import NewsIntelEngine

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
        self._intel_engine = NewsIntelEngine()

    async def generate(
        self,
        ticker: str,
        current_close: float,
        model_output: dict[str, Any],
        risk_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Generate full prediction payload matching the TerminalPayload (v5.0) contract.

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

        # --- Sentiment Agent (Phase 3) ---
        horizon = model_output.get("horizon", "short")
        sentiment_data = await self._intel_engine.get_latest_intelligence(ticker, horizon=horizon)
        
        sentiment_payload = None
        if sentiment_data:
            sentiment_payload = {
                "sentiment_score": float(sentiment_data["sentiment_score"]),
                "sentiment_regime": sentiment_data["trend"].lower(),
                "sentiment_confidence": 0.85, # Default confidence for LLM analysis
                "source_breakdown": [], # Placeholder for detailed sources
                "market_psychology_tags": [], # Future enhancement
                "narrative_risk_flags": [],
            }

        # --- Phase 4: Decision Fusion (Agent Matrix) ---
        from src.api.schemas import QuantitativeSignals, QualitativeAnalysis
        
        # Adapt v5.0 inputs for compat with matrix.py
        quant_model = QuantitativeSignals(
            trend_probabilities=trend_probs,
            expected_range=expected_range,
            max_upside_pct=0.0,
            max_downside_pct=0.0,
            horizon="short",
            feature_set_version="v5.0",
            action_plan={"recommendation": "STAND_ASIDE", "entry_zone": [], "stop_loss": 0, "take_profit": 0}
        )
        # Handle recommendation from raw trend_probs for matrix
        if trend_probs.get("up", 0) > 0.6: quant_model.action_plan.recommendation = "BUY"
        elif trend_probs.get("down", 0) > 0.6: quant_model.action_plan.recommendation = "SELL"
        
        qual_model = QualitativeAnalysis(
            ticker=ticker,
            sentiment=sentiment_payload["sentiment_regime"] if sentiment_payload else "neutral", # Use actual sentiment
            confidence=sentiment_payload["sentiment_confidence"] if sentiment_payload else 0.5, # Use actual confidence
            analysis_status="success",
            reasoning="Phase 3 Intelligence"
        )
        
        matrix_decision, consensus = evaluate_decision_matrix(
            quant=quant_model,
            qual=qual_model,
            weights={"technical": 0.6, "sentiment": 0.4}
        )
        
        # --- Phase 4: Risk Overlay ---
        # Placeholder ATR (In production this would come from the live feature stream)
        mock_atr = current_close * 0.03 # 3% daily volatility approx
        
        order_payload, risk_record = apply_risk_constraints(
            ticker=ticker,
            action_plan=quant_model.action_plan,
            real_time_price=current_close,
            atr_14=mock_atr,
            applied_risk_tolerance=min(risk_tolerance if risk_tolerance is not None else 0.0, 0.70)
        )
        
        # --- Terminal Payload (Standard Contract v5.0) ---
        return {
            "ticker": ticker,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "technical": {
                "horizons": [{
                    "horizon": "1w",
                    "trend_probs": trend_probs,
                    "expected_range": expected_range,
                    "model_confidence": 0.85
                }],
                "agent_weights": {"technical": 0.6, "sentiment": 0.4},
                "regime_detected": "trend"
            },
            "sentiment": sentiment_payload, # Use the raw sentiment_payload
            "fusion": {
                "regime_detected": "trend",
            },
            "risk": {
                "position_size_suggestion": 0.0,
                "veto_flag": False,
                "constraints_hit": [],
                "risk_budget_consumed": 0.0,
            },
            "run_id": f"run_{int(dt.datetime.now().timestamp())}",
            "status": "success",
        }

        logger.info(
            "signal_generated_v5",
            ticker=ticker,
            action=action_plan["recommendation"],
            horizon=horizon,
            has_sentiment=bool(sentiment_payload),
        )

        return payload

    # ── Action Plan Logic ─────────────────────────────────────

    def _generate_action_plan(
        self,
        trend_probs: dict[str, float],
        expected_range: dict[str, float],
        current_close: float,
    ) -> dict[str, Any]:
        """Map trend probabilities + range to specific entry/exit zones.

        Phase 2 Upgrade: Uses quantile ranges for strict zone definition.
        """
        p_up = trend_probs["up"]
        p_down = trend_probs["down"]

        q10 = expected_range["bottom_10th"]
        q50 = expected_range["median_50th"]
        q90 = expected_range["ceiling_90th"]

        if p_up > DIRECTIONAL_THRESHOLD:
            # ── BULLISH: BUY ZONE ──
            # Entry: Between current price and Q10 support
            entry_low = min(q10, current_close)
            entry_high = max(q10, current_close)
            
            return {
                "recommendation": ACTION_BUY,
                "entry_zone": [round(entry_low, 2), round(entry_high, 2)],
                "exit_zones": {
                    "take_profit_target": round(q90, 2),
                    "exit_conservative": round(q50, 2),
                    "stop_loss_hard": round(q10 * 0.985, 2), # 1.5% buffer below Q10
                },
                "rationale": f"Strong bullish conviction ({p_up:.1%}) with target at {q90}"
            }

        elif p_down > DIRECTIONAL_THRESHOLD:
            # ── BEARISH: SELL / SHORT ZONE ──
            return {
                "recommendation": ACTION_SELL,
                "entry_zone": [round(current_close, 2), round(q90, 2)],
                "exit_zones": {
                    "take_profit_target": round(q10, 2),
                    "exit_conservative": round(q50, 2),
                    "stop_loss_hard": round(q90 * 1.015, 2),
                },
                "rationale": f"Bearish regime detected ({p_down:.1%}). Risk of drop to {q10}"
            }

        else:
            # ── NEUTRAL: RANGE TRADE ──
            return {
                "recommendation": ACTION_RANGE_TRADE,
                "entry_zone": [round(q10, 2), round(q10 * 1.02, 2)], # Buy near support
                "exit_zones": {
                    "take_profit_target": round(q90, 2),
                    "stop_loss_hard": round(q10 * 0.97, 2),
                },
                "rationale": "Neutral trend. Buy near lower support (Q10) for range target (Q90)."
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
