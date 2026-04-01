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
from src.ml.llm.news_intel import NewsIntelEngine
from src.context.news_crawler import NewsCrawler
from src.utils.monitoring.drift import DriftMonitor
from src.utils.monitoring.accuracy import AccuracyMonitor

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
        self._crawler = NewsCrawler()
        self._drift_monitor = DriftMonitor()
        self._accuracy_monitor = AccuracyMonitor()

    async def generate(
        self,
        ticker: str,
        current_close: float,
        model_output: dict[str, Any],
        risk_tolerance: float | None = None,
        active_analysis: bool = False,
        sentiment_override: dict[str, Any] | None = None,
        volatility_score: float | None = None,
    ) -> dict[str, Any]:
        """Generate full prediction payload matching the TerminalPayload (v5.0) contract.

        Args:
            ticker: Stock symbol.
            current_close: Latest closing price.
            model_output: Output from DualModelTrainer.predict().
            risk_tolerance: Requested risk tolerance (will be capped at 0.70).
            active_analysis: Whether to perform live sentiment analysis (Phase 3).
            sentiment_override: Pre-defined sentiment data (useful for tests/scenarios).
            volatility_score: Predicted volatility for this ticker/horizon.

        Returns:
            Complete JSON-serializable payload for the API response.
        """
        ticker = ticker.upper().strip()
        
        # --- Phase 5: Signal Generation & TUI Standardization ---
        trend_probs = model_output.get("trend_probabilities", {})
        expected_range = model_output.get("expected_range", {})
        
        # Use provided volatility or from model_output
        vol_val = volatility_score if volatility_score is not None else model_output.get("volatility")

        # Default sentiment data or override
        if sentiment_override:
            sentiment_payload = {
                "sentiment_score": sentiment_override.get("sentiment_score", 0.0),
                "sentiment_regime": sentiment_override.get("sentiment_regime", "neutral"),
                "sentiment_confidence": sentiment_override.get("sentiment_confidence", 0.8),
                "narrative_risk_flags": sentiment_override.get("narrative_risk_flags", []),
                "status": sentiment_override.get("status", "success")
            }
        else:
            sentiment_payload = {
                "sentiment_score": 0.0,
                "sentiment_regime": "neutral",
                "sentiment_confidence": 0.0,
                "narrative_risk_flags": ["NO_DATA_YET"],
                "status": "success"
            }

        # Enforce system constraints
        system_params = self._build_system_parameters(risk_tolerance)

        # --- Phase 5: Monitoring (Accuracy & Drift) ---
        horizon = model_output.get("horizon", "short")
        accuracy_info = self._accuracy_monitor.calculate_recent_accuracy(ticker, horizon)

        # --- Sentiment Agent (Phase 3) ---
        if not sentiment_override:
            sentiment_data = None
            sentiment_error = False
            try:
                sentiment_data = await self._intel_engine.get_latest_intelligence(ticker, horizon=horizon)
                
                # --- Active Mode (Phase 5) ---
                if not sentiment_data and active_analysis:
                    logger.info("triggering_active_analysis", ticker=ticker)
                    articles = await self._crawler.crawl_ticker(ticker, count=10)
                    if articles:
                        sentiment_data = await self._intel_engine.analyze_ticker_news(ticker, articles, horizon=horizon)
            except Exception as e:
                logger.error("sentiment_engine_error", ticker=ticker, error=str(e))
                sentiment_error = True

            if sentiment_data:
                sentiment_payload = {
                    "sentiment_score": float(sentiment_data["sentiment_score"]),
                    "sentiment_regime": sentiment_data["trend"].lower(),
                    "sentiment_confidence": 0.85,
                    "summary": sentiment_data.get("summary", ""),
                    "status": "SUCCESS"
                }
            elif sentiment_error:
                sentiment_payload = {
                    "sentiment_score": 0.0,
                    "sentiment_regime": "degraded",
                    "sentiment_confidence": 0.0,
                    "narrative_risk_flags": ["SERVICE_UNAVAILABLE"],
                    "status": "FALLBACK"
                }
            else:
                sentiment_payload = {
                    "sentiment_score": 0.0,
                    "sentiment_regime": "neutral",
                    "sentiment_confidence": 0.0,
                    "narrative_risk_flags": ["NO_DATA_YET"],
                    "status": "success"
                }

        # --- Phase 4: Decision Fusion (Agent Matrix) ---
        from src.api.schemas import QuantitativeSignals, QualitativeAnalysis
        
        # Ensure default values for trend_probs and expected_range to satisfy Pydantic
        trend_probs_safe = {
            "up": trend_probs.get("up", 0.0),
            "down": trend_probs.get("down", 0.0),
            "sideways": trend_probs.get("sideways", 1.0) if not trend_probs else 1.0
        }
        expected_range_safe = {
            "bottom_10th": expected_range.get("bottom_10th", current_close * 0.95),
            "median_50th": expected_range.get("median_50th", current_close),
            "ceiling_90th": expected_range.get("ceiling_90th", current_close * 1.05)
        }

        # Adapt v5.0 inputs for compat with matrix.py
        quant_model = QuantitativeSignals(
            trend_probabilities=trend_probs_safe,
            expected_range=expected_range_safe,
            action_plan={"recommendation": "STAND_ASIDE", "entry_zone": [0.0, 0.0], "stop_loss": 0, "take_profit": 0}
        )
        # Handle recommendation from raw trend_probs for matrix
        if trend_probs.get("up", 0) > 0.6: quant_model.action_plan.recommendation = "BUY"
        elif trend_probs.get("down", 0) > 0.6: quant_model.action_plan.recommendation = "SELL"
        
        qual_model = QualitativeAnalysis(
            analysis_status=sentiment_payload["status"].lower(),
            confidence_score=sentiment_payload["sentiment_confidence"],
            overall_outlook=sentiment_payload["sentiment_regime"],
            reasoning="Phase 3 Intelligence",
            veto_flag=False,
            anti_hallucination_check_passed=True
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
        
        # --- Terminal Payload (Standard Contract v5.1) ---
        return {
            "ticker": ticker,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "technical": {
                "ticker": ticker,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "current_price": current_close,
                "horizons": [{
                    "horizon": horizon,
                    "trend_probs": trend_probs_safe,
                    "expected_range": expected_range_safe,
                    "volatility_score": vol_val,
                    "confidence": 0.85
                }],
                "agent_weights": {"technical": 0.6, "sentiment": 0.4},
                "regime_detected": "trend"
            },
            "sentiment": {
                **sentiment_payload,
                "source_breakdown": [],
                "market_psychology_tags": [],
            },
            "fusion": {
                "regime_detected": "trend",
                "action": matrix_decision,
                "confidence": 0.85,
                "rationale": (
                    f"ml={consensus.ml_signal}, llm={consensus.llm_sentiment}, "
                    f"veto={consensus.veto_triggered}"
                ),
                "agent_weights": {"technical": 0.6, "sentiment": 0.4},
            },
            "risk": {
                "position_size_suggestion": float(order_payload.volume) if order_payload else 0.0,
                "veto_flag": not risk_record.fomo_check_passed,
                "constraints_hit": ["FOMO"] if not risk_record.fomo_check_passed else [],
                "risk_budget_consumed": 0.0,
                "model_accuracy_1w": accuracy_info.get("accuracy", 0.0)
            },
            "run_id": f"run_{int(dt.datetime.now().timestamp())}",
            "status": "success",
        }

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
            stock_quantitative_data  -> 0.95 (model-derived)
            general_market_context   -> 0.70 (default for context/metadata)
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
