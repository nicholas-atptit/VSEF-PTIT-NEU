"""Pipeline execution for Phase 3 LLM Analysis.

Leverages the local Ollama LLM to perform qualitative analysis by merging
quantitative signals and vectorized RAG context into a unified JSON response.
"""

from __future__ import annotations

import json
from typing import Any

from config.settings import get_settings
from src.llm.client import get_llm_client
from src.llm.prompts import build_system_prompt, build_user_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def run_qualitative_analysis(
    ticker: str,
    quant_data: dict[str, Any],
    rag_context: str,
    news_context: str = "",
    user_risk_input: float = 1.0,
) -> dict[str, Any]:
    """Execute the local LLM to generate qualitative JSON analysis.

    Args:
        ticker: Stock symbol (e.g., 'SSI')
        quant_data: Quantitative payload from Phase 2 ML models
        rag_context: Text extracted from Phase 1 Vector DB
        news_context: Latest news headlines from market sources
        user_risk_input: User's raw requested risk tolerance (0.0 to 1.0)

    Returns:
        A dictionary parsed from the LLM's JSON output containing
        sentiment, risk_factor, and reasoning.
    """
    settings = get_settings()
    client = get_llm_client()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        ticker=ticker,
        user_risk_input=user_risk_input,
        quant_data=quant_data,
        rag_context=rag_context,
        news_context=news_context,
    )

    logger.info(
        "llm_analysis_started",
        ticker=ticker,
        model=settings.llm_model_name,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            timeout=15.0,  # Strict timeout to prevent UI network errors
        )
    except Exception as e:
        logger.warning(
            "llm_timeout_or_error_fallback",
            error=str(e),
            msg="Ollama took too long to respond or failed. Falling back to default analysis."
        )
        return {
            "sentiment": "NEUTRAL",
            "risk_factor": "MODERATE",
            "reasoning": f"Local LLM ({settings.llm_model_name}) is currently overloaded or starting up. Analysis falls back to quantitative signals only.",
        }

    # ── Parse LLM response (now correctly outside the except block) ──
    try:
        result_text = response.choices[0].message.content
        if not result_text:
            raise ValueError("LLM returned an empty response")

        parsed_result = json.loads(result_text)

        # Validate required keys
        required_keys = {"analysis_status", "sentiment", "risk_factor", "reasoning", "system_parameters", "sources_used"}
        missing = required_keys - set(parsed_result.keys())
        if missing:
            logger.warning("llm_missing_keys", ticker=ticker, missing=list(missing))
            return {
                "analysis_status": "insufficient_data",
                "sentiment": parsed_result.get("sentiment", "neutral"),
                "risk_factor": parsed_result.get("risk_factor", "high"),
                "reasoning": f"Fallback: LLM thiếu khóa {missing}. {parsed_result.get('reasoning', '')}",
                "system_parameters": parsed_result.get("system_parameters", {
                    "applied_risk_tolerance": min(user_risk_input, 0.70),
                    "confidence_metrics": {
                        "stock_quantitative_data": 0.95,
                        "rag_context_data": 0.70
                    }
                }),
                "sources_used": parsed_result.get("sources_used", [])
            }

        logger.info("llm_analysis_completed", ticker=ticker, status=parsed_result["analysis_status"])
        return parsed_result

    except json.JSONDecodeError as e:
        logger.error("llm_json_decode_error", ticker=ticker, error=str(e), raw_output=result_text)
        return {
            "analysis_status": "insufficient_data",
            "sentiment": "neutral",
            "risk_factor": "high",
            "reasoning": f"Lỗi parse JSON từ LLM: {str(e)}",
            "system_parameters": {
                "applied_risk_tolerance": min(user_risk_input, 0.70),
                "confidence_metrics": {"stock_quantitative_data": 0.95, "rag_context_data": 0.70}
            },
            "sources_used": []
        }
    except Exception as e:
        logger.error("llm_connection_failed", ticker=ticker, error=str(e))
        return {
            "analysis_status": "insufficient_data",
            "sentiment": "neutral",
            "risk_factor": "high",
            "reasoning": f"Lỗi kết nối Ollama LLM Local: {str(e)}",
            "system_parameters": {
                "applied_risk_tolerance": min(user_risk_input, 0.70),
                "confidence_metrics": {"stock_quantitative_data": 0.95, "rag_context_data": 0.70}
            },
            "sources_used": []
        }
