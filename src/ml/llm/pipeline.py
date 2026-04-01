"""Pipeline execution for Phase 3 LLM Analysis.

Leverages the local Ollama LLM to perform qualitative analysis by merging
quantitative signals and vectorized RAG context into a unified JSON response.
"""

from __future__ import annotations

import json
from typing import Any

from config.settings import get_settings
from src.ml.llm.client import get_llm_client
from src.ml.llm.prompts import build_system_prompt, build_user_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def run_qualitative_analysis(
    ticker: str,
    quant_data: dict[str, Any],
    rag_context: str,
    news_context: str = "",
    dl_data: dict[str, Any] = None,
    rl_data: dict[str, Any] = None,
) -> dict[str, Any]:
    """Execute the local LLM to generate qualitative JSON analysis.

    Args:
        ticker: Stock symbol (e.g., 'SSI')
        quant_data: Quantitative payload from Phase 2 ML models
        rag_context: Text extracted from Phase 1 Vector DB
        news_context: Latest news headlines from market sources
        dl_data: Phase 10 Deep Learning signals (TFT, CNN)
        rl_data: Phase 10 RL Portfolio Manager recommendations

    Returns:
        A dictionary parsed from the LLM's JSON output.
    """
    settings = get_settings()
    client = get_llm_client()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        ticker=ticker,
        quant_data=quant_data,
        rag_context=rag_context,
        news_context=news_context,
        dl_data=dl_data,
        rl_data=rl_data,
    )

    # ── Resolve Model Name based on Provider ──
    provider = settings.llm_provider
    if provider == "openai":
        model_name = settings.openai_model_name
    elif provider == "groq":
        model_name = settings.groq_model_name
    elif provider == "gemini":
        model_name = settings.gemini_model_name
    else:
        model_name = settings.ollama_model_name

    logger.debug("llm_analysis_started", ticker=ticker, provider=provider, model=model_name)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            timeout=25.0,  # Increased timeout for complex SOP reasoning
        )
    except Exception as e:
        logger.warning(
            "llm_error_fallback",
            error=str(e),
            ticker=ticker
        )
        return {
            "analysis_status": "insufficient_data",
            "confidence_score": 0.0,
            "veto_flag": False,
            "overall_outlook": None,
            "reasoning": f"Local LLM Error: {str(e)}",
            "signals": None,
            "deep_learning_context": None,
            "rl_recommendation": None,
            "main_risks": None
        }

    try:
        result_text = response.choices[0].message.content
        if not result_text:
            raise ValueError("LLM returned an empty response")

        parsed_result = json.loads(result_text)

        # ── SOP Validation & Consistency Checks ──
        status = parsed_result.get("analysis_status", "insufficient_data")
        score = parsed_result.get("confidence_score", 0.0)
        
        # Enforce Kill Switch (SOP Step 7)
        if score < 0.7:
            status = "insufficient_data"

        if status == "insufficient_data":
            return {
                "analysis_status": "insufficient_data",
                "confidence_score": score,
                "veto_flag": False,
                "overall_outlook": None,
                "reasoning": parsed_result.get("reasoning", "Dữ liệu không đủ hoặc mâu thuẫn theo SOP."),
                "signals": None,
                "deep_learning_context": None,
                "rl_recommendation": None,
                "main_risks": None
            }

        # Enforce Veto Logic (SOP Section III)
        veto_triggered = parsed_result.get("veto_flag", False)
        if veto_triggered:
            parsed_result["overall_outlook"] = "negative"

        logger.info("llm_analysis_completed", ticker=ticker, status=status, score=score, veto=veto_triggered)
        return parsed_result

    except Exception as e:
        logger.error("llm_parse_failed", ticker=ticker, error=str(e))
        return {
            "analysis_status": "insufficient_data",
            "confidence_score": 0.0,
            "veto_flag": False,
            "overall_outlook": None,
            "reasoning": f"Lỗi hệ thống khi xử lý kết quả LLM: {str(e)}",
            "signals": None,
            "deep_learning_context": None,
            "rl_recommendation": None,
            "main_risks": None
        }
