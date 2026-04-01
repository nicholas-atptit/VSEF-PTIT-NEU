"""Tests for Phase 3: Qualitative Risk Analyst Engine.

Validates the strict SOP compliance: evidence-locked reasoning, 
zero-price prediction, veto logic, and JSON schema enforcement.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.ml.llm.pipeline import run_qualitative_analysis
from src.ml.llm.prompts import build_system_prompt, build_user_prompt


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── Testing Prompts Logic ──────────────────────────────────────────

class TestPrompts:
    def test_system_prompt_enforces_sop(self):
        prompt = build_system_prompt()
        assert "Risk Analyst Engine" in prompt
        assert "EVIDENCE-LOCKED REASONING" in prompt
        assert "NO PRICE / NO TRADING LOGIC" in prompt
        assert "VETO LOGIC" in prompt
        assert "ZONE_DATA" in prompt

    def test_user_prompt_formats_zones(self):
        quant_mock = {"trend_probabilities": {"up": 0.8}, "expected_range": {"median": 100}}
        rag_mock = "Zone 1 data text"
        news_mock = "Zone 2 news text"

        prompt = build_user_prompt(
            ticker="SSI",
            quant_data=quant_mock,
            rag_context=rag_mock,
            news_context=news_mock
        )

        assert "SSI" in prompt
        assert "[Zone 1: Fundamental/RAG]" in prompt
        assert "Zone 1 data text" in prompt
        assert "[Zone 2: Latest News]" in prompt
        assert "0.8" in prompt


# ── Testing LLM Pipeline (SOP Logic) ──────────────────────────────

class TestQualitativeAnalysisSOP:
    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_success_sop(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        mock_choice.message.content = json.dumps({
            "analysis_status": "success",
            "confidence_score": 0.85,
            "veto_flag": False,
            "overall_outlook": "positive",
            "reasoning": "Dòng tiền mạnh từ khối ngoại và TFT dự báo UP.",
            "signals": {
                "bullish": [{"evidence": "Khối ngoại mua ròng", "zone": "zone_2"}],
                "bearish": []
            },
            "deep_learning_context": {
                "tft_forecast": "Uptrend confirmed by sequence models",
                "cnn_microstructure": "Neutral"
            },
            "rl_recommendation": {
                "suggested_allocation_pct": 0.45,
                "justification": "Optimal risk/reward"
            },
            "main_risks": [],
            "anti_hallucination_check_passed": True
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        dl_mock = {"tft_sequence_forecast": {"expected_trend": "UP"}}
        rl_mock = {"suggested_allocation_pct": 0.45}

        result = await run_qualitative_analysis(
            ticker="SSI",
            quant_data={},
            rag_context="Data",
            dl_data=dl_mock,
            rl_data=rl_mock
        )

        assert result["analysis_status"] == "success"
        assert result["deep_learning_context"]["tft_forecast"] == "Uptrend confirmed by sequence models"
        assert result["rl_recommendation"]["suggested_allocation_pct"] == 0.45

    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_veto_logic(self, mock_get_client):
        """If veto_flag is True, result should be negative regardless of other fields."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        mock_choice.message.content = json.dumps({
            "analysis_status": "success",
            "confidence_score": 0.9,
            "veto_flag": True,  # VETO TRIGGERED
            "overall_outlook": "positive", # Conflicting outlook from model
            "reasoning": "Phát hiện dấu hiệu gian lận tài chính.",
            "signals": {"bullish": [], "bearish": []},
            "main_risks": [{"risk_type": "legal", "description": "Gian lận", "zone": "zone_1"}],
            "anti_hallucination_check_passed": True
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = await run_qualitative_analysis(ticker="SSI", quant_data={}, rag_context="Data")

        # Pipeline must override to negative due to Veto Logic
        assert result["veto_flag"] is True
        assert result["overall_outlook"] == "negative"

    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_kill_switch(self, mock_get_client):
        """If confidence < 0.7, trigger insufficient_data."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        mock_choice.message.content = json.dumps({
            "analysis_status": "success",
            "confidence_score": 0.4, # TOO LOW
            "veto_flag": False,
            "overall_outlook": "positive",
            "reasoning": "Dữ liệu quá mờ nhạt.",
            "signals": {"bullish": [], "bearish": []},
            "main_risks": [],
            "anti_hallucination_check_passed": True
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = await run_qualitative_analysis(ticker="SSI", quant_data={}, rag_context="Data")

        assert result["analysis_status"] == "insufficient_data"
        assert result["overall_outlook"] is None
        assert result["signals"] is None
