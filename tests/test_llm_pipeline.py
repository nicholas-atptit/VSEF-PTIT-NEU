"""Tests for Phase 3: LLM Qualitative Analysis Pipeline.

Validates the Ollama OpenAI client logic, Prompt building, JSON schema
enforcement, and the new `/analyze` API endpoint using mocks.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.llm.pipeline import run_qualitative_analysis
from src.llm.prompts import build_system_prompt, build_user_prompt


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── Testing Prompts Logic ──────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_enforces_json(self):
        prompt = build_system_prompt()
        assert "định dạng JSON" in prompt
        assert "Kịch bản A" in prompt
        assert "Kịch bản B" in prompt
        assert "sentiment" in prompt
        assert "risk_factor" in prompt
        assert "0.70" in prompt  # System ensures risk cap rule description

    def test_user_prompt_injects_data(self):
        quant_mock = {"mock_key": "mock_value", "trend": "UP"}
        rag_mock = "Tin đồn doanh thu giảm."

        prompt = build_user_prompt(
            ticker="SSI",
            user_risk_input=1.0,
            quant_data=quant_mock,
            rag_context=rag_mock,
        )

        assert "SSI" in prompt
        assert "100.0%" in prompt
        assert "mock_value" in prompt
        assert "doanh thu giảm" in prompt


# ── Testing LLM Pipeline (with Mocks) ──────────────────────────────


class TestQualitativeAnalysis:
    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_success(self, mock_get_client):
        # 1. Setup the AsyncMock for the OpenAI client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # 2. Mock the JSON returned by the model
        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        mock_choice.message.content = json.dumps({
            "analysis_status": "success",
            "sentiment": "positive",
            "risk_factor": "medium",
            "reasoning": "Doanh thu tăng trưởng mạnh mẽ.",
            "system_parameters": {
                "applied_risk_tolerance": 0.70,
                "confidence_metrics": {"stock_quantitative_data": 0.95, "rag_context_data": 0.70}
            },
            "sources_used": ["zone_1"]
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # 3. Call the pipeline
        result = await run_qualitative_analysis(
            ticker="SSI",
            quant_data={"mock": "data"},
            rag_context="[zone_1] BCTC tốt",
        )

        # 4. Assertions
        assert result["analysis_status"] == "success"
        assert result["sentiment"] == "positive"
        assert result["risk_factor"] == "medium"
        assert "Doanh thu" in result["reasoning"]
        assert result["sources_used"] == ["zone_1"]

    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_missing_keys(self, mock_get_client):
        """If the LLM omits keys, the pipeline must fallback to Kịch bản B (insufficient_data)."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        # Missing 'sentiment', 'risk_factor', 'analysis_status', 'sources_used', 'system_parameters'
        mock_choice.message.content = json.dumps({
            "reasoning": "Model failed to output standard limits.",
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = await run_qualitative_analysis(
            ticker="SSI",
            quant_data={},
            rag_context="",
        )

        assert result["analysis_status"] == "insufficient_data"
        assert result["sentiment"] == "neutral"  # Fallback default
        assert result["risk_factor"] == "high"    # Fallback default
        assert "Model failed to output" in result["reasoning"]
        assert result["sources_used"] == []

    @pytest.mark.asyncio
    @patch("src.llm.pipeline.get_llm_client")
    async def test_llm_pipeline_bad_json(self, mock_get_client):
        """If the LLM returns completely broken JSON, the pipeline must catch it gracefully."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        mock_response = AsyncMock()
        mock_choice = AsyncMock()
        mock_choice.message.content = "This is not JSON at all."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = await run_qualitative_analysis(
            ticker="SSI",
            quant_data={},
            rag_context="",
        )

        assert result["analysis_status"] == "insufficient_data"
        assert result["sentiment"] == "neutral"  # Fallback default
        assert result["risk_factor"] == "high"
        assert "Lỗi parse JSON" in result["reasoning"]
        assert result["sources_used"] == []


# ── Testing /analyze API ──────────────────────────────────────────


class TestAnalyzeEndpoint:
    @pytest.fixture(autouse=True)
    def _train_first(self, client: TestClient):
        """Ensure a model is trained before prediction tests."""
        client.post(
            "/api/v1/train",
            json={"ticker": "PRED", "use_mock": True},
        )

    @patch("src.api.routes.run_qualitative_analysis")
    @patch("src.api.routes.ticker_news")
    def test_analyze_endpoint_success(self, mock_ticker_news, mock_llm, client: TestClient):
        """The /analyze endpoint should return both quantitative and qualitative data."""
        # Force the mocked LLM pipeline to return sync standard response
        # Since the route is async, we mock it via AsyncMock logic
        async def mock_qualitative(*args, **kwargs):
            return {
                "analysis_status": "success",
                "sentiment": "negative",
                "risk_factor": "high",
                "reasoning": "Lãi suất tăng.",
                "system_parameters": {
                    "applied_risk_tolerance": 0.70,
                    "confidence_metrics": {"stock_quantitative_data": 0.95, "rag_context_data": 0.70}
                },
                "sources_used": ["zone_2", "zone_3"]
            }

        async def mock_news(*args, **kwargs):
            return {"ticker": "PRED", "news": []}

        mock_ticker_news.side_effect = mock_news
        mock_llm.side_effect = mock_qualitative

        response = client.get("/api/v1/analyze?ticker=PRED&use_mock=true&allowed_zones=zone_2&allowed_zones=zone_3")
        assert response.status_code == 200
        assert "X-Stage-Timings" in response.headers

        data = response.json()
        
        # Core Predict Contract
        assert "quantitative_signals" in data
        assert "system_parameters" in data
        
        # New Phase 3 Contract
        assert "qualitative_analysis" in data
        qual = data["qualitative_analysis"]
        assert qual["analysis_status"] == "success"
        assert qual["sentiment"] == "negative"
        assert qual["risk_factor"] == "high"
        assert qual["reasoning"] == "Lãi suất tăng."
        assert "system_parameters" in qual
        assert qual["sources_used"] == ["zone_2", "zone_3"]
