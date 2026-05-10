"""Tests for Module 4: FastAPI Microservice.

Validates API endpoints, JSON schema compliance, and system constraints.
Uses FastAPI TestClient (synchronous) for testing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas_v2 import TerminalPayload
import src.api.routes as api_routes


FORBIDDEN_AUTHORITY_TERMS = (
    "BUY",
    "SELL",
    "STRONG_BUY",
    "STRONG_SELL",
    "recommendation",
    "execute",
    "execution",
    "order_payload",
    "final_order",
    "broker",
    "trade now",
)


def assert_no_authority_terms(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False).lower()
    for term in FORBIDDEN_AUTHORITY_TERMS:
        assert term.lower() not in text


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """GET /api/v1/health should always return 200."""

    def test_health_ok(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "5.0.0"
        assert "X-Process-Time-Ms" in response.headers
        assert "X-Trace-Id" in response.headers

    def test_health_phase(self, client: TestClient):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["phase"] == "Diagnostic research API"


class TestRootEndpoint:
    """GET / should return service info."""

    def test_root(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data
        assert "predict" in data
        assert data["service"] == "Diagnostic Research API"
        assert "forecast_diagnostics" in data["scope"]
        assert "risk_diagnostics" in data["scope"]
        assert "route_diagnostics" in data["scope"]
        assert "non_executing" in data["boundaries"]
        assert "non_advisory" in data["boundaries"]
        assert_no_authority_terms(data)
        assert "/web" not in str(data)
        assert "/dashboard" not in str(data)

    def test_fastapi_metadata_is_diagnostic_only(self):
        metadata = f"{app.title} {app.description}"
        assert app.title == "Diagnostic Research API"
        assert "Forecast" in app.description
        assert "risk" in app.description
        assert "route diagnostics" in app.description
        assert "Algo Trading System" not in metadata
        assert "paper trading" not in metadata
        assert "trading assistant" not in metadata
        assert "final order" not in metadata
        assert "broker-like" not in metadata

    def test_web_dashboard_removed(self, client: TestClient):
        response = client.get("/web/index.html")
        assert response.status_code == 404

        dashboard_response = client.get("/dashboard")
        assert dashboard_response.status_code == 200
        data = dashboard_response.json()
        assert data["status"] == "removed"
        assert "/web/index.html" not in str(data)


class TestTrainEndpoint:
    """POST /api/v1/train should train models using mock data."""

    def test_train_with_mock(self, client: TestClient):
        """Training with mock data should succeed."""
        response = client.post(
            "/api/v1/train",
            json={"ticker": "TEST", "use_mock": True, "runtime_mode": "demo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "trained"
        assert data["ticker"] == "TEST"
        assert "metrics" in data
        assert data["data_provenance"]["uses_mock_data"] is True
        assert data["data_provenance"]["runtime_mode"] == "demo"


class TestPredictEndpoint:
    """GET /api/v1/predict — full pipeline with mock data."""

    @pytest.fixture(autouse=True)
    def _train_first(self, client: TestClient):
        """Ensure a model is trained before prediction tests."""
        client.post(
            "/api/v1/train",
            json={"ticker": "PRED", "use_mock": True, "runtime_mode": "demo"},
        )

    def test_predict_success(self, client: TestClient):
        """Predict should return 200 with valid JSON."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        assert response.status_code == 200

    def test_predict_json_contract(self, client: TestClient):
        """Response must match the TerminalPayload schema."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()

        # Validate with Pydantic model
        parsed = TerminalPayload(**data)
        assert parsed.ticker == "PRED"
        assert parsed.data_provenance["uses_mock_data"] is True
        assert parsed.data_provenance["runtime_mode"] == "demo"

    def test_predict_has_trend_probabilities(self, client: TestClient):
        """Response must include up/down/sideways probabilities."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        probs = data["technical"]["horizons"][0]["trend_probs"]
        assert "up" in probs
        assert "down" in probs
        assert "sideways" in probs
        # Probabilities should sum to ~1.0
        total = probs["up"] + probs["down"] + probs["sideways"]
        assert 0.95 <= total <= 1.05

    def test_predict_has_expected_range(self, client: TestClient):
        """Response must include quantile price ranges."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        rng = data["technical"]["horizons"][0]["expected_range"]
        assert "bottom_10th" in rng
        assert "median_50th" in rng
        assert "ceiling_90th" in rng
        # Bottom should be <= median <= ceiling
        assert rng["bottom_10th"] <= rng["median_50th"] <= rng["ceiling_90th"]

    def test_predict_has_diagnostic_route_fields(self, client: TestClient):
        """Response must include diagnostic route fields."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        fusion = data["fusion"]
        assert fusion["diagnostic_signal"] in (
            "upward_bias",
            "downward_bias",
            "high_upward_bias",
            "high_downward_bias",
            "range_bound",
            "hold_review",
            "risk_blocked",
            "review_required",
        )
        assert "route_decision" in fusion
        assert "decision_lane" in fusion
        assert "diagnostic_summary" in fusion
        assert fusion["review_required"] is True
        assert data["candidate_status"] == "diagnostic_only"
        assert_no_authority_terms(data)

    def test_predict_risk_payload(self, client: TestClient):
        """Response must include risk monitoring."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        risk = data["risk"]
        assert "allocation_candidate_weight" in risk
        assert "risk_flag" in risk
        assert "review_required" in risk
        assert isinstance(risk["review_required"], bool)
        assert_no_authority_terms(data)

    def test_analyze_sanitizes_qualitative_summary(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        async def fake_qualitative_analysis(*args, **kwargs):
            return {
                "analysis_status": "success",
                "reasoning": "BUY recommendation with broker execution language",
            }

        async def fake_news(*args, **kwargs):
            return {"ticker": "PRED", "news": [], "count": 0}

        monkeypatch.setattr(api_routes, "run_qualitative_analysis", fake_qualitative_analysis)
        monkeypatch.setattr(api_routes, "ticker_news", fake_news)

        response = client.get("/api/v1/analyze?ticker=PRED&use_mock=true&runtime_mode=demo")
        assert response.status_code == 200
        assert_no_authority_terms(response.json())


class TestPredictErrors:
    """Error handling for predict endpoint."""

    def test_predict_missing_ticker(self, client: TestClient):
        """Missing ticker should return 422."""
        response = client.get("/api/v1/predict")
        assert response.status_code == 422

    def test_predict_untrained_ticker(self, client: TestClient):
        """Untrained ticker still returns 404 when demo data is explicitly requested."""
        response = client.get(
            "/api/v1/predict?ticker=NONEXISTENT_TICKER_XYZ&use_mock=true&runtime_mode=demo"
        )
        assert response.status_code == 404


class TestChatGovernance:
    """Chat route must keep prompt boundaries diagnostic-only."""

    def test_chat_prompt_declares_no_authority(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        captured: dict = {}

        class FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="diagnostic response")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        monkeypatch.setattr("src.ml.llm.client.get_llm_client", lambda: fake_client)

        response = client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Cho toi boi canh rui ro"}]},
        )

        assert response.status_code == 200
        assert response.json()["response"] == "diagnostic response"
        system_prompt = captured["messages"][0]["content"]
        assert "research diagnostics only" in system_prompt
        assert "no financial advice" in system_prompt
        assert "no BUY/SELL recommendation authority" in system_prompt
        assert "no trade execution instructions" in system_prompt
        assert "no broker authority" in system_prompt
        assert "no order authority" in system_prompt
        assert_no_authority_terms(response.json())


class TestRuntimeModeGovernance:
    """Mock usage must be explicit and mode-gated."""

    def test_audit_mode_blocks_explicit_mock_prediction(self, client: TestClient):
        response = client.get("/api/v1/predict?ticker=AUDIT&use_mock=true&runtime_mode=audit")

        assert response.status_code == 403
        assert "Mock data disabled for audit mode" in response.json()["detail"]

    def test_research_mode_blocks_silent_mock_fallback(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        def fail_provider(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(api_routes, "load_ohlcv_from_db", fail_provider)
        monkeypatch.setattr(api_routes, "load_ohlcv_from_vnstock", fail_provider)

        response = client.get("/api/v1/predict?ticker=FAIL&runtime_mode=research")

        assert response.status_code == 403
        assert "Mock fallback disabled for research mode" in response.json()["detail"]

    def test_research_mode_allows_explicit_mock_request(self):
        df = api_routes._load_governed_ohlcv("RESEARCH", use_mock=True, runtime_mode="research")
        provenance = df.attrs["data_provenance"]

        assert provenance["uses_mock_data"] is True
        assert provenance["fallback_triggered"] is False
        assert provenance["runtime_mode"] == "research"

    def test_demo_mode_fallback_is_explicitly_provenanced(self, monkeypatch: pytest.MonkeyPatch):
        def fail_provider(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(api_routes, "load_ohlcv_from_db", fail_provider)
        monkeypatch.setattr(api_routes, "load_ohlcv_from_vnstock", fail_provider)

        df = api_routes._load_governed_ohlcv("DEMO", use_mock=False, runtime_mode="demo")
        provenance = df.attrs["data_provenance"]

        assert provenance["source"] == "synthetic_mock_data"
        assert provenance["uses_mock_data"] is True
        assert provenance["fallback_triggered"] is True
        assert provenance["runtime_mode"] == "demo"

    def test_v2_demo_endpoint_declares_mock_provenance(self, client: TestClient):
        response = client.get("/predict/technical?ticker=SSI&runtime_mode=demo")

        assert response.status_code == 200
        provenance = response.json()["data_provenance"]
        assert provenance["uses_mock_data"] is True
        assert provenance["fallback_triggered"] is False
        assert provenance["runtime_mode"] == "demo"

    def test_v2_fused_demo_is_non_authoritative(self, client: TestClient):
        response = client.get("/predict/fused?ticker=SSI&runtime_mode=demo")

        assert response.status_code == 200
        data = response.json()
        assert data["candidate_status"] == "demo_diagnostic_only"
        assert data["fusion"]["diagnostic_signal"] == "upward_bias"
        assert "allocation_candidate_weight" in data["risk"]
        assert_no_authority_terms(data)

    def test_v2_audit_mode_blocks_demo_endpoint_mock(self, client: TestClient):
        response = client.get("/predict/technical?ticker=SSI&runtime_mode=audit")

        assert response.status_code == 403
        assert "Mock data disabled for audit mode" in response.json()["detail"]

    def test_v2_audit_mode_blocks_fused_demo_payload(self, client: TestClient):
        response = client.get("/predict/fused?ticker=SSI&runtime_mode=audit")

        assert response.status_code == 403
        assert "Mock data disabled for audit mode" in response.json()["detail"]

    def test_legacy_v1_routes_are_gated(self, client: TestClient):
        for path in (
            "/api/v1/execute?ticker=SSI&runtime_mode=research",
            "/api/v1/paper-trade?ticker=SSI&runtime_mode=research",
        ):
            response = client.get(path)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "legacy_route_gated"
            assert data["candidate_status"] == "legacy_diagnostic_only"
            assert data["review_required"] is True
            assert_no_authority_terms(data)

    def test_v2_debate_route_is_gated(self, client: TestClient):
        response = client.get("/debate?ticker=SSI&runtime_mode=research")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "legacy_route_gated"
        assert data["candidate_status"] == "legacy_diagnostic_only"
        assert_no_authority_terms(data)
