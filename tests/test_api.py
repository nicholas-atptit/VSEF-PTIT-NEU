"""Tests for Module 4: FastAPI Microservice.

Validates API endpoints, JSON schema compliance, and system constraints.
Uses FastAPI TestClient (synchronous) for testing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas_v2 import TerminalPayload
import src.api.routes as api_routes


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
        assert "Phase 1-5" in data["phase"]


class TestRootEndpoint:
    """GET / should return service info."""

    def test_root(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data
        assert "predict" in data


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

    def test_predict_has_action_plan(self, client: TestClient):
        """Response must include a fusion decision."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        fusion = data["fusion"]
        assert fusion["action"] in ("BUY", "SELL", "RANGE_TRADE", "STAND_ASIDE", "STRONG_BUY", "STRONG_SELL", "CANCEL_ORDER", "STANDBY")
        assert "rationale" in fusion

    def test_predict_risk_payload(self, client: TestClient):
        """Response must include risk monitoring."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&runtime_mode=demo")
        data = response.json()
        risk = data["risk"]
        assert "position_size_suggestion" in risk
        assert "veto_flag" in risk
        assert isinstance(risk["veto_flag"], bool)


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

    def test_v2_audit_mode_blocks_demo_endpoint_mock(self, client: TestClient):
        response = client.get("/predict/technical?ticker=SSI&runtime_mode=audit")

        assert response.status_code == 403
        assert "Mock data disabled for audit mode" in response.json()["detail"]
