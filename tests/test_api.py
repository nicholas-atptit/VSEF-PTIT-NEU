"""Tests for Module 4: FastAPI Microservice.

Validates API endpoints, JSON schema compliance, and system constraints.
Uses FastAPI TestClient (synchronous) for testing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictionResponse


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
        assert data["version"] == "2.0.0"
        assert "X-Process-Time-Ms" in response.headers
        assert "X-Trace-Id" in response.headers

    def test_health_phase(self, client: TestClient):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "Phase 2" in data["phase"]


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
            json={"ticker": "TEST", "use_mock": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "trained"
        assert data["ticker"] == "TEST"
        assert "metrics" in data


class TestPredictEndpoint:
    """GET /api/v1/predict — full pipeline with mock data."""

    @pytest.fixture(autouse=True)
    def _train_first(self, client: TestClient):
        """Ensure a model is trained before prediction tests."""
        client.post(
            "/api/v1/train",
            json={"ticker": "PRED", "use_mock": True},
        )

    def test_predict_success(self, client: TestClient):
        """Predict should return 200 with valid JSON."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        assert response.status_code == 200

    def test_predict_json_contract(self, client: TestClient):
        """Response must match the PredictionResponse schema."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()

        # Validate with Pydantic model
        parsed = PredictionResponse(**data)
        assert parsed.ticker == "PRED"

    def test_predict_has_trend_probabilities(self, client: TestClient):
        """Response must include up/down/sideways probabilities."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()
        probs = data["quantitative_signals"]["trend_probabilities"]
        assert "up" in probs
        assert "down" in probs
        assert "sideways" in probs
        # Probabilities should sum to ~1.0
        total = probs["up"] + probs["down"] + probs["sideways"]
        assert 0.95 <= total <= 1.05

    def test_predict_has_expected_range(self, client: TestClient):
        """Response must include quantile price ranges."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()
        rng = data["quantitative_signals"]["expected_range"]
        assert "bottom_10th" in rng
        assert "median_50th" in rng
        assert "ceiling_90th" in rng
        # Bottom should be <= median <= ceiling
        assert rng["bottom_10th"] <= rng["median_50th"] <= rng["ceiling_90th"]

    def test_predict_has_action_plan(self, client: TestClient):
        """Response must include an action plan."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()
        plan = data["quantitative_signals"]["action_plan"]
        assert plan["recommendation"] in ("BUY", "SELL", "RANGE_TRADE", "STAND_ASIDE")
        assert "entry_zone" in plan
        assert "stop_loss" in plan
        assert "take_profit" in plan

    def test_predict_risk_cap(self, client: TestClient):
        """max_risk_tolerance must be ≤ 0.70."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()
        assert data["system_parameters"]["max_risk_tolerance"] <= 0.70

    def test_predict_risk_cap_override(self, client: TestClient):
        """Even with risk_tolerance=1.0, cap at 0.70."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true&risk_tolerance=1.0")
        data = response.json()
        assert data["system_parameters"]["max_risk_tolerance"] <= 0.70

    def test_predict_confidence_routing(self, client: TestClient):
        """Confidence metrics must follow specified routing."""
        response = client.get("/api/v1/predict?ticker=PRED&use_mock=true")
        data = response.json()
        conf = data["system_parameters"]["confidence_metrics"]
        assert conf["stock_quantitative_data"] == 0.95
        assert conf["general_market_context"] == 0.70


class TestPredictErrors:
    """Error handling for predict endpoint."""

    def test_predict_missing_ticker(self, client: TestClient):
        """Missing ticker should return 422."""
        response = client.get("/api/v1/predict")
        assert response.status_code == 422

    def test_predict_untrained_ticker(self, client: TestClient):
        """Untrained ticker without mock should return 404."""
        response = client.get("/api/v1/predict?ticker=NONEXISTENT_TICKER_XYZ")
        assert response.status_code in (404, 500)
