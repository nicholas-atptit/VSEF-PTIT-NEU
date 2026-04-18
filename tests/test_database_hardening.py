from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data.database.decision_card_schema import AuditMetadata, DecisionCard
from src.data.database.decision_repository import DecisionRepository
import src.data.database.decision_repository as decision_repository_module


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)


class FakeSession:
    def __init__(self, storage: dict[str, dict]):
        self.storage = storage

    async def execute(self, statement, params=None):
        sql = str(statement)
        payload = dict(params or {})

        if "INSERT INTO decision_audit" in sql:
            self.storage[payload["decision_id"]] = payload
            return FakeResult([])

        if "FROM decision_audit" in sql:
            rows = [
                {
                    **record,
                    "created_at": record.get("timestamp"),
                }
                for record in self.storage.values()
                if record.get("ticker") == payload["ticker"]
            ]
            rows.sort(key=lambda row: row["timestamp"], reverse=True)
            return FakeResult(rows[: payload["limit"]])

        raise AssertionError(f"Unexpected SQL executed in test: {sql}")


@pytest.mark.asyncio
async def test_decision_repository_persists_and_reads_back(monkeypatch, tmp_path) -> None:
    storage: dict[str, dict] = {}

    @asynccontextmanager
    async def fake_get_session():
        yield FakeSession(storage)

    monkeypatch.setattr(decision_repository_module, "get_session", fake_get_session)

    repo = DecisionRepository(
        output_dir=str(tmp_path / "decision_cards"),
        write_json_artifacts=True,
        allow_json_fallback=False,
    )
    decision = DecisionCard(
        meta=AuditMetadata(
            decision_id="decision-001",
            ticker="SSI",
            timestamp=datetime(2026, 4, 16, 0, 0, tzinfo=timezone.utc),
            provider="test",
            model_version="unit-test",
            latency_sec=0.12,
        ),
        tech_summary={"close": 100.5},
        news_summary={"sentiment": "positive"},
        evidence_ids=["news-1", "tech-1"],
        consensus_score=0.8,
        regime_label="trend",
        dynamic_confidence_threshold=0.7,
        bull_thesis="Momentum remains intact.",
        bear_thesis="Liquidity could fade.",
        risk_veto=False,
        risk_reason="none",
        action="BUY",
        target_weight=0.15,
        execution_shares=300,
        rationale="Conviction remains above threshold.",
        confidence=0.82,
    )

    await repo.save_decision(decision)
    rows = await repo.get_decisions_by_ticker("SSI")

    assert rows[0]["meta"]["decision_id"] == "decision-001"
    assert rows[0]["action"] == "BUY"
    assert rows[0]["tech_summary"]["close"] == 100.5
    assert repo.log_file.exists()
    assert any(path.suffix == ".json" for path in Path(repo.output_dir).iterdir())


def test_database_hardening_migration_declares_required_guards() -> None:
    migration_path = Path("alembic/versions/004_database_hardening.py")
    content = migration_path.read_text(encoding="utf-8")

    assert "uq_agent_predictions_run_horizon" in content
    assert "ck_news_intelligence_sentiment_score_range" in content
    assert "ix_raw_prices_lookup" in content
    assert "decision_audit" in content
