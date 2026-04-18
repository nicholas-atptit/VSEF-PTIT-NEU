from __future__ import annotations

import asyncio
import json
import uuid

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from config.settings import get_settings
from src.data.database.connection import dispose_engine, get_db
from src.data.database.decision_card_schema import AuditMetadata, DecisionCard
from src.data.database.decision_repository import DecisionRepository


def upgrade_database() -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", get_settings().timescale_sync_url.replace("%", "%%"))
    command.upgrade(cfg, "head")


def validate_agent_prediction_uniqueness() -> None:
    run_id = uuid.uuid4()
    engine = get_db()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO agent_runs (
                    id,
                    ticker,
                    status,
                    final_action,
                    fusion_confidence,
                    veto_triggered,
                    suggested_allocation,
                    payload
                ) VALUES (
                    :id,
                    :ticker,
                    :status,
                    :final_action,
                    :fusion_confidence,
                    :veto_triggered,
                    :suggested_allocation,
                    CAST(:payload AS JSONB)
                )
                """
            ),
            {
                "id": run_id,
                "ticker": "VALUNI",
                "status": "success",
                "final_action": "BUY",
                "fusion_confidence": 0.6,
                "veto_triggered": False,
                "suggested_allocation": 0.1,
                "payload": json.dumps({"source": "validation"}),
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO agent_predictions (
                    run_id,
                    horizon,
                    trend,
                    probability_up,
                    probability_down,
                    target_ceiling,
                    target_floor
                ) VALUES (
                    :run_id,
                    :horizon,
                    :trend,
                    :probability_up,
                    :probability_down,
                    :target_ceiling,
                    :target_floor
                )
                """
            ),
            {
                "run_id": run_id,
                "horizon": "short",
                "trend": "UP",
                "probability_up": 0.7,
                "probability_down": 0.2,
                "target_ceiling": 110.0,
                "target_floor": 95.0,
            },
        )

    duplicate_blocked = False
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO agent_predictions (
                        run_id,
                        horizon,
                        trend,
                        probability_up,
                        probability_down,
                        target_ceiling,
                        target_floor
                    ) VALUES (
                        :run_id,
                        :horizon,
                        :trend,
                        :probability_up,
                        :probability_down,
                        :target_ceiling,
                        :target_floor
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "horizon": "short",
                    "trend": "DOWN",
                    "probability_up": 0.3,
                    "probability_down": 0.6,
                    "target_ceiling": 109.0,
                    "target_floor": 90.0,
                },
            )
    except IntegrityError:
        duplicate_blocked = True

    if not duplicate_blocked:
        raise RuntimeError("agent_predictions(run_id, horizon) uniqueness was not enforced")

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_predictions WHERE run_id = :run_id"), {"run_id": run_id})
        conn.execute(text("DELETE FROM agent_runs WHERE id = :run_id"), {"run_id": run_id})


async def validate_decision_repository() -> None:
    decision_id = f"validation-{uuid.uuid4()}"
    repo = DecisionRepository(write_json_artifacts=False, allow_json_fallback=False)
    decision = DecisionCard(
        meta=AuditMetadata(
            decision_id=decision_id,
            ticker="VALDB",
            provider="validation",
            model_version="validation-script",
            latency_sec=0.01,
        ),
        tech_summary={"close": 101.0},
        news_summary={"trend": "neutral"},
        evidence_ids=["validation"],
        consensus_score=0.5,
        regime_label="sideways",
        dynamic_confidence_threshold=0.6,
        bull_thesis="Validation only.",
        bear_thesis="Validation only.",
        risk_veto=False,
        risk_reason="none",
        action="HOLD",
        target_weight=0.0,
        execution_shares=0,
        rationale="Round-trip persistence check.",
        confidence=0.5,
    )

    await repo.save_decision(decision)
    rows = await repo.get_decisions_by_ticker("VALDB", limit=5)
    if not any(row["meta"]["decision_id"] == decision_id for row in rows):
        raise RuntimeError("decision repository round-trip validation failed")

    engine = get_db()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM decision_audit WHERE decision_id = :decision_id"), {"decision_id": decision_id})


def main() -> None:
    upgrade_database()
    validate_agent_prediction_uniqueness()
    asyncio.run(validate_decision_repository())
    asyncio.run(dispose_engine())
    print("database hardening validation passed")


if __name__ == "__main__":
    main()
