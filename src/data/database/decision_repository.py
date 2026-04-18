from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from src.data.database.connection import get_session
from src.utils.logging import get_logger

from .decision_card_schema import DecisionCard

logger = get_logger(__name__)

_UPSERT_DECISION_SQL = (
    sa.text(
        """
        INSERT INTO decision_audit (
            decision_id,
            ticker,
            timestamp,
            provider,
            model_version,
            latency_sec,
            tech_summary,
            news_summary,
            evidence_ids,
            consensus_score,
            regime_label,
            dynamic_confidence_threshold,
            bull_thesis,
            bear_thesis,
            risk_veto,
            risk_reason,
            action,
            target_weight,
            execution_shares,
            rationale,
            confidence
        ) VALUES (
            :decision_id,
            :ticker,
            :timestamp,
            :provider,
            :model_version,
            :latency_sec,
            :tech_summary,
            :news_summary,
            :evidence_ids,
            :consensus_score,
            :regime_label,
            :dynamic_confidence_threshold,
            :bull_thesis,
            :bear_thesis,
            :risk_veto,
            :risk_reason,
            :action,
            :target_weight,
            :execution_shares,
            :rationale,
            :confidence
        )
        ON CONFLICT (decision_id) DO UPDATE SET
            ticker = EXCLUDED.ticker,
            timestamp = EXCLUDED.timestamp,
            provider = EXCLUDED.provider,
            model_version = EXCLUDED.model_version,
            latency_sec = EXCLUDED.latency_sec,
            tech_summary = EXCLUDED.tech_summary,
            news_summary = EXCLUDED.news_summary,
            evidence_ids = EXCLUDED.evidence_ids,
            consensus_score = EXCLUDED.consensus_score,
            regime_label = EXCLUDED.regime_label,
            dynamic_confidence_threshold = EXCLUDED.dynamic_confidence_threshold,
            bull_thesis = EXCLUDED.bull_thesis,
            bear_thesis = EXCLUDED.bear_thesis,
            risk_veto = EXCLUDED.risk_veto,
            risk_reason = EXCLUDED.risk_reason,
            action = EXCLUDED.action,
            target_weight = EXCLUDED.target_weight,
            execution_shares = EXCLUDED.execution_shares,
            rationale = EXCLUDED.rationale,
            confidence = EXCLUDED.confidence
        """
    ).bindparams(
        sa.bindparam("tech_summary", type_=postgresql.JSONB(astext_type=sa.Text())),
        sa.bindparam("news_summary", type_=postgresql.JSONB(astext_type=sa.Text())),
        sa.bindparam("evidence_ids", type_=postgresql.JSONB(astext_type=sa.Text())),
    )
)

_SELECT_BY_TICKER_SQL = sa.text(
    """
    SELECT
        decision_id,
        ticker,
        timestamp,
        provider,
        model_version,
        latency_sec,
        tech_summary,
        news_summary,
        evidence_ids,
        consensus_score,
        regime_label,
        dynamic_confidence_threshold,
        bull_thesis,
        bear_thesis,
        risk_veto,
        risk_reason,
        action,
        target_weight,
        execution_shares,
        rationale,
        confidence,
        created_at
    FROM decision_audit
    WHERE ticker = :ticker
    ORDER BY timestamp DESC, created_at DESC
    LIMIT :limit
    """
)


class DecisionRepository:
    """Persist decision cards to the database with optional JSON artifacts."""

    def __init__(
        self,
        output_dir: str = "reports/decision_cards",
        *,
        write_json_artifacts: bool = True,
        allow_json_fallback: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "decisions_log.jsonl"
        self.write_json_artifacts = write_json_artifacts
        self.allow_json_fallback = allow_json_fallback

    async def save_decision(self, decision: DecisionCard) -> dict[str, Any]:
        """Persist a decision to TimescaleDB and optionally mirror it to JSON artifacts."""
        payload = self._to_db_payload(decision)
        json_payload = self._to_json_payload(decision)

        try:
            async with get_session() as session:
                await session.execute(_UPSERT_DECISION_SQL, payload)
        except Exception as exc:
            logger.warning(
                "decision_repo_db_write_failed",
                decision_id=payload["decision_id"],
                ticker=payload["ticker"],
                error=str(exc),
            )
            if not self.allow_json_fallback:
                raise
        finally:
            if self.write_json_artifacts:
                self._write_json_artifacts(json_payload)

        return json_payload

    async def get_decisions_by_ticker(self, ticker: str, limit: int = 50) -> list[dict[str, Any]]:
        """Load decision history from the database and fall back to JSON artifacts if needed."""
        ticker = ticker.upper().strip()
        try:
            async with get_session() as session:
                result = await session.execute(_SELECT_BY_TICKER_SQL, {"ticker": ticker, "limit": limit})
                rows = result.mappings().all()
            return [self._row_to_payload(row) for row in rows]
        except Exception as exc:
            logger.warning("decision_repo_db_read_failed", ticker=ticker, error=str(exc))
            if self.allow_json_fallback:
                return self.get_decisions_by_ticker_from_artifacts(ticker, limit=limit)
            raise

    def get_decisions_by_ticker_from_artifacts(self, ticker: str, limit: int = 50) -> list[dict[str, Any]]:
        """Read decision history back from JSONL artifacts."""
        ticker = ticker.upper().strip()
        results: list[dict[str, Any]] = []
        if not self.log_file.exists():
            return results

        with self.log_file.open("r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                if data.get("meta", {}).get("ticker") == ticker:
                    results.append(data)

        results.sort(key=lambda item: item.get("meta", {}).get("timestamp", ""), reverse=True)
        return results[:limit]

    @staticmethod
    def _to_db_payload(decision: DecisionCard) -> dict[str, Any]:
        payload = decision.model_dump(mode="python")
        meta = payload["meta"]
        return {
            "decision_id": meta["decision_id"],
            "ticker": str(meta["ticker"]).upper().strip(),
            "timestamp": meta["timestamp"],
            "provider": meta["provider"],
            "model_version": meta["model_version"],
            "latency_sec": meta["latency_sec"],
            "tech_summary": payload["tech_summary"],
            "news_summary": payload["news_summary"],
            "evidence_ids": payload["evidence_ids"],
            "consensus_score": payload["consensus_score"],
            "regime_label": payload["regime_label"],
            "dynamic_confidence_threshold": payload["dynamic_confidence_threshold"],
            "bull_thesis": payload["bull_thesis"],
            "bear_thesis": payload["bear_thesis"],
            "risk_veto": payload["risk_veto"],
            "risk_reason": payload["risk_reason"],
            "action": payload["action"],
            "target_weight": payload["target_weight"],
            "execution_shares": payload["execution_shares"],
            "rationale": payload["rationale"],
            "confidence": payload["confidence"],
        }

    @staticmethod
    def _to_json_payload(decision: DecisionCard) -> dict[str, Any]:
        payload = decision.model_dump(mode="json")
        payload["meta"]["ticker"] = str(payload["meta"]["ticker"]).upper().strip()
        return payload

    @staticmethod
    def _row_to_payload(row: Any) -> dict[str, Any]:
        timestamp = row["timestamp"]
        return {
            "meta": {
                "decision_id": row["decision_id"],
                "ticker": row["ticker"],
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                "provider": row["provider"],
                "model_version": row["model_version"],
                "latency_sec": row["latency_sec"],
            },
            "tech_summary": row["tech_summary"],
            "news_summary": row["news_summary"],
            "evidence_ids": row["evidence_ids"] or [],
            "consensus_score": row["consensus_score"],
            "regime_label": row["regime_label"],
            "dynamic_confidence_threshold": row["dynamic_confidence_threshold"],
            "bull_thesis": row["bull_thesis"],
            "bear_thesis": row["bear_thesis"],
            "risk_veto": row["risk_veto"],
            "risk_reason": row["risk_reason"],
            "action": row["action"],
            "target_weight": row["target_weight"],
            "execution_shares": row["execution_shares"],
            "rationale": row["rationale"],
            "confidence": row["confidence"],
        }

    def _write_json_artifacts(self, data: dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

        ticker = data["meta"]["ticker"]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_file = self.output_dir / f"{ticker}_{timestamp_str}_{data['meta']['decision_id']}.json"
        with artifact_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
