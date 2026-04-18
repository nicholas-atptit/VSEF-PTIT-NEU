"""Harden core tables and add decision audit storage.

Revision ID: 004_database_hardening
Revises: 003_news_intel_upgrade
Create Date: 2026-04-16
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004_database_hardening"
down_revision: Union[str, None] = "003_news_intel_upgrade"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_check_not_valid(table_name: str, constraint_name: str, condition: str) -> None:
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = '{constraint_name}'
            ) THEN
                ALTER TABLE {table_name}
                ADD CONSTRAINT {constraint_name}
                CHECK ({condition}) NOT VALID;
            END IF;
        END
        $$;
        """
    )


def _drop_constraint_if_exists(table_name: str, constraint_name: str) -> None:
    op.execute(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {constraint_name};")


def upgrade() -> None:
    # Keep the existing raw/adjusted price primary keys intact. The current PK
    # shape does not fully match the multi-timeframe columns, but rewriting it
    # in-place would be a disruptive data migration. Add safer lookup indexes now.
    _add_check_not_valid("raw_prices", "ck_raw_prices_open_positive", '"open" > 0')
    _add_check_not_valid("raw_prices", "ck_raw_prices_high_positive", "high > 0")
    _add_check_not_valid("raw_prices", "ck_raw_prices_low_positive", "low > 0")
    _add_check_not_valid("raw_prices", "ck_raw_prices_close_positive", "close > 0")
    _add_check_not_valid("raw_prices", "ck_raw_prices_high_gte_low", "high >= low")
    _add_check_not_valid("raw_prices", "ck_raw_prices_volume_nonnegative", "volume >= 0")

    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_open_positive", '"open" > 0')
    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_high_positive", "high > 0")
    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_low_positive", "low > 0")
    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_close_positive", "close > 0")
    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_high_gte_low", "high >= low")
    _add_check_not_valid("adjusted_prices", "ck_adjusted_prices_volume_nonnegative", "volume >= 0")
    _add_check_not_valid(
        "adjusted_prices",
        "ck_adjusted_prices_adjustment_factor_positive",
        "adjustment_factor > 0",
    )

    _add_check_not_valid(
        "agent_runs",
        "ck_agent_runs_fusion_confidence_unit_interval",
        "fusion_confidence >= 0 AND fusion_confidence <= 1",
    )
    _add_check_not_valid(
        "agent_runs",
        "ck_agent_runs_suggested_allocation_unit_interval",
        "suggested_allocation >= 0 AND suggested_allocation <= 1",
    )
    _add_check_not_valid(
        "agent_predictions",
        "ck_agent_predictions_probability_up_unit_interval",
        "probability_up >= 0 AND probability_up <= 1",
    )
    _add_check_not_valid(
        "agent_predictions",
        "ck_agent_predictions_probability_down_unit_interval",
        "probability_down >= 0 AND probability_down <= 1",
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'uq_agent_predictions_run_horizon'
            ) THEN
                ALTER TABLE agent_predictions
                ADD CONSTRAINT uq_agent_predictions_run_horizon
                UNIQUE (run_id, horizon);
            END IF;
        END
        $$;
        """
    )

    _add_check_not_valid(
        "news_intelligence",
        "ck_news_intelligence_sentiment_score_range",
        "sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)",
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_raw_prices_lookup
        ON raw_prices (ticker, exchange, timeframe, timestamp DESC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_adjusted_prices_lookup
        ON adjusted_prices (ticker, exchange, timeframe, timestamp DESC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_news_intelligence_lookup
        ON news_intelligence (ticker, horizon, timestamp DESC);
        """
    )

    op.create_table(
        "decision_audit",
        sa.Column("decision_id", sa.String(length=64), primary_key=True),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("latency_sec", sa.Float(), nullable=False, server_default="0"),
        sa.Column("tech_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("news_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "evidence_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("consensus_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("regime_label", sa.String(length=50), nullable=False),
        sa.Column("dynamic_confidence_threshold", sa.Float(), nullable=False),
        sa.Column("bull_thesis", sa.Text(), nullable=False),
        sa.Column("bear_thesis", sa.Text(), nullable=False),
        sa.Column("risk_veto", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("risk_reason", sa.Text(), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("execution_shares", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("latency_sec >= 0", name="ck_decision_audit_latency_nonnegative"),
        sa.CheckConstraint(
            "consensus_score >= 0 AND consensus_score <= 1",
            name="ck_decision_audit_consensus_unit_interval",
        ),
        sa.CheckConstraint(
            "dynamic_confidence_threshold >= 0 AND dynamic_confidence_threshold <= 1",
            name="ck_decision_audit_dynamic_threshold_unit_interval",
        ),
        sa.CheckConstraint(
            "target_weight >= 0 AND target_weight <= 1",
            name="ck_decision_audit_target_weight_unit_interval",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_decision_audit_confidence_unit_interval",
        ),
        sa.CheckConstraint(
            "execution_shares >= 0",
            name="ck_decision_audit_execution_shares_nonnegative",
        ),
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_decision_audit_ticker_timestamp
        ON decision_audit (ticker, timestamp DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_decision_audit_ticker_timestamp;")
    op.drop_table("decision_audit")

    op.execute("DROP INDEX IF EXISTS ix_news_intelligence_lookup;")
    op.execute("DROP INDEX IF EXISTS ix_adjusted_prices_lookup;")
    op.execute("DROP INDEX IF EXISTS ix_raw_prices_lookup;")

    _drop_constraint_if_exists("news_intelligence", "ck_news_intelligence_sentiment_score_range")
    _drop_constraint_if_exists("agent_predictions", "uq_agent_predictions_run_horizon")
    _drop_constraint_if_exists("agent_predictions", "ck_agent_predictions_probability_down_unit_interval")
    _drop_constraint_if_exists("agent_predictions", "ck_agent_predictions_probability_up_unit_interval")
    _drop_constraint_if_exists("agent_runs", "ck_agent_runs_suggested_allocation_unit_interval")
    _drop_constraint_if_exists("agent_runs", "ck_agent_runs_fusion_confidence_unit_interval")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_adjustment_factor_positive")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_volume_nonnegative")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_high_gte_low")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_close_positive")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_low_positive")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_high_positive")
    _drop_constraint_if_exists("adjusted_prices", "ck_adjusted_prices_open_positive")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_volume_nonnegative")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_high_gte_low")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_close_positive")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_low_positive")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_high_positive")
    _drop_constraint_if_exists("raw_prices", "ck_raw_prices_open_positive")
