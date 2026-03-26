"""Add agent run and prediction tables.

Revision ID: 002_agent_tables
Revises: 001_initial
Create Date: 2026-03-26
"""

from typing import Sequence, Union
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "002_agent_tables"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # ── agent_runs ──────────────────────────────────────────
    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), server_default="success", nullable=False),
        sa.Column("final_action", sa.String(20), nullable=False),
        sa.Column("fusion_confidence", sa.Float, nullable=False),
        sa.Column("veto_triggered", sa.Boolean, server_default="false", nullable=False),
        sa.Column("suggested_allocation", sa.Float, server_default="0.0", nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_runs_ticker", "agent_runs", ["ticker"])

    # ── agent_predictions ───────────────────────────────────
    op.create_table(
        "agent_predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("horizon", sa.String(10), nullable=False),
        sa.Column("trend", sa.String(10), nullable=False),
        sa.Column("probability_up", sa.Float, nullable=False),
        sa.Column("probability_down", sa.Float, nullable=False),
        sa.Column("target_ceiling", sa.Float, nullable=False),
        sa.Column("target_floor", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_predictions_run_id", "agent_predictions", ["run_id"])

def downgrade() -> None:
    op.drop_table("agent_predictions")
    op.drop_table("agent_runs")
