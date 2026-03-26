"""Update news_intelligence with horizons.

Revision ID: 003_news_intel_upgrade
Revises: 002_agent_tables
Create Date: 2026-03-26
"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "003_news_intel_upgrade"
down_revision: Union[str, None] = "002_agent_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # ── news_intelligence ───────────────────────────────────
    # Table to store structured LLM-analyzed intelligence for news articles
    op.create_table(
        "news_intelligence",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("trend", sa.String(20)),     # Bullish, Bearish, Neutral
        sa.Column("sentiment_score", sa.Float), # -1.0 to 1.0
        sa.Column("summary", sa.Text),          # Concise summary
        sa.Column("full_report", sa.Text),      # Detailed analysis
        sa.Column("article_ids", sa.ARRAY(sa.Text)),
        sa.Column("source_sites", sa.ARRAY(sa.Text)),
        sa.Column("horizon", sa.String(20), server_default="short", nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Fast lookup by ticker and horizon
    op.create_index("ix_news_intel_ticker_horizon", "news_intelligence", ["ticker", "horizon", "timestamp"])

def downgrade() -> None:
    op.drop_table("news_intelligence")
