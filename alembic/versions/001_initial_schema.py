"""Initial schema — create all tables and hypertables.

Revision ID: 001_initial
Revises: None
Create Date: 2026-03-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Ensure TimescaleDB extension ──────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # ── raw_prices ────────────────────────────────────────────
    op.create_table(
        "raw_prices",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(10), nullable=False, server_default="HOSE"),
        sa.Column("timeframe", sa.String(5), nullable=False, server_default="1m"),
        sa.Column("open", sa.Numeric(12, 2), nullable=False),
        sa.Column("high", sa.Numeric(12, 2), nullable=False),
        sa.Column("low", sa.Numeric(12, 2), nullable=False),
        sa.Column("close", sa.Numeric(12, 2), nullable=False),
        sa.Column("volume", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("source", sa.String(20), nullable=False, server_default="websocket"),
        sa.PrimaryKeyConstraint("timestamp", "ticker"),
    )
    op.create_index("ix_raw_prices_ticker_ts", "raw_prices", ["ticker", "timestamp"])
    op.create_index("ix_raw_prices_exchange", "raw_prices", ["exchange"])

    # Convert to hypertable
    op.execute(
        "SELECT create_hypertable('raw_prices', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 month', "
        "if_not_exists => TRUE);"
    )

    # ── adjusted_prices ──────────────────────────────────────
    op.create_table(
        "adjusted_prices",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("exchange", sa.String(10), nullable=False, server_default="HOSE"),
        sa.Column("timeframe", sa.String(5), nullable=False, server_default="1m"),
        sa.Column("open", sa.Numeric(12, 2), nullable=False),
        sa.Column("high", sa.Numeric(12, 2), nullable=False),
        sa.Column("low", sa.Numeric(12, 2), nullable=False),
        sa.Column("close", sa.Numeric(12, 2), nullable=False),
        sa.Column("volume", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("adjustment_factor", sa.Numeric(12, 8), nullable=False, server_default="1.0"),
        sa.Column("adjustment_reason", sa.Text, nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="computed"),
        sa.PrimaryKeyConstraint("timestamp", "ticker"),
    )
    op.create_index("ix_adj_prices_ticker_ts", "adjusted_prices", ["ticker", "timestamp"])
    op.create_index("ix_adj_prices_exchange", "adjusted_prices", ["exchange"])

    op.execute(
        "SELECT create_hypertable('adjusted_prices', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 month', "
        "if_not_exists => TRUE);"
    )

    # ── corporate_actions ────────────────────────────────────
    op.create_table(
        "corporate_actions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("event_date", sa.Date, nullable=False),
        sa.Column("action_type", sa.String(50), nullable=False),
        sa.Column("factor", sa.Numeric(12, 8), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
    )
    op.create_index("ix_corp_actions_ticker_date", "corporate_actions", ["ticker", "event_date"])

    # ── watchlist ────────────────────────────────────────────
    op.create_table(
        "watchlist",
        sa.Column("ticker", sa.String(20), primary_key=True),
        sa.Column("exchange", sa.String(10), nullable=False, server_default="HOSE"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("added_by", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── blacklist ────────────────────────────────────────────
    op.create_table(
        "blacklist",
        sa.Column("ticker", sa.String(20), primary_key=True),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("source", sa.String(100), nullable=False, server_default="manual"),
        sa.Column("banned_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── macro_indicators ─────────────────────────────────────
    op.create_table(
        "macro_indicators",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("indicator_name", sa.String(50), nullable=False),
        sa.Column("value", sa.Numeric(18, 4), nullable=False),
        sa.Column("open", sa.Numeric(18, 4), nullable=True),
        sa.Column("high", sa.Numeric(18, 4), nullable=True),
        sa.Column("low", sa.Numeric(18, 4), nullable=True),
        sa.Column("close", sa.Numeric(18, 4), nullable=True),
        sa.Column("volume", sa.Integer, nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="api"),
        sa.Column("unit", sa.String(20), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "indicator_name"),
    )
    op.create_index("ix_macro_indicator_name_ts", "macro_indicators", ["indicator_name", "timestamp"])

    op.execute(
        "SELECT create_hypertable('macro_indicators', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 month', "
        "if_not_exists => TRUE);"
    )

    # ── signal_events ────────────────────────────────────────
    op.create_table(
        "signal_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("event_date", sa.Date, nullable=False),
        sa.Column("event_end_date", sa.Date, nullable=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="medium"),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("affected_tickers", postgresql.ARRAY(sa.String(20)), nullable=True),
        sa.Column("affected_sectors", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("market_impact_pct", sa.Float, nullable=True),
        sa.Column("source", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_signal_event_date", "signal_events", ["event_date"])
    op.create_index("ix_signal_event_type", "signal_events", ["event_type"])

    # ── Enable compression on historical hypertables ─────────
    op.execute(
        "ALTER TABLE raw_prices SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'ticker', "
        "timescaledb.compress_orderby = 'timestamp DESC');"
    )
    op.execute(
        "SELECT add_compression_policy('raw_prices', INTERVAL '30 days');"
    )

    op.execute(
        "ALTER TABLE adjusted_prices SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'ticker', "
        "timescaledb.compress_orderby = 'timestamp DESC');"
    )
    op.execute(
        "SELECT add_compression_policy('adjusted_prices', INTERVAL '30 days');"
    )


def downgrade() -> None:
    op.execute("SELECT remove_compression_policy('adjusted_prices', if_exists => TRUE);")
    op.execute("SELECT remove_compression_policy('raw_prices', if_exists => TRUE);")

    op.drop_table("signal_events")
    op.drop_table("macro_indicators")
    op.drop_table("blacklist")
    op.drop_table("watchlist")
    op.drop_table("corporate_actions")
    op.drop_table("adjusted_prices")
    op.drop_table("raw_prices")
