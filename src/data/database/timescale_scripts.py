"""TimescaleDB helper utilities aligned with the Alembic-owned schema.

The canonical candle source is `raw_prices`. Continuous aggregates should be
built from raw market data by default because `adjusted_prices` is a derived
table that may be refreshed after new corporate actions arrive.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_AGGREGATE_SOURCES = {"raw_prices", "adjusted_prices"}

_REFRESH_ADJUSTED_PRICES_SQL = text(
    """
    WITH refreshed AS (
        SELECT
            rp.timestamp,
            rp.ticker,
            rp.exchange,
            rp.timeframe,
            rp.open,
            rp.high,
            rp.low,
            rp.close,
            rp.volume,
            COALESCE(
                EXP(
                    SUM(
                        LN(ca.factor::double precision)
                    ) FILTER (
                        WHERE ca.event_date > rp.timestamp::date
                    )
                ),
                1.0
            ) AS cumulative_factor,
            STRING_AGG(
                COALESCE(ca.description, ca.action_type || ' @ ' || ca.event_date::text),
                '; ' ORDER BY ca.event_date
            ) FILTER (
                WHERE ca.event_date > rp.timestamp::date
            ) AS adjustment_reason
        FROM raw_prices rp
        LEFT JOIN corporate_actions ca
            ON ca.ticker = rp.ticker
        WHERE (:ticker IS NULL OR rp.ticker = :ticker)
        GROUP BY
            rp.timestamp,
            rp.ticker,
            rp.exchange,
            rp.timeframe,
            rp.open,
            rp.high,
            rp.low,
            rp.close,
            rp.volume
    )
    INSERT INTO adjusted_prices (
        timestamp,
        ticker,
        exchange,
        timeframe,
        open,
        high,
        low,
        close,
        volume,
        adjustment_factor,
        adjustment_reason,
        source
    )
    SELECT
        timestamp,
        ticker,
        exchange,
        timeframe,
        ROUND((open * cumulative_factor)::numeric, 2),
        ROUND((high * cumulative_factor)::numeric, 2),
        ROUND((low * cumulative_factor)::numeric, 2),
        ROUND((close * cumulative_factor)::numeric, 2),
        CASE
            WHEN cumulative_factor = 0 THEN volume
            ELSE GREATEST(0, ROUND((volume / cumulative_factor)::numeric))::bigint
        END,
        ROUND(cumulative_factor::numeric, 8),
        adjustment_reason,
        'computed'
    FROM refreshed
    ON CONFLICT (timestamp, ticker) DO UPDATE SET
        exchange = EXCLUDED.exchange,
        timeframe = EXCLUDED.timeframe,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        adjustment_factor = EXCLUDED.adjustment_factor,
        adjustment_reason = EXCLUDED.adjustment_reason,
        source = EXCLUDED.source
    """
)


async def ensure_price_hypertables(session: AsyncSession) -> None:
    """Ensure the canonical price tables are Timescale hypertables."""
    sql = """
    SELECT create_hypertable(
        'raw_prices',
        'timestamp',
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );

    SELECT create_hypertable(
        'adjusted_prices',
        'timestamp',
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );

    CREATE INDEX IF NOT EXISTS ix_raw_prices_lookup
        ON raw_prices (ticker, exchange, timeframe, timestamp DESC);

    CREATE INDEX IF NOT EXISTS ix_adjusted_prices_lookup
        ON adjusted_prices (ticker, exchange, timeframe, timestamp DESC);

    CREATE INDEX IF NOT EXISTS ix_corp_actions_ticker_date
        ON corporate_actions (ticker, event_date);
    """
    await session.execute(text(sql))
    await session.commit()


async def create_continuous_aggregates(
    session: AsyncSession,
    *,
    source_table: str = "raw_prices",
    base_timeframe: str = "1m",
) -> None:
    """Create idempotent Timescale continuous aggregates for candle rollups."""
    if source_table not in _ALLOWED_AGGREGATE_SOURCES:
        raise ValueError(f"Unsupported aggregate source table: {source_table}")

    for view_name, bucket, schedule in (
        ("ohlcv_5m", "5 minutes", "5 minutes"),
        ("ohlcv_15m", "15 minutes", "15 minutes"),
        ("ohlcv_1h", "1 hour", "1 hour"),
    ):
        sql = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
        WITH (timescaledb.continuous) AS
        SELECT
            ticker,
            exchange,
            time_bucket(INTERVAL '{bucket}', timestamp) AS bucket,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume
        FROM {source_table}
        WHERE timeframe = :base_timeframe
        GROUP BY ticker, exchange, bucket
        WITH NO DATA;
        """
        await session.execute(text(sql), {"base_timeframe": base_timeframe})
        await session.execute(
            text(
                f"""
                DO $$
                BEGIN
                    PERFORM add_continuous_aggregate_policy(
                        '{view_name}',
                        start_offset => INTERVAL '7 days',
                        end_offset => INTERVAL '5 minutes',
                        schedule_interval => INTERVAL '{schedule}'
                    );
                EXCEPTION
                    WHEN duplicate_object THEN NULL;
                END
                $$;
                """
            )
        )

    await session.commit()


async def refresh_adjusted_prices_from_corporate_actions(
    session: AsyncSession,
    *,
    ticker: str | None = None,
) -> None:
    """Explicitly rebuild adjusted prices from `raw_prices` and `corporate_actions`.

    This helper intentionally does not install an automatic trigger. Blind
    backwards adjustments are risky because `adjusted_prices` is derived state
    and the current schema still carries a non-trivial multi-timeframe key
    design issue. Call this utility after loading or modifying corporate
    actions instead.
    """
    normalized_ticker = ticker.upper().strip() if ticker else None
    await session.execute(_REFRESH_ADJUSTED_PRICES_SQL, {"ticker": normalized_ticker})
    await session.commit()


async def init_timescaledb_features(
    session: AsyncSession,
    *,
    aggregate_source_table: str = "raw_prices",
    base_timeframe: str = "1m",
) -> None:
    """Initialize non-destructive Timescale features for canonical tables."""
    try:
        await ensure_price_hypertables(session)
        await create_continuous_aggregates(
            session,
            source_table=aggregate_source_table,
            base_timeframe=base_timeframe,
        )
    except Exception as exc:
        logger.warning("timescaledb_init_failed", error=str(exc))
