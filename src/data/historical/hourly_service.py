"""Shared service for hourly data extraction."""

import asyncio
import datetime as dt
import pandas as pd
from src.data.providers.vn_price_gateway import fetch_price_history
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName
from src.utils.logging import get_logger

logger = get_logger("hourly_service")

async def fetch_hourly_data(ticker: str, days: int = 30) -> pd.DataFrame | None:
    """Fetch hourly history for a ticker using the canonical provider gateway."""
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days)
        response = fetch_price_history(
            FetchRequest(
                symbol=ticker.upper(),
                asset_type=AssetType.STOCK,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                frequency=Frequency.HOURLY,
                preferred_sources=(SourceName.KBS, SourceName.VCI),
                allow_legacy_fallback=True,
                allow_daily=False,
                allow_resample=False,
            )
        )
        df = response.data.rename(columns={"datetime": "time"}).drop(columns=["frequency"], errors="ignore")
        return df
    except Exception as e:
        logger.debug("fetch_error", ticker=ticker, error=str(e))
        return None

def format_row_to_text(ticker: str, row: pd.Series) -> str:
    """Convert a price row into a natural language sentence with decimal precision."""
    timestamp = str(row.get("time", ""))
    open_p = row.get("open", 0.0)
    close_p = row.get("close", 0.0)
    high_p = row.get("high", 0.0)
    low_p = row.get("low", 0.0)
    vol = row.get("volume", 0)
    
    return (
        f"At {timestamp}, the stock {ticker} opened at {open_p:,.2f} VND, "
        f"reached a high of {high_p:,.2f} VND, a low of {low_p:,.2f} VND, "
        f"and closed the hour at {close_p:,.2f} VND with a trading volume of {vol:,.0f} shares."
    )
