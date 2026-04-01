import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.database.connection import get_session
from src.ml.models.price import RawPrice
from sqlalchemy import select, func

async def check_ticker(ticker):
    async with get_session() as session:
        stmt = select(func.count()).select_from(RawPrice).filter(RawPrice.ticker == ticker)
        res = await session.execute(stmt)
        count = res.scalar()
        print(f"Ticker {ticker} has {count} rows in RawPrice.")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ACB"
    asyncio.run(check_ticker(ticker))
