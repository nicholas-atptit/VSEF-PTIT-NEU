import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.database.connection import get_session
from sqlalchemy import text
from src.utils.time_utils import now_vn

async def check():
    now = now_vn()
    print(f"Current VN Time: {now}")
    async with get_session() as session:
        # Check latest 10 tickers in raw_prices
        try:
            res = await session.execute(text("SELECT ticker, MAX(timestamp) as latest FROM raw_prices GROUP BY ticker ORDER BY latest DESC LIMIT 10"))
            rows = res.all()
            print("\nLatest 10 tickers in DB (raw_prices):")
            for r in rows:
                print(f"  {r[0]}: {r[1]}")
        except Exception as e:
            print(f"Error checking raw_prices: {e}")

if __name__ == "__main__":
    asyncio.run(check())
