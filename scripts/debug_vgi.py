import asyncio
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.database.connection import get_session
from sqlalchemy import text

async def main():
    ticker = "VGI"
    try:
        async with get_session() as session:
            # 1. Check raw_prices
            res = await session.execute(text("SELECT COUNT(*) FROM raw_prices WHERE ticker = :t"), {"t": ticker})
            count_raw = res.scalar()
            print(f"raw_prices for {ticker}: {count_raw} rows")
            
            # 2. Check adjusted_prices
            res = await session.execute(text("SELECT COUNT(*) FROM adjusted_prices WHERE ticker = :t"), {"t": ticker})
            count_adj = res.scalar()
            print(f"adjusted_prices for {ticker}: {count_adj} rows")
            
            if count_adj > 0:
                res = await session.execute(text("SELECT timestamp, close FROM adjusted_prices WHERE ticker = :t ORDER BY timestamp DESC LIMIT 1"), {"t": ticker})
                last = res.fetchone()
                print(f"Last adjusted price for {ticker}: {last}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
