import asyncio
import pandas as pd
from sqlalchemy import text
from src.database.connection import get_session
from config.settings import get_settings

async def test_query(ticker):
    print(f"Testing query for {ticker}...")
    try:
        async with get_session() as session:
            res = await session.execute(text("""
                SELECT 
                    DATE(timestamp) as date, 
                    (ARRAY_AGG(close ORDER BY timestamp DESC))[1] as close,
                    SUM(volume) as volume
                FROM raw_prices 
                WHERE ticker = :t 
                GROUP BY DATE(timestamp) 
                ORDER BY date DESC LIMIT 200
            """), {"t": ticker})
            rows = res.fetchall()
            print(f"Found {len(rows)} rows.")
            if len(rows) > 0:
                df = pd.DataFrame(rows, columns=["date", "close", "volume"]).sort_values("date")
                print(df.tail())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    os.environ["PYTHONPATH"] = "."
    asyncio.run(test_query("DXG"))
