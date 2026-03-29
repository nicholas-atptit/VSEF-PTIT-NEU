import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv

load_dotenv()

async def check():
    url = os.getenv("TIMESCALE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/algo_trading")
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT * FROM news_intelligence LIMIT 1;"))
        print(f"Columns: {res.keys()}")
        
        # Try a safe query based on known columns
        res = await conn.execute(text("SELECT ticker, sentiment_score, trend FROM news_intelligence ORDER BY ticker DESC LIMIT 5;"))
        rows = res.fetchall()
        print("=== LATEST RECORDS ===")
        for r in rows:
            print(r)
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check())
