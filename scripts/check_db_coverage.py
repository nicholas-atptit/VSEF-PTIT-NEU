import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

async def check_db():
    url = os.getenv("TIMESCALE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/algo_trading")
    engine = create_async_engine(url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Check News Intelligence
        query = text("SELECT ticker, COUNT(*) FROM news_intelligence GROUP BY ticker ORDER BY COUNT(*) DESC")
        result = await session.execute(query)
        news_data = result.fetchall()
        print("--- NEWS INTELLIGENCE ---")
        for row in news_data:
            print(f"Ticker: {row[0]}, Count: {row[1]}")

        # Check Decision Cards
        query = text("SELECT ticker, COUNT(*) FROM decision_cards GROUP BY ticker ORDER BY COUNT(*) DESC")
        result = await session.execute(query)
        decision_data = result.fetchall()
        print("\n--- DECISION CARDS ---")
        for row in decision_data:
            print(f"Ticker: {row[0]}, Count: {row[1]}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_db())
