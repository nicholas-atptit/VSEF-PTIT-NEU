import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

async def list_tables():
    url = os.getenv("TIMESCALE_URL")
    if not url:
        print("TIMESCALE_URL not found in .env")
        return
    
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        query = text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'")
        result = await conn.execute(query)
        tables = result.fetchall()
        print("--- TABLES ---")
        for table in tables:
            print(table[0])

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(list_tables())
