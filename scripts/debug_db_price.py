
import asyncio
from sqlalchemy import select, desc
from src.database.connection import get_session
from src.models.price import RawPrice

async def check_price():
    async with get_session() as session:
        stmt = select(RawPrice).filter(RawPrice.ticker == "FPT").order_by(desc(RawPrice.timestamp)).limit(5)
        res = await session.execute(stmt)
        recs = res.scalars().all()
        for r in recs:
            print(f"TS: {r.timestamp} | Close: {r.close}")

if __name__ == "__main__":
    asyncio.run(check_price())
