
import asyncio
import datetime as dt
from src.historical.backdate import BackdateIngestor

async def main():
    ingestor = BackdateIngestor()
    # Fetch last 30 days to keep it quick
    start_date = dt.date.today() - dt.timedelta(days=30)
    await ingestor.run(tickers=["FPT", "VGI"], start_date=start_date)

if __name__ == "__main__":
    asyncio.run(main())
