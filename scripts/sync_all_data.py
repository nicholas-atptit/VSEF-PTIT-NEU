
import asyncio
import datetime as dt
import json
import os
from src.historical.backdate import BackdateIngestor
from src.utils.logging import get_logger

logger = get_logger("sync_all")

async def main():
    vip_path = 'H:/AI-ML-LLM in Stock_march26_PTIT_NEU/data/listing/danh_sach_VIP_LLM_ready.jsonl'
    if not os.path.exists(vip_path):
        logger.error("vip_list_not_found", path=vip_path)
        return

    tickers = []
    with open(vip_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tickers.append(data['symbol'])

    logger.info("sync_starting", ticker_count=len(tickers))
    
    # Fetch 6 months of data for comprehensive coverage
    start_date = dt.date.today() - dt.timedelta(days=180)
    
    # Large scale parallel ingestion
    ingestor = BackdateIngestor()
    logger.info("sync_executing_parallel", start=start_date)
    await ingestor.run(tickers=tickers, start_date=start_date)

    logger.info("sync_complete")

if __name__ == "__main__":
    asyncio.run(main())
