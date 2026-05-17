"""Legacy module.
Retained for historical compatibility or migration reference.
Not part of canonical governed runtime.
"""

import asyncio
import os
import sys

# Add project root to sys path
sys.path.append(os.getcwd())

from scripts.run_news_crawler import crawl_vnstock_news

async def test_crawler():
    ticker = 'VGI'
    print(f"Testing vnstock news crawl for {ticker}...")
    articles = crawl_vnstock_news(ticker)
    print(f"Found {len(articles)} articles.")
    for a in articles[:3]:
        print(f" - {a.title} ({a.publish_date})")

if __name__ == "__main__":
    asyncio.run(test_crawler())
