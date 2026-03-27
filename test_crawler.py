import asyncio
from src.context.news_crawler import NewsCrawler

async def main():
    crawler = NewsCrawler()
    print("Testing crawl_ticker with max_pages...")
    try:
        docs = await crawler.crawl_ticker("SSI", max_pages=1)
        print(f"Success! Found {len(docs)} docs.")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nTesting crawl_watchlist...")
    try:
        docs = await crawler.crawl_watchlist(["SSI", "HPG"], max_pages_per_ticker=1)
        print(f"Success! Found {len(docs)} docs total.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
