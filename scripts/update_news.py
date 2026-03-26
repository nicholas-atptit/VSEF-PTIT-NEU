import asyncio
import sys
import argparse
import datetime as dt
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.context.news_crawler import NewsCrawler
from src.llm.news_intel import NewsIntelEngine
import json
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def main(tickers=None, update_all=False, limit=None):
    print("📰 [NEWS INTELLIGENCE] Starting high-fidelity news extraction & analysis...")
    
    # 1. Determine Tickers
    if update_all:
        print("🔍 Fetching all tickers from database...")
        from src.database.connection import get_session
        from sqlalchemy import text
        async with get_session() as session:
            try:
                # Try company_profiles first
                res = await session.execute(text("SELECT ticker FROM company_profiles ORDER BY ticker"))
                db_tickers = [r[0] for r in res.fetchall()]
            except Exception:
                # Rollback failed transaction before trying fallback
                await session.rollback()
                print("⚠️ company_profiles not found. Fetching from raw_prices...")
                res = await session.execute(text("SELECT DISTINCT ticker FROM raw_prices ORDER BY ticker"))
                db_tickers = [r[0] for r in res.fetchall()]
            
            if db_tickers:
                tickers = db_tickers
                if limit:
                    tickers = tickers[:limit]
            else:
                print("⚠️ No tickers found in database. Falling back to default list.")
    
    if not tickers:
        tickers = ["FPT", "VGI", "VHM", "VIC", "SSI", "TCB", "VNM", "HPG", "MWG", "DGC"]
    
    crawler = NewsCrawler()
    intel_engine = NewsIntelEngine()
    
    print(f"🔍 Analyzing news for: {', '.join(tickers)}")
    
    # Load existing cache for merging
    cache_path = root_dir / "data" / "latest_predictions.json"
    current_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                current_cache = json.load(f)
        except Exception: pass

    total_analyzed = 0
    for ticker in tickers:
        print(f"\n📡 Processing {ticker}...")
        # 1. Extraction (Upgraded to 10 sources)
        articles = await crawler.crawl_ticker(ticker, max_pages=3)
        if not articles:
            print(f"   ⚠️ No recent news found for {ticker}.")
            continue
        
        # 2. Intelligence Analysis (LLM)
        print(f"   🧠 Running AI Summary & Trend Analysis ({len(articles)} articles)...")
        intel = await intel_engine.analyze_ticker_news(ticker, articles)
        if intel:
            print(f"   ✅ Trend: {intel.get('trend')} | Sentiment: {intel.get('sentiment_score')}")
            total_analyzed += 1
            
            # 3. Update Cache Fallback
            if ticker in current_cache:
                current_cache[ticker]["llm_analysis"] = {
                    "overall_outlook": intel.get("trend"),
                    "reasoning": intel.get("summary"),
                    "news_headlines": "\n".join([f"• {a.title[:80]}..." for a in articles[:3]]) if hasattr(articles[0], 'title') else "",
                    "timestamp": dt.datetime.now().isoformat()
                }
                # Periodically save cache safely
                if total_analyzed % 10 == 0:
                    try:
                        tmp_path = cache_path.with_suffix(".tmp")
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(current_cache, f, indent=4)
                        if cache_path.exists(): cache_path.unlink()
                        tmp_path.rename(cache_path)
                    except Exception: pass
        else:
            print(f"   ❌ AI analysis failed for {ticker}.")

    # Final safe save
    try:
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(current_cache, f, indent=4)
        if cache_path.exists(): cache_path.unlink()
        tmp_path.rename(cache_path)
    except Exception: pass

    print(f"\n✅ [SUCCESS] Intelligence Engine complete. Processed {total_analyzed} tickers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual News Update Script")
    parser.add_argument("tickers", nargs="*", help="Optional list of tickers to crawl")
    parser.add_argument("--all", action="store_true", help="Update all tickers in database")
    parser.add_argument("--limit", type=int, help="Limit number of tickers when using --all")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.tickers, args.all, args.limit))
    except KeyboardInterrupt:
        print("\n👋 Canceled.")
