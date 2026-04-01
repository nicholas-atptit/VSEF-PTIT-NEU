import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import pandas as pd
import feedparser
import time
from datetime import datetime

# Attempt to load vnstock
try:
    import vnstock 
    from vnstock import Vnstock
except (ImportError, ModuleNotFoundError):
    Vnstock = None

from config.settings import get_settings
from src.ml.llm.news_intel import NewsIntelEngine

class CrawledArticle:
    def __init__(self, title, content, source, date_str):
        self.title = title
        self.content = content
        self.source = source
        self.publish_date = date_str

def crawl_google_news(ticker, query, start_year=2021):
    import urllib.parse
    """Crawl from Google News RSS using Custom Boolean Query from CSV"""
    articles = []
    # If query is too complex for Google RSS, we might need to trim or refine.
    # But for now, we use the query_news column directly.
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=vi&gl=VN&ceid=VN:vi"
    
    print(f"[{ticker}] Fetching Google News RSS with ADVANCED QUERY: {query[:100]}...")
    feed = feedparser.parse(url)
    
    for entry in feed.entries[:100]:  # Increased from 30 to capture more data
        title = entry.get('title', '')
        summary = entry.get('summary', title)
        published = entry.get('published', '')
        articles.append(CrawledArticle(title, summary, "Google_News", published))
    return articles

def crawl_vnstock_news(ticker, start_year=2021):
    """Crawl from VNStock API"""
    articles = []
    if not Vnstock:
        print(f"[{ticker}] vnstock not installed. Skipping.")
        return articles
        
    print(f"[{ticker}] Fetching VNStock corporate news...")
    try:
        settings = get_settings()
        # Set API Key in environment if available
        if hasattr(settings, 'vnstock_api_key') and settings.vnstock_api_key:
            os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key
            
        vn = Vnstock()
        # Use VCI as a reliable source for news in vnstock v3
        stock = vn.stock(symbol=ticker, source='VCI')
        df = stock.company.news()
        
        if df is None or df.empty:
            return articles
            
        for _, row in df.iterrows():
            title = row.get('news_title', '')
            public_date = row.get('public_date', '')
            
            date_str = ""
            if public_date:
                try:
                    # public_date can be an int (ms timestamp) or a string
                    if isinstance(public_date, (int, float)):
                        # Convert ms to seconds if it's too large
                        ts = public_date / 1000.0 if public_date > 1e12 else public_date
                        dt_obj = datetime.fromtimestamp(ts)
                        year = dt_obj.year
                        date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = str(public_date)
                        year = int(date_str.split('-')[0][:4])
                        
                    if year < start_year: continue
                except: pass
            
            # Use news_title as both title and content if content is missing
            content = row.get('news_short_content', title)
            articles.append(CrawledArticle(title, content, "VNStock", date_str))
    except Exception as e:
        print(f"[{ticker}] Lỗi khi chạy vnstock: {e}")
        
    return articles  # No cap — push 100% to LLM

async def process_ticker(ticker, query_news, negative_list=None):
    print(f"=== Bắt đầu Crawl tin tức mã: {ticker} ===")
    
    # 1. Crawl
    gg_arts = crawl_google_news(ticker, query_news, start_year=2021)
    vn_arts = crawl_vnstock_news(ticker, start_year=2021)
    
    all_articles = gg_arts + vn_arts
    print(f"[{ticker}] Thu thập: {len(all_articles)} bài.")
    
    # 2. Filter using Negative Keywords from CSV
    filtered_articles = []
    if negative_list:
        neg_words = [w.strip().lower() for w in negative_list.split('|') if w.strip()]
        for art in all_articles:
            if any(nw in art.title.lower() for nw in neg_words):
                continue
            filtered_articles.append(art)
        print(f"[{ticker}] Sau lọc Negative Keywords: {len(filtered_articles)} bài.")
    else:
        filtered_articles = all_articles

    # 3. Phân tích qua LLM (NewsIntelEngine)
    if not filtered_articles:
        print(f"[{ticker}] Không tìm thấy tin tức nào sau khi lọc.")
        return
        
    print(f"[{ticker}] Đẩy 100% ({len(filtered_articles)} bài) vào LLM để phân tích (Ollama qwen3)...")
    engine = NewsIntelEngine()
    
    art_dicts = []
    for a in filtered_articles:
         art_dicts.append({
             "title": a.title,
             "content": a.content,
             "source": a.source
         })
         
    try:
        result = await engine.analyze_ticker_news(ticker, art_dicts, horizon="short")
        if result:
            print(f"[{ticker}] XONG! Sentiment: {result.get('sentiment_score')} | Trend: {result.get('trend')}")
        else:
            print(f"[{ticker}] Phân tích thất bại (LLM returns None).")
    except Exception as e:
        print(f"[{ticker}] Lỗi phân tích LLM: {e}")

async def main():
    csv_path = "reports/news_keywords_baseline.csv"
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy file {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Use prioritization: Viettel Group first, then VN100
    viettel_tickers = ['VGI', 'CTR', 'VTP', 'VTK']
    all_tickers = df['ticker'].tolist()
    
    ordered_tickers = viettel_tickers + [t for t in all_tickers if t not in viettel_tickers]
    
    print(f"Sẽ xử lý tổng cộng {len(ordered_tickers)} mã (Ưu tiên Viettel Group...)")
    
    for ticker in ordered_tickers:
        row_matches = df[df['ticker'] == ticker]
        if row_matches.empty: continue
        
        row = row_matches.iloc[0]
        query = row['query_news']
        negatives = row['negative_keywords'] if not pd.isna(row['negative_keywords']) else None
        
        await process_ticker(ticker, str(query), negatives)
        # Sleep 1s to respect Google News / Local Ollama workload
        await asyncio.sleep(1)
            
    print("=== ĐÃ HOÀN THÀNH TOÀN BỘ CHIẾN DỊCH QUÉT TIN TỨC ===")

if __name__ == "__main__":
    asyncio.run(main())
