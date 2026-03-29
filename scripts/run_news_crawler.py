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
    from vnstock import company_news
except ImportError:
    company_news = None

from config.settings import get_settings
from src.llm.news_intel import NewsIntelEngine

class CrawledArticle:
    def __init__(self, title, content, source, date_str):
        self.title = title
        self.content = content
        self.source = source
        self.publish_date = date_str

def crawl_google_news(ticker, keywords, start_year=2021):
    import urllib.parse
    """Crawl from Google News RSS using Custom Keywords"""
    articles = []
    query = f'"{keywords}" after:{start_year}-01-01'
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=vi&gl=VN&ceid=VN:vi"
    
    print(f"[{ticker}] Fetching Google News RSS: {query}")
    feed = feedparser.parse(url)
    
    for entry in feed.entries[:30]: 
        title = entry.get('title', '')
        summary = entry.get('summary', title)
        published = entry.get('published', '')
        articles.append(CrawledArticle(title, summary, "Google_News", published))
    return articles

def crawl_vnstock_news(ticker, start_year=2021):
    """Crawl from VNStock API"""
    articles = []
    if not company_news:
        print(f"[{ticker}] vnstock not installed. Skipping.")
        return articles
        
    print(f"[{ticker}] Fetching VNStock corporate news...")
    try:
        for page in range(0, 3): 
            df = company_news(symbol=ticker, page_size=100, page_num=page)
            if df is None or df.empty:
                break
                
            for _, row in df.iterrows():
                title = row.get('title', '')
                date_str = str(row.get('publishDate', ''))
                
                if date_str:
                    try:
                        year = int(date_str.split('-')[0][:4])
                        if year < start_year: continue
                    except: pass
                
                articles.append(CrawledArticle(title, title, "VNStock", date_str))
    except Exception as e:
        print(f"[{ticker}] Lỗi khi chạy vnstock: {e}")
        
    return articles[:50] 

async def process_ticker(ticker, keywords):
    print(f"=== Bắt đầu Crawl tin tức mã: {ticker} ===")
    
    # Run synchronous IO in executor or just sync since it's a CLI
    gg_arts = crawl_google_news(ticker, keywords, start_year=2021)
    vn_arts = crawl_vnstock_news(ticker, start_year=2021)
    
    all_articles = gg_arts + vn_arts
    print(f"[{ticker}] Tổng số bài thu thập: {len(all_articles)} (Google: {len(gg_arts)}, VNStock: {len(vn_arts)})")
    
    # Sắp xếp và chọn lọc Top 20 bài báo để tránh tràn Context Window
    # LLM không thể đọc 500 bài cùng lúc!
    selected_articles = all_articles[:20] 
    
    # 2. Phân tích qua LLM (NewsIntelEngine)
    if not selected_articles:
        print(f"[{ticker}] Không tìm thấy tin tức nào.")
        return
        
    print(f"[{ticker}] Đẩy {len(selected_articles)} bài vào LLM để phân tích Sentiment...")
    engine = NewsIntelEngine()
    
    # Convert format for NewsIntel
    art_dicts = []
    for a in selected_articles:
         art_dicts.append({
             "title": a.title,
             "content": a.content,
             "source": a.source
         })
         
    result = await engine.analyze_ticker_news(ticker, art_dicts, horizon="short")
    if result:
        print(f"[{ticker}] XONG! Sentiment: {result.get('sentiment_score')} | Trend: {result.get('trend')}")
    else:
        print(f"[{ticker}] Phân tích thất bại hoặc LLM Timeout.")

async def main():
    csv_path = "reports/news_keywords_baseline.csv"
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy file {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # User requirement: "quét trước t xem 3 mã HPG, DGC và VGI"
    target_tickers = ['HPG', 'DGC', 'VGI']
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        if ticker in target_tickers:
            keywords = row['Custom_News_Keywords_User_Fill']
            # Fallback if empty
            if pd.isna(keywords) or str(keywords).strip() == "":
                 keywords = row['Baseline_Keywords']
                 
            # Take only the short name or company name to optimize Google Query
            # Because full CSV string might be "VGI, Viettel Global"
            # We'll just pass the whole thing and let Google handle it.
            
            await process_ticker(ticker, str(keywords))
            
    print("=== ĐÃ HOÀN THÀNH TOÀN BỘ ===")

if __name__ == "__main__":
    asyncio.run(main())
