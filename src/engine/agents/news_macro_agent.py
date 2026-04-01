from typing import Dict, Any
from .base import BaseAgent

class NewsMacroAgent(BaseAgent):
    """
    Phân tích Tin tức và Vĩ mô (News & Macro).
    Mô tả: Tiêu thụ dữ liệu sentiment, tin tức và các báo cáo vĩ mô gần đây.
    """
    def __init__(self):
        super().__init__(name="News & Macro Agent")

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: data = {"ticker": "SSI", "news": [...], "macro": {...}}
        Output: news sentiment score.
        """
        ticker = data.get("ticker", "UNKNOWN")
        
        try:
            from src.ml.llm.news_intel import NewsIntelEngine
            engine = NewsIntelEngine()
            latest = await engine.get_latest_intelligence(ticker)
        except Exception as e:
            latest = None
            
        if latest:
            return {
                "agent": self.name,
                "ticker": ticker,
                "sentiment_score": latest.get("sentiment_score", 0.5),
                "trend": latest.get("trend", "Neutral"),
                "summary": latest.get("summary", "Không có tin tức"),
                "confidence": 0.85
            }
            
        # Fallback if no news in DB
        return {
            "agent": self.name,
            "ticker": ticker,
            "sentiment_score": 0.5,
            "trend": "Neutral",
            "summary": "Không có dữ liệu tin tức cập nhật trong CSDL.",
            "confidence": 0.3
        }
