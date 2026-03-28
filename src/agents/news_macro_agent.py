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
        
        # NOTE: Kết nối với DB Vector (Chroma) để lấy news sentiment.
        # Mock sentiment
        return {
            "agent": self.name,
            "ticker": ticker,
            "sentiment_score": 0.6,
            "key_drivers": ["Ngân hàng nhà nước hỗ trợ thanh khoản", "Triển vọng nới lỏng tiền tệ"],
            "confidence": 0.8
        }
