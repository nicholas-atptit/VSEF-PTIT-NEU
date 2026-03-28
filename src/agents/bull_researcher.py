from typing import Dict, Any
from .base import BaseAgent

import json
from src.llm.client import get_llm_client
from config.settings import get_settings

class BullResearcherAgent(BaseAgent):
    """
    Bull Researcher Agent.
    Vai trò: Tiếp nhận tổng hợp Technical + News và bằng mọi giá tìm kiếm lập luận TĂNG GIÁ.
    """
    def __init__(self):
        super().__init__(name="Bull Researcher")

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: Dữ liệu (State, Market Context, Technical_Score, News_Score).
        Output: Lập luận Bull.
        """
        # Gọi tới LLM với prompt đóng vai "BULL"
        prompt = f"""
        Theo thông tin thị trường sau đây của {data.get("ticker", "UNKNOWN")}:
        - Kỹ thuật: {json.dumps(data.get("technical", {}), ensure_ascii=False)}
        - Tin tức: {json.dumps(data.get("news", {}), ensure_ascii=False)}
        
        Nhiệm vụ: Bạn là một chuyên gia đầu tư theo phe "BÒ TÓT" cực kỳ lạc quan.
        Hãy tìm mọi lập luận từ số liệu kỹ thuật và tin tức ở trên để chứng minh giá cổ phiếu này SẼ TĂNG.
        
        Trả về JSON với cấu trúc:
        {{
            "thesis": "Lập luận chính (khoảng 3-4 câu)",
            "confidence": 0.0 - 1.0 (mức độ tự tin của bạn)
        }}
        """
        
        client = get_llm_client()
        settings = get_settings()
        
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = json.loads(response.choices[0].message.content)
            
            return {
                "agent": self.name,
                "thesis": content.get("thesis", "Lỗi sinh lập luận."),
                "prompt_used": prompt,
                "confidence": content.get("confidence", 0.5)
            }
        except Exception as e:
            return {
                "agent": self.name,
                "thesis": f"Lỗi gọi LLM: {str(e)}",
                "prompt_used": prompt,
                "confidence": 0.0
            }
