from typing import Dict, Any
from .base import BaseAgent

import json
from src.ml.llm.client import get_llm_client
from config.settings import get_settings

class BearResearcherAgent(BaseAgent):
    """
    Bear Researcher Agent.
    Vai trò: Tiếp nhận tổng hợp Technical + News và bằng mọi giá tìm kiếm lập luận GIẢM GIÁ (Rủi ro).
    """
    def __init__(self):
        super().__init__(name="Bear Researcher")

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: Dữ liệu (State, Market Context, Technical_Score, News_Score).
        Output: Lập luận Bear.
        """
        # Gắn thêm lịch sử tranh luận từ vòng trước nếu có (Round 2+)
        prev_round_text = ""
        if "previous_round" in data:
            prev_round_text = f"Lịch sử tranh luận vòng trước:\n- Phe Đối Lập (Bull): {data['previous_round'].get('bull_thesis', 'Không có')}\nHãy phản bác lại lập luận của Bull và củng cố thêm thesis của bạn.\n"

        prompt = f"""
        Theo thông tin thị trường sau đây của {data.get("ticker", "UNKNOWN")}:
        - Kỹ thuật: {json.dumps(data.get("technical", {}), ensure_ascii=False)}
        - Tin tức: {json.dumps(data.get("news", {}), ensure_ascii=False)}
        
        {prev_round_text}
        
        Nhiệm vụ: Bạn là một chuyên gia quản trị rủi ro theo phe "GẤU" cực kỳ bi quan.
        Hãy tìm mọi điểm yếu, rủi ro tiềm ẩn hoặc kháng cự để phản biện và cho rằng giá cổ phiếu này SẼ GIẢM hoặc KHÔNG NÊN MUA CHÚT NÀO.
        
        Trả về JSON với cấu trúc (lưu ý không dùng code block, chỉ trả object):
        {{
            "thesis": "Lập luận phản biện chính (khoảng 3-4 câu)",
            "confidence": 0.0 - 1.0 (mức độ tự tin của bạn)
        }}
        """
        
        client = get_llm_client()
        settings = get_settings()
        
        try:
            model_name = getattr(settings, f"{settings.llm_provider}_model_name")
            response = await client.chat.completions.create(
                model=model_name,
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
