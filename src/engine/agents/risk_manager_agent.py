from typing import Dict, Any
from .base import BaseAgent

class RiskManagerAgent(BaseAgent):
    """
    Risk Manager / Veto Agent.
    Vai trò: Tiếp nhận quyết định sơ bộ của Bull/Bear và quyết định Veto (phủ quyết) nếu nhận thấy rủi ro Volatility Shock,
    hoặc Confidence của AI quá thấp, hoặc Conflict quá cao để đảm bảo an toàn.
    """
    def __init__(self):
        super().__init__(name="Risk Manager")

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: Dữ liệu (Bull Thesis, Bear Thesis, Technicals, Context).
        Output: Quyết định Phủ quyết / Vượt qua.
        """
        # Logic giả lập Risk (có thể thay bằng quy tắc cứng hoặc gọi LLM đánh giá logic).
        # Nếu confidence của Bull < 0.5 --> VETO BUY
        
        return {
            "agent": self.name,
            "veto": False,
            "risk_flag": "NORMAL",
            "reason": "Chưa ghi nhận biến động vol shock lớn hoặc low confidence.",
            "max_position_size": 0.1 # Cho phép không quá 10% NAV
        }
