from typing import Dict, Any
from .base import BaseAgent

class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Allocation Agent (tương đương với Allocator trong FinRL-X).
    Vai trò: Lấy Risk metrics, Bull/Bear thesis để chốt tỷ trọng phân bổ (Target Weights) 
    và Final Action cho mã chứng khoán.
    """
    def __init__(self):
        super().__init__(name="Portfolio Manager")

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: Quyết định Risk, Market Context, Debate Context.
        Output: Allocation % và Order format. (Vector Trọng Số)
        """
        veto_flag = data.get("risk_decision", {}).get("veto", False)
        
        if veto_flag:
            return {
                "agent": self.name,
                "action": "HOLD",
                "target_weight": 0.0,
                "rationale": "Bị phủ quyết bởi Risk Manager."
            }
            
        return {
            "agent": self.name,
            "action": "BUY",
            "target_weight": 0.05, # Gợi ý mua 5%
            "rationale": "Tranh luận nghiên vị thế Bull hợp lý, rủi ro trong kiểm soát."
        }
