import numpy as np
from typing import List, Dict, Any
from src.data.database.decision_card_schema import DecisionCard

class MetricsEvaluator:
    """
    Kho tính toán chỉ số tài chính và Model Performance.
    """
    def __init__(self):
        pass

    def evaluate_financial_metrics(self, decisions: List[DecisionCard]) -> Dict[str, Any]:
        """
        Dùng mảng Decision (Target Weights) so với Return thị trường để ra PnL.
        Phiên bản Mock: In ra các hệ số Random hợp lý để kiểm thử khung luồng.
        """
        if not decisions:
            return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            
        # Giả lập tính toán trên Pandas Series thay cho code thật
        return {
            "cagr": round(np.random.uniform(0.05, 0.25), 4),
            "sharpe": round(np.random.uniform(0.5, 2.5), 2),
            "max_drawdown": round(np.random.uniform(-0.3, -0.05), 4),
            "turnover_rate": round(np.random.uniform(0.1, 1.0), 2),
            "fee_adjusted_pnl": round(np.random.uniform(10.0, 50.0), 2)
        }

    def evaluate_model_quality(self, decisions: List[DecisionCard]) -> Dict[str, Any]:
        """
        Đánh giá Hành vi của LLM trong Multi-Agent Framework.
        """
        if not decisions:
            return {}
            
        veto_count = sum(1 for d in decisions if d.risk_veto)
        avg_latency = float(np.mean([d.meta.latency_sec for d in decisions]))
        
        # Tỷ lệ mà Bull vs Bear hoàn toàn ngược điểm nhưng vẫn bị/không bị Veto
        contradiction_rate = round(np.random.uniform(0.1, 0.3), 3) # Mock
        
        return {
            "total_decisions_made": len(decisions),
            "veto_rate": round(veto_count / len(decisions), 3),
            "avg_latency_sec": round(avg_latency, 2),
            "contradiction_rate": contradiction_rate
        }
