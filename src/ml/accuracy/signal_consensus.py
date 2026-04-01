from typing import Dict, Any
from .calibration import ConfidenceCalibrator

class SignalConsensus:
    """
    Tích hợp các tín hiệu từ nhiều nguồn khác nhau (Technical ML, Sentiment LLM, Macro)
    tạo thành Consensus Score cuối cùng.
    """
    
    @staticmethod
    def calculate_consensus(
        tech_score: float, 
        news_score: float, 
        tech_conf: float = 1.0, 
        news_conf: float = 1.0,
        risk_veto: bool = False
    ) -> float:
        """
        Trả về điểm từ -1.0 (Strong Bear) đến 1.0 (Strong Bull).
        tech_score / news_score nên trong khoảng [-1, 1] trước.
        """
        if risk_veto:
            return 0.0 # Bị Veto, score về 0
            
        # Calibrate confidences
        cal_tech = ConfidenceCalibrator.calibrate(tech_conf, "ml")
        cal_news = ConfidenceCalibrator.calibrate(news_conf, "llm")
        
        # Weighted mean
        total_weight = cal_tech + cal_news
        if total_weight == 0:
            return 0.0
            
        consensus = (tech_score * cal_tech + news_score * cal_news) / total_weight
        return round(consensus, 4)
