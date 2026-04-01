from typing import Dict, Any

class ConfidenceCalibrator:
    """
    Hiệu chỉnh xác suất (Confidence / Probability) học từ ML model hoặc LLM.
    Trong production, có thể thay bằng Isotonic Regression hoặc Platt Scaling.
    Ở bản POC này dùng Heuristic Scaling an toàn: phạt Overconfidence.
    """
    
    @staticmethod
    def calibrate(raw_confidence: float, source: str = "llm") -> float:
        """
        Calibrate confidence score.
        Ví dụ: LLM (thường tự tin thái quá -> phạt 20%)
        """
        if not (0.0 <= raw_confidence <= 1.0):
            raw_confidence = max(0.0, min(1.0, float(raw_confidence)))
            
        if source.lower() == "llm":
            # Heuristic penalty for LLM overconfidence
            return round(raw_confidence * 0.8, 4)
        elif source.lower() == "ml":
            # Typical ML might be under-confident or noisy depending on the label smoothing
            return round(raw_confidence * 0.95, 4)
        return raw_confidence
