from typing import Dict, Any

class RegimeDetector:
    """
    Detects the current market regime based on technical indicators or historical variance.
    Supported Regimes: 'uptrend', 'downtrend', 'sideways', 'high_volatility'
    """
    
    @staticmethod
    def detect_regime(technical_data: Dict[str, Any]) -> str:
        """
        Phân tích data input (VD: MACD, RSI, ADX) để trả về label Regime.
        Mặc định mock up logic nếu không có đủ evidence.
        """
        # Giả định technical_data chứa 1 key là "trend_probs" hay "trend_direction"
        # Dummy logic
        if not technical_data:
            return "sideways"
            
        trend = technical_data.get("trend", "sideways").lower()
        volatility = technical_data.get("volatility", "normal").lower()
        
        if volatility == "high":
            return "high_volatility"
            
        if trend in ["up", "bull"]:
            return "uptrend"
        elif trend in ["down", "bear"]:
            return "downtrend"
            
        return "sideways"
