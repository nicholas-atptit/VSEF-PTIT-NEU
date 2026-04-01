class DecisionThresholds:
    """
    Cung cấp Dynamic Confidence Thresholds phân mảnh theo Market Regime.
    VD: Downtrend thì cần tự tin > 0.8 mới mua ngược dòng.
    """
    
    # Base thresholds
    THRESHOLDS = {
        "uptrend": 0.60,
        "sideways": 0.75,
        "downtrend": 0.85, # Rất cẩn trọng khi mua trong downtrend
        "high_volatility": 0.80
    }
    
    @staticmethod
    def get_dynamic_threshold(regime: str) -> float:
        return DecisionThresholds.THRESHOLDS.get(regime, 0.75)
