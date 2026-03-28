from typing import List
from src.portfolio.weights_schema import PortfolioTarget, AssetWeight
import datetime

class AllocatorRegistry:
    """Kho thuật toán chia vốn."""
    pass

class BaseAllocator:
    def allocate(self, tickers: List[str], current_date: datetime.datetime, **kwargs) -> PortfolioTarget:
        pass

class EqualWeightAllocator(BaseAllocator):
    """Phương pháp chia đều tiền cho các mã."""
    def allocate(self, tickers: List[str], current_date: datetime.datetime, **kwargs) -> PortfolioTarget:
        n = len(tickers)
        w = round(1.0 / n, 4) if n > 0 else 0.0
        
        assets = [AssetWeight(ticker=t, target_weight=w) for t in tickers]
        # Xử lý phần dư cash nếu chia không hết làm tròn
        target = PortfolioTarget(
            timestamp=current_date,
            allocator_name="EqualWeight",
            assets=assets
        )
        target.validate_weights()
        return target

class DRLAllocator(BaseAllocator):
    """
    Dùng Model PPO (đã train) từ stable-baselines3.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        # Lẽ ra ở đây sẽ: self.model = PPO.load(model_path)
    
    def allocate(self, tickers: List[str], current_date: datetime.datetime, **kwargs) -> PortfolioTarget:
        # Mock Observation -> Predict -> Weights
        # V1 POC: Trả về random chuẩn hoá
        import numpy as np
        raw_weights = np.random.rand(len(tickers))
        weights = raw_weights / np.sum(raw_weights)
        
        assets = [AssetWeight(ticker=t, target_weight=round(float(w), 4)) for t, w in zip(tickers, weights)]
        target = PortfolioTarget(
            timestamp=current_date,
            allocator_name="DRL_PPO",
            assets=assets
        )
        target.validate_weights()
        return target
