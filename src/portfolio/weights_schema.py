from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import datetime

class AssetWeight(BaseModel):
    """
    Trọng số mục tiêu cho 1 mã cổ phiếu.
    Weight-centric mindset của FinRL-X.
    """
    ticker: str
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Tỷ lệ vốn phân bổ, 0.0 -> 1.0")
    confidence: float = Field(default=0.0, description="Độ tự tin từ thuật toán / LLM")
    
class PortfolioTarget(BaseModel):
    """
    Vector phân bổ danh mục mục tiêu cho toàn bộ thời điểm t.
    """
    timestamp: datetime.datetime
    allocator_name: str = "MultiAgent_RiskOverlay"
    assets: List[AssetWeight]
    cash_weight: float = Field(default=1.0, description="Tỷ lệ tiền mặt còn lại")
    
    def validate_weights(self):
        """Đảm bảo tổng trọng số + tiền mặt = 1.0."""
        total_asset_weight = sum(a.target_weight for a in self.assets)
        if total_asset_weight > 1.0:
            raise ValueError("Tổng trọng số tài sản lớn hơn vốn khả dụng (1.0).")
        self.cash_weight = round(1.0 - total_asset_weight, 4)
