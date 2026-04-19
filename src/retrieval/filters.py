"""Canonical filtering models for retrieval queries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RetrievalFilters(BaseModel):
    """Filter criteria for retrieval queries."""
    ticker: Optional[str] = None
    ticker_group: Optional[str] = None
    sector: Optional[str] = None
    horizon: Optional[int] = None
    target_type: Optional[str] = None
    regime_label: Optional[str] = None
    volatility_bucket: Optional[str] = None
    signal_strength_bucket: Optional[str] = None
    agreement_bucket: Optional[str] = None
    run_mode: Optional[str] = None
    realized_outcome_label: Optional[str] = None
    source_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert non-None filters to a flat dictionary."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
