"""Feature generator orchestration."""

from __future__ import annotations

import pandas as pd
from src.ml.features.technical import TechnicalFeatures
from src.ml.features.fundamental import FundamentalFeatures
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FeatureGenerator:
    """Orchestrates technical and fundamental feature generation."""

    def __init__(self) -> None:
        self.tech = TechnicalFeatures()
        self.fund = FundamentalFeatures()

    def generate(self, ohlc_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """Merge and generate all features for a symbol.
        
        Aligns fundamental data with the latest OHLC period.
        """
        if ohlc_df.empty:
            return pd.DataFrame()
            
        logger.info("generating_all_features", rows=len(ohlc_df))
        
        # 1. Technical Features
        df = self.tech.add_all_indicators(ohlc_df)
        
        # 2. Fundamental Features
        fund_processed = self.fund.process_ratios(fundamental_df)
        if not fund_processed.empty:
            # Broadcast latest fundamentals to all OHLC rows (static for now)
            for col in fund_processed.columns:
                df[f'fund_{col}'] = fund_processed[col].iloc[0]
                
        return df
