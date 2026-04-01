"""Technical indicator calculation module."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TechnicalFeatures:
    """Calculate technical indicators from OHLC data."""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a standard set of technical indicators."""
        if df.empty or 'close' not in df.columns:
            return df
            
        logger.debug("calculating_technical_indicators", rows=len(df))
        try:
            import pandas_ta as ta
            # Ensure index is datetime if needed
            df_ta = df.copy()
            
            # RSI
            df_ta['rsi'] = ta.rsi(df_ta['close'], length=14)
            
            # SMAs
            df_ta['sma_20'] = ta.sma(df_ta['close'], length=20)
            df_ta['sma_50'] = ta.sma(df_ta['close'], length=50)
            df_ta['sma_200'] = ta.sma(df_ta['close'], length=200)
            
            # MACD
            macd = ta.macd(df_ta['close'])
            if macd is not None:
                df_ta = pd.concat([df_ta, macd], axis=1)
                
            return df_ta
        except Exception as e:
            logger.error("technical_indicator_error", error=str(e))
            return df

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add specific momentum-based features."""
        if df.empty or 'close' not in df.columns:
            return df
        # TODO: Implement complex momentum features
        return df
