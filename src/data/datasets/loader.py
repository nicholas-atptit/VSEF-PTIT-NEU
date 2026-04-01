"""Dataset preparation and loading module."""

from __future__ import annotations

import pandas as pd
from typing import Tuple, List, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DatasetLoader:
    """Prepares training, validation, and test datasets."""

    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols

    def create_features_labels(self) -> pd.DataFrame:
        """Load features and labels and merge them for all symbols.
        
        Fetches OHLC and fundamentals via adapter and processes through generators.
        """
        from src.data.adapters.vnstock_adapter import VnstockAdapter
        from src.ml.features.generator import FeatureGenerator
        from src.labels.regression import RegressionLabels
        
        adapter = VnstockAdapter()
        generator = FeatureGenerator()
        labeler = RegressionLabels()
        
        all_data = []
        
        # Use a lookback for OHLC (e.g., 2 years for training)
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        
        for symbol in self.symbols:
            logger.info("processing_symbol_for_dataset", symbol=symbol)
            ohlc = adapter.get_ohlc(symbol, start_date, end_date)
            fund = adapter.get_financial_ratios(symbol)
            
            if ohlc.empty:
                continue
                
            # Generate features
            df = generator.generate(ohlc, fund)
            # Generate labels (1d return)
            df = labeler.add_future_return(df, horizon=1)
            
            df['symbol'] = symbol
            all_data.append(df)
            
        if not all_data:
            return pd.DataFrame()
            
        full_df = pd.concat(all_data).dropna()
        logger.info("features_labels_created", total_rows=len(full_df))
        return full_df

    def temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically by sorting timestamps."""
        if 'timestamp' not in df.columns:
            return df, pd.DataFrame()
            
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * (1 - test_size))
        
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        logger.info("temporal_split_complete", train_rows=len(train), test_rows=len(test))
        return train, test
