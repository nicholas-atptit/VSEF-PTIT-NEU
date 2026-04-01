"""Fundamental feature engineering module."""

from __future__ import annotations

import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FundamentalFeatures:
    """Process fundamental data into ML-ready features."""

    @staticmethod
    def process_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize key financial ratios from the vnstock ratio DataFrame.
        
        Expected input: vnstock finance.ratio() output with indicators as index or 'Chỉ số tài chính' column.
        """
        if df.empty:
            return df
            
        logger.debug("processing_fundamental_ratios")
        try:
            # Vnstock 3.x usually has the indicator name in 'Chỉ số tài chính'
            id_col = 'Chỉ số tài chính' if 'Chỉ số tài chính' in df.columns else df.index.name
            
            # Map of interest: (Indicator Name -> New Column Name)
            mapping = {
                'P/E': 'pe',
                'P/B': 'pb',
                'ROE': 'roe',
                'ROA': 'roa',
                'Nợ/Vốn CSH': 'debt_to_equity',
                'Tỉ suất lợi nhuận gộp': 'gross_margin'
            }
            
            # Pivot if needed or filter rows
            # Assuming the columns are 'Chỉ số tài chính' and year columns
            # We take the latest year (first numeric column after 'Chỉ số tài chính')
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return pd.DataFrame()
                
            latest_year = numeric_cols[0] # Usually the most recent is the first column in vnstock
            
            result_dict = {}
            for row_idx, row in df.iterrows():
                indicator = row[id_col] if id_col in df.columns else row_idx
                if indicator in mapping:
                    result_dict[mapping[indicator]] = row[latest_year]
            
            return pd.DataFrame([result_dict])
        except Exception as e:
            logger.error("fundamental_processing_error", error=str(e))
            return pd.DataFrame()
