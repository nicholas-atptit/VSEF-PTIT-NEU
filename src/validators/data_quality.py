"""Data quality validation layer for stock market data.

Provides checks for common data issues such as missing fields, negative prices,
high/low violations, and suspicious volume patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DataQualityValidator:
    """Validator for OHLCV and feature datasets."""

    def __init__(self, ticker: Optional[str] = None):
        self.ticker = ticker or "Unknown"

    def validate_ohlcv(self, df: pd.DataFrame, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
        """Validate raw OHLCV data.
        
        Checks:
        - Required columns present
        - Basic numeric types
        - No negative prices or volume
        - High >= Low
        - No duplicate dates
        - Empty dataframe
        """
        errors = []
        warnings = []
        
        if df is None or df.empty:
            msg = f"[{self.ticker}] Dataframe is empty."
            if raise_on_error: raise ValueError(msg)
            return False, [msg]

        # 1. Required Columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"[{self.ticker}] Missing columns: {missing}")

        # 2. Date/Time checks
        date_col = "date" if "date" in df.columns else ("time" if "time" in df.columns else None)
        if not date_col:
            errors.append(f"[{self.ticker}] No date/time column found.")
        else:
            if df[date_col].duplicated().any():
                errors.append(f"[{self.ticker}] Duplicate dates detected.")

        # 3. Numeric checks
        for col in required:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"[{self.ticker}] Column '{col}' is not numeric.")
                    continue
                
                # Negative values
                if (df[col] < 0).any():
                    errors.append(f"[{self.ticker}] Negative values found in '{col}'.")
                
                # Missing values (hard check for OHLC)
                if col != "volume" and df[col].isna().any():
                    errors.append(f"[{self.ticker}] NaN values found in '{col}'.")

        # 4. Consistency checks
        if "high" in df.columns and "low" in df.columns:
            violations = (df["high"] < df["low"]).sum()
            if violations > 0:
                errors.append(f"[{self.ticker}] High < Low found in {violations} rows.")

        # 5. Suspicious Zeros
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol > len(df) * 0.1: # More than 10% zero volume
                warnings.append(f"[{self.ticker}] Suspiciously high zero-volume rate: {zero_vol/len(df):.1%}")

        # Handle findings
        for warn in warnings:
            logger.warning("data_quality_warning", ticker=self.ticker, message=warn)
            
        if errors:
            for err in errors:
                logger.error("data_quality_error", ticker=self.ticker, message=err)
            if raise_on_error:
                raise ValueError(f"Data quality validation failed for {self.ticker}: {'; '.join(errors)}")
            return False, errors + warnings

        return True, warnings

    def validate_features(self, df: pd.DataFrame, feature_cols: List[str], null_threshold: float = 0.2) -> Tuple[bool, List[str]]:
        """Validate computed features.
        
        Checks:
        - Excessive null rates per feature
        - Inf/-Inf values
        """
        warnings = []
        
        if df.empty:
            return False, ["Feature dataframe is empty."]

        # Check null rates
        null_rates = df[feature_cols].isna().mean()
        high_nulls = null_rates[null_rates > null_threshold]
        if not high_nulls.empty:
            for feat, rate in high_nulls.items():
                warnings.append(f"[{self.ticker}] High null rate for feature '{feat}': {rate:.1%}")

        # Check for Infs
        has_inf = np.isinf(df[feature_cols]).any().any()
        if has_inf:
            warnings.append(f"[{self.ticker}] Infinite values found in feature matrix.")

        for warn in warnings:
            logger.warning("feature_quality_warning", ticker=self.ticker, message=warn)

        return True, warnings

def validate_ohlcv_standalone(df: pd.DataFrame, ticker: str = "Unknown") -> bool:
    """Helper for one-off validation."""
    validator = DataQualityValidator(ticker=ticker)
    success, _ = validator.validate_ohlcv(df, raise_on_error=False)
    return success
