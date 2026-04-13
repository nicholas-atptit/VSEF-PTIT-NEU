"""Module 1: Advanced Feature Engineering — OHLCV -> Feature Vectors.

Transforms raw daily OHLCV data into a rich feature matrix suitable for
ML model training and inference.  All features use only past data via
rolling windows to prevent data leakage.

Rolling windows: 5 (short), 20 (medium), 60 (long) sessions.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Rolling window sizes for multi-timeframe analysis
WINDOWS = [5, 20, 60]
LEGACY_COMPATIBILITY_COLUMNS = {
    "hv_20",
    "bb_bandwidth_5",
    "bb_bandwidth_20",
    "bb_bandwidth_60",
    "pivot",
    "resistance_1",
    "support_1",
}

# ── VN100 Daily Feature Catalogue ────────────────────────────────────────
VN100_DAILY_FEATURES: List[str] = [
    "prev_close",
    "close_to_close_return_1d",
    "open_to_close_return_1d",
    "overnight_return_1d",
    "high_low_range_pct",
    "true_range",
    "atr_14",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "value_ratio_5",
    "value_ratio_20",
    "rolling_volatility_5",
    "rolling_volatility_20",
]


def apply_kalman_filter(data):
    """Simple 1D Kalman Filter for price denoising."""
    if len(data) == 0: return np.array([])
    n_iter = len(data)
    xhat = np.zeros(n_iter)
    P = np.zeros(n_iter)
    xhatminus = np.zeros(n_iter)
    Pminus = np.zeros(n_iter)
    K = np.zeros(n_iter)

    Q = 1e-5
    R = 0.1**2

    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, lower

class FeatureEngineer:
    """Computes quantitative features from daily OHLCV data."""

    def transform(self, df: pd.DataFrame, fundamentals_df: pd.DataFrame = None, 
                  sentiment_df: pd.DataFrame = None, drop_na: bool = True) -> pd.DataFrame:
        """Unified transformation pipeline for both training and inference."""
        df = df.copy().sort_values("date").reset_index(drop=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        # 0. Preserve raw Close for safe target/label generation
        df["close_raw"] = df["close"].copy()
        
        # 1. Merge external contexts accurately
        if fundamentals_df is not None and not fundamentals_df.empty:
            df = df.merge(fundamentals_df, on='date', how='left').ffill()

        if sentiment_df is not None and not sentiment_df.empty:
            if 'date' in sentiment_df.columns:
                sentiment_df = sentiment_df.copy()
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
            df = df.merge(sentiment_df, on='date', how='left').fillna(0)

        df = df.ffill() # Forward fill only for time-safety. 
        # bfill() removed to prevent future leakage in merged contexts.

        # 2. Base Returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["pct_return"] = df["close"].pct_change()

        # 3. Technical Indicators
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_structure_features(df)
        df = self._add_volume_features(df)
        df = self._add_advanced_indicators(df)
        df = self._add_lagged_features(df)
        df = self.add_kalman_filter(df)

        # 4. Multi-timeframe
        for w in WINDOWS:
            df[f"return_roll_{w}"] = df["close"].pct_change(w)
            df[f"volume_ratio_{w}"] = df["volume"] / (df["volume"].rolling(w).mean() + 1e-9)

        # 5. VN100 Features
        df = self._add_vn100_daily_features(df)
        df = self._add_legacy_compatibility_features(df)

        # 6. Delta features (Mirroring legacy script)
        # This MUST happen BEFORE dropna but AFTER all base features are computed
        df = self._add_delta_features(df)

        # 7. Final Handling of NaN (Inference vs Train)
        if drop_na:
            df = df.dropna().reset_index(drop=True)
        else:
            # For inference, preserve latest rows via filling
            # Note: ffill is safe; bfill at very end is only for inference edge cases.
            df = df.ffill().fillna(0).reset_index(drop=True)

        return df

    def add_kalman_filter(self, df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        if col in df.columns:
            df[f"{col}_kalman"] = apply_kalman_filter(df[col].values)
        return df

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        exclude = {"date", "ticker", "open", "high", "low", "close", "volume"}
        feat_cols = [c for c in df.columns if c not in exclude and not c.startswith('d_')]
        delta_map = {f"d_{col}": df[col] - df[col].shift(1) for col in feat_cols}
        if not delta_map:
            return df
        return pd.concat([df, pd.DataFrame(delta_map, index=df.index)], axis=1)

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        window = 20
        # Parkinson
        hl_term = np.log(df["high"] / (df["low"] + 1e-9))**2
        df["park_vol_20"] = np.sqrt(hl_term.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
        # RS
        rs_term = np.log(df['high']/df['close']) * np.log(df['high']/df['open']) + \
                  np.log(df['low']/df['close']) * np.log(df['low']/df['open'])
        df["rs_vol_20"] = np.sqrt(rs_term.rolling(window).mean().clip(lower=0)) * np.sqrt(252)
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for n in [5, 20]:
            df[f"range_{n}"] = (df["high"].rolling(n).max() - df["low"].rolling(n).min()) / (df["close"] + 1e-9)
        df["price_gap"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)
        for w in WINDOWS:
            df[f"roc_{w}"] = df["close"].pct_change(w)
        return df

    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in WINDOWS:
            df[f"dist_ma_{w}"] = (df["close"] - df["close"].rolling(w).mean()) / (df["close"].rolling(w).mean() + 1e-9)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in WINDOWS:
            df[f"vroc_{w}"] = df["volume"].pct_change(w)
        return df

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = compute_rsi(df["close"], window=14)
        bb_upper, bb_lower = compute_bollinger_bands(df["close"], window=20, num_std=2.0)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["close"] + 1e-9)
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["pct_return", "rsi"]:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def _add_vn100_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prev_close"] = df["close"].shift(1)
        df["close_to_close_return_1d"] = df["close"].pct_change()
        df["open_to_close_return_1d"] = (df["close"] - df["open"]) / (df["open"] + 1e-9)
        df["overnight_return_1d"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)
        df["high_low_range_pct"] = (df["high"] - df["low"]) / (df["low"] + 1e-9)
        df["true_range"] = np.maximum(df["high"] - df["low"], 
                                      np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                                 abs(df["low"] - df["close"].shift(1))))
        df["atr_14"] = df["true_range"].rolling(14).mean()
        df["return_3d"] = df["close"].pct_change(3)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)
        
        for w in [5, 20]:
            df[f"volume_ma_{w}"] = df["volume"].rolling(w).mean()
            df[f"rolling_volatility_{w}"] = df["pct_return"].rolling(w).std()
            _turnover = df["close"] * df["volume"]
            df[f"value_ratio_{w}"] = _turnover / (_turnover.rolling(w).mean() + 1e-9)
            
        return df

    def _add_legacy_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore legacy feature names without changing the canonical feature set."""
        if "rolling_volatility_20" in df.columns:
            df["hv_20"] = df["rolling_volatility_20"] * np.sqrt(252)
        else:
            df["hv_20"] = df["pct_return"].rolling(20).std() * np.sqrt(252)

        for window in WINDOWS:
            rolling_mean = df["close"].rolling(window).mean()
            rolling_std = df["close"].rolling(window).std()
            df[f"bb_bandwidth_{window}"] = (4.0 * rolling_std) / (rolling_mean.abs() + 1e-9)

        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)
        df["pivot"] = (prev_high + prev_low + prev_close) / 3.0
        df["resistance_1"] = (2.0 * df["pivot"]) - prev_low
        df["support_1"] = (2.0 * df["pivot"]) - prev_high
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = {"date", "open", "high", "low", "close", "close_raw", "volume", "ticker", "time"}
        cols = [c for c in df.columns if c not in exclude and not str(c).startswith("target_")]
        cols = [c for c in cols if c not in LEGACY_COMPATIBILITY_COLUMNS]
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
