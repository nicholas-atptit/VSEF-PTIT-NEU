"""Module 1: Advanced Feature Engineering — OHLCV → Feature Vectors.

Transforms raw daily OHLCV data into a rich feature matrix suitable for
ML model training and inference.  All features use only past data via
rolling windows to prevent data leakage.

Rolling windows: 5 (short), 20 (medium), 60 (long) sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Rolling window sizes for multi-timeframe analysis
WINDOWS = [5, 20, 60]


def apply_kalman_filter(data):
    """Simple 1D Kalman Filter for price denoising."""
    n_iter = len(data)
    sz = (n_iter,)
    xhat = np.zeros(sz)      # a posteri estimate of x
    P = np.zeros(sz)         # a posteri error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor

    Q = 1e-5 # process variance
    R = 0.1**2 # estimate of measurement variance

    # intial guesses
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

class FeatureEngineer:
    """Computes quantitative features from daily OHLCV data.

    Usage::

        fe = FeatureEngineer()
        features_df = fe.transform(ohlcv_df, fundamentals_df=f_df)
    """

    # ── Public API ────────────────────────────────────────────

    def transform(self, df: pd.DataFrame, fundamentals_df: pd.DataFrame = None) -> pd.DataFrame:
        """Transform OHLCV DataFrame into feature matrix.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume].
                Must be sorted ascending by date.
            fundamentals_df: Optional DataFrame with quarterly ratios merged daily.

        Returns:
            DataFrame with original OHLCV columns + computed features.
            Rows with NaN (from insufficient history) are dropped.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        
        # --- Step 0: Merge Fundamentals if provided (forward-filled) ---
        if fundamentals_df is not None and not fundamentals_df.empty:
            # Assumes fundamentals_df has 'date' and 'ticker' or matched to df
            df = df.merge(fundamentals_df, on='date', how='left').ffill()

        # ── Step 1: Forward-fill only (no backward-fill = no leakage) ──
        df = df.ffill()

        # ── Step 2: Base returns ──
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["pct_return"] = df["close"].pct_change()

        # ── Step 3: Feature groups ──
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_structure_features(df)
        df = self._add_volume_features(df)
        df = self._add_advanced_indicators(df)
        df = self._add_lagged_features(df)

        # ── Step 4: Multi-timeframe rolling returns ──
        for w in WINDOWS:
            df[f"return_roll_{w}"] = df["close"].pct_change(w)
            df[f"volume_ratio_{w}"] = df["volume"] / df["volume"].rolling(w).mean()

        # Drop rows where features couldn't be computed
        df = df.dropna().reset_index(drop=True)

        logger.info(
            "features_computed",
            rows=len(df),
            features=len([c for c in df.columns if c not in ("date",)]),
        )
        return df

    # ── Volatility Features ───────────────────────────────────

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR-14, Historical Volatility (HV-20), Bollinger Bandwidth."""

        # --- ATR (Average True Range, 14-day) ---
        # Using pandas-ta library correctly
        atr_col = df.ta.atr(length=14)
        if atr_col is not None:
            df["atr_14"] = atr_col
        else:
            df["atr_14"] = 0.0

        # --- Historical Volatility (annualized) ---
        # HV = σ(log_return) × √252,  rolling 20-day
        df["hv_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

        # --- Bollinger Bandwidth ---
        for w in WINDOWS:
            # ta.bbands returns lower, mid, upper, bandwidth, percent
            bb = df.ta.bbands(length=w, std=2.0)
            if bb is not None:
                bb_col = f"BBB_{w}_2.0"
                if bb_col in bb.columns:
                    df[f"bb_bandwidth_{w}"] = bb[bb_col]
                else:
                    ma = df["close"].rolling(w).mean()
                    std = df["close"].rolling(w).std()
                    df[f"bb_bandwidth_{w}"] = (4 * std) / ma
            else:
                ma = df["close"].rolling(w).mean()
                std = df["close"].rolling(w).std()
                df[f"bb_bandwidth_{w}"] = (4 * std) / ma

        return df

    # ── Momentum Features ─────────────────────────────────────

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """N-Session Range, Price Gap, rolling momentum."""

        # --- N-Session Range: (High_max - Low_min) over N sessions ---
        for n in [5, 20]:
            df[f"range_{n}"] = (
                df["high"].rolling(n).max() - df["low"].rolling(n).min()
            )

        # --- Price Gap: (Open_t - Close_{t-1}) / Close_{t-1} ---
        df["price_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # --- ROC (Rate of Change) for multiple windows ---
        for w in WINDOWS:
            df[f"roc_{w}"] = df["close"].pct_change(w)

        return df

    # ── Structure Features ────────────────────────────────────

    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dynamic Pivot Points (P, R1, S1) based on prior session H, L, C."""

        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)

        # Classic pivot point formulas
        df["pivot"] = (prev_high + prev_low + prev_close) / 3
        df["resistance_1"] = 2 * df["pivot"] - prev_low
        df["support_1"] = 2 * df["pivot"] - prev_high

        # Distance from close to pivot levels (normalized)
        df["dist_to_pivot"] = (df["close"] - df["pivot"]) / df["close"]
        df["dist_to_r1"] = (df["close"] - df["resistance_1"]) / df["close"]
        df["dist_to_s1"] = (df["close"] - df["support_1"]) / df["close"]

        return df

    # ── Volume Features ───────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features for liquidity analysis."""
        # OBV-like cumulative direction volume
        direction = np.sign(df["close"] - df["close"].shift(1))
        df["volume_direction"] = direction * df["volume"]
        
        # Relative Volume (10d, 30d)
        df["rel_vol_10"] = df["volume"] / df["volume"].rolling(10).mean()
        df["rel_vol_30"] = df["volume"] / df["volume"].rolling(30).mean()

        # Money flow approximation
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["money_flow"] = typical_price * df["volume"]

        for w in WINDOWS:
            df[f"mf_ratio_{w}"] = df["money_flow"].rolling(w).mean()

        return df

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Smart Money and Oscillator indicators (OBV, MFI, Stoch, Williams %R)."""
        # --- OBV (On-Balance Volume) ---
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['obv_ema'] = ta.ema(df['obv'], length=10)
        df['obv_slope'] = df['obv'].pct_change(3)

        # --- MFI (Money Flow Index) ---
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

        # --- Stochastic Oscillator ---
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']

        # --- Williams %R ---
        df['wpr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # --- Phase 36: High-Octane Factors ---
        # --- Phase 41: Kalman Smoothing (Noise Reduction) ---
        df['close_raw'] = df['close'] # Keep raw for target calculation
        df['close'] = apply_kalman_filter(df['close'].values)
        
        # Standard indicators Needed for Phase 36
        rsi = ta.rsi(df['close'], length=14)
        df['rsi'] = rsi
        df['rsi_signal'] = ta.sma(rsi, length=5)
        df['rsi_diff'] = df['rsi'] - df['rsi_signal']

        # 1. Bollinger Band Squeeze
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
             # Safe column access for different pandas_ta versions
             upper_col = [c for c in bb.columns if 'BBU' in c][0]
             lower_col = [c for c in bb.columns if 'BBL' in c][0]
             df['bb_width'] = (bb[upper_col] - bb[lower_col]) / df['close']
             df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(60).mean()
             df['bb_percent'] = (df['close'] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])

        # 2. RSI Divergence Proxy (Recent High vs RSI High)
        df['rsi_20_max'] = df['rsi'].rolling(20).max()
        df['price_20_max'] = df['close'].rolling(20).max()
        df['rsi_divergence'] = (df['close'] >= df['price_20_max']) & (df['rsi'] < df['rsi_20_max'])

        # 3. Volume Intensity
        df['vol_intensity'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_surge'] = df['vol_intensity'] > 2.0

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of primary signals to capture temporal context."""
        lag_cols = ['pct_return', 'rsi', 'mfi', 'stoch_k', 'wpr', 'rel_vol_10']
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # --- Rolling Trend ---
        df['trend_5d'] = df['pct_return'].rolling(5).mean()
        df['trend_20d'] = df['pct_return'].rolling(20).mean()
        
        return df

    # ── Utility ───────────────────────────────────────────────

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of computed feature column names (excludes leakage & metadata)."""
        exclude = {"date", "open", "high", "low", "close", "volume", "ticker", "time"}
        # Select columns in exclude set, then return others if they are numeric AND not targets
        cols = [c for c in df.columns if c not in exclude and not c.startswith("target_")]
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
