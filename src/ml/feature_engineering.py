"""Module 1: Advanced Feature Engineering — OHLCV -> Feature Vectors.

Transforms raw daily OHLCV data into a rich feature matrix suitable for
ML model training and inference.  All features use only past data via
rolling windows to prevent data leakage.

Rolling windows: 5 (short), 20 (medium), 60 (long) sessions.

VN100 Extensions:
    Added ``_add_vn100_daily_features`` for daily prediction features
    required by the VN100 batch pipeline.  See ``VN100_DAILY_FEATURES``
    for the canonical list.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Rolling window sizes for multi-timeframe analysis
WINDOWS = [5, 20, 60]

# ── VN100 Daily Feature Catalogue ────────────────────────────────────────
# Canonical names for features required by the VN100 daily prediction
# pipeline.  Features marked "(alias)" are pointers to pre-existing
# columns computed by the legacy pipeline.
VN100_DAILY_FEATURES: List[str] = [
    # Price-level / return features
    "prev_close",
    "close_to_close_return_1d",   # alias -> pct_return
    "open_to_close_return_1d",
    "overnight_return_1d",
    "high_low_range_pct",
    "true_range",
    "atr_14",                     # already computed by legacy pipeline
    "return_3d",
    "return_5d",                  # alias -> return_roll_5
    "return_10d",
    "return_20d",                 # alias -> return_roll_20
    # Volume features
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5",             # already computed by legacy pipeline
    "volume_ratio_20",            # already computed by legacy pipeline
    # Value (turnover) features
    "value_ratio_5",
    "value_ratio_20",
    # Volatility features
    "rolling_volatility_5",
    "rolling_volatility_20",
]


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

    def transform(self, df: pd.DataFrame, fundamentals_df: pd.DataFrame = None, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """Transform OHLCV DataFrame into feature matrix.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume].
                Must be sorted ascending by date.
            fundamentals_df: Optional DataFrame with quarterly ratios merged daily.
            sentiment_df: Optional DataFrame with [ticker, date, sentiment_avg, ...]

        Returns:
            DataFrame with original OHLCV columns + computed features + sentiment features.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        # Convert date to standard format for merging
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # --- Step 0: Merge Fundamentals if provided ---
        if fundamentals_df is not None and not fundamentals_df.empty:
            df = df.merge(fundamentals_df, on='date', how='left').ffill()

        # --- Step 0.5: Merge Sentiment if provided ---
        if sentiment_df is not None and not sentiment_df.empty:
            # Ensure sentiment_df date is standard
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
            df = df.merge(sentiment_df, on='date', how='left').fillna(0) # 0 for no newsdays

        # ── Step 1: Forward-fill OHLCV ──
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

        # ── Step 5: VN100 daily prediction features ──
        df = self._add_vn100_daily_features(df)

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
        """ATR-14, Historical Volatility, Parkinson Volatility, Bollinger Bandwidth."""

        # --- ATR (Average True Range, 14-day) ---
        atr_col = df.ta.atr(length=14)
        if atr_col is not None:
            df["atr_14"] = atr_col
        else:
            df["atr_14"] = 0.0

        # --- Historical Volatility (Standard Std Dev) ---
        df["hv_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

        # --- Parkinson Volatility (High-Low Range-based) ---
        # Formula: sqrt( (1 / (4 * n * ln(2))) * sum( ln(H/L)^2 ) ) * sqrt(252)
        window = 20
        hl_term = np.log(df["high"] / df["low"])**2
        park_vol = np.sqrt(hl_term.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
        df["park_vol_20"] = park_vol

        # --- Rogers-Satchell Volatility ---
        # Formula: Sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
        rs_term = np.log(df['high']/df['close']) * np.log(df['high']/df['open']) + \
                  np.log(df['low']/df['close']) * np.log(df['low']/df['open'])
        rs_vol = np.sqrt(rs_term.rolling(window).mean()) * np.sqrt(252)
        df["rs_vol_20"] = rs_vol

        # --- Yang-Zhang Volatility (Combining Overnight, Open-to-Close, and RS) ---
        # Formula: sqrt(sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2)
        # where k = 0.34 / (1.34 + (n+1)/(n-1))
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Overnight Vol (log(O_t / C_{t-1}))
        overnight_ret = np.log(df['open'] / df['close'].shift(1))
        overnight_vol_sq = overnight_ret.rolling(window).var()
        
        # Open-to-Close Vol (log(C_t / O_t))
        open_to_close_ret = np.log(df['close'] / df['open'])
        open_to_close_vol_sq = open_to_close_ret.rolling(window).var()
        
        # Rogers-Satchell Vol Squre (already computed in rs_term)
        rs_vol_sq = rs_term.rolling(window).mean()
        
        yz_vol = np.sqrt(overnight_vol_sq + k * open_to_close_vol_sq + (1 - k) * rs_vol_sq) * np.sqrt(252)
        df["yz_vol_20"] = yz_vol

        # --- Bollinger Bandwidth ---
        for w in WINDOWS:
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

    # ── VN100 Daily Prediction Features ──────────────────────

    def _add_vn100_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardised daily features for the VN100 prediction pipeline.

        This method only creates features that do not already exist as
        named columns.  Where the legacy pipeline computes an equivalent
        value under a different name (e.g. ``pct_return`` ≡
        ``close_to_close_return_1d``), an alias is created.

        All features are time-safe: they use only current-row or past
        data via ``.shift()`` / ``.rolling()`` — no look-ahead.

        Feature groups:
            1. Price-level / return features
            2. Volume features
            3. Value (turnover) features
            4. Volatility features
        """
        # ── 1. Price-level / return features ─────────────────

        # Previous session close
        if "prev_close" not in df.columns:
            df["prev_close"] = df["close"].shift(1)

        # Close-to-close 1-day return  (alias for pct_return)
        if "close_to_close_return_1d" not in df.columns:
            if "pct_return" in df.columns:
                df["close_to_close_return_1d"] = df["pct_return"]
            else:
                df["close_to_close_return_1d"] = df["close"].pct_change()

        # Open-to-close intraday return
        if "open_to_close_return_1d" not in df.columns:
            df["open_to_close_return_1d"] = (df["close"] - df["open"]) / df["open"]

        # Overnight return (gap: open_t vs close_{t-1})
        if "overnight_return_1d" not in df.columns:
            _prev_close = df["prev_close"] if "prev_close" in df.columns else df["close"].shift(1)
            df["overnight_return_1d"] = (df["open"] - _prev_close) / _prev_close

        # High-low range as percentage of close
        if "high_low_range_pct" not in df.columns:
            df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"]

        # True Range (scalar, not averaged)
        if "true_range" not in df.columns:
            _prev_close = df["close"].shift(1)
            tr_hl = df["high"] - df["low"]
            tr_hc = (df["high"] - _prev_close).abs()
            tr_lc = (df["low"] - _prev_close).abs()
            df["true_range"] = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)

        # atr_14 — already computed by _add_volatility_features; skip.

        # Multi-day returns
        if "return_3d" not in df.columns:
            df["return_3d"] = df["close"].pct_change(3)

        if "return_5d" not in df.columns:
            # Alias for return_roll_5 if it exists
            if "return_roll_5" in df.columns:
                df["return_5d"] = df["return_roll_5"]
            else:
                df["return_5d"] = df["close"].pct_change(5)

        if "return_10d" not in df.columns:
            df["return_10d"] = df["close"].pct_change(10)

        if "return_20d" not in df.columns:
            if "return_roll_20" in df.columns:
                df["return_20d"] = df["return_roll_20"]
            else:
                df["return_20d"] = df["close"].pct_change(20)

        # ── 2. Volume features ───────────────────────────────

        # Volume moving averages (raw values, useful for downstream)
        if "volume_ma_5" not in df.columns:
            df["volume_ma_5"] = df["volume"].rolling(5).mean()

        if "volume_ma_20" not in df.columns:
            df["volume_ma_20"] = df["volume"].rolling(20).mean()

        # volume_ratio_5 / volume_ratio_20 — already computed in Step 4; skip.

        # ── 3. Value (turnover) features ─────────────────────

        # Turnover proxy = close × volume  (approximation for daily value)
        _turnover = df["close"] * df["volume"]

        if "value_ratio_5" not in df.columns:
            _turnover_ma_5 = _turnover.rolling(5).mean()
            df["value_ratio_5"] = _turnover / _turnover_ma_5

        if "value_ratio_20" not in df.columns:
            _turnover_ma_20 = _turnover.rolling(20).mean()
            df["value_ratio_20"] = _turnover / _turnover_ma_20

        # ── 4. Volatility features ───────────────────────────

        # Rolling volatility (std of pct_return over window)
        _ret = df["pct_return"] if "pct_return" in df.columns else df["close"].pct_change()

        if "rolling_volatility_5" not in df.columns:
            df["rolling_volatility_5"] = _ret.rolling(5).std()

        if "rolling_volatility_20" not in df.columns:
            df["rolling_volatility_20"] = _ret.rolling(20).std()

        # ── Logging ──────────────────────────────────────────
        vn100_present = [f for f in VN100_DAILY_FEATURES if f in df.columns]
        logger.info(
            "vn100_daily_features_added",
            features_added=len(vn100_present),
            total_requested=len(VN100_DAILY_FEATURES),
        )

        return df

    # ── Utility ───────────────────────────────────────────────

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of computed feature column names (excludes leakage & metadata)."""
        exclude = {
            "date", "open", "high", "low", "close", "close_raw",
            "volume", "ticker", "time",
        }
        # Select columns in exclude set, then return others if they are numeric AND not targets
        cols = [c for c in df.columns if c not in exclude and not c.startswith("target_")]
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
