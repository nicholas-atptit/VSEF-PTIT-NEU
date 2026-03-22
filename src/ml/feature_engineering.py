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


class FeatureEngineer:
    """Computes quantitative features from daily OHLCV data.

    Usage::

        fe = FeatureEngineer()
        features_df = fe.transform(ohlcv_df)
    """

    # ── Public API ────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform OHLCV DataFrame into feature matrix.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume].
                Must be sorted ascending by date.

        Returns:
            DataFrame with original OHLCV columns + computed features.
            Rows with NaN (from insufficient history) are dropped.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

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

        # Money flow approximation
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["money_flow"] = typical_price * df["volume"]

        for w in WINDOWS:
            df[f"mf_ratio_{w}"] = df["money_flow"].rolling(w).mean()

        return df

    # ── Utility ───────────────────────────────────────────────

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of computed feature column names (excludes date & OHLCV)."""
        exclude = {"date", "open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in exclude]
