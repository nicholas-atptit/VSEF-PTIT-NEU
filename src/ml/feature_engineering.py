"""Module 1: Advanced Feature Engineering — OHLCV -> feature vectors.

All deterministic features use only information available on or before date ``t``.
No feature in this module should require future prices to produce values for the
current row.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, List

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Performance Instrumentation
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _timing_context(stage_name: str, attrs_dict: dict | None = None) -> Any:
    """Context manager to measure and log stage timing."""
    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms = elapsed_ns / 1e6
        if attrs_dict is not None:
            if "feature_build_stages" not in attrs_dict:
                attrs_dict["feature_build_stages"] = {}
            attrs_dict["feature_build_stages"][stage_name] = f"{elapsed_ms:.2f}ms"
        logger.debug(f"feature_stage_completed", stage=stage_name, elapsed_ms=f"{elapsed_ms:.2f}")

WINDOWS = [5, 20, 60]
MOVING_AVERAGE_WINDOWS = [20, 50, 200]
DEFAULT_FEATURE_BUILD_MODE = "full_research_mode"
FEATURE_BUILD_MODE_CONFIGS = {
    "fast_core_mode": {
        "description": "Compute the governed core forecasting surface only; skip research-only expansion blocks.",
        "include_context": True,
        "include_regime": False,
        "include_corporate_action_diagnostics": False,
        "include_kalman": False,
        "include_legacy_compatibility": False,
        "include_delta_features": False,
    },
    "regime_risk_mode": {
        "description": "Compute core plus regime/risk context without research-only delta or compatibility aliases.",
        "include_context": True,
        "include_regime": True,
        "include_corporate_action_diagnostics": True,
        "include_kalman": False,
        "include_legacy_compatibility": False,
        "include_delta_features": False,
    },
    "full_research_mode": {
        "description": "Backward-compatible full feature surface, including compatibility aliases and delta expansion.",
        "include_context": True,
        "include_regime": True,
        "include_corporate_action_diagnostics": True,
        "include_kalman": True,
        "include_legacy_compatibility": True,
        "include_delta_features": True,
    },
}
TREND_REGIME_TO_CODE = {"bear": -1, "sideways": 0, "bull": 1}
VOLATILITY_REGIME_TO_CODE = {"low": 0, "normal": 1, "high": 2}
TREND_STATE_TO_CODE = {"downtrend": -1, "neutral": 0, "uptrend": 1}
ABNORMAL_GAP_ABS_MIN = 0.08
ABNORMAL_GAP_MULTIPLIER = 3.0
CORPORATE_ACTION_MOVE_MIN = 0.12
CORPORATE_ACTION_INTRADAY_RETRACE_MAX = 0.35
CORPORATE_ACTION_VOLUME_ZSCORE_MIN = 2.0
TREND_REGIME_STRENGTH_THRESHOLD = 0.5
PRICE_REFERENCE_COLUMNS = {"raw_close", "model_close_reference", "adjusted_close", "close_raw"}
NON_FEATURE_SUPPORT_COLUMNS = {
    "ohlcv_source",
    "price_adjustment_status",
    "price_adjustment_available",
    "price_reference_semantics",
}
LEGACY_COMPATIBILITY_COLUMNS = {
    "hv_20",
    "bb_bandwidth_5",
    "bb_bandwidth_20",
    "bb_bandwidth_60",
    "pivot",
    "resistance_1",
    "support_1",
}
NON_CANONICAL_ALIAS_COLUMNS = {
    "rsi",
    "pct_return",
    "close_return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "price_gap",
    "roc_5",
    "roc_20",
    "return_roll_5",
    "return_roll_20",
    "return_roll_60",
    "close_mean_20",
    "prev_close",
    "value_ratio_5",
    "value_ratio_20",
    "volume_shock_5",
    "volume_shock_20",
}

VN100_DAILY_FEATURES: List[str] = [
    "prev_close",
    "close_to_close_return_1d",
    "close_return_2d",
    "close_return_3d",
    "open_to_close_return_1d",
    "overnight_return_1d",
    "open_close_spread",
    "open_close_spread_pct",
    "high_low_range",
    "high_low_range_pct",
    "true_range",
    "atr_14",
    "atr_proxy_5",
    "atr_proxy_10",
    "close_mean_5",
    "close_mean_10",
    "close_mean_20",
    "close_std_5",
    "close_std_10",
    "close_std_20",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volume_ma_5",
    "volume_ma_10",
    "volume_ma_20",
    "volume_shock_5",
    "volume_shock_10",
    "volume_shock_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "value_ratio_5",
    "value_ratio_20",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
]


def apply_kalman_filter(data: np.ndarray) -> np.ndarray:
    """Simple 1D Kalman filter for price denoising."""
    if len(data) == 0:
        return np.array([])
    n_iter = len(data)
    xhat = np.zeros(n_iter)
    p = np.zeros(n_iter)
    xhatminus = np.zeros(n_iter)
    pminus = np.zeros(n_iter)
    k_gain = np.zeros(n_iter)

    q = 1e-5
    r = 0.1**2

    xhat[0] = data[0]
    p[0] = 1.0

    for idx in range(1, n_iter):
        xhatminus[idx] = xhat[idx - 1]
        pminus[idx] = p[idx - 1] + q
        k_gain[idx] = pminus[idx] / (pminus[idx] + r)
        xhat[idx] = xhatminus[idx] + k_gain[idx] * (data[idx] - xhatminus[idx])
        p[idx] = (1 - k_gain[idx]) * pminus[idx]

    return xhat


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, lower


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    fast_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    slow_k = fast_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(smooth_d).mean()
    return slow_k, slow_d


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / (atr + 1e-9)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / (atr + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + pd.to_numeric(series, errors="coerce")).rolling(window).apply(np.prod, raw=True) - 1.0


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.rolling(window).mean()
    std = numeric.rolling(window).std()
    return (numeric - mean) / (std + 1e-9)


class FeatureEngineer:
    """Computes quantitative features from daily OHLCV data."""

    @staticmethod
    def normalize_build_mode(build_mode: str | None = None) -> str:
        mode = str(build_mode or DEFAULT_FEATURE_BUILD_MODE).strip()
        if mode not in FEATURE_BUILD_MODE_CONFIGS:
            raise ValueError(
                f"Unsupported feature build mode '{mode}'. Available: {sorted(FEATURE_BUILD_MODE_CONFIGS)}"
            )
        return mode

    @staticmethod
    def build_mode_manifest() -> dict[str, Any]:
        return {
            "default_mode": DEFAULT_FEATURE_BUILD_MODE,
            "modes": {
                name: {
                    **config,
                    "documented_behavior": "explicit_opt_in_no_hidden_semantic_change",
                }
                for name, config in FEATURE_BUILD_MODE_CONFIGS.items()
            },
        }

    @staticmethod
    def _feature_block_frame(
        feature_map: dict[str, Any] | pd.DataFrame,
        index: pd.Index,
    ) -> pd.DataFrame:
        if isinstance(feature_map, pd.DataFrame):
            if feature_map.empty and len(feature_map.columns) == 0:
                return pd.DataFrame(index=index)
            block = feature_map.copy()
            if not block.index.equals(index):
                block = block.reindex(index)
            return block
        if not feature_map:
            return pd.DataFrame(index=index)
        return pd.DataFrame(feature_map, index=index)

    @staticmethod
    def _concat_feature_block(
        df: pd.DataFrame,
        feature_map: dict[str, Any] | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(feature_map, pd.DataFrame):
            if feature_map.empty and len(feature_map.columns) == 0:
                return df
        elif not feature_map:
            return df
        attrs = dict(getattr(df, "attrs", {}))
        block = FeatureEngineer._feature_block_frame(feature_map, df.index)
        overlap = [column for column in block.columns if column in df.columns]
        base = df.drop(columns=overlap) if overlap else df
        result = pd.concat([base, block], axis=1)
        result.attrs.update(attrs)
        return result

    def build_feature_frame(
        self,
        df: pd.DataFrame,
        fundamentals_df: pd.DataFrame = None,
        sentiment_df: pd.DataFrame = None,
        build_mode: str = DEFAULT_FEATURE_BUILD_MODE,
    ) -> pd.DataFrame:
        mode = self.normalize_build_mode(build_mode)
        mode_config = FEATURE_BUILD_MODE_CONFIGS[mode]
        frame = df.copy().sort_values("date").reset_index(drop=True)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

        frame = self._attach_price_reference_semantics(frame)
        frame.attrs["feature_build_mode"] = mode
        frame.attrs["feature_build_stages"] = {}

        if fundamentals_df is not None and not fundamentals_df.empty:
            fundamentals_df = fundamentals_df.copy()
            if "date" in fundamentals_df.columns:
                fundamentals_df["date"] = pd.to_datetime(fundamentals_df["date"]).dt.normalize()
                frame = frame.merge(fundamentals_df, on="date", how="left")

        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df = sentiment_df.copy()
            if "date" in sentiment_df.columns:
                sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
            frame = frame.merge(sentiment_df, on="date", how="left").fillna(0)

        frame = frame.ffill()

        with _timing_context("returns", frame.attrs):
            frame = self._concat_feature_block(
                frame,
                {
                    "log_return": np.log(frame["close"] / frame["close"].shift(1)),
                    "pct_return": frame["close"].pct_change(),
                },
            )

        with _timing_context("volatility", frame.attrs):
            frame = self._add_volatility_features(frame)

        with _timing_context("momentum", frame.attrs):
            frame = self._add_momentum_features(frame)

        with _timing_context("structure", frame.attrs):
            frame = self._add_structure_features(frame)

        with _timing_context("volume", frame.attrs):
            frame = self._add_volume_features(frame)

        with _timing_context("advanced_indicators", frame.attrs):
            frame = self._add_advanced_indicators(frame)

        with _timing_context("lagged_features", frame.attrs):
            frame = self._add_lagged_features(frame)

        if mode_config["include_kalman"]:
            with _timing_context("kalman_filter", frame.attrs):
                frame = self.add_kalman_filter(frame)

        with _timing_context("rolling_features", frame.attrs):
            rolling_block: dict[str, pd.Series] = {}
            for window in WINDOWS:
                rolling_block[f"return_roll_{window}"] = frame["close"].pct_change(window)
                rolling_block[f"volume_ratio_{window}"] = frame["volume"] / (frame["volume"].rolling(window).mean() + 1e-9)
            frame = self._concat_feature_block(frame, rolling_block)

        with _timing_context("vn100_daily", frame.attrs):
            frame = self._add_vn100_daily_features(frame)

        if mode_config["include_context"]:
            with _timing_context("context_features", frame.attrs):
                frame = self._add_context_features(frame)

        if mode_config["include_regime"]:
            with _timing_context("regime_features", frame.attrs):
                frame = self._add_regime_features(frame)

        if mode_config["include_corporate_action_diagnostics"]:
            with _timing_context("corporate_action_diagnostics", frame.attrs):
                frame = self._add_corporate_action_diagnostics(frame)

        if mode_config["include_legacy_compatibility"]:
            with _timing_context("legacy_compatibility", frame.attrs):
                frame = self._add_legacy_compatibility_features(frame)

        if mode_config["include_delta_features"]:
            with _timing_context("delta_features", frame.attrs):
                frame = self._add_delta_features(frame)

        return frame.copy()

    def transform(
        self,
        df: pd.DataFrame,
        fundamentals_df: pd.DataFrame = None,
        sentiment_df: pd.DataFrame = None,
        drop_na: bool = True,
        build_mode: str = DEFAULT_FEATURE_BUILD_MODE,
    ) -> pd.DataFrame:
        df = self.build_feature_frame(
            df,
            fundamentals_df=fundamentals_df,
            sentiment_df=sentiment_df,
            build_mode=build_mode,
        )
        semantic_cols = [col for col in PRICE_REFERENCE_COLUMNS if col in df.columns]
        support_cols = [col for col in NON_FEATURE_SUPPORT_COLUMNS if col in df.columns]
        non_semantic_cols = [col for col in df.columns if col not in semantic_cols]
        completeness_cols = [col for col in non_semantic_cols if col not in support_cols]

        structurally_missing_cols = [col for col in completeness_cols if df[col].isna().all()]
        if structurally_missing_cols:
            numeric_structural = [col for col in structurally_missing_cols if pd.api.types.is_numeric_dtype(df[col])]
            non_numeric_structural = [col for col in structurally_missing_cols if col not in numeric_structural]
            if numeric_structural:
                df[numeric_structural] = df[numeric_structural].fillna(0.0)
            if non_numeric_structural:
                df[non_numeric_structural] = df[non_numeric_structural].fillna("unavailable")
            logger.debug(
                "feature_structural_missing_filled",
                columns=sorted(structurally_missing_cols),
                count=len(structurally_missing_cols),
            )

        if drop_na:
            df = df.dropna(subset=completeness_cols).reset_index(drop=True)
        else:
            df[non_semantic_cols] = df[non_semantic_cols].ffill().fillna(0)
            df = df.reset_index(drop=True)

        self.validate_features(df, drop_na)
        return df

    def validate_features(self, df: pd.DataFrame, drop_na: bool) -> None:
        if len(df) == 0:
            raise ValueError("[feature_validation_failed] DataFrame is empty after feature generation")

        if not drop_na:
            numeric_cols = [
                col
                for col in df.select_dtypes(include=[np.number]).columns
                if col not in PRICE_REFERENCE_COLUMNS
            ]
            last_row = df[numeric_cols].iloc[-1]
            is_bad = np.isinf(last_row) | np.isnan(last_row)
            if is_bad.any():
                bad_cols = last_row.index[is_bad].tolist()
                raise ValueError(
                    f"[feature_validation_failed] NaN or Inf found in latest row for columns: {bad_cols}"
                )

    @staticmethod
    def _attach_price_reference_semantics(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        price_status = "unknown"
        if "price_adjustment_status" in frame.columns and frame["price_adjustment_status"].notna().any():
            price_status = str(frame["price_adjustment_status"].dropna().iloc[0]).strip().lower()
        elif "adjustment_status" in frame.attrs:
            price_status = str(frame.attrs["adjustment_status"]).strip().lower()

        model_close_reference = pd.to_numeric(frame["close"], errors="coerce")
        feature_block: dict[str, Any] = {
            "model_close_reference": model_close_reference,
            "price_reference_semantics": pd.Series(price_status, index=frame.index, dtype="object"),
        }
        if price_status == "adjusted":
            feature_block["adjusted_close"] = model_close_reference.copy()
            feature_block["raw_close"] = pd.Series(np.nan, index=frame.index)
            feature_block["close_raw"] = pd.Series(np.nan, index=frame.index)
            feature_block["price_adjustment_available"] = pd.Series(True, index=frame.index)
        else:
            feature_block["raw_close"] = model_close_reference.copy()
            feature_block["close_raw"] = model_close_reference.copy()
            feature_block["price_adjustment_available"] = pd.Series(False, index=frame.index)
        return FeatureEngineer._concat_feature_block(frame, feature_block)

    def add_kalman_filter(self, df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
        if col in df.columns:
            return self._concat_feature_block(
                df,
                {
                    f"{col}_kalman": apply_kalman_filter(
                        pd.to_numeric(df[col], errors="coerce").ffill().fillna(0).values
                    )
                },
            )
        return df

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        exclude = {"date", "ticker", "open", "high", "low", "close", "volume"}
        exclude |= PRICE_REFERENCE_COLUMNS
        exclude |= NON_FEATURE_SUPPORT_COLUMNS
        feat_cols = [
            column
            for column in df.select_dtypes(include=[np.number]).columns
            if column not in exclude and not str(column).startswith("d_")
        ]
        if not feat_cols:
            return df
        numeric_block = df[feat_cols]
        valid_columns = numeric_block.notna().sum(axis=0) >= 2
        if not bool(valid_columns.any()):
            return df
        delta_block = numeric_block.loc[:, valid_columns[valid_columns].index].diff().add_prefix("d_")
        return self._concat_feature_block(df, delta_block)

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        hl_term = np.log(df["high"] / (df["low"] + 1e-9)) ** 2
        park_vol_20 = np.sqrt(hl_term.rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
        rs_term = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) + np.log(
            df["low"] / df["close"]
        ) * np.log(df["low"] / df["open"])
        rs_vol_20 = np.sqrt(rs_term.rolling(20).mean().clip(lower=0)) * np.sqrt(252)
        return self._concat_feature_block(
            df,
            {
                "park_vol_20": park_vol_20,
                "rs_vol_20": rs_vol_20,
            },
        )

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        for n in [5, 20]:
            feature_map[f"range_{n}"] = (df["high"].rolling(n).max() - df["low"].rolling(n).min()) / (df["close"] + 1e-9)
        feature_map["price_gap"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)
        for window in WINDOWS:
            feature_map[f"roc_{window}"] = df["close"].pct_change(window)
        for window in [5, 10, 20]:
            feature_map[f"momentum_{window}"] = df["close"] - df["close"].shift(window)
        return self._concat_feature_block(df, feature_map)

    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        sma_200 = None
        for window in WINDOWS:
            rolling_mean = df["close"].rolling(window).mean()
            feature_map[f"dist_ma_{window}"] = (df["close"] - rolling_mean) / (rolling_mean + 1e-9)
            feature_map[f"rolling_min_{window}"] = df["close"].rolling(window).min()
            feature_map[f"rolling_max_{window}"] = df["close"].rolling(window).max()
        for window in MOVING_AVERAGE_WINDOWS:
            moving_average = df["close"].rolling(window).mean()
            feature_map[f"sma_{window}"] = moving_average
            if window == 200:
                sma_200 = moving_average
        for window in [12, 20, 50]:
            feature_map[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False, min_periods=window).mean()
        if sma_200 is None:
            sma_200 = df["close"].rolling(200).mean()
        feature_map["close_to_sma_200"] = (df["close"] - sma_200) / (sma_200 + 1e-9)
        return self._concat_feature_block(df, feature_map)

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        turnover = df["close"] * df["volume"]
        feature_map: dict[str, pd.Series] = {"turnover": turnover}
        for window in WINDOWS:
            turnover_ma = turnover.rolling(window).mean()
            feature_map[f"vroc_{window}"] = df["volume"].pct_change(window)
            feature_map[f"volume_std_{window}"] = df["volume"].rolling(window).std()
            feature_map[f"volume_min_{window}"] = df["volume"].rolling(window).min()
            feature_map[f"volume_max_{window}"] = df["volume"].rolling(window).max()
            feature_map[f"turnover_ma_{window}"] = turnover_ma
            feature_map[f"turnover_ratio_{window}"] = turnover / (turnover_ma + 1e-9)
        feature_map["volume_spike_zscore_20"] = (
            df["volume"] - df["volume"].rolling(20).mean()
        ) / (df["volume"].rolling(20).std() + 1e-9)
        feature_map["amihud_20"] = (df["pct_return"].abs() / (turnover + 1e-9)).rolling(20).mean()
        return self._concat_feature_block(df, feature_map)

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = compute_rsi(df["close"], window=14)
        macd_line, signal_line, macd_hist = compute_macd(df["close"])
        bb_upper, bb_lower = compute_bollinger_bands(df["close"], window=20, num_std=2.0)
        stoch_k, stoch_d = compute_stochastic(df["high"], df["low"], df["close"])
        adx = compute_adx(df["high"], df["low"], df["close"], window=14)
        return self._concat_feature_block(
            df,
            {
                "rsi": rsi,
                "rsi_14": rsi,
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_hist": macd_hist,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_width": (bb_upper - bb_lower) / (df["close"] + 1e-9),
                "bb_percent_b": (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9),
                "stoch_k_14": stoch_k,
                "stoch_d_14": stoch_d,
                "adx_14": adx,
            },
        )

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_columns = ["close", "volume", "pct_return", "log_return", "rsi_14"]
        feature_map: dict[str, pd.Series] = {}
        for col in lag_columns:
            if col in df.columns:
                for lag in [1, 2, 3, 5]:
                    feature_map[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return self._concat_feature_block(df, feature_map)

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        if "m_ret" in df.columns:
            market_ret = pd.to_numeric(df["m_ret"], errors="coerce").fillna(0.0)
            market_return_20d = rolling_compound_return(market_ret, 20)
            market_return_60d = rolling_compound_return(market_ret, 60)
            feature_map["m_ret"] = market_ret
            feature_map["market_return_20d"] = market_return_20d
            feature_map["market_return_60d"] = market_return_60d
            if "close_return_20d" in df.columns:
                feature_map["relative_strength_market_20"] = df["close_return_20d"] - market_return_20d
            if "roc_60" in df.columns:
                feature_map["relative_strength_market_60"] = df["roc_60"] - market_return_60d
            cov_20 = df["pct_return"].rolling(20).cov(market_ret)
            var_20 = market_ret.rolling(20).var()
            cov_60 = df["pct_return"].rolling(60).cov(market_ret)
            var_60 = market_ret.rolling(60).var()
            feature_map["rolling_beta_market_20"] = cov_20 / (var_20 + 1e-9)
            feature_map["rolling_beta_market_60"] = cov_60 / (var_60 + 1e-9)
            feature_map["rolling_corr_market_20"] = df["pct_return"].rolling(20).corr(market_ret)
            feature_map["rolling_corr_market_60"] = df["pct_return"].rolling(60).corr(market_ret)

        if "s_ret" in df.columns:
            sector_ret = pd.to_numeric(df["s_ret"], errors="coerce").fillna(0.0)
            sector_return_20d = rolling_compound_return(sector_ret, 20)
            sector_return_60d = rolling_compound_return(sector_ret, 60)
            feature_map["s_ret"] = sector_ret
            feature_map["sector_return_20d"] = sector_return_20d
            feature_map["sector_return_60d"] = sector_return_60d
            if "close_return_20d" in df.columns:
                feature_map["relative_strength_sector_20"] = df["close_return_20d"] - sector_return_20d
            if "roc_60" in df.columns:
                feature_map["relative_strength_sector_60"] = df["roc_60"] - sector_return_60d
            cov_20 = df["pct_return"].rolling(20).cov(sector_ret)
            var_20 = sector_ret.rolling(20).var()
            cov_60 = df["pct_return"].rolling(60).cov(sector_ret)
            var_60 = sector_ret.rolling(60).var()
            feature_map["rolling_beta_sector_20"] = cov_20 / (var_20 + 1e-9)
            feature_map["rolling_beta_sector_60"] = cov_60 / (var_60 + 1e-9)
            feature_map["rolling_corr_sector_20"] = df["pct_return"].rolling(20).corr(sector_ret)
            feature_map["rolling_corr_sector_60"] = df["pct_return"].rolling(60).corr(sector_ret)
            if "rel_to_sector" in df.columns:
                feature_map["sector_relative_zscore_20"] = rolling_zscore(df["rel_to_sector"], 20)
        if "sector_dispersion" in df.columns:
            sector_dispersion = pd.to_numeric(df["sector_dispersion"], errors="coerce")
            feature_map["sector_dispersion"] = sector_dispersion
            feature_map["sector_dispersion_zscore_20"] = rolling_zscore(sector_dispersion, 20)

        if "market_breadth" in df.columns:
            breadth_base = pd.to_numeric(
                df["advancing_share"] if "advancing_share" in df.columns else df["market_breadth"],
                errors="coerce",
            )
            feature_map["breadth_thrust_5"] = breadth_base.rolling(5).mean()
            feature_map["breadth_thrust_10"] = breadth_base.rolling(10).mean()
            feature_map["breadth_momentum_5"] = breadth_base - breadth_base.shift(5)
        if "advance_decline_ratio" in df.columns:
            feature_map["advance_decline_ratio_5"] = pd.to_numeric(df["advance_decline_ratio"], errors="coerce").rolling(5).mean()
        if {"new_highs_252", "new_lows_252"} <= set(df.columns):
            feature_map["new_high_low_ratio"] = pd.to_numeric(df["new_highs_252"], errors="coerce") / (
                pd.to_numeric(df["new_lows_252"], errors="coerce") + 1e-9
            )
        if "new_high_low_spread" in df.columns:
            feature_map["new_high_low_spread_5"] = pd.to_numeric(df["new_high_low_spread"], errors="coerce").rolling(5).mean()
        if {"up_volume", "down_volume"} <= set(df.columns):
            up_volume = pd.to_numeric(df["up_volume"], errors="coerce")
            down_volume = pd.to_numeric(df["down_volume"], errors="coerce")
            feature_map["up_down_volume_ratio_5"] = (up_volume / (down_volume + 1e-9)).rolling(5).mean()
            feature_map["up_down_volume_spread"] = (up_volume - down_volume) / (up_volume + down_volume + 1e-9)

        if {"foreign_net_volume", "volume"} <= set(df.columns):
            feature_map["foreign_net_volume_ratio"] = pd.to_numeric(df["foreign_net_volume"], errors="coerce") / (
                df["volume"] + 1e-9
            )
        if {"foreign_net_value", "turnover"} <= set(df.columns):
            foreign_net_value_ratio = pd.to_numeric(df["foreign_net_value"], errors="coerce") / (
                df["turnover"] + 1e-9
            )
            feature_map["foreign_net_value_ratio"] = foreign_net_value_ratio
            feature_map["foreign_participation_20"] = foreign_net_value_ratio.abs().rolling(20).mean()
            feature_map["foreign_flow_intensity_zscore_20"] = rolling_zscore(foreign_net_value_ratio.abs(), 20)
            feature_map["abnormal_foreign_participation_20"] = feature_map["foreign_flow_intensity_zscore_20"]
        for raw_flow_col in ["foreign_net_volume", "foreign_net_value"]:
            if raw_flow_col in df.columns:
                numeric_flow = pd.to_numeric(df[raw_flow_col], errors="coerce")
                feature_map[f"{raw_flow_col}_1d"] = numeric_flow
                for window in [3, 5, 10]:
                    feature_map[f"{raw_flow_col}_{window}d"] = numeric_flow.rolling(window).sum()
        if {"foreign_net_value", "turnover"} <= set(df.columns):
            rolling_turnover_5 = pd.to_numeric(df["turnover"], errors="coerce").rolling(5).sum()
            foreign_net_value_5d = feature_map.get("foreign_net_value_5d")
            if foreign_net_value_5d is None and "foreign_net_value_5d" in df.columns:
                foreign_net_value_5d = pd.to_numeric(df["foreign_net_value_5d"], errors="coerce")
            if foreign_net_value_5d is not None:
                feature_map["foreign_net_value_5d_ratio"] = pd.to_numeric(foreign_net_value_5d, errors="coerce") / (
                    rolling_turnover_5 + 1e-9
                )
        if "close_return_5d" in df.columns and ("foreign_net_value_5d" in feature_map or "foreign_net_value_5d" in df.columns):
            foreign_net_value_5d = feature_map.get("foreign_net_value_5d")
            if foreign_net_value_5d is None:
                foreign_net_value_5d = pd.to_numeric(df["foreign_net_value_5d"], errors="coerce")
            feature_map["foreign_flow_divergence_5d_flag"] = (
                np.sign(pd.to_numeric(df["close_return_5d"], errors="coerce"))
                != np.sign(pd.to_numeric(foreign_net_value_5d, errors="coerce"))
            ).astype(float)

        for raw_macro_col in ["fx_usdvnd", "interest_rate", "gold_price", "oil_price"]:
            if raw_macro_col in df.columns:
                numeric = pd.to_numeric(df[raw_macro_col], errors="coerce")
                feature_map[f"{raw_macro_col}_ret_1d"] = numeric.pct_change()
                feature_map[f"{raw_macro_col}_ret_5d"] = numeric.pct_change(5)
                feature_map[f"{raw_macro_col}_zscore_20"] = rolling_zscore(numeric, 20)
                delta = numeric.diff() if raw_macro_col == "interest_rate" else numeric.pct_change()
                feature_map[f"{raw_macro_col}_delta_1d"] = delta
                feature_map[f"{raw_macro_col}_delta_5d"] = (
                    numeric.diff(5) if raw_macro_col == "interest_rate" else numeric.pct_change(5)
                )
                feature_map[f"{raw_macro_col}_shock_20"] = rolling_zscore(delta, 20)

        macro_risk_components = [
            column
            for column in ["fx_usdvnd_shock_20", "interest_rate_shock_20", "gold_price_shock_20"]
            if column in feature_map or column in df.columns
        ]
        macro_shock_components = [
            column
            for column in ["fx_usdvnd_shock_20", "interest_rate_shock_20", "gold_price_shock_20", "oil_price_shock_20"]
            if column in feature_map or column in df.columns
        ]
        if macro_risk_components:
            macro_risk_series = []
            for column in macro_risk_components:
                source = feature_map.get(column)
                if source is None and column in df.columns:
                    source = df[column]
                if source is not None:
                    macro_risk_series.append(pd.to_numeric(source, errors="coerce"))
            feature_map["macro_risk_off_score"] = pd.concat(
                macro_risk_series,
                axis=1,
            ).mean(axis=1)
        if macro_shock_components:
            macro_shock_series = []
            for column in macro_shock_components:
                source = feature_map.get(column)
                if source is None and column in df.columns:
                    source = df[column]
                if source is not None:
                    macro_shock_series.append(pd.to_numeric(source, errors="coerce").abs())
            feature_map["macro_shock_index"] = pd.concat(
                macro_shock_series,
                axis=1,
            ).mean(axis=1)

        return self._concat_feature_block(df, feature_map)

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        bull_mask = (
            (df["close"] > df["sma_20"])
            & (df["sma_20"] > df["sma_50"])
            & (df["close_return_20d"] > 0)
        )
        bear_mask = (
            (df["close"] < df["sma_20"])
            & (df["sma_20"] < df["sma_50"])
            & (df["close_return_20d"] < 0)
        )
        market_regime = np.select(
            [bull_mask, bear_mask],
            ["bull", "bear"],
            default="sideways",
        )
        market_regime = pd.Series(market_regime, index=df.index)
        market_regime_code = market_regime.map(TREND_REGIME_TO_CODE).astype(float)
        trend_regime_strength = (df["sma_20"] - df["sma_50"]) / (df["atr_14"] + 1e-9)
        trend_up = (trend_regime_strength > TREND_REGIME_STRENGTH_THRESHOLD) & (df["close"] > df["sma_200"])
        trend_down = (trend_regime_strength < -TREND_REGIME_STRENGTH_THRESHOLD) & (df["close"] < df["sma_200"])
        trend_regime = np.select(
            [trend_up, trend_down],
            ["uptrend", "downtrend"],
            default="neutral",
        )
        trend_regime = pd.Series(trend_regime, index=df.index)
        trend_regime_code = trend_regime.map(TREND_STATE_TO_CODE).astype(float)

        vol_ref = df["rolling_volatility_20"].rolling(60, min_periods=20).median()
        high_vol = df["rolling_volatility_20"] > (vol_ref * 1.25)
        low_vol = df["rolling_volatility_20"] < (vol_ref * 0.75)
        volatility_regime = np.select(
            [high_vol, low_vol],
            ["high", "low"],
            default="normal",
        )
        volatility_regime = pd.Series(volatility_regime, index=df.index)
        volatility_regime_code = volatility_regime.map(VOLATILITY_REGIME_TO_CODE).astype(float)
        feature_map: dict[str, Any] = {
            "market_regime": market_regime,
            "market_regime_code": market_regime_code,
            "trend_regime_strength": trend_regime_strength,
            "trend_regime": trend_regime,
            "trend_regime_code": trend_regime_code,
            "volatility_regime": volatility_regime,
            "volatility_regime_code": volatility_regime_code,
        }
        for label_column, code_column in [
            ("market_regime", "market_regime_code"),
            ("volatility_regime", "volatility_regime_code"),
            ("trend_regime", "trend_regime_code"),
        ]:
            regime_code = pd.to_numeric(feature_map[code_column], errors="coerce")
            change_group = regime_code.ne(regime_code.shift(1)).cumsum()
            persistence = regime_code.groupby(change_group).cumcount() + 1
            transition = regime_code.ne(regime_code.shift(1)).astype(float)
            if not transition.empty:
                transition.iloc[0] = 0.0
            feature_map[f"{label_column}_persistence_days"] = persistence.astype(float)
            feature_map[f"{label_column}_transition_flag"] = transition
        return self._concat_feature_block(df, feature_map)

    def _add_corporate_action_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        trailing_gap_abs = pd.to_numeric(df["overnight_return_1d"], errors="coerce").abs()
        gap_baseline = trailing_gap_abs.rolling(20, min_periods=5).median().shift(1)
        abnormal_gap_sigma_20 = trailing_gap_abs / (gap_baseline + 1e-9)
        gap_threshold = np.maximum(ABNORMAL_GAP_ABS_MIN, gap_baseline * ABNORMAL_GAP_MULTIPLIER)
        abnormal_gap_flag = (trailing_gap_abs > gap_threshold).astype(float)

        full_day_move = pd.to_numeric(df["close_to_close_return_1d"], errors="coerce").abs()
        intraday_retrace_ratio = pd.to_numeric(df["open_to_close_return_1d"], errors="coerce").abs() / (
            trailing_gap_abs + 1e-9
        )
        potential_price_event_flag = (
            (abnormal_gap_flag > 0)
            & (full_day_move > CORPORATE_ACTION_MOVE_MIN)
        ).astype(float)
        volume_confirmation = (
            pd.to_numeric(df["volume_spike_zscore_20"], errors="coerce") > CORPORATE_ACTION_VOLUME_ZSCORE_MIN
            if "volume_spike_zscore_20" in df.columns
            else False
        )
        potential_corporate_action_flag = (
            (potential_price_event_flag > 0)
            & (
                (intraday_retrace_ratio <= CORPORATE_ACTION_INTRADAY_RETRACE_MAX)
                | volume_confirmation
            )
        ).astype(float)
        return self._concat_feature_block(
            df,
            {
                "abnormal_gap_sigma_20": abnormal_gap_sigma_20,
                "abnormal_gap_flag": abnormal_gap_flag,
                "potential_price_event_flag": potential_price_event_flag,
                "potential_corporate_action_flag": potential_corporate_action_flag,
                "recent_price_event_risk_5d": potential_price_event_flag.rolling(5, min_periods=1).max(),
                "recent_corporate_action_risk_20d": potential_corporate_action_flag.rolling(20, min_periods=1).max(),
            },
        )

    @staticmethod
    def regime_definition_manifest() -> dict[str, Any]:
        return {
            "market_regime": {
                "type": "rule_based",
                "labels": ["bear", "sideways", "bull"],
                "logic": "bull when close > sma_20, sma_20 > sma_50, and close_return_20d > 0; bear when the inverse holds; otherwise sideways.",
            },
            "volatility_regime": {
                "type": "rule_based",
                "labels": ["low", "normal", "high"],
                "logic": "high when rolling_volatility_20 > 1.25 * trailing 60-day median; low when < 0.75 * trailing 60-day median; otherwise normal.",
            },
            "trend_regime": {
                "type": "rule_based",
                "labels": ["downtrend", "neutral", "uptrend"],
                "logic": (
                    "uptrend when trend_regime_strength > 0.5 and close > sma_200; "
                    "downtrend when trend_regime_strength < -0.5 and close < sma_200; otherwise neutral."
                ),
            },
            "persistence": "Current consecutive-day streak length within the same regime label.",
            "transition": "1 when the current regime label differs from the previous trading day, else 0.",
        }

    @staticmethod
    def corporate_action_diagnostic_manifest() -> dict[str, Any]:
        return {
            "adjusted_close_available_from_live_vnstock_data": False,
            "diagnostic_only": True,
            "abnormal_gap_logic": (
                "Flag when |overnight_return_1d| exceeds max(8%, 3x the prior 20-day median absolute overnight gap)."
            ),
            "potential_price_event_logic": "Abnormal gap plus |close_to_close_return_1d| > 12%.",
            "potential_corporate_action_logic": (
                "Potential price event plus either limited intraday retrace "
                "(|open_to_close_return_1d| / |overnight_return_1d| <= 0.35) or volume_spike_zscore_20 > 2.0."
            ),
            "risk_marker_logic": "Trailing max over 5-day and 20-day lookback windows; no future propagation.",
            "note": "These flags are diagnostics for suspicious discontinuities, not confirmed corporate-action labels.",
        }

    def _add_vn100_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        close_to_close_return_1d = df["close"].pct_change()
        true_range = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ),
        )
        turnover = df["close"] * df["volume"]
        feature_map: dict[str, pd.Series] = {
            "prev_close": prev_close,
            "close_to_close_return_1d": close_to_close_return_1d,
            "close_return_1d": close_to_close_return_1d,
            "close_return_2d": df["close"].pct_change(2),
            "close_return_3d": df["close"].pct_change(3),
            "close_return_5d": df["close"].pct_change(5),
            "close_return_10d": df["close"].pct_change(10),
            "close_return_20d": df["close"].pct_change(20),
            "open_to_close_return_1d": (df["close"] - df["open"]) / (df["open"] + 1e-9),
            "open_close_spread": df["open"] - df["close"],
            "open_close_spread_pct": (df["open"] - df["close"]) / (df["close"] + 1e-9),
            "overnight_return_1d": (df["open"] - prev_close) / (prev_close + 1e-9),
            "high_low_range": df["high"] - df["low"],
            "high_low_range_pct": (df["high"] - df["low"]) / (df["low"] + 1e-9),
            "true_range": true_range,
            "atr_14": true_range.rolling(14).mean(),
            "atr_proxy_5": true_range.rolling(5).mean(),
            "atr_proxy_10": true_range.rolling(10).mean(),
            "return_3d": df["close"].pct_change(3),
            "return_5d": df["close"].pct_change(5),
            "return_10d": df["close"].pct_change(10),
            "return_20d": df["close"].pct_change(20),
        }

        for window in [5, 10, 20]:
            volume_ma = df["volume"].rolling(window).mean()
            feature_map[f"close_mean_{window}"] = df["close"].rolling(window).mean()
            feature_map[f"close_std_{window}"] = df["close"].rolling(window).std()
            feature_map[f"volume_ma_{window}"] = volume_ma
            feature_map[f"volume_shock_{window}"] = df["volume"] / (volume_ma + 1e-9)
            feature_map[f"rolling_volatility_{window}"] = df["pct_return"].rolling(window).std()
            feature_map[f"value_ratio_{window}"] = turnover / (turnover.rolling(window).mean() + 1e-9)

        feature_map["rolling_volatility_60"] = df["pct_return"].rolling(60).std()
        return self._concat_feature_block(df, feature_map)

    def _add_legacy_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        if "rolling_volatility_20" in df.columns:
            feature_map["hv_20"] = df["rolling_volatility_20"] * np.sqrt(252)
        else:
            feature_map["hv_20"] = df["pct_return"].rolling(20).std() * np.sqrt(252)

        for window in WINDOWS:
            rolling_mean = df["close"].rolling(window).mean()
            rolling_std = df["close"].rolling(window).std()
            feature_map[f"bb_bandwidth_{window}"] = (4.0 * rolling_std) / (rolling_mean.abs() + 1e-9)

        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0
        feature_map["pivot"] = pivot
        feature_map["resistance_1"] = (2.0 * pivot) - prev_low
        feature_map["support_1"] = (2.0 * pivot) - prev_high
        return self._concat_feature_block(df, feature_map)

    def get_registry_candidate_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticker",
            "time",
        }
        exclude |= PRICE_REFERENCE_COLUMNS
        exclude |= NON_FEATURE_SUPPORT_COLUMNS
        cols = [c for c in df.columns if c not in exclude and not str(c).startswith("target_")]
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = self.get_registry_candidate_columns(df)
        filtered: list[str] = []
        for col in numeric_cols:
            if col in LEGACY_COMPATIBILITY_COLUMNS:
                continue
            if col in NON_CANONICAL_ALIAS_COLUMNS:
                continue
            if col.startswith("d_") and col[2:] in NON_CANONICAL_ALIAS_COLUMNS:
                continue
            filtered.append(col)
        return filtered

    def build_feature_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for col in self.get_registry_candidate_columns(df):
            rows.append(
                {
                    "feature_name": col,
                    "category": self._feature_category(col),
                    "input_source": self._feature_input_source(col),
                    "exact_provenance": self._feature_provenance(col),
                    "formula_logic": self._feature_formula(col),
                    "expected_availability": self._feature_expected_availability(col),
                    "leakage_risk_note": self._feature_leakage_note(col),
                    "usable_for_forecast": self._feature_use_case(col)[0],
                    "usable_for_regime": self._feature_use_case(col)[1],
                    "usable_for_risk": self._feature_use_case(col)[2],
                    "status": self._feature_status(col),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _feature_category(col: str) -> str:
        if col in {"advancers", "decliners", "unchanged", "net_advancers"}:
            return "market_context"
        if col.startswith(("m_", "s_", "rel_to_", "market_", "sector_", "rolling_beta_", "rolling_corr_", "breadth_", "advance_decline")):
            return "market_context"
        if col.startswith(("pct_above_ma", "new_high", "new_low", "up_volume", "down_volume", "up_down_volume", "advancing_share", "declining_share")):
            return "market_context"
        if col.startswith(("abnormal_gap", "potential_price_event", "potential_corporate_action", "recent_price_event", "recent_corporate_action")):
            return "corporate_action_diagnostics"
        if col.startswith(("fx_", "interest_", "gold_", "oil_", "macro_")):
            return "macro_cross_asset"
        if col.startswith(("foreign_", "abnormal_foreign")) or col.startswith(("turnover", "amihud", "volume_spike")):
            return "flow_microstructure"
        if "regime" in col:
            return "regime_state"
        if "sentiment" in col or "news_" in col:
            return "sentiment_news"
        if any(token in col for token in ("rsi", "macd", "bb_", "atr", "roc", "stoch", "adx", "sma_", "ema_", "pivot", "support", "resistance")):
            return "technical_indicator"
        return "price_volume_core"

    @staticmethod
    def _feature_input_source(col: str) -> str:
        if FeatureEngineer._feature_category(col) == "corporate_action_diagnostics":
            return "ohlcv"
        if FeatureEngineer._feature_category(col) == "macro_cross_asset":
            return "macro/commodity context"
        if FeatureEngineer._feature_category(col) == "market_context":
            return "market/sector/breadth context"
        if FeatureEngineer._feature_category(col) == "sentiment_news":
            return "sentiment/news context"
        return "ohlcv"

    @staticmethod
    def _feature_provenance(col: str) -> str:
        category = FeatureEngineer._feature_category(col)
        if category == "sentiment_news":
            return "local computation"
        if category in {"market_context", "macro_cross_asset", "flow_microstructure"}:
            return "derived from vnstock_data"
        if category == "corporate_action_diagnostics":
            return "derived from vnstock_data"
        return "derived from vnstock_data"

    @staticmethod
    def _feature_formula(col: str) -> str:
        if col.startswith("close_lag_") or col.startswith("volume_lag_") or col.startswith("pct_return_lag_"):
            return "lagged value"
        if col.startswith("rolling_min_"):
            return "rolling minimum of close"
        if col.startswith("rolling_max_"):
            return "rolling maximum of close"
        if col.startswith("volume_std_"):
            return "rolling std of volume"
        if col.startswith("rolling_beta_market_"):
            return "rolling covariance(asset, market) / variance(market)"
        if col.startswith("rolling_corr_"):
            return "rolling correlation"
        if col == "abnormal_gap_flag":
            return "1 when overnight gap exceeds trailing abnormal-gap threshold"
        if col == "potential_corporate_action_flag":
            return "1 when a suspicious discontinuity resembles a split/dividend/event pattern"
        if col.endswith("_persistence_days"):
            return "consecutive-day streak length for the current regime label"
        if col.endswith("_transition_flag"):
            return "1 when the current regime label differs from the previous day"
        if col == "market_regime_code":
            return "bull=1, sideways=0, bear=-1 from SMA/trailing-return rules"
        if col == "volatility_regime_code":
            return "low=0, normal=1, high=2 from rolling volatility vs trailing median"
        if col == "trend_regime_code":
            return "uptrend=1, neutral=0, downtrend=-1 from ATR-normalized MA spread plus SMA-200 filter"
        if col.startswith("relative_strength_"):
            return "asset trailing return minus benchmark trailing return"
        if col == "sector_relative_zscore_20":
            return "20-day z-score of return relative to sector proxy"
        if col.startswith("turnover_ratio_"):
            return "turnover / rolling mean turnover"
        if col == "amihud_20":
            return "rolling mean of |return| / traded value"
        if col == "foreign_flow_intensity_zscore_20":
            return "20-day z-score of absolute foreign net value ratio"
        if col == "macro_risk_off_score":
            return "mean of standardized FX, interest-rate, and gold shocks"
        if col == "macro_shock_index":
            return "mean absolute standardized macro shock across available series"
        if col == "stoch_k_14":
            return "smoothed stochastic %K"
        if col == "stoch_d_14":
            return "smoothed stochastic %D"
        if col == "adx_14":
            return "14-day average directional index"
        return "deterministic rolling/lagged transformation"

    @staticmethod
    def _feature_expected_availability(col: str) -> str:
        category = FeatureEngineer._feature_category(col)
        if category == "macro_cross_asset":
            return "requires macro_context artifact or live vnstock_data macro/commodity fetch"
        if category == "corporate_action_diagnostics":
            return "available from OHLCV when adjusted prices are unavailable and diagnostics are needed"
        if category == "sentiment_news":
            return "requires sentiment artifact; otherwise absent by design"
        if category == "market_context":
            return "requires market/sector/breadth context joins"
        if col.startswith(("foreign_", "abnormal_foreign")):
            return "requires foreign_flow artifact or live vnstock_data trading fetch"
        if col in {"sma_200", "close_to_sma_200"} or col.endswith("_60"):
            return "available with extended OHLCV history after warmup"
        return "available from OHLCV after standard warmup"

    @staticmethod
    def _feature_leakage_note(col: str) -> str:
        if "regime" in col:
            return "Rule-based label uses trailing moving averages and trailing volatility only."
        if FeatureEngineer._feature_category(col) == "corporate_action_diagnostics":
            return "Diagnostic flags use current-day OHLCV and prior rolling baselines only."
        if col.startswith(("fx_", "interest_", "gold_", "oil_")):
            return "Requires backward-looking date alignment from released macro series."
        if col.startswith(("m_", "s_", "rel_to_", "market_", "sector_", "rolling_beta_", "rolling_corr_", "breadth_", "advance_decline", "foreign_")):
            return "Safe only if context inputs are aligned on date and never forward-filled from the future."
        return "Computed from current and past rows only."

    @staticmethod
    def _feature_use_case(col: str) -> tuple[bool, bool, bool]:
        category = FeatureEngineer._feature_category(col)
        if category == "regime_state":
            return True, True, True
        if category == "market_context":
            return True, True, True
        if category == "corporate_action_diagnostics":
            return True, False, True
        if category == "flow_microstructure":
            return True, False, True
        if category == "macro_cross_asset":
            return True, True, True
        if category == "sentiment_news":
            return True, False, False
        return True, False, True

    @staticmethod
    def _feature_status(col: str) -> str:
        if col in LEGACY_COMPATIBILITY_COLUMNS or col in NON_CANONICAL_ALIAS_COLUMNS:
            return "deprecated"
        if FeatureEngineer._feature_category(col) == "sentiment_news":
            return "experimental"
        if col.startswith("d_") or col in {
            "close_kalman",
            "park_vol_20",
            "rs_vol_20",
            "fx_usdvnd",
            "interest_rate",
            "gold_price",
            "oil_price",
            "sentiment_std",
        }:
            return "experimental"
        return "active"
