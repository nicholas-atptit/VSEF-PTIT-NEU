"""Leakage-safe Track A regime and feature v2 builder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, build_feature_set_c  # noqa: E402

FEATURE_SET_NAME = "regime_feature_v2"
SECTOR_PATH = REPO_ROOT / "data" / "ticker_sectors.csv"


def _volatility_ratio_regime(short_vol: pd.Series, long_vol: pd.Series) -> pd.Series:
    ratio = short_vol / long_vol.replace(0.0, np.nan)
    regime = pd.Series(1.0, index=ratio.index)
    regime.loc[ratio < 0.75] = 0.0
    regime.loc[ratio > 1.25] = 2.0
    regime.loc[ratio.isna()] = np.nan
    return regime


def load_sector_map() -> dict[str, int]:
    if not SECTOR_PATH.exists():
        return {}
    sectors = pd.read_csv(SECTOR_PATH, low_memory=False)
    if "ticker" not in sectors.columns or "industry_code" not in sectors.columns:
        return {}
    sectors["ticker"] = sectors["ticker"].astype(str).str.upper()
    sectors["industry_code"] = pd.to_numeric(sectors["industry_code"], errors="coerce")
    sectors = sectors.dropna(subset=["ticker", "industry_code"]).drop_duplicates("ticker", keep="first")
    return {str(row["ticker"]): int(row["industry_code"]) for _, row in sectors.iterrows()}


def _index_regime_features(index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    feature_frames: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    for code in ["VNINDEX", "VN30"]:
        if code not in index_data:
            continue
        frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        prefix = code.lower()
        close = pd.to_numeric(frame["close"], errors="coerce")
        ret = close.pct_change(periods=1, fill_method=None)
        frame[f"{prefix}_ret_lag_1_v2"] = ret.shift(1)
        frame[f"{prefix}_ret_lag_2_v2"] = ret.shift(2)
        frame[f"{prefix}_trend_20_lag_v2"] = (close / close.shift(20) - 1.0).shift(1)
        frame[f"{prefix}_trend_60_lag_v2"] = (close / close.shift(60) - 1.0).shift(1)
        frame[f"{prefix}_vol_20_lag_v2"] = ret.rolling(20, min_periods=5).std().shift(1)
        frame[f"{prefix}_vol_60_lag_v2"] = ret.rolling(60, min_periods=15).std().shift(1)
        frame[f"{prefix}_vol_regime_v2"] = _volatility_ratio_regime(frame[f"{prefix}_vol_20_lag_v2"], frame[f"{prefix}_vol_60_lag_v2"])
        trend = frame[f"{prefix}_trend_60_lag_v2"]
        frame[f"{prefix}_bull_proxy_v2"] = (trend > 0.02).astype(float)
        frame[f"{prefix}_bear_proxy_v2"] = (trend < -0.02).astype(float)
        frame[f"{prefix}_sideway_proxy_v2"] = ((trend >= -0.02) & (trend <= 0.02)).astype(float)
        cols = [col for col in frame.columns if col not in {"datetime", "close"}]
        feature_cols.extend(cols)
        feature_frames.append(frame[["datetime", *cols]])
    if not feature_frames:
        return pd.DataFrame(columns=["datetime"]), []
    merged = feature_frames[0]
    for frame in feature_frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer")
    return merged.sort_values("datetime"), feature_cols


def build_regime_feature_v2(stock_df: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    base, base_cols = build_feature_set_c(stock_df, index_data)
    prepared = base.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feature_cols = list(base_cols)
    index_features, index_cols = _index_regime_features(index_data)
    if not index_features.empty:
        prepared = prepared.merge(index_features.drop_duplicates("datetime", keep="last"), on="datetime", how="left")
        feature_cols.extend(index_cols)

    sector_map = load_sector_map()
    prepared["ticker_code_v2"] = pd.Categorical(prepared["ticker"]).codes.astype(float)
    feature_cols.append("ticker_code_v2")
    if sector_map:
        prepared["sector_code_v2"] = prepared["ticker"].astype(str).str.upper().map(sector_map).astype(float)
        feature_cols.append("sector_code_v2")

    for _ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        stock_ret = close.pct_change(periods=1, fill_method=None)
        range_ratio = (group["high"].astype(float) - group["low"].astype(float)) / close.replace(0.0, np.nan)
        prepared.loc[idx, "stock_ret_lag_1_v2"] = stock_ret.shift(1)
        prepared.loc[idx, "stock_ret_lag_2_v2"] = stock_ret.shift(2)
        prepared.loc[idx, "stock_roll_vol_20_lag_v2"] = stock_ret.rolling(20, min_periods=5).std().shift(1)
        prepared.loc[idx, "stock_roll_vol_60_lag_v2"] = stock_ret.rolling(60, min_periods=15).std().shift(1)
        prepared.loc[idx, "stock_vol_regime_v2"] = _volatility_ratio_regime(prepared.loc[idx, "stock_roll_vol_20_lag_v2"], prepared.loc[idx, "stock_roll_vol_60_lag_v2"])
        prepared.loc[idx, "volume_shock_60_lag_v2"] = (volume / volume.rolling(60, min_periods=10).mean() - 1.0).shift(1)
        prepared.loc[idx, "range_shock_20_lag_v2"] = (range_ratio / range_ratio.rolling(20, min_periods=5).mean() - 1.0).shift(1)
        prepared.loc[idx, "range_vol_20_lag_v2"] = range_ratio.rolling(20, min_periods=5).std().shift(1)
        if "vnindex_ret_lag_1_v2" in prepared.columns:
            prepared.loc[idx, "stock_minus_vnindex_lag_1_v2"] = prepared.loc[idx, "stock_ret_lag_1_v2"] - prepared.loc[idx, "vnindex_ret_lag_1_v2"]
        if "vn30_ret_lag_1_v2" in prepared.columns:
            prepared.loc[idx, "stock_minus_vn30_lag_1_v2"] = prepared.loc[idx, "stock_ret_lag_1_v2"] - prepared.loc[idx, "vn30_ret_lag_1_v2"]

    session = prepared["datetime"].dt.hour
    prepared["session_open_v2"] = session.between(9, 10).astype(float)
    prepared["session_midday_v2"] = session.between(11, 13).astype(float)
    prepared["session_close_v2"] = session.between(14, 15).astype(float)
    v2_cols = [
        "stock_ret_lag_1_v2",
        "stock_ret_lag_2_v2",
        "stock_roll_vol_20_lag_v2",
        "stock_roll_vol_60_lag_v2",
        "stock_vol_regime_v2",
        "volume_shock_60_lag_v2",
        "range_shock_20_lag_v2",
        "range_vol_20_lag_v2",
        "stock_minus_vnindex_lag_1_v2",
        "stock_minus_vn30_lag_1_v2",
        "session_open_v2",
        "session_midday_v2",
        "session_close_v2",
    ]
    feature_cols.extend([col for col in v2_cols if col in prepared.columns])
    feature_cols = sorted({col for col in feature_cols if col in prepared.columns})
    prepared[feature_cols] = prepared[feature_cols].replace([np.inf, -np.inf], np.nan)
    regime_col = "vnindex_bull_proxy_v2"
    if regime_col in prepared.columns:
        prepared["market_regime_v2"] = "sideway"
        prepared.loc[prepared["vnindex_bull_proxy_v2"] == 1.0, "market_regime_v2"] = "bull"
        prepared.loc[prepared["vnindex_bear_proxy_v2"] == 1.0, "market_regime_v2"] = "bear"
    else:
        prepared["market_regime_v2"] = "unknown"
    manifest = {
        "feature_set": FEATURE_SET_NAME,
        "feature_count": len(feature_cols),
        "sector_encoding_used": bool(sector_map),
        "leakage_safe": True,
        "future_return_features": False,
        "future_regime_features": False,
        "final_label_derived_features": False,
        "final_period_manual_filters": False,
    }
    return prepared, feature_cols, manifest
