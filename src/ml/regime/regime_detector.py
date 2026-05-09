"""Rule-based market regime detection for diagnostic research.

The Phase 4 detector labels ticker-level historical windows from rolling
returns and realized volatility. All rolling features use only current and
past observations. The legacy risk-regime API remains available for older
tests and modules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd


REGIME_TO_CODE = {"NORMAL": 0, "HIGH_VOL": 1, "CRISIS": 2}

REGIME_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "close",
    "daily_return",
    "rolling_return",
    "realized_volatility",
    "trend_regime",
    "volatility_regime",
    "combined_regime",
    "regime_window",
    "regime_policy_id",
    "diagnostics",
]

DEFAULT_PHASE4_CONFIG: dict[str, Any] = {
    "policy": {
        "id": "regime_policy_v1",
        "diagnostic_only": True,
    },
    "windows": {
        "return_window": 20,
        "volatility_window": 20,
        "min_periods": 10,
    },
    "trend_regime": {
        "method": "rolling_return_threshold",
        "bull_threshold": 0.03,
        "bear_threshold": -0.03,
        "sideway_band": {
            "lower": -0.03,
            "upper": 0.03,
        },
    },
    "volatility_regime": {
        "method": "quantile_threshold",
        "high_vol_quantile": 0.70,
        "low_vol_quantile": 0.30,
        "fallback_method": "median_split",
    },
    "combined_regime": {
        "enabled": True,
        "format": "{trend_regime}_{volatility_regime}",
    },
    "governance": {
        "no_future_data": True,
        "diagnostic_only": True,
    },
}


@dataclass(frozen=True)
class RegimeDetectionResult:
    labels: pd.Series
    encoded_labels: pd.Series
    probabilities: pd.DataFrame


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    return str(value)


class RegimeDetector:
    """Transparent rule-based detector for Phase 4 regime labels.

    Parameters
    ----------
    config:
        Phase 4 regime policy mapping. If omitted, documented defaults are
        used. For backward compatibility, legacy keyword thresholds for the
        older NORMAL/HIGH_VOL/CRISIS detector are also accepted.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        method: str = "threshold",
        high_vol_threshold: float = 0.03,
        crisis_drawdown_threshold: float = -0.12,
        crisis_delta_covar_threshold: float = 0.015,
    ) -> None:
        self.config = self._normalize_config(config)
        self.method = method
        self.high_vol_threshold = high_vol_threshold
        self.crisis_drawdown_threshold = crisis_drawdown_threshold
        self.crisis_delta_covar_threshold = crisis_delta_covar_threshold

    @staticmethod
    def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
        merged = deepcopy(DEFAULT_PHASE4_CONFIG)
        if config:
            _deep_update(merged, deepcopy(config))
            windows = merged.setdefault("windows", {})
            if "return_window" in config:
                windows["return_window"] = config["return_window"]
            if "volatility_window" in config:
                windows["volatility_window"] = config["volatility_window"]
            if "min_periods" in config:
                windows["min_periods"] = config["min_periods"]
        return merged

    @property
    def policy_id(self) -> str:
        return str(self.config.get("policy", {}).get("id") or "regime_policy_v1")

    @property
    def return_window(self) -> int:
        return int(self.config.get("windows", {}).get("return_window") or 20)

    @property
    def volatility_window(self) -> int:
        return int(self.config.get("windows", {}).get("volatility_window") or 20)

    @property
    def min_periods(self) -> int:
        configured = int(self.config.get("windows", {}).get("min_periods") or 10)
        return max(1, configured)

    def compute_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ticker-level daily returns without using future rows."""
        frame = self._normalize_ohlcv(data)
        frame["daily_return"] = frame.groupby("ticker", sort=False)["close"].pct_change()
        return frame

    def compute_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling return and realized volatility using past/current rows."""
        frame = data.copy()
        if "daily_return" not in frame.columns:
            frame = self.compute_returns(frame)
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce")

        log_returns = np.log1p(frame["daily_return"].where(frame["daily_return"] > -1.0))
        frame["_log_return"] = log_returns
        grouped = frame.groupby("ticker", sort=False)

        rolling_log_return = grouped["_log_return"].transform(
            lambda values: values.rolling(self.return_window, min_periods=self.min_periods).sum()
        )
        frame["rolling_return"] = np.expm1(rolling_log_return)
        frame["realized_volatility"] = grouped["daily_return"].transform(
            lambda values: values.rolling(self.volatility_window, min_periods=self.min_periods).std(ddof=0)
        )
        frame["_return_count"] = grouped["daily_return"].transform(
            lambda values: values.rolling(self.return_window, min_periods=1).count()
        )
        frame["_volatility_count"] = grouped["daily_return"].transform(
            lambda values: values.rolling(self.volatility_window, min_periods=1).count()
        )
        frame["_sufficient_history"] = (
            (frame["_return_count"] >= self.min_periods)
            & (frame["_volatility_count"] >= self.min_periods)
            & frame["rolling_return"].notna()
            & frame["realized_volatility"].notna()
        )
        return frame

    def assign_trend_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign bull, bear, sideway, or insufficient_history."""
        frame = data.copy()
        trend_cfg = self.config.get("trend_regime", {}) or {}
        bull_threshold = float(trend_cfg.get("bull_threshold", 0.03))
        bear_threshold = float(trend_cfg.get("bear_threshold", -0.03))
        band = trend_cfg.get("sideway_band", {}) or {}
        sideway_lower = float(band.get("lower", bear_threshold))
        sideway_upper = float(band.get("upper", bull_threshold))

        frame["trend_regime"] = "insufficient_history"
        sufficient = frame.get("_sufficient_history", pd.Series(False, index=frame.index)).astype(bool)
        rolling_return = pd.to_numeric(frame["rolling_return"], errors="coerce")
        frame.loc[sufficient & (rolling_return >= bull_threshold), "trend_regime"] = "bull"
        frame.loc[sufficient & (rolling_return <= bear_threshold), "trend_regime"] = "bear"
        sideway = sufficient & (rolling_return > sideway_lower) & (rolling_return < sideway_upper)
        frame.loc[sideway, "trend_regime"] = "sideway"
        remaining = sufficient & frame["trend_regime"].eq("insufficient_history")
        frame.loc[remaining, "trend_regime"] = "sideway"
        return frame

    def assign_volatility_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign high_vol, low_vol, or insufficient_history.

        Quantile thresholds are computed as expanding ticker-level thresholds,
        so each row only sees volatility values available up to that date.
        """
        frame = data.copy()
        vol_cfg = self.config.get("volatility_regime", {}) or {}
        high_quantile = float(vol_cfg.get("high_vol_quantile", 0.70))
        low_quantile = float(vol_cfg.get("low_vol_quantile", 0.30))
        fallback = str(vol_cfg.get("fallback_method") or "median_split")

        grouped = frame.groupby("ticker", sort=False)
        frame["_high_vol_threshold"] = grouped["realized_volatility"].transform(
            lambda values: values.expanding(min_periods=self.min_periods).quantile(high_quantile)
        )
        frame["_low_vol_threshold"] = grouped["realized_volatility"].transform(
            lambda values: values.expanding(min_periods=self.min_periods).quantile(low_quantile)
        )
        frame["_median_vol_threshold"] = grouped["realized_volatility"].transform(
            lambda values: values.expanding(min_periods=self.min_periods).median()
        )

        frame["volatility_regime"] = "insufficient_history"
        sufficient = (
            frame.get("_sufficient_history", pd.Series(False, index=frame.index)).astype(bool)
            & frame["realized_volatility"].notna()
            & frame["_high_vol_threshold"].notna()
            & frame["_low_vol_threshold"].notna()
        )
        vol = pd.to_numeric(frame["realized_volatility"], errors="coerce")
        high_mask = sufficient & (vol >= frame["_high_vol_threshold"])
        low_mask = sufficient & (vol <= frame["_low_vol_threshold"])
        frame.loc[high_mask, "volatility_regime"] = "high_vol"
        frame.loc[low_mask, "volatility_regime"] = "low_vol"

        middle = sufficient & frame["volatility_regime"].eq("insufficient_history")
        if fallback == "median_split":
            frame.loc[middle & (vol >= frame["_median_vol_threshold"]), "volatility_regime"] = "high_vol"
            frame.loc[middle & (vol < frame["_median_vol_threshold"]), "volatility_regime"] = "low_vol"
        else:
            frame.loc[middle, "volatility_regime"] = "normal_vol"
        return frame

    def assign_combined_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Combine trend and volatility labels when both are available."""
        frame = data.copy()
        combined_cfg = self.config.get("combined_regime", {}) or {}
        enabled = bool(combined_cfg.get("enabled", True))
        label_format = str(combined_cfg.get("format") or "{trend_regime}_{volatility_regime}")

        frame["combined_regime"] = "insufficient_history"
        valid = (
            frame["trend_regime"].ne("insufficient_history")
            & frame["volatility_regime"].ne("insufficient_history")
        )
        if enabled:
            frame.loc[valid, "combined_regime"] = frame.loc[valid].apply(
                lambda row: label_format.format(
                    trend_regime=row["trend_regime"],
                    volatility_regime=row["volatility_regime"],
                ),
                axis=1,
            )
        else:
            frame.loc[valid, "combined_regime"] = frame.loc[valid, "trend_regime"]
        return frame

    def tag(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the required Phase 4 regime label columns."""
        frame = self.compute_returns(data)
        frame = self.compute_rolling_features(frame)
        frame = self.assign_trend_regime(frame)
        frame = self.assign_volatility_regime(frame)
        frame = self.assign_combined_regime(frame)

        frame["regime_window"] = (
            "return_window="
            + str(self.return_window)
            + ";volatility_window="
            + str(self.volatility_window)
            + ";min_periods="
            + str(self.min_periods)
        )
        frame["regime_policy_id"] = self.policy_id
        frame["diagnostics"] = frame.apply(self._row_diagnostics, axis=1)
        return frame[REGIME_OUTPUT_COLUMNS].reset_index(drop=True)

    def summarize(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Summarize regime label distribution and basic return behavior."""
        if labels.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "trend_regime",
                    "volatility_regime",
                    "combined_regime",
                    "observation_count",
                    "start_date",
                    "end_date",
                    "mean_return",
                    "mean_realized_volatility",
                ]
            )
        frame = labels.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("daily_return", "realized_volatility"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        summary = (
            frame.groupby(["ticker", "trend_regime", "volatility_regime", "combined_regime"], dropna=False)
            .agg(
                observation_count=("date", "size"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                mean_return=("daily_return", "mean"),
                mean_realized_volatility=("realized_volatility", "mean"),
            )
            .reset_index()
        )
        summary["start_date"] = summary["start_date"].dt.date.astype(str)
        summary["end_date"] = summary["end_date"].dt.date.astype(str)
        return summary

    def _row_diagnostics(self, row: pd.Series) -> str:
        insufficient = (
            row.get("trend_regime") == "insufficient_history"
            or row.get("volatility_regime") == "insufficient_history"
        )
        payload = {
            "policy_id": self.policy_id,
            "diagnostic_only": bool(self.config.get("governance", {}).get("diagnostic_only", True)),
            "no_future_data": bool(self.config.get("governance", {}).get("no_future_data", True)),
            "trend_method": self.config.get("trend_regime", {}).get("method", "rolling_return_threshold"),
            "volatility_method": self.config.get("volatility_regime", {}).get("method", "quantile_threshold"),
            "return_observations": row.get("_return_count"),
            "volatility_observations": row.get("_volatility_count"),
            "insufficient_history": bool(insufficient),
        }
        if insufficient:
            payload["reason"] = "rolling_window_min_periods_not_met"
        else:
            payload["high_vol_threshold"] = row.get("_high_vol_threshold")
            payload["low_vol_threshold"] = row.get("_low_vol_threshold")
        return json.dumps(payload, sort_keys=True, default=_json_default)

    @staticmethod
    def _normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["date", "ticker", "close"])
        frame = data.copy()
        if "date" not in frame.columns and "time" in frame.columns:
            frame = frame.rename(columns={"time": "date"})
        required = {"date", "ticker", "close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required regime input columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
        frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        return frame

    # Legacy risk-regime API -------------------------------------------------

    @staticmethod
    def _safe_series(values: pd.Series | None, index: pd.Index, fill: float = 0.0) -> pd.Series:
        if values is None:
            return pd.Series(fill, index=index)
        return pd.to_numeric(values.reindex(index), errors="coerce").fillna(fill)

    def detect(
        self,
        *,
        volatility: pd.Series,
        drawdown: pd.Series,
        delta_covar: pd.Series | None = None,
    ) -> RegimeDetectionResult:
        index = volatility.index
        vol = self._safe_series(volatility, index=index)
        dd = self._safe_series(drawdown, index=index)
        dc = self._safe_series(delta_covar, index=index)

        crisis_mask = (dd <= self.crisis_drawdown_threshold) | (dc >= self.crisis_delta_covar_threshold)
        high_vol_mask = (~crisis_mask) & (vol >= self.high_vol_threshold)

        labels = pd.Series("NORMAL", index=index, dtype="object")
        labels.loc[high_vol_mask] = "HIGH_VOL"
        labels.loc[crisis_mask] = "CRISIS"

        vol_score = (vol / max(self.high_vol_threshold, 1e-9)).clip(lower=0.0, upper=3.0)
        dd_score = (-dd / max(abs(self.crisis_drawdown_threshold), 1e-9)).clip(lower=0.0, upper=3.0)
        dc_score = (dc / max(self.crisis_delta_covar_threshold, 1e-9)).clip(lower=0.0, upper=3.0)

        crisis_score = ((dd_score + dc_score) / 2.0).clip(0.0, 1.0)
        high_vol_score = (vol_score - crisis_score * 0.5).clip(0.0, 1.0)
        normal_score = (1.0 - np.maximum(crisis_score, high_vol_score)).clip(0.0, 1.0)

        probs = pd.DataFrame(
            {
                "NORMAL": normal_score,
                "HIGH_VOL": high_vol_score,
                "CRISIS": crisis_score,
            },
            index=index,
        )
        probs = probs.div(probs.sum(axis=1).replace(0.0, 1.0), axis=0)
        encoded = labels.map(REGIME_TO_CODE).astype(float).rename("regime_label")
        return RegimeDetectionResult(
            labels=labels.rename("regime_name"),
            encoded_labels=encoded,
            probabilities=probs,
        )

    def detect_from_frame(
        self,
        frame: pd.DataFrame,
        *,
        volatility_col: str = "rolling_volatility_20",
        drawdown_col: str = "rolling_drawdown",
        delta_covar_col: str = "delta_covar",
    ) -> RegimeDetectionResult:
        volatility = frame[volatility_col] if volatility_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        drawdown = frame[drawdown_col] if drawdown_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        delta_covar = frame[delta_covar_col] if delta_covar_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        return self.detect(
            volatility=pd.to_numeric(volatility, errors="coerce"),
            drawdown=pd.to_numeric(drawdown, errors="coerce"),
            delta_covar=pd.to_numeric(delta_covar, errors="coerce"),
        )
