"""Probability calibration helpers for Scenario Evaluation Engine v1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scenario.schema import SCENARIO_LABELS


CALIBRATION_COLUMNS: tuple[str, ...] = (
    "scenario_label",
    "probability_bin",
    "bin_low",
    "bin_high",
    "prediction_count",
    "observed_count",
    "observed_frequency",
    "mean_probability",
    "calibration_error",
    "brier_score",
    "expected_calibration_error",
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric) or not np.isfinite(numeric):
        return default
    return float(numeric)


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return lower
    return float(min(max(value, lower), upper))


def _return_threshold(realized: pd.Series) -> float:
    clean = pd.to_numeric(realized, errors="coerce").dropna().abs()
    if clean.empty:
        return 0.0025
    return float(max(0.0025, min(0.01, clean.median() * 0.25)))


def _probability_bin(probability: float, bins: int) -> tuple[int, float, float, str]:
    probability = _bounded(probability)
    index = min(int(probability * bins), bins - 1)
    low = index / bins
    high = (index + 1) / bins
    return index, low, high, f"{low:.2f}-{high:.2f}"


def _event_observed(row: pd.Series, threshold: float) -> float:
    realized = _safe_float(row.get("realized_outcome"), default=float("nan"))
    if pd.isna(realized):
        return float("nan")
    label = str(row.get("scenario_label"))
    target_text = f"{row.get('target_type', '')} {row.get('target_family', '')}".lower()
    if "direction" in target_text or "binary" in target_text:
        bullish = realized >= 0.5
        if label in {"bull", "recovery"}:
            return float(bullish)
        if label in {"bear", "drawdown"}:
            return float(not bullish)
        if label == "sideway":
            return 0.0
        if label == "high_volatility":
            return 0.0
        return 0.0

    if label == "bull":
        return float(realized > threshold)
    if label == "bear":
        return float(realized < -threshold)
    if label == "sideway":
        return float(abs(realized) <= threshold)
    if label == "high_volatility":
        return float(abs(realized) >= max(2.0 * threshold, 0.03))
    if label == "drawdown":
        return float(realized <= -max(2.0 * threshold, 0.03))
    if label == "recovery":
        return float(realized >= max(2.0 * threshold, 0.03))
    return float(abs(realized) <= threshold)


def attach_realized_events(probability_df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic observed-event flags used for calibration."""

    if probability_df.empty:
        result = probability_df.copy()
        result["scenario_event_observed"] = pd.Series(dtype=float)
        return result
    result = probability_df.copy()
    threshold = _return_threshold(result.get("realized_outcome", pd.Series(dtype=float)))
    result["scenario_event_observed"] = result.apply(lambda row: _event_observed(row, threshold), axis=1)
    return result


def build_calibration_summary(
    probability_df: pd.DataFrame,
    *,
    bins: int = 5,
    lookback: int | None = 252,
) -> pd.DataFrame:
    """Build scenario-label probability-bin calibration statistics."""

    if probability_df.empty:
        return pd.DataFrame(columns=list(CALIBRATION_COLUMNS))
    prepared = attach_realized_events(probability_df)
    prepared = prepared[pd.to_numeric(prepared["scenario_event_observed"], errors="coerce").notna()].copy()
    if prepared.empty:
        return pd.DataFrame(columns=list(CALIBRATION_COLUMNS))

    prepared["timestamp"] = pd.to_datetime(prepared.get("timestamp"), errors="coerce")
    if lookback is not None and int(lookback) > 0:
        prepared = (
            prepared.sort_values(["scenario_label", "timestamp", "scenario_id"])
            .groupby("scenario_label", sort=True, dropna=False)
            .tail(int(lookback))
            .reset_index(drop=True)
        )

    bin_parts = prepared["scenario_probability"].apply(lambda value: _probability_bin(_safe_float(value, 0.0), bins))
    prepared["probability_bin_index"] = bin_parts.apply(lambda item: item[0])
    prepared["bin_low"] = bin_parts.apply(lambda item: item[1])
    prepared["bin_high"] = bin_parts.apply(lambda item: item[2])
    prepared["probability_bin"] = bin_parts.apply(lambda item: item[3])

    rows: list[dict[str, Any]] = []
    for label in SCENARIO_LABELS:
        label_frame = prepared[prepared["scenario_label"].astype(str) == label].copy()
        if label_frame.empty:
            continue
        label_rows: list[dict[str, Any]] = []
        for _, group in label_frame.groupby(["probability_bin_index", "probability_bin", "bin_low", "bin_high"], sort=True):
            probabilities = pd.to_numeric(group["scenario_probability"], errors="coerce").astype(float)
            observed = pd.to_numeric(group["scenario_event_observed"], errors="coerce").astype(float)
            mean_probability = float(probabilities.mean())
            observed_frequency = float(observed.mean())
            calibration_error = abs(observed_frequency - mean_probability)
            brier_score = float(((probabilities - observed) ** 2).mean())
            label_rows.append(
                {
                    "scenario_label": label,
                    "probability_bin": str(group["probability_bin"].iloc[0]),
                    "bin_low": float(group["bin_low"].iloc[0]),
                    "bin_high": float(group["bin_high"].iloc[0]),
                    "prediction_count": int(len(group)),
                    "observed_count": int(observed.sum()),
                    "observed_frequency": observed_frequency,
                    "mean_probability": mean_probability,
                    "calibration_error": float(calibration_error),
                    "brier_score": brier_score,
                }
            )
        total_count = sum(row["prediction_count"] for row in label_rows)
        expected_calibration_error = (
            sum(row["calibration_error"] * row["prediction_count"] for row in label_rows) / total_count
            if total_count
            else float("nan")
        )
        for row in label_rows:
            row["expected_calibration_error"] = float(expected_calibration_error)
            rows.append(row)

    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS)).sort_values(
        ["scenario_label", "bin_low", "bin_high"]
    ).reset_index(drop=True)


def apply_probability_calibration(
    probability_df: pd.DataFrame,
    *,
    bins: int = 5,
    lookback: int | None = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply bin calibration and return adjusted probabilities plus bin summary."""

    if probability_df.empty:
        result = probability_df.copy()
        for column in ("probability_bin", "historical_hit_rate", "calibration_error", "brier_score", "expected_calibration_error"):
            if column not in result.columns:
                result[column] = pd.Series(dtype=float)
        return result, pd.DataFrame(columns=list(CALIBRATION_COLUMNS))

    result = attach_realized_events(probability_df)
    calibration = build_calibration_summary(result, bins=bins, lookback=lookback)
    bin_parts = result["scenario_probability"].apply(lambda value: _probability_bin(_safe_float(value, 0.0), bins))
    result["probability_bin"] = bin_parts.apply(lambda item: item[3])

    if not calibration.empty:
        merge_columns = ["scenario_label", "probability_bin"]
        result = result.merge(
            calibration[
                [
                    *merge_columns,
                    "observed_frequency",
                    "calibration_error",
                    "brier_score",
                    "expected_calibration_error",
                ]
            ],
            on=merge_columns,
            how="left",
            suffixes=("", "_calibrated"),
        )
        if "calibration_error_calibrated" in result.columns:
            result["calibration_error"] = result["calibration_error_calibrated"]
            result = result.drop(columns=["calibration_error_calibrated"])
    else:
        result["observed_frequency"] = float("nan")
        result["brier_score"] = float("nan")
        result["expected_calibration_error"] = float("nan")

    result["historical_hit_rate"] = pd.to_numeric(result.get("observed_frequency"), errors="coerce")
    calibration_error = pd.to_numeric(result.get("calibration_error"), errors="coerce")
    probability = pd.to_numeric(result["scenario_probability"], errors="coerce").fillna(0.0)
    reliability = (1.0 - calibration_error).clip(lower=0.05, upper=1.0)
    result["confidence_adjusted_probability"] = np.where(
        calibration_error.notna(),
        probability * reliability,
        probability,
    )
    return result.reset_index(drop=True), calibration
