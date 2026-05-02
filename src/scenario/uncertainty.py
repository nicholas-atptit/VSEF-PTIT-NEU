"""Scenario uncertainty scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scenario.schema import SCENARIO_CONTEXT_COLUMNS, present_columns


UNCERTAINTY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
    "scenario_count",
    "probability_entropy",
    "top_probability",
    "second_probability",
    "probability_gap",
    "uncertainty_score",
    "dispersion_score",
    "mean_calibration_error",
    "missing_calibration_share",
    "confidence_bucket",
)


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return lower
    return float(min(max(value, lower), upper))


def _group_columns(frame: pd.DataFrame) -> list[str]:
    columns = present_columns(frame, SCENARIO_CONTEXT_COLUMNS)
    if columns:
        return columns
    return [column for column in ("timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id") if column in frame.columns]


def _confidence_bucket(uncertainty_score: float, missing_calibration_share: float) -> str:
    if missing_calibration_share >= 1.0:
        return "uncalibrated"
    if uncertainty_score <= 0.35:
        return "high"
    if uncertainty_score <= 0.60:
        return "medium"
    return "low"


def attach_uncertainty_scores(probability_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach packet-level uncertainty scores to scenario probability rows."""

    if probability_df.empty:
        result = probability_df.copy()
        result["uncertainty_score"] = pd.Series(dtype=float)
        return result, pd.DataFrame(columns=list(UNCERTAINTY_COLUMNS))

    result = probability_df.copy()
    group_columns = _group_columns(result)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in result.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys))
        probabilities = pd.to_numeric(group["scenario_probability"], errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(probabilities.sum())
        normalized = probabilities / total if total > 0.0 else pd.Series(1.0 / len(group), index=group.index)
        entropy = float(-(normalized * np.log(normalized.replace(0.0, np.nan))).sum(skipna=True))
        normalized_entropy = entropy / np.log(max(len(group), 2))
        ordered = normalized.sort_values(ascending=False)
        top_probability = float(ordered.iloc[0]) if len(ordered) else 0.0
        second_probability = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
        probability_gap = top_probability - second_probability
        dispersion_norm = pd.to_numeric(group.get("dispersion_normalized"), errors="coerce").dropna()
        if dispersion_norm.empty:
            raw_dispersion = pd.to_numeric(group.get("dispersion_score"), errors="coerce").dropna()
            dispersion_component = _bounded(float(raw_dispersion.mean()) / 0.05) if not raw_dispersion.empty else 0.0
        else:
            dispersion_component = _bounded(float(dispersion_norm.mean()))
        disagreement = 1.0 - float(pd.to_numeric(group.get("model_agreement_score"), errors="coerce").dropna().mean())
        disagreement = _bounded(disagreement)
        missing_context = pd.to_numeric(group.get("missing_context_share"), errors="coerce").dropna()
        missing_context_share = _bounded(float(missing_context.mean())) if not missing_context.empty else 0.0
        calibration_error = pd.to_numeric(group.get("calibration_error"), errors="coerce")
        mean_calibration_error = float(calibration_error.dropna().mean()) if calibration_error.notna().any() else float("nan")
        missing_calibration_share = float(calibration_error.isna().mean())
        calibration_component = _bounded(mean_calibration_error) if pd.notna(mean_calibration_error) else 0.15
        uncertainty_score = _bounded(
            0.40 * normalized_entropy
            + 0.20 * dispersion_component
            + 0.20 * disagreement
            + 0.10 * missing_context_share
            + 0.10 * calibration_component
        )
        result.loc[group.index, "uncertainty_score"] = uncertainty_score
        summary_rows.append(
            {
                **context,
                "scenario_count": int(len(group)),
                "probability_entropy": float(normalized_entropy),
                "top_probability": top_probability,
                "second_probability": second_probability,
                "probability_gap": float(probability_gap),
                "uncertainty_score": uncertainty_score,
                "dispersion_score": float(pd.to_numeric(group.get("dispersion_score"), errors="coerce").dropna().mean())
                if pd.to_numeric(group.get("dispersion_score"), errors="coerce").notna().any()
                else float("nan"),
                "mean_calibration_error": mean_calibration_error,
                "missing_calibration_share": missing_calibration_share,
                "confidence_bucket": _confidence_bucket(uncertainty_score, missing_calibration_share),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return result.reset_index(drop=True), summary[[column for column in UNCERTAINTY_COLUMNS if column in summary.columns]].reset_index(drop=True)
