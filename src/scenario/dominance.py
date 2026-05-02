"""Dominant scenario selection and ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scenario.schema import DOMINANCE_LABELS, SCENARIO_CONTEXT_COLUMNS, present_columns


DOMINANCE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
    "dominant_scenario",
    "dominant_scenario_probability",
    "dominant_scenario_adjusted_probability",
    "second_scenario",
    "second_scenario_adjusted_probability",
    "probability_gap",
    "dominance_score",
    "dominance_label",
    "dominant_scenario_flag",
    "uncertainty_score",
    "calibration_error",
    "downside_risk",
    "scenario_confidence_bucket",
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


def _dominance_label(
    *,
    top_label: str,
    top_adjusted: float,
    probability_gap: float,
    dominance_score: float,
    uncertainty_score: float,
    calibration_error: float,
    downside_risk: float,
    risk_scenario_pressure: float,
) -> str:
    uncalibrated = pd.isna(calibration_error)
    calibration_penalty = 0.20 if uncalibrated else _bounded(calibration_error)
    if top_label in {"bull", "recovery"} and downside_risk >= 0.08 and risk_scenario_pressure >= top_adjusted * 0.80:
        return "risk_overrides_dominance"
    if uncalibrated and top_adjusted >= 0.35 and probability_gap >= 0.12 and uncertainty_score <= 0.60:
        return "uncalibrated_dominance"
    if (
        top_adjusted >= 0.35
        and probability_gap >= 0.12
        and uncertainty_score <= 0.55
        and calibration_penalty <= 0.30
        and dominance_score >= 0.15
    ):
        return "dominant"
    if top_adjusted >= 0.25 and probability_gap >= 0.05 and uncertainty_score <= 0.70:
        return "weak_dominance"
    return "no_clear_dominance"


def _confidence_bucket(label: str, uncertainty_score: float, calibration_error: float) -> str:
    if label == "uncalibrated_dominance":
        return "uncalibrated"
    if label == "risk_overrides_dominance":
        return "risk_overridden"
    if label == "no_clear_dominance":
        return "low"
    if label == "dominant" and uncertainty_score <= 0.35 and (pd.isna(calibration_error) or calibration_error <= 0.15):
        return "high"
    if label in {"dominant", "weak_dominance"}:
        return "medium"
    return "low"


def evaluate_scenario_dominance(probability_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank scenarios and attach dominant-scenario diagnostics."""

    if probability_df.empty:
        result = probability_df.copy()
        for column in ("scenario_rank", "dominance_label"):
            result[column] = pd.Series(dtype=object)
        return result, result.copy(), pd.DataFrame(columns=list(DOMINANCE_SUMMARY_COLUMNS))

    result = probability_df.copy()
    result["dominant_scenario_flag"] = False
    group_columns = _group_columns(result)
    ranking_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for keys, group in result.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys))
        ranked = group.copy()
        ranked["_adjusted_sort"] = pd.to_numeric(
            ranked.get("confidence_adjusted_probability"),
            errors="coerce",
        ).fillna(pd.to_numeric(ranked.get("scenario_probability"), errors="coerce").fillna(0.0))
        ranked = ranked.sort_values(
            ["_adjusted_sort", "scenario_probability", "scenario_label"],
            ascending=[False, False, True],
        ).reset_index()
        ranked["scenario_rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else top
        top_adjusted = float(top["_adjusted_sort"])
        second_adjusted = float(second["_adjusted_sort"]) if len(ranked) > 1 else 0.0
        probability_gap = top_adjusted - second_adjusted
        uncertainty_score = float(pd.to_numeric(group.get("uncertainty_score"), errors="coerce").dropna().mean())
        if pd.isna(uncertainty_score):
            uncertainty_score = 0.5
        calibration_error = float(top.get("calibration_error")) if pd.notna(top.get("calibration_error")) else float("nan")
        downside_risk = float(top.get("downside_risk")) if pd.notna(top.get("downside_risk")) else 0.0
        downside_penalty = _bounded(downside_risk / 0.12) if str(top.get("scenario_label")) in {"bull", "recovery"} else 0.0
        calibration_penalty = 0.20 if pd.isna(calibration_error) else _bounded(calibration_error)
        risk_pressure = float(
            ranked[ranked["scenario_label"].isin(["bear", "drawdown", "high_volatility"])]["_adjusted_sort"].sum()
        )
        dominance_score = _bounded(
            probability_gap
            - 0.20 * _bounded(uncertainty_score)
            - 0.20 * calibration_penalty
            - 0.15 * downside_penalty
        )
        label = _dominance_label(
            top_label=str(top.get("scenario_label")),
            top_adjusted=top_adjusted,
            probability_gap=probability_gap,
            dominance_score=dominance_score,
            uncertainty_score=uncertainty_score,
            calibration_error=calibration_error,
            downside_risk=downside_risk,
            risk_scenario_pressure=risk_pressure,
        )
        if label not in DOMINANCE_LABELS:
            label = "no_clear_dominance"
        flag = label != "no_clear_dominance"
        original_top_index = int(top["index"])
        result.loc[group.index, "dominance_score"] = dominance_score
        result.loc[group.index, "dominance_label"] = label
        result.loc[group.index, "scenario_rank"] = result.loc[group.index].index.map(
            dict(zip(ranked["index"].astype(int), ranked["scenario_rank"].astype(int)))
        )
        result.loc[original_top_index, "dominant_scenario_flag"] = bool(flag)

        ranked["dominance_score"] = dominance_score
        ranked["dominance_label"] = label
        ranked["dominant_scenario_flag"] = ranked["index"].astype(int).eq(original_top_index) & bool(flag)
        ranked["probability_gap"] = probability_gap
        ranking_frames.append(ranked.drop(columns=["_adjusted_sort", "index"]))
        summary_rows.append(
            {
                **context,
                "dominant_scenario": str(top.get("scenario_label")),
                "dominant_scenario_probability": float(top.get("scenario_probability")),
                "dominant_scenario_adjusted_probability": top_adjusted,
                "second_scenario": str(second.get("scenario_label")) if len(ranked) > 1 else "",
                "second_scenario_adjusted_probability": second_adjusted,
                "probability_gap": float(probability_gap),
                "dominance_score": dominance_score,
                "dominance_label": label,
                "dominant_scenario_flag": bool(flag),
                "uncertainty_score": uncertainty_score,
                "calibration_error": calibration_error,
                "downside_risk": downside_risk,
                "scenario_confidence_bucket": _confidence_bucket(label, uncertainty_score, calibration_error),
            }
        )

    rankings = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    rankings = rankings.sort_values([*group_columns, "scenario_rank"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    summary = summary[[column for column in DOMINANCE_SUMMARY_COLUMNS if column in summary.columns]].sort_values(
        group_columns
    ).reset_index(drop=True)
    result = result.sort_values([*group_columns, "scenario_label"]).reset_index(drop=True)
    return result, rankings, summary
