"""Fold-level feature-importance diagnostics for tree and boosting models."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS = [
    "fold_id",
    "step_size",
    "forecast_sequence_index",
    "ticker",
    "prediction_date",
    "model",
    "horizon",
    "task",
    "feature",
    "importance",
    "importance_rank",
    "importance_normalized",
    "train_start",
    "train_end",
    "eval_start",
    "eval_end",
]

FEATURE_IMPORTANCE_STABILITY_COLUMNS = [
    "model",
    "horizon",
    "task",
    "feature",
    "fold_count",
    "mean_importance",
    "std_importance",
    "mean_importance_normalized",
    "std_importance_normalized",
    "mean_rank",
    "best_rank",
    "top_5_count",
    "top_10_count",
    "top_5_ratio",
    "top_10_ratio",
    "importance_stability_level",
]

LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS = [
    "horizon",
    "task",
    "feature",
    "linear_models_present",
    "linear_mean_abs_coefficient",
    "linear_best_sign_consistency_ratio",
    "linear_best_stability_level",
    "importance_models_present",
    "mean_importance_normalized",
    "best_top_10_ratio",
    "best_importance_stability_level",
    "alignment_label",
]

SUPPORTED_IMPORTANCE_MODELS = {"cart", "xgboost", "lightgbm", "random_forest"}
STABILITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def empty_feature_importance_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS)


def empty_feature_importance_stability_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_IMPORTANCE_STABILITY_COLUMNS)


def empty_linear_vs_importance_comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS)


def importance_stability_level(top_10_ratio: float, fold_count: int) -> str:
    if int(fold_count) < 3 or pd.isna(top_10_ratio):
        return "low"
    ratio = float(top_10_ratio)
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def _extract_importances(model: Any) -> np.ndarray | None:
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(model, "estimator", None),
    ):
        if candidate is None:
            continue
        values = getattr(candidate, "feature_importances_", None)
        if values is not None:
            return np.asarray(values, dtype=float).reshape(-1)
    return None


def extract_feature_importance_rows(
    *,
    model_name: str,
    model: Any,
    feature_columns: Iterable[str],
    fold_context: dict[str, Any],
) -> pd.DataFrame:
    normalized_model_name = str(model_name).strip().lower()
    if normalized_model_name not in SUPPORTED_IMPORTANCE_MODELS:
        return empty_feature_importance_diagnostics_frame()

    features = [str(column) for column in feature_columns]
    importances = _extract_importances(model)
    if importances is None or len(importances) != len(features):
        return empty_feature_importance_diagnostics_frame()

    importances = np.nan_to_num(importances.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    total_importance = float(importances.sum())
    if total_importance > 0.0:
        normalized = importances / total_importance
    else:
        normalized = np.zeros_like(importances, dtype=float)

    order = np.argsort(-importances, kind="mergesort")
    ranks = np.empty(len(importances), dtype=int)
    ranks[order] = np.arange(1, len(importances) + 1)

    rows = []
    for index, feature in enumerate(features):
        rows.append(
            {
                **fold_context,
                "model": normalized_model_name,
                "feature": feature,
                "importance": float(importances[index]),
                "importance_rank": int(ranks[index]),
                "importance_normalized": float(normalized[index]),
            }
        )
    return pd.DataFrame(rows).reindex(columns=FEATURE_IMPORTANCE_DIAGNOSTIC_COLUMNS)


def summarize_feature_importance_stability(importance_rows: pd.DataFrame) -> pd.DataFrame:
    if importance_rows is None or importance_rows.empty:
        return empty_feature_importance_stability_summary_frame()

    working = importance_rows.copy()
    working["importance"] = pd.to_numeric(working["importance"], errors="coerce")
    working["importance_normalized"] = pd.to_numeric(working["importance_normalized"], errors="coerce")
    working["importance_rank"] = pd.to_numeric(working["importance_rank"], errors="coerce")
    working = working.dropna(subset=["importance", "importance_normalized", "importance_rank"])
    if working.empty:
        return empty_feature_importance_stability_summary_frame()

    rows: list[dict[str, Any]] = []
    group_columns = ["model", "horizon", "task", "feature"]
    for keys, group in working.groupby(group_columns, sort=True):
        fold_count = int(group["fold_id"].nunique())
        top_5_count = int((group["importance_rank"] <= 5).sum())
        top_10_count = int((group["importance_rank"] <= 10).sum())
        top_5_ratio = float(top_5_count / fold_count) if fold_count else np.nan
        top_10_ratio = float(top_10_count / fold_count) if fold_count else np.nan
        rows.append(
            {
                "model": keys[0],
                "horizon": keys[1],
                "task": keys[2],
                "feature": keys[3],
                "fold_count": fold_count,
                "mean_importance": float(group["importance"].mean()),
                "std_importance": float(group["importance"].std(ddof=0)),
                "mean_importance_normalized": float(group["importance_normalized"].mean()),
                "std_importance_normalized": float(group["importance_normalized"].std(ddof=0)),
                "mean_rank": float(group["importance_rank"].mean()),
                "best_rank": int(group["importance_rank"].min()),
                "top_5_count": top_5_count,
                "top_10_count": top_10_count,
                "top_5_ratio": top_5_ratio,
                "top_10_ratio": top_10_ratio,
                "importance_stability_level": importance_stability_level(top_10_ratio, fold_count),
            }
        )

    return pd.DataFrame(rows).reindex(columns=FEATURE_IMPORTANCE_STABILITY_COLUMNS)


def _best_stability_level(values: pd.Series) -> str:
    levels = [str(value).lower() for value in values.dropna()]
    if not levels:
        return "missing"
    return max(levels, key=lambda value: STABILITY_ORDER.get(value, -1))


def _stable_or_medium(level: Any) -> bool:
    return str(level).lower() in {"high", "medium"}


def _model_list(values: pd.Series) -> str:
    models = sorted({str(value).lower() for value in values.dropna() if str(value).strip()})
    return ",".join(models)


def compare_linear_and_importance_diagnostics(
    *,
    linear_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
) -> pd.DataFrame:
    linear_agg = pd.DataFrame(
        columns=[
            "horizon",
            "task",
            "feature",
            "linear_models_present",
            "linear_mean_abs_coefficient",
            "linear_best_sign_consistency_ratio",
            "linear_best_stability_level",
        ]
    )
    if linear_summary is not None and not linear_summary.empty:
        linear_working = linear_summary.copy()
        linear_working["mean_abs_coefficient"] = pd.to_numeric(
            linear_working["mean_abs_coefficient"],
            errors="coerce",
        )
        linear_working["sign_consistency_ratio"] = pd.to_numeric(
            linear_working["sign_consistency_ratio"],
            errors="coerce",
        )
        linear_agg = (
            linear_working.groupby(["horizon", "task", "feature"], sort=True)
            .agg(
                linear_models_present=("model", _model_list),
                linear_mean_abs_coefficient=("mean_abs_coefficient", "mean"),
                linear_best_sign_consistency_ratio=("sign_consistency_ratio", "max"),
                linear_best_stability_level=("stability_level", _best_stability_level),
            )
            .reset_index()
        )

    importance_agg = pd.DataFrame(
        columns=[
            "horizon",
            "task",
            "feature",
            "importance_models_present",
            "mean_importance_normalized",
            "best_top_10_ratio",
            "best_importance_stability_level",
        ]
    )
    if importance_summary is not None and not importance_summary.empty:
        importance_working = importance_summary.copy()
        importance_working["mean_importance_normalized"] = pd.to_numeric(
            importance_working["mean_importance_normalized"],
            errors="coerce",
        )
        importance_working["top_10_ratio"] = pd.to_numeric(
            importance_working["top_10_ratio"],
            errors="coerce",
        )
        importance_agg = (
            importance_working.groupby(["horizon", "task", "feature"], sort=True)
            .agg(
                importance_models_present=("model", _model_list),
                mean_importance_normalized=("mean_importance_normalized", "mean"),
                best_top_10_ratio=("top_10_ratio", "max"),
                best_importance_stability_level=("importance_stability_level", _best_stability_level),
            )
            .reset_index()
        )

    if linear_agg.empty and importance_agg.empty:
        return empty_linear_vs_importance_comparison_frame()

    merged = pd.merge(
        linear_agg,
        importance_agg,
        on=["horizon", "task", "feature"],
        how="outer",
    )
    for column in ["linear_models_present", "importance_models_present"]:
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("")
    for column in ["linear_best_stability_level", "best_importance_stability_level"]:
        if column not in merged.columns:
            merged[column] = "missing"
        merged[column] = merged[column].fillna("missing")

    labels = []
    for row in merged.itertuples(index=False):
        linear_stable = _stable_or_medium(getattr(row, "linear_best_stability_level"))
        importance_stable = _stable_or_medium(getattr(row, "best_importance_stability_level"))
        if linear_stable and importance_stable:
            labels.append("aligned_stable")
        elif linear_stable:
            labels.append("linear_only")
        elif importance_stable:
            labels.append("importance_only")
        else:
            labels.append("unstable_or_missing")
    merged["alignment_label"] = labels

    return merged.reindex(columns=LINEAR_VS_IMPORTANCE_COMPARISON_COLUMNS).sort_values(
        ["horizon", "task", "feature"]
    ).reset_index(drop=True)
