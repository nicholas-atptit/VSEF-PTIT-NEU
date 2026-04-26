"""Fold-level coefficient diagnostics for linear forecast baselines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.forecast.ml.lasso import LassoForecastModel
from src.forecast.ml.linear import LinearForecastModel
from src.forecast.ml.ridge import RidgeForecastModel
from src.ml.features.registry import resolve_feature_set, resolve_task_feature_set

LINEAR_DIAGNOSTIC_MODELS = {
    "linear": LinearForecastModel,
    "ridge": RidgeForecastModel,
    "lasso": LassoForecastModel,
}

COEFFICIENT_DIAGNOSTIC_COLUMNS = [
    "fold_id",
    "step_size",
    "forecast_sequence_index",
    "ticker",
    "prediction_date",
    "model",
    "horizon",
    "task",
    "feature",
    "coefficient",
    "coefficient_sign",
    "coefficient_magnitude",
    "intercept",
    "nonzero_coefficient_count",
    "feature_count",
    "train_start",
    "train_end",
    "eval_start",
    "eval_end",
]

COEFFICIENT_STABILITY_COLUMNS = [
    "model",
    "horizon",
    "task",
    "feature",
    "fold_count",
    "mean_coefficient",
    "std_coefficient",
    "mean_abs_coefficient",
    "sign_positive_count",
    "sign_negative_count",
    "sign_zero_count",
    "sign_consistency_ratio",
    "coefficient_cv",
    "stability_level",
]


def empty_coefficient_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COEFFICIENT_DIAGNOSTIC_COLUMNS)


def empty_coefficient_stability_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COEFFICIENT_STABILITY_COLUMNS)


def select_linear_diagnostic_features(frame: pd.DataFrame) -> list[str]:
    selected = resolve_task_feature_set(
        "regression_forecasting",
        available_columns=frame.columns,
    )
    if not selected:
        selected = resolve_feature_set(
            "forecast_core_features",
            available_columns=frame.columns,
        )
    return selected


def coefficient_sign(value: float) -> str:
    coefficient = float(value)
    if abs(coefficient) <= 1e-12:
        return "zero"
    if coefficient > 0.0:
        return "positive"
    return "negative"


def coefficient_stability_level(sign_consistency_ratio: float, fold_count: int) -> str:
    if int(fold_count) < 3 or pd.isna(sign_consistency_ratio):
        return "low"
    ratio = float(sign_consistency_ratio)
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.6:
        return "medium"
    return "low"


def coefficient_rows_from_metadata(
    *,
    model_name: str,
    metadata: dict[str, Any],
    fold_context: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = metadata.get("coefficient_diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics.get("available"):
        return []

    rows: list[dict[str, Any]] = []
    feature_count = int(diagnostics.get("coefficient_count", len(diagnostics.get("selected_feature_names", []))))
    nonzero_count = int(diagnostics.get("nonzero_coefficient_count", 0))
    intercept = diagnostics.get("intercept")
    if isinstance(intercept, list):
        intercept = np.nan

    for item in diagnostics.get("coefficients", []):
        coefficient = float(item["coefficient"])
        rows.append(
            {
                **fold_context,
                "model": str(model_name).lower(),
                "feature": str(item["feature"]),
                "coefficient": coefficient,
                "coefficient_sign": str(item.get("sign") or coefficient_sign(coefficient)),
                "coefficient_magnitude": float(item.get("magnitude", abs(coefficient))),
                "intercept": intercept,
                "nonzero_coefficient_count": nonzero_count,
                "feature_count": feature_count,
            }
        )
    return rows


def fit_linear_fold_diagnostics(
    *,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    fold_context: dict[str, Any],
    model_names: Iterable[str] = tuple(LINEAR_DIAGNOSTIC_MODELS),
) -> pd.DataFrame:
    if train_frame.empty or not feature_columns or target_column not in train_frame.columns:
        return empty_coefficient_diagnostics_frame()

    rows: list[dict[str, Any]] = []
    for raw_name in model_names:
        model_name = str(raw_name).strip().lower()
        model_cls = LINEAR_DIAGNOSTIC_MODELS.get(model_name)
        if model_cls is None:
            continue
        model = model_cls()
        model.fit(
            train_df=train_frame,
            features=feature_columns,
            target=target_column,
            horizon=int(fold_context.get("horizon_days", 1)),
            config={"diagnostic_scope": "walk_forward_linear_coefficient_stability"},
        )
        rows.extend(
            coefficient_rows_from_metadata(
                model_name=model_name,
                metadata=model.get_metadata(),
                fold_context=fold_context,
            )
        )

    if not rows:
        return empty_coefficient_diagnostics_frame()
    return pd.DataFrame(rows).reindex(columns=COEFFICIENT_DIAGNOSTIC_COLUMNS)


def summarize_coefficient_stability(coefficient_rows: pd.DataFrame) -> pd.DataFrame:
    if coefficient_rows is None or coefficient_rows.empty:
        return empty_coefficient_stability_summary_frame()

    working = coefficient_rows.copy()
    working["coefficient"] = pd.to_numeric(working["coefficient"], errors="coerce")
    working["coefficient_magnitude"] = pd.to_numeric(working["coefficient_magnitude"], errors="coerce")
    working["coefficient_sign"] = working["coefficient_sign"].astype(str)
    working = working.dropna(subset=["coefficient"])
    if working.empty:
        return empty_coefficient_stability_summary_frame()

    rows: list[dict[str, Any]] = []
    group_columns = ["model", "horizon", "task", "feature"]
    for keys, group in working.groupby(group_columns, sort=True):
        signs = group["coefficient_sign"].value_counts()
        positive_count = int(signs.get("positive", 0))
        negative_count = int(signs.get("negative", 0))
        zero_count = int(signs.get("zero", 0))
        fold_count = int(group["fold_id"].nunique())
        max_sign_count = max(positive_count, negative_count, zero_count)
        sign_consistency_ratio = float(max_sign_count / fold_count) if fold_count else np.nan
        mean_coefficient = float(group["coefficient"].mean())
        std_coefficient = float(group["coefficient"].std(ddof=0))
        coefficient_cv = (
            float(std_coefficient / abs(mean_coefficient))
            if abs(mean_coefficient) > 1e-12
            else np.nan
        )
        rows.append(
            {
                "model": keys[0],
                "horizon": keys[1],
                "task": keys[2],
                "feature": keys[3],
                "fold_count": fold_count,
                "mean_coefficient": mean_coefficient,
                "std_coefficient": std_coefficient,
                "mean_abs_coefficient": float(group["coefficient_magnitude"].mean()),
                "sign_positive_count": positive_count,
                "sign_negative_count": negative_count,
                "sign_zero_count": zero_count,
                "sign_consistency_ratio": sign_consistency_ratio,
                "coefficient_cv": coefficient_cv,
                "stability_level": coefficient_stability_level(sign_consistency_ratio, fold_count),
            }
        )

    return pd.DataFrame(rows).reindex(columns=COEFFICIENT_STABILITY_COLUMNS)
