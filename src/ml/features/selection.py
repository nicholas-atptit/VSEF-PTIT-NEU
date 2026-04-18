from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from src.ml.feature_engineering import FeatureEngineer
from src.utils.logging import get_logger

logger = get_logger(__name__)

REGIME_BUCKETS = ("bull", "bear", "sideways", "high_vol", "low_vol")
MIN_REGIME_OBSERVATIONS = 20


@dataclass(frozen=True)
class WalkForwardFeatureSelectionConfig:
    horizon_days: int = 5
    train_window_days: int = 756
    validation_window_days: int = 126
    step_days: int = 63
    max_folds: int = 4
    gap_days: int = 5
    top_k: int = 18
    permutation_top_n: int = 36
    permutation_repeats: int = 5
    min_train_rows: int = 400
    min_validation_rows: int = 120
    random_state: int = 42
    rolling_train: bool = True
    regression_max_features: int = 18
    classification_max_features: int = 16
    regime_max_features: int = 14
    risk_max_features: int = 16
    dedupe_correlation_threshold: float = 0.97


TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "regression_forecasting": {
        "target_column": "target_return_5d",
        "task_type": "regression",
        "registry_flag": "usable_for_forecast",
        "max_features_field": "regression_max_features",
    },
    "directional_classification": {
        "target_column": "target_direction_5d",
        "task_type": "classification",
        "registry_flag": "usable_for_forecast",
        "max_features_field": "classification_max_features",
    },
    "regime_detection": {
        "target_column": "market_regime_code",
        "task_type": "classification",
        "registry_flag": "usable_for_regime",
        "max_features_field": "regime_max_features",
    },
    "risk_layer": {
        "target_column": "target_abs_return_5d",
        "task_type": "regression",
        "registry_flag": "usable_for_risk",
        "max_features_field": "risk_max_features",
    },
}


def _rank_normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric.empty:
        return numeric
    if float(numeric.abs().sum()) == 0.0:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    return numeric.rank(method="average", pct=True).astype(float)


def _safe_spearman(feature: pd.Series, target: pd.Series) -> float:
    joined = pd.concat([pd.to_numeric(feature, errors="coerce"), pd.to_numeric(target, errors="coerce")], axis=1).dropna()
    if len(joined) < MIN_REGIME_OBSERVATIONS:
        return float("nan")
    if joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))


def _regime_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    market = frame.get("market_regime", pd.Series(index=frame.index, dtype=object)).astype(str).str.lower()
    volatility = frame.get("volatility_regime", pd.Series(index=frame.index, dtype=object)).astype(str).str.lower()
    return {
        "bull": market.eq("bull"),
        "bear": market.eq("bear"),
        "sideways": market.eq("sideways"),
        "high_vol": volatility.eq("high"),
        "low_vol": volatility.eq("low"),
    }


def detect_future_join_leakage(
    frame: pd.DataFrame,
    *,
    target_date_column: str = "date",
    source_date_column: str = "source_date",
) -> int:
    if target_date_column not in frame.columns or source_date_column not in frame.columns:
        return 0
    left = pd.to_datetime(frame[target_date_column], errors="coerce")
    right = pd.to_datetime(frame[source_date_column], errors="coerce")
    return int(((right.notna()) & (left.notna()) & (right > left)).sum())


def detect_forward_fill_boundary_leakage(
    frame: pd.DataFrame,
    *,
    group_column: str = "ticker",
    source_group_column: str = "source_ticker",
) -> int:
    if group_column not in frame.columns or source_group_column not in frame.columns:
        return 0
    left = frame[group_column].astype(str).str.upper()
    right = frame[source_group_column].astype(str).str.upper()
    return int(((right != "") & (left != "") & (left != right)).sum())


def validate_walk_forward_folds(folds: pd.DataFrame) -> None:
    if folds.empty:
        raise ValueError("Walk-forward folds must not be empty")
    required = {"fold_id", "train_start", "train_end", "validation_start", "validation_end"}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"Walk-forward folds missing required columns: {sorted(missing)}")
    for row in folds.itertuples(index=False):
        train_start = pd.Timestamp(row.train_start)
        train_end = pd.Timestamp(row.train_end)
        validation_start = pd.Timestamp(row.validation_start)
        validation_end = pd.Timestamp(row.validation_end)
        if not (train_start <= train_end < validation_start <= validation_end):
            raise ValueError(
                "Walk-forward fold contamination detected: "
                f"train=({train_start}, {train_end}) validation=({validation_start}, {validation_end})"
            )


def add_phase3_targets(frame: pd.DataFrame, *, horizon_days: int = 5) -> pd.DataFrame:
    enriched = frame.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "ticker" not in enriched.columns or "close" not in enriched.columns:
        raise ValueError("Phase 3 targets require ticker/date/close columns")

    grouped = enriched.groupby("ticker", sort=False)
    future_close = grouped["close"].shift(-int(horizon_days))
    target_return = future_close / pd.to_numeric(enriched["close"], errors="coerce") - 1.0
    enriched["target_return_5d"] = pd.to_numeric(target_return, errors="coerce")
    enriched["target_direction_5d"] = np.where(
        enriched["target_return_5d"].notna(),
        (enriched["target_return_5d"] > 0.0).astype(float),
        np.nan,
    )
    enriched["target_abs_return_5d"] = enriched["target_return_5d"].abs()
    enriched["target_date_5d"] = grouped["date"].shift(-int(horizon_days))
    return enriched


def build_feature_panel(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    if raw_dataset.empty:
        return raw_dataset.copy()

    engineer = FeatureEngineer()
    panels: list[pd.DataFrame] = []
    for ticker, group in raw_dataset.groupby("ticker", sort=False):
        transformed = engineer.transform(group.copy(), drop_na=True)
        transformed["ticker"] = str(ticker).upper()
        panels.append(transformed)
    if not panels:
        return pd.DataFrame()
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return add_phase3_targets(panel)


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    config: WalkForwardFeatureSelectionConfig,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("Walk-forward fold generation requires a date column")
    dates = pd.Index(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize().unique()).sort_values()
    if dates.empty:
        raise ValueError("Cannot generate walk-forward folds from an empty date index")
    if len(dates) < int(config.train_window_days + config.gap_days + config.validation_window_days):
        raise ValueError(
            "Not enough trading sessions to generate walk-forward folds: "
            f"sessions={len(dates)} required>={int(config.train_window_days + config.gap_days + config.validation_window_days)}"
        )

    folds: list[dict[str, Any]] = []
    validation_start_idx = int(config.train_window_days + config.gap_days)
    date_count = len(dates)
    for fold_number in range(1, int(config.max_folds) + 1):
        validation_end_idx = validation_start_idx + int(config.validation_window_days) - 1
        if validation_end_idx >= date_count:
            break
        train_end_idx = validation_start_idx - int(config.gap_days) - 1
        if train_end_idx < 0:
            break
        train_start_idx = train_end_idx - int(config.train_window_days) + 1 if config.rolling_train else 0
        if train_start_idx < 0:
            validation_start_idx = validation_start_idx + int(config.step_days)
            continue
        train_start = pd.Timestamp(dates[train_start_idx])
        train_end = pd.Timestamp(dates[train_end_idx])
        validation_start = pd.Timestamp(dates[validation_start_idx])
        validation_end = pd.Timestamp(dates[validation_end_idx])
        train_mask = (frame["date"] >= train_start) & (frame["date"] <= train_end)
        validation_mask = (frame["date"] >= validation_start) & (frame["date"] <= validation_end)
        if int(train_mask.sum()) < int(config.min_train_rows) or int(validation_mask.sum()) < int(config.min_validation_rows):
            validation_start_idx = validation_start_idx + int(config.step_days)
            continue
        folds.append(
            {
                "fold_id": f"fold_{fold_number:03d}",
                "train_start": str(pd.Timestamp(train_start).date()),
                "train_end": str(pd.Timestamp(train_end).date()),
                "validation_start": str(pd.Timestamp(validation_start).date()),
                "validation_end": str(pd.Timestamp(validation_end).date()),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "gap_days": int(config.gap_days),
            }
        )
        validation_start_idx = validation_start_idx + int(config.step_days)
    folds_df = pd.DataFrame(folds)
    validate_walk_forward_folds(folds_df)
    return folds_df


def _fit_selection_model(task_type: str, *, random_state: int) -> Any:
    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=96,
            max_depth=6,
            min_samples_leaf=8,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    return RandomForestRegressor(
        n_estimators=96,
        max_depth=6,
        min_samples_leaf=8,
        random_state=random_state,
        n_jobs=1,
    )


def _mutual_information_scores(X: pd.DataFrame, y: pd.Series, *, task_type: str, random_state: int) -> pd.Series:
    if task_type == "classification":
        scores = mutual_info_classif(X, y.astype(int), random_state=random_state)
    else:
        scores = mutual_info_regression(X, y.astype(float), random_state=random_state)
    return pd.Series(scores, index=X.columns, dtype=float)


def _validation_scoring_name(task_type: str, y: pd.Series) -> str:
    if task_type != "classification":
        return "neg_root_mean_squared_error"
    unique_count = int(pd.Series(y).dropna().nunique())
    return "f1_macro" if unique_count > 2 else "f1"


def _candidate_feature_columns(
    frame: pd.DataFrame,
    *,
    registry: dict[str, Any],
    task_name: str,
) -> list[str]:
    spec = TASK_DEFINITIONS[task_name]
    lookup = {str(entry["feature_name"]): entry for entry in registry.get("features", [])}
    selected: list[str] = []
    for column in FeatureEngineer().get_feature_columns(frame):
        entry = lookup.get(column)
        if not entry:
            continue
        if str(entry.get("status", "")).lower() != "active":
            continue
        if not bool(entry.get(spec["registry_flag"], False)):
            continue
        if task_name == "regime_detection" and "regime" in column:
            continue
        selected.append(column)
    return selected


def _filter_quality_issues(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    keep: list[str] = []
    for column in feature_columns:
        train_col = pd.to_numeric(train_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        val_col = pd.to_numeric(validation_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_missing = float(train_col.isna().mean())
        val_missing = float(val_col.isna().mean())
        unique_count = int(train_col.nunique(dropna=True))
        issue = None
        action = "keep"
        if unique_count <= 1:
            issue = "near_constant"
            action = "demote"
        elif max(train_missing, val_missing) >= 0.35:
            issue = "high_nan_ratio"
            action = "demote"
        else:
            keep.append(column)
        rows.append(
            {
                "feature_name": column,
                "train_missing_ratio": train_missing,
                "validation_missing_ratio": val_missing,
                "train_unique_count": unique_count,
                "quality_issue": issue or "",
                "quality_action": action,
            }
        )
    return keep, pd.DataFrame(rows)


def _score_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    task_name: str,
    config: WalkForwardFeatureSelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = TASK_DEFINITIONS[task_name]
    task_type = str(spec["task_type"])
    target_column = str(spec["target_column"])

    train = train_frame.dropna(subset=[target_column]).copy()
    validation = validation_frame.dropna(subset=[target_column]).copy()
    kept_columns, quality_frame = _filter_quality_issues(train, validation, feature_columns)
    if not kept_columns:
        return pd.DataFrame(), quality_frame

    X_train = train[kept_columns].astype(float)
    y_train = train[target_column]
    X_val = validation[kept_columns].astype(float)
    y_val = validation[target_column]
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    train_medians = X_train.median(axis=0, numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_val = X_val.fillna(train_medians).fillna(0.0)

    mi_scores = _mutual_information_scores(
        X_train,
        y_train,
        task_type=task_type,
        random_state=int(config.random_state),
    )
    model = _fit_selection_model(task_type, random_state=int(config.random_state))
    model.fit(X_train, y_train.astype(int) if task_type == "classification" else y_train.astype(float))
    model_scores = pd.Series(getattr(model, "feature_importances_", np.zeros(len(kept_columns))), index=kept_columns, dtype=float)

    precombined = (0.5 * _rank_normalize(mi_scores)) + (0.5 * _rank_normalize(model_scores))
    top_for_permutation = (
        precombined.sort_values(ascending=False)
        .head(max(int(config.permutation_top_n), int(config.top_k)))
        .index
        .tolist()
    )
    perm_scores = pd.Series(0.0, index=kept_columns, dtype=float)
    if len(X_val) >= int(config.min_validation_rows) and top_for_permutation:
        permutation_model = _fit_selection_model(task_type, random_state=int(config.random_state))
        permutation_model.fit(
            X_train[top_for_permutation],
            y_train.astype(int) if task_type == "classification" else y_train.astype(float),
        )
        perm_result = permutation_importance(
            permutation_model,
            X_val[top_for_permutation],
            y_val.astype(int) if task_type == "classification" else y_val.astype(float),
            scoring=_validation_scoring_name(task_type, y_train),
            n_repeats=int(config.permutation_repeats),
            random_state=int(config.random_state),
        )
        perm_scores.loc[top_for_permutation] = np.maximum(perm_result.importances_mean, 0.0)

    combined_score = (
        0.35 * _rank_normalize(mi_scores)
        + 0.35 * _rank_normalize(model_scores)
        + 0.30 * _rank_normalize(perm_scores)
    )
    top_ranked = set(combined_score.sort_values(ascending=False).head(int(config.top_k)).index.tolist())

    fold_scores = pd.DataFrame(
        {
            "feature_name": kept_columns,
            "mutual_information": mi_scores.reindex(kept_columns).to_numpy(dtype=float),
            "model_importance": model_scores.reindex(kept_columns).to_numpy(dtype=float),
            "permutation_importance": perm_scores.reindex(kept_columns).to_numpy(dtype=float),
            "combined_score": combined_score.reindex(kept_columns).to_numpy(dtype=float),
            "selected_top_k": [1.0 if feature in top_ranked else 0.0 for feature in kept_columns],
        }
    )
    return fold_scores, quality_frame


def _feature_regime_effects(
    validation_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    task_name: str,
) -> pd.DataFrame:
    spec = TASK_DEFINITIONS[task_name]
    target_column = str(spec["target_column"])
    rows: list[dict[str, Any]] = []
    masks = _regime_masks(validation_frame)
    for regime_name, regime_mask in masks.items():
        scoped = validation_frame.loc[regime_mask].copy()
        if len(scoped) < MIN_REGIME_OBSERVATIONS:
            continue
        for column in feature_columns:
            ic = _safe_spearman(scoped[column], scoped[target_column])
            rows.append(
                {
                    "feature_name": column,
                    "regime_bucket": regime_name,
                    "abs_information_coefficient": abs(ic) if pd.notna(ic) else np.nan,
                    "signed_information_coefficient": ic,
                    "observations": int(len(scoped)),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_task_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    features: list[str],
    task_name: str,
    config: WalkForwardFeatureSelectionConfig,
) -> pd.DataFrame:
    spec = TASK_DEFINITIONS[task_name]
    target_column = str(spec["target_column"])
    task_type = str(spec["task_type"])
    model = _fit_selection_model(task_type, random_state=int(config.random_state))

    train = train_frame.dropna(subset=[target_column]).copy()
    validation = validation_frame.dropna(subset=[target_column]).copy()
    X_train = train[features].astype(float)
    y_train = train[target_column].astype(int) if task_type == "classification" else train[target_column].astype(float)
    X_val = validation[features].astype(float)
    y_val = validation[target_column].astype(int) if task_type == "classification" else validation[target_column].astype(float)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(axis=0, numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_val = X_val.fillna(medians).fillna(0.0)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    masks = _regime_masks(validation)
    rows: list[dict[str, Any]] = []
    for regime_name, regime_mask in masks.items():
        scoped = validation.loc[regime_mask].copy()
        if len(scoped) < MIN_REGIME_OBSERVATIONS:
            continue
        scoped_positions = np.flatnonzero(regime_mask.to_numpy(dtype=bool))
        scoped_pred = np.asarray(predictions)[scoped_positions]
        scoped_true = y_val.iloc[scoped_positions]
        if task_type == "classification":
            accuracy = accuracy_score(scoped_true, scoped_pred)
            average = "macro" if int(pd.Series(y_train).nunique()) > 2 else "binary"
            f1 = f1_score(scoped_true, scoped_pred, zero_division=0, average=average)
            rows.append(
                {
                    "regime_bucket": regime_name,
                    "observations": int(len(scoped_true)),
                    "primary_metric": float(f1),
                    "secondary_metric": float(accuracy),
                    "metric_name": "f1",
                    "secondary_metric_name": "accuracy",
                }
            )
        else:
            rmse = float(math.sqrt(mean_squared_error(scoped_true, scoped_pred)))
            mae = float(mean_absolute_error(scoped_true, scoped_pred))
            rows.append(
                {
                    "regime_bucket": regime_name,
                    "observations": int(len(scoped_true)),
                    "primary_metric": rmse,
                    "secondary_metric": mae,
                    "metric_name": "rmse",
                    "secondary_metric_name": "mae",
                }
            )
    return pd.DataFrame(rows)


def _greedy_deduplicate(
    frame: pd.DataFrame,
    scored_features: pd.DataFrame,
    *,
    max_features: int,
    correlation_threshold: float,
) -> list[str]:
    selected: list[str] = []
    for feature in scored_features["feature_name"].tolist():
        if feature not in frame.columns:
            continue
        if not selected:
            selected.append(feature)
            if len(selected) >= max_features:
                break
            continue
        candidate = pd.to_numeric(frame[feature], errors="coerce")
        redundant = False
        for incumbent in selected:
            incumbent_series = pd.to_numeric(frame[incumbent], errors="coerce")
            pair = pd.concat([candidate, incumbent_series], axis=1).dropna()
            if len(pair) < 50:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if pd.notna(corr) and abs(corr) >= float(correlation_threshold):
                redundant = True
                break
        if redundant:
            continue
        selected.append(feature)
        if len(selected) >= max_features:
            break
    return selected


def run_walk_forward_feature_selection(
    frame: pd.DataFrame,
    *,
    registry: dict[str, Any],
    config: WalkForwardFeatureSelectionConfig | None = None,
) -> dict[str, Any]:
    selection_config = config or WalkForwardFeatureSelectionConfig()
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.sort_values(["date", "ticker"]).reset_index(drop=True)

    duplicate_keys = int(working.duplicated(subset=["ticker", "date"]).sum())
    if duplicate_keys:
        raise ValueError(f"Feature selection input contains duplicate ticker/date rows: {duplicate_keys}")

    folds = build_walk_forward_folds(working, config=selection_config)
    fold_score_frames: list[pd.DataFrame] = []
    feature_summary_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    regime_effect_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    final_sets: dict[str, list[str]] = {}
    demotion_rows: list[dict[str, Any]] = []

    for task_name in TASK_DEFINITIONS:
        candidate_columns = _candidate_feature_columns(working, registry=registry, task_name=task_name)
        if not candidate_columns:
            logger.warning("feature_selection_no_candidates", task=task_name)
            final_sets[task_name] = []
            continue

        task_fold_scores: list[pd.DataFrame] = []
        task_quality: list[pd.DataFrame] = []
        task_regime_effects: list[pd.DataFrame] = []
        for fold in folds.itertuples(index=False):
            train_mask = (
                (working["date"] >= pd.Timestamp(fold.train_start))
                & (working["date"] <= pd.Timestamp(fold.train_end))
            )
            validation_mask = (
                (working["date"] >= pd.Timestamp(fold.validation_start))
                & (working["date"] <= pd.Timestamp(fold.validation_end))
            )
            train_frame = working.loc[train_mask].copy()
            validation_frame = working.loc[validation_mask].copy()
            fold_scores, quality_frame = _score_fold(
                train_frame,
                validation_frame,
                feature_columns=candidate_columns,
                task_name=task_name,
                config=selection_config,
            )
            if fold_scores.empty:
                continue
            fold_scores["task_name"] = task_name
            fold_scores["fold_id"] = str(fold.fold_id)
            fold_scores["train_start"] = str(fold.train_start)
            fold_scores["train_end"] = str(fold.train_end)
            fold_scores["validation_start"] = str(fold.validation_start)
            fold_scores["validation_end"] = str(fold.validation_end)
            task_fold_scores.append(fold_scores)

            quality_frame["task_name"] = task_name
            quality_frame["fold_id"] = str(fold.fold_id)
            task_quality.append(quality_frame)

            regime_effect = _feature_regime_effects(
                validation_frame,
                feature_columns=fold_scores["feature_name"].tolist(),
                task_name=task_name,
            )
            if not regime_effect.empty:
                regime_effect["task_name"] = task_name
                regime_effect["fold_id"] = str(fold.fold_id)
                task_regime_effects.append(regime_effect)

        if not task_fold_scores:
            final_sets[task_name] = []
            continue

        fold_score_frame = pd.concat(task_fold_scores, ignore_index=True)
        fold_score_frames.append(fold_score_frame)
        quality_frames.extend(task_quality)
        if task_regime_effects:
            regime_effect_frames.append(pd.concat(task_regime_effects, ignore_index=True))

        summary = (
            fold_score_frame.groupby("feature_name", as_index=False)
            .agg(
                mean_mutual_information=("mutual_information", "mean"),
                mean_model_importance=("model_importance", "mean"),
                mean_permutation_importance=("permutation_importance", "mean"),
                mean_combined_score=("combined_score", "mean"),
                score_std=("combined_score", "std"),
                selection_hit_rate=("selected_top_k", "mean"),
                folds_selected=("selected_top_k", "sum"),
            )
            .sort_values(["mean_combined_score", "selection_hit_rate"], ascending=[False, False])
            .reset_index(drop=True)
        )

        task_regime_frame = pd.concat(task_regime_effects, ignore_index=True) if task_regime_effects else pd.DataFrame()
        if not task_regime_frame.empty:
            regime_summary = (
                task_regime_frame.groupby("feature_name", as_index=False)
                .agg(
                    regime_mean_abs_ic=("abs_information_coefficient", "mean"),
                    regime_max_abs_ic=("abs_information_coefficient", "max"),
                    regime_bucket_count=("regime_bucket", "nunique"),
                )
            )
            summary = summary.merge(regime_summary, on="feature_name", how="left")
        else:
            summary["regime_mean_abs_ic"] = np.nan
            summary["regime_max_abs_ic"] = np.nan
            summary["regime_bucket_count"] = 0

        ordered_folds = sorted(fold_score_frame["fold_id"].unique().tolist())
        midpoint = max(len(ordered_folds) // 2, 1)
        early_folds = set(ordered_folds[:midpoint])
        late_folds = set(ordered_folds[midpoint:] or ordered_folds)
        early_scores = (
            fold_score_frame[fold_score_frame["fold_id"].isin(early_folds)]
            .groupby("feature_name")["combined_score"]
            .mean()
        )
        late_scores = (
            fold_score_frame[fold_score_frame["fold_id"].isin(late_folds)]
            .groupby("feature_name")["combined_score"]
            .mean()
        )
        summary["early_score"] = summary["feature_name"].map(early_scores).astype(float)
        summary["late_score"] = summary["feature_name"].map(late_scores).astype(float)
        summary["decay_ratio"] = summary["late_score"] / (summary["early_score"] + 1e-9)
        summary["freshness_label"] = np.select(
            [
                summary["selection_hit_rate"] < 0.25,
                summary["decay_ratio"] < 0.67,
                summary["decay_ratio"] > 1.50,
                summary["score_std"].fillna(0.0) <= (summary["mean_combined_score"] * 0.40),
            ],
            ["weak", "decaying", "emerging", "stable"],
            default="transient",
        )
        summary["regime_behavior"] = np.select(
            [
                (summary["regime_bucket_count"] >= 3) & (summary["regime_mean_abs_ic"].fillna(0.0) >= 0.04),
                summary["regime_max_abs_ic"].fillna(0.0) >= (summary["regime_mean_abs_ic"].fillna(0.0) * 1.8 + 1e-9),
            ],
            ["robust", "regime_specific"],
            default="mixed",
        )
        summary["evidence_score"] = (
            0.45 * _rank_normalize(summary["mean_combined_score"])
            + 0.25 * _rank_normalize(summary["selection_hit_rate"])
            + 0.15 * _rank_normalize(summary["regime_mean_abs_ic"].fillna(0.0))
            + 0.15 * _rank_normalize(1.0 / (summary["score_std"].fillna(0.0) + 1e-6))
        )
        summary["task_name"] = task_name
        feature_summary_frames.append(summary)

        max_features = int(getattr(selection_config, str(TASK_DEFINITIONS[task_name]["max_features_field"])))
        selected = _greedy_deduplicate(
            working,
            summary.sort_values(["evidence_score", "mean_combined_score"], ascending=[False, False]),
            max_features=max_features,
            correlation_threshold=float(selection_config.dedupe_correlation_threshold),
        )
        final_sets[task_name] = selected

        for fold in folds.itertuples(index=False):
            train_mask = (
                (working["date"] >= pd.Timestamp(fold.train_start))
                & (working["date"] <= pd.Timestamp(fold.train_end))
            )
            validation_mask = (
                (working["date"] >= pd.Timestamp(fold.validation_start))
                & (working["date"] <= pd.Timestamp(fold.validation_end))
            )
            if not selected:
                continue
            evaluation = _evaluate_task_model(
                working.loc[train_mask].copy(),
                working.loc[validation_mask].copy(),
                features=selected,
                task_name=task_name,
                config=selection_config,
            )
            if evaluation.empty:
                continue
            evaluation["task_name"] = task_name
            evaluation["fold_id"] = str(fold.fold_id)
            evaluation_frames.append(evaluation)

        threshold = float(summary["mean_combined_score"].median())
        for row in summary.itertuples(index=False):
            if (
                float(row.selection_hit_rate) < 0.25
                and float(row.mean_combined_score) < threshold
                and str(row.freshness_label) in {"weak", "decaying"}
            ):
                demotion_rows.append(
                    {
                        "task_name": task_name,
                        "feature_name": str(row.feature_name),
                        "proposed_action": "demote",
                        "issue": f"{row.freshness_label}_low_stability",
                        "rationale": "Low walk-forward selection hit rate and weak evidence score across folds.",
                    }
                )

    return {
        "config": {key: getattr(selection_config, key) for key in selection_config.__dataclass_fields__},
        "folds": folds,
        "fold_scores": pd.concat(fold_score_frames, ignore_index=True) if fold_score_frames else pd.DataFrame(),
        "feature_summary": pd.concat(feature_summary_frames, ignore_index=True) if feature_summary_frames else pd.DataFrame(),
        "quality_summary": pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame(),
        "regime_feature_effects": pd.concat(regime_effect_frames, ignore_index=True) if regime_effect_frames else pd.DataFrame(),
        "regime_model_evaluation": pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame(),
        "final_task_feature_sets": final_sets,
        "demotion_candidates": pd.DataFrame(demotion_rows),
        "validation": {
            "duplicate_key_count": duplicate_keys,
            "selection_scope": "walk_forward_train_only",
            "fold_count": int(len(folds)),
            "future_join_leakage_count": int(
                detect_future_join_leakage(working, target_date_column="date", source_date_column="source_date")
            ),
        },
    }


def write_phase3_reports(
    results: dict[str, Any],
    *,
    output_dir: Path | str,
    as_of_date: str,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    csv_outputs = {
        "folds": results["folds"],
        "feature_selection_scores": results["fold_scores"],
        "feature_selection_summary": results["feature_summary"],
        "feature_quality_summary": results["quality_summary"],
        "regime_feature_effects": results["regime_feature_effects"],
        "regime_model_evaluation": results["regime_model_evaluation"],
        "demotion_candidates": results["demotion_candidates"],
    }
    for name, frame in csv_outputs.items():
        path = out_dir / f"{as_of_date}_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    final_sets = results["final_task_feature_sets"]
    validation = results["validation"]
    demotion = results["demotion_candidates"]

    feature_report_path = out_dir / f"{as_of_date}_feature_selection_report.md"
    feature_lines = [
        "# Feature Selection Report",
        "",
        "## Method",
        f"- Selection scope: `{validation['selection_scope']}`",
        f"- Fold count: `{validation['fold_count']}`",
        f"- Duplicate key count in analysis frame: `{validation['duplicate_key_count']}`",
        f"- Future-join leakage count in analysis frame: `{validation['future_join_leakage_count']}`",
        "",
        "## Final Task Feature Sets",
    ]
    for task_name, columns in final_sets.items():
        feature_lines.append(f"- `{task_name}`: {', '.join(columns) if columns else 'none'}")
    feature_lines.extend(["", "## Demotion Candidates"])
    if demotion.empty:
        feature_lines.append("- none")
    else:
        for row in demotion.head(20).itertuples(index=False):
            feature_lines.append(
                f"- `{row.feature_name}` ({row.task_name}): {row.proposed_action} because {row.issue}"
            )
    feature_report_path.write_text("\n".join(feature_lines), encoding="utf-8")
    paths["feature_selection_report"] = str(feature_report_path)

    regime_report_path = out_dir / f"{as_of_date}_regime_evaluation_report.md"
    regime_lines = ["# Regime Evaluation Report", "", "## Task-Level Regime Metrics"]
    regime_metrics = results["regime_model_evaluation"]
    if regime_metrics.empty:
        regime_lines.append("- no regime evaluation rows were generated")
    else:
        for task_name, group in regime_metrics.groupby("task_name", sort=False):
            regime_lines.append(f"### {task_name}")
            ordered = group.groupby("regime_bucket", as_index=False).agg(
                observations=("observations", "sum"),
                mean_primary_metric=("primary_metric", "mean"),
                mean_secondary_metric=("secondary_metric", "mean"),
            )
            for row in ordered.itertuples(index=False):
                regime_lines.append(
                    f"- `{row.regime_bucket}`: obs={int(row.observations)}, "
                    f"primary={float(row.mean_primary_metric):.4f}, secondary={float(row.mean_secondary_metric):.4f}"
                )
    regime_report_path.write_text("\n".join(regime_lines), encoding="utf-8")
    paths["regime_evaluation_report"] = str(regime_report_path)

    json_path = out_dir / f"{as_of_date}_task_feature_sets.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_on": as_of_date,
                "final_task_feature_sets": final_sets,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["task_feature_sets"] = str(json_path)
    return paths
