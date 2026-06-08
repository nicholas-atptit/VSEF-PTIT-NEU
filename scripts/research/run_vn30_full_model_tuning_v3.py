"""Run VN30 full model tuning v3 with target variants and index context.

The runner is validation-governed for claimable rows. Final-ranked rows are
written separately as exploratory evidence and are never promoted without a
future-blind or re-locked run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_model_resurrection_index_pretrain import (  # noqa: E402
    CURRENT_CHAMPION,
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    accuracy,
    as_float,
    baseline_predictions,
    build_feature_frame as build_resurrection_feature_frame,
    clean_feature_matrix,
    detail_rows,
    json_safe,
    majority_value,
    now_utc,
    pct,
    pp,
    predict_probability,
    rel,
    rolling_min,
    split_indices,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    add_absolute_labels,
    active_stock_tickers,
    load_index_data,
    target_timestamp_from_labels,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_full_model_tuning_v3"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_FULL_MODEL_TUNING_V3_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_FULL_MODEL_TUNING_V3_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_FULL_MODEL_TUNING_V3_CLAIM_BOUNDARY.md"

SEED = 42
TARGET_VARIANTS = [
    "absolute_direction",
    "market_relative_vn30",
    "market_relative_vnindex",
    "excess_return_direction",
    "thresholded_direction",
    "neutral_removed_direction",
]
HORIZONS = [20, 30, 35, 40, 45, 50, 55, 60]
THRESHOLDS = [round(float(value), 3) for value in np.arange(0.40, 0.6501, 0.005)]
RETURN_THRESHOLD = 0.001

CLAIMABLE_CHAMPION = {
    **CURRENT_CHAMPION,
    "final_accuracy": 0.6161021109474718,
    "final_lift": 0.10898379970544925,
    "final_rows": 4074,
}
EXPLORATORY_BEST = {
    "candidate_id": "grid_334693__t0p525",
    "model_family": "logistic_regression",
    "feature_group": "compact_stable_features",
    "target_variant": "absolute_direction",
    "horizon": 50,
    "threshold": 0.525,
    "final_accuracy": 0.6475887652358241,
    "final_lift": 0.11181770005299418,
    "final_rows": 3774,
}

FEATURE_GROUP_ORDER = [
    "feature_set_C_closest",
    "compact_stable_features",
    "feature_set_C_closest_plus_index_context",
    "feature_set_C_closest_plus_relative_strength",
    "feature_set_C_closest_plus_volatility_regime",
    "feature_set_C_closest_plus_volume_shock",
    "compact_stable_plus_index_context",
    "market_context_features",
    "low_noise_features",
    "all_stable_features",
    "regime_interaction_features",
]

MODEL_FAMILIES = [
    "logistic_regression",
    "elasticnet_logistic",
    "calibrated_logistic",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
]

MODEL_COMPLEXITY = {
    "logistic_regression": 1,
    "elasticnet_logistic": 1,
    "calibrated_logistic": 2,
    "random_forest": 3,
    "extra_trees": 3,
    "xgboost": 4,
    "lightgbm": 4,
    "hist_gradient_boosting": 4,
    "soft_vote_ensemble": 5,
    "regime_gated_ensemble": 5,
    "historical_replay": 3,
}


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def normalize_feature_columns(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return sorted({col for col in cols if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])})


def build_v3_feature_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    features, prior_groups, prior_manifest, prior_audit, prior_index_manifest = build_resurrection_feature_frame()
    out = features.copy()

    if "index_agreement_score" in out.columns:
        out["cross_index_agreement_score"] = pd.to_numeric(out["index_agreement_score"], errors="coerce")
    if "cross_index_breadth_proxy" in out.columns:
        out["index_breadth_proxy"] = pd.to_numeric(out["cross_index_breadth_proxy"], errors="coerce")
    if "market_direction_regime_code" in out.columns:
        out["market_regime"] = pd.to_numeric(out["market_direction_regime_code"], errors="coerce")

    vol5 = pd.to_numeric(out.get("market_volatility_5", pd.Series(np.nan, index=out.index)), errors="coerce")
    vol20 = pd.to_numeric(out.get("market_volatility_20", pd.Series(np.nan, index=out.index)), errors="coerce")
    vol_ratio = vol5 / vol20.replace(0.0, np.nan)
    out["high_volatility_flag"] = (vol_ratio > 1.10).astype(float)
    out.loc[vol_ratio.isna(), "high_volatility_flag"] = np.nan
    out["low_volatility_flag"] = (vol_ratio < 0.90).astype(float)
    out.loc[vol_ratio.isna(), "low_volatility_flag"] = np.nan
    market_momentum_20 = pd.to_numeric(out.get("market_momentum_20", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["sideway_flag"] = (market_momentum_20.abs() <= 0.02).astype(float)
    out.loc[market_momentum_20.isna(), "sideway_flag"] = np.nan

    interaction_specs = [
        ("return_1_lag_1", "market_return_lag1", "stock_lag1_x_market_return_lag1"),
        ("return_1_lag_5", "market_return_lag5", "stock_lag5_x_market_return_lag5"),
        ("rolling_return_vol_20", "market_volatility_20", "stock_vol20_x_market_vol20"),
        ("momentum_20", "market_momentum_20", "stock_mom20_x_market_mom20"),
        ("relative_strength_vs_market_lag1", "risk_on_risk_off_state", "rel_strength_x_risk_state"),
    ]
    interaction_cols: list[str] = []
    for left, right, name in interaction_specs:
        if left in out.columns and right in out.columns:
            out[name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
            interaction_cols.append(name)

    v3_required_context = [
        "market_direction_lag1",
        "market_direction_lag5",
        "market_return_lag1",
        "market_return_lag5",
        "market_momentum_5",
        "market_momentum_20",
        "market_volatility_5",
        "market_volatility_20",
        "cross_index_agreement_score",
        "index_breadth_proxy",
        "risk_on_risk_off_state",
        "market_regime",
        "high_volatility_flag",
        "low_volatility_flag",
        "sideway_flag",
    ]
    context_cols = normalize_feature_columns(
        out,
        [
            *v3_required_context,
            *[col for col in out.columns if "_ctx_" in col],
            "market_momentum_60",
            "market_direction_regime_code",
            "market_volatility_regime_code",
            "cross_index_breadth_proxy",
            "index_agreement_score",
        ],
    )
    base_cols = normalize_feature_columns(out, prior_groups.get("feature_set_C_closest", []))
    compact_cols = normalize_feature_columns(out, prior_groups.get("compact_stable_features", []))
    relative_cols = normalize_feature_columns(out, [col for col in out.columns if col.startswith("relative_strength")])
    volatility_cols = normalize_feature_columns(
        out,
        [
            "market_volatility_5",
            "market_volatility_20",
            "market_volatility_regime_code",
            "high_volatility_flag",
            "low_volatility_flag",
            "sideway_flag",
            *[col for col in out.columns if col.startswith("stock_volatility_") or col.startswith("stock_atr_proxy_")],
        ],
    )
    volume_cols = normalize_feature_columns(out, [col for col in out.columns if "volume" in col and ("shock" in col or "zscore" in col or "change" in col)])
    low_noise_cols = normalize_feature_columns(
        out,
        [
            "return_1_lag_1",
            "return_1_lag_2",
            "return_1_lag_3",
            "return_1_lag_5",
            "rolling_return_mean_5",
            "rolling_return_vol_5",
            "rolling_return_mean_20",
            "rolling_return_vol_20",
            "momentum_5",
            "momentum_20",
            "market_return_lag1",
            "market_return_lag5",
            "market_momentum_5",
            "market_momentum_20",
            "market_volatility_20",
            "cross_index_agreement_score",
            "index_breadth_proxy",
            "risk_on_risk_off_state",
            "high_volatility_flag",
            "low_volatility_flag",
            "sideway_flag",
            "day_of_week",
            "hour",
        ],
    )
    all_stable_cols = normalize_feature_columns(out, [*base_cols, *compact_cols, *context_cols, *relative_cols, *volatility_cols, *volume_cols])
    regime_cols = normalize_feature_columns(out, [*base_cols, *context_cols, *interaction_cols])
    feature_groups = {
        "feature_set_C_closest": base_cols,
        "compact_stable_features": compact_cols,
        "feature_set_C_closest_plus_index_context": normalize_feature_columns(out, [*base_cols, *context_cols]),
        "feature_set_C_closest_plus_relative_strength": normalize_feature_columns(out, [*base_cols, *relative_cols]),
        "feature_set_C_closest_plus_volatility_regime": normalize_feature_columns(out, [*base_cols, *volatility_cols]),
        "feature_set_C_closest_plus_volume_shock": normalize_feature_columns(out, [*base_cols, *volume_cols]),
        "compact_stable_plus_index_context": normalize_feature_columns(out, [*compact_cols, *context_cols]),
        "market_context_features": context_cols,
        "low_noise_features": low_noise_cols,
        "all_stable_features": all_stable_cols,
        "regime_interaction_features": regime_cols,
    }
    all_cols = sorted({col for cols in feature_groups.values() for col in cols})
    out[all_cols] = out[all_cols].replace([np.inf, -np.inf], np.nan)
    out["feature_timestamp"] = out["datetime"]

    audit_rows: list[dict[str, Any]] = []
    prior_lookup = prior_audit.set_index("feature_name").to_dict("index") if not prior_audit.empty and "feature_name" in prior_audit.columns else {}
    for col in context_cols:
        prior = prior_lookup.get(col, {})
        audit_rows.append(
            {
                "feature_name": col,
                "source_indices": prior.get("source_indices", ",".join(prior_index_manifest.get("loaded_index_codes", []))),
                "lag_rule": prior.get(
                    "lag_rule",
                    "derived from lagged index features already shifted at least one source bar before stock merge",
                ),
                "feature_timestamp_safe": True,
                "uses_future_index_label": False,
                "uses_stock_final_label": False,
                "non_null_rows": int(out[col].notna().sum()),
                "nan_rate": float(out[col].isna().mean()),
                "first_timestamp": str(out["datetime"].min()),
                "last_timestamp": str(out["datetime"].max()),
            }
        )
    index_manifest = {
        "created_at_utc": now_utc(),
        "loaded_index_codes": prior_index_manifest.get("loaded_index_codes", []),
        "required_context_features": v3_required_context,
        "context_feature_count": len(context_cols),
        "point_in_time_rule": "all market-state features are lagged, shifted rolling-window values, or aliases/interactions of those lagged values before stock-row merge",
        "future_index_features_used": False,
        "stock_final_labels_used": False,
        "aliases_added": {
            "cross_index_agreement_score": "index_agreement_score",
            "index_breadth_proxy": "cross_index_breadth_proxy",
            "market_regime": "market_direction_regime_code",
        },
    }
    feature_manifest = {
        "stock_ticker_count": len(active_stock_tickers()),
        "stock_rows": int(len(out)),
        "feature_groups": {name: {"feature_count": len(cols), "columns": cols} for name, cols in feature_groups.items()},
        "index_context_manifest": index_manifest,
        "target_variants": TARGET_VARIANTS,
        "feature_timestamp_column": "feature_timestamp",
        "features_are_point_in_time_or_lagged": True,
        "final_performance_used_for_selection": False,
    }
    return out, feature_groups, pd.DataFrame(audit_rows), index_manifest, feature_manifest


def stock_forward_return_and_timestamp(features: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    returns: list[pd.Series] = []
    timestamps: list[pd.Series] = []
    for _ticker, group in features.groupby("ticker", sort=True):
        future_close = pd.to_numeric(group["close"], errors="coerce").shift(-horizon)
        current_close = pd.to_numeric(group["close"], errors="coerce")
        future_datetime = group["datetime"].shift(-horizon)
        forward_return = future_close / current_close.replace(0.0, np.nan) - 1.0
        forward_return.loc[future_close.isna() | future_datetime.isna()] = np.nan
        returns.append(pd.Series(forward_return.to_numpy(dtype=float), index=group.index))
        timestamps.append(pd.Series(future_datetime.to_numpy(), index=group.index))
    ret = pd.concat(returns).sort_index() if returns else pd.Series(dtype=float)
    target_ts = pd.concat(timestamps).sort_index() if timestamps else pd.Series(dtype="datetime64[ns]")
    return ret, pd.to_datetime(target_ts, errors="coerce")


def index_forward_return(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], code: str, target_ts: pd.Series) -> pd.Series:
    if code not in index_data:
        return pd.Series(np.nan, index=features.index, dtype=float)
    frame = index_data[code].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    lookup = frame.dropna(subset=["datetime", "close"]).drop_duplicates("datetime", keep="last").sort_values("datetime")[["datetime", "close"]]
    if lookup.empty:
        return pd.Series(np.nan, index=features.index, dtype=float)

    def asof_close(timestamps: pd.Series) -> pd.Series:
        query = pd.DataFrame({"_row_index": features.index, "_timestamp": pd.to_datetime(timestamps, errors="coerce")})
        query = query.dropna(subset=["_timestamp"]).sort_values("_timestamp")
        if query.empty:
            return pd.Series(np.nan, index=features.index, dtype=float)
        merged = pd.merge_asof(query, lookup, left_on="_timestamp", right_on="datetime", direction="backward")
        return merged.set_index("_row_index")["close"].reindex(features.index).astype(float)

    current = asof_close(features["datetime"])
    future = asof_close(target_ts)
    return future / current.replace(0.0, np.nan) - 1.0


def build_target_labels(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], horizon: int, target_variant: str) -> pd.Series:
    stock_return, target_ts = stock_forward_return_and_timestamp(features, horizon)
    if target_variant == "absolute_direction":
        labels = (stock_return > 0.0).astype(float)
    elif target_variant == "market_relative_vn30":
        market_return = index_forward_return(features, index_data, "VN30", target_ts)
        labels = (stock_return > market_return).astype(float)
        labels.loc[market_return.isna()] = np.nan
    elif target_variant == "market_relative_vnindex":
        market_return = index_forward_return(features, index_data, "VNINDEX", target_ts)
        labels = (stock_return > market_return).astype(float)
        labels.loc[market_return.isna()] = np.nan
    elif target_variant == "excess_return_direction":
        market_return = index_forward_return(features, index_data, "VNINDEX", target_ts)
        excess = stock_return - market_return
        labels = (excess > 0.0).astype(float)
        labels.loc[market_return.isna()] = np.nan
    elif target_variant == "thresholded_direction":
        labels = (stock_return > RETURN_THRESHOLD).astype(float)
    elif target_variant == "neutral_removed_direction":
        labels = (stock_return > 0.0).astype(float)
        labels.loc[stock_return.abs() <= RETURN_THRESHOLD] = np.nan
    else:
        raise ValueError(f"unknown target variant: {target_variant}")
    labels.loc[stock_return.isna() | target_ts.isna()] = np.nan
    labels = pd.Series(labels.to_numpy(dtype=float), index=features.index)
    labels.attrs["target_timestamp"] = target_ts.reindex(features.index)
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["target_variant"] = target_variant
    labels.attrs["target_definition"] = target_variant_definition(target_variant)
    labels.attrs["label_cutoff_rule"] = "split rows require feature_timestamp and target_timestamp inside split boundaries"
    return labels


def target_variant_definition(target_variant: str) -> str:
    definitions = {
        "absolute_direction": "stock_forward_return > 0",
        "market_relative_vn30": "stock_forward_return > VN30_forward_return",
        "market_relative_vnindex": "stock_forward_return > VNINDEX_forward_return",
        "excess_return_direction": "stock_forward_return - VNINDEX_forward_return > 0",
        "thresholded_direction": f"stock_forward_return > {RETURN_THRESHOLD}",
        "neutral_removed_direction": f"stock_forward_return > 0 after excluding abs(stock_forward_return) <= {RETURN_THRESHOLD}",
    }
    return definitions[target_variant]


def baseline_frames_for_split(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    split: str,
    horizon: int,
    target_variant: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    idx = splits[split]
    train_y = labels.reindex(splits["train"]).dropna().astype(int)
    y_true = labels.reindex(idx).dropna().astype(int)
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    base_cols = features.loc[idx, ["datetime", "ticker"]].copy()
    for baseline_name, pred in baseline_predictions(features, labels, idx, train_y).items():
        frame = base_cols.copy()
        frame["split"] = split
        frame["target_variant"] = target_variant
        frame["horizon"] = horizon
        frame["baseline_name"] = baseline_name
        frame["y_true"] = y_true.to_numpy(dtype=int)
        frame["y_pred"] = pred
        frame["correct"] = (frame["y_true"].to_numpy(dtype=int) == frame["y_pred"].to_numpy(dtype=int)).astype(int)
        frames[baseline_name] = frame
        rows.append(
            {
                "split": split,
                "target_variant": target_variant,
                "horizon": horizon,
                "baseline_name": baseline_name,
                "accuracy": float(frame["correct"].mean()) if len(frame) else math.nan,
                "rows": int(len(frame)),
                "selection_role": "validation_baseline" if split == "validation" else "post_lock_or_exploratory_final_baseline",
            }
        )
    strongest = max(rows, key=lambda row: float(row["accuracy"])) if rows else {"baseline_name": "", "accuracy": math.nan}
    return pd.DataFrame(rows), strongest, frames


def target_audit_row(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index], horizon: int, target_variant: str) -> dict[str, Any]:
    target_ts = target_timestamp_from_labels(labels)
    return {
        "target_variant": target_variant,
        "horizon": horizon,
        "definition": target_variant_definition(target_variant),
        "return_threshold": RETURN_THRESHOLD if target_variant in {"thresholded_direction", "neutral_removed_direction"} else "",
        "rows_total": int(labels.notna().sum()),
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "overall_positive_rate": float(labels.dropna().mean()) if labels.notna().any() else math.nan,
        "validation_positive_rate": float(labels.reindex(splits["validation"]).dropna().mean()) if len(splits["validation"]) else math.nan,
        "final_positive_rate": float(labels.reindex(splits["final"]).dropna().mean()) if len(splits["final"]) else math.nan,
        "feature_timestamp_min": str(features["datetime"].min()),
        "feature_timestamp_max": str(features["datetime"].max()),
        "target_timestamp_min": str(target_ts.dropna().min()) if target_ts.notna().any() else "",
        "target_timestamp_max": str(target_ts.dropna().max()) if target_ts.notna().any() else "",
        "split_guard_passed": True,
        "mixed_with_other_targets": False,
    }


def build_label_and_baseline_cache(
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
) -> tuple[
    dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]],
    dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]],
    pd.DataFrame,
    pd.DataFrame,
]:
    label_cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]] = {}
    baseline_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]] = {}
    audit_rows: list[dict[str, Any]] = []
    baseline_rows: list[pd.DataFrame] = []
    for target_variant in TARGET_VARIANTS:
        for horizon in HORIZONS:
            labels = build_target_labels(features, index_data, horizon, target_variant)
            splits = split_indices(features, labels)
            label_cache[(target_variant, horizon)] = (labels, splits)
            audit_rows.append(target_audit_row(features, labels, splits, horizon, target_variant))
            for split in ("validation", "final"):
                baseline_df, strongest, frames = baseline_frames_for_split(features, labels, splits, split, horizon, target_variant)
                baseline_rows.append(baseline_df)
                baseline_cache[(target_variant, horizon, split)] = (baseline_df, strongest, frames)
    return (
        label_cache,
        baseline_cache,
        pd.DataFrame(audit_rows),
        pd.concat(baseline_rows, ignore_index=True) if baseline_rows else pd.DataFrame(),
    )


def logistic_full_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c_values = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
    for penalty, c_value, class_weight, solver in itertools.product(["l1", "l2"], c_values, [None, "balanced"], ["liblinear", "saga"]):
        rows.append({"model_family": "logistic_regression", "penalty": penalty, "C": c_value, "class_weight": class_weight, "solver": solver, "l1_ratio": None})
    for c_value, class_weight in itertools.product(c_values, [None, "balanced"]):
        rows.append({"model_family": "elasticnet_logistic", "penalty": "elasticnet", "C": c_value, "class_weight": class_weight, "solver": "saga", "l1_ratio": 0.5})
    return rows


def tree_full_grid(model_family: str) -> list[dict[str, Any]]:
    return [
        {
            "model_family": model_family,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features, class_weight in itertools.product(
            [200, 500, 800],
            [3, 5, 8, None],
            [5, 10, 20, 50, 100],
            ["sqrt", "log2", 0.5],
            [None, "balanced"],
        )
    ]


def xgb_full_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "xgboost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
        }
        for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_lambda, min_child_weight in itertools.product(
            [100, 200, 400, 600],
            [2, 3, 4],
            [0.005, 0.01, 0.03, 0.05],
            [0.7, 0.85, 1.0],
            [0.7, 0.85, 1.0],
            [1, 5, 10, 20, 50],
            [5, 10, 20, 50],
        )
    ]


def lgbm_full_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "lightgbm",
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
        }
        for n_estimators, num_leaves, learning_rate, min_child_samples, subsample, colsample_bytree, reg_lambda in itertools.product(
            [100, 200, 400, 600],
            [7, 15, 31, 63],
            [0.005, 0.01, 0.03, 0.05],
            [20, 30, 50, 100],
            [0.7, 0.85, 1.0],
            [0.7, 0.85, 1.0],
            [1, 5, 10, 20, 50],
        )
    ]


def hist_full_grid() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "hist_gradient_boosting",
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
        }
        for max_iter, max_leaf_nodes, learning_rate, l2_regularization in itertools.product(
            [100, 200, 400],
            [7, 15, 31],
            [0.005, 0.01, 0.03, 0.05],
            [0, 1, 5, 10],
        )
    ]


def full_param_grids() -> dict[str, list[dict[str, Any]]]:
    logistic_rows = logistic_full_grid()
    return {
        "logistic_regression": [row for row in logistic_rows if row["model_family"] == "logistic_regression"],
        "elasticnet_logistic": [row for row in logistic_rows if row["model_family"] == "elasticnet_logistic"],
        "calibrated_logistic": [
            {"model_family": "calibrated_logistic", "penalty": "l2", "C": c_value, "class_weight": class_weight, "solver": "liblinear", "calibration": "sigmoid_cv3"}
            for c_value, class_weight in itertools.product([0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10], [None, "balanced"])
        ],
        "random_forest": tree_full_grid("random_forest"),
        "extra_trees": tree_full_grid("extra_trees"),
        "xgboost": xgb_full_grid(),
        "lightgbm": lgbm_full_grid(),
        "hist_gradient_boosting": hist_full_grid(),
    }


def compact_candidate_grid(feature_groups: dict[str, list[str]], param_grids: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    threshold_count = len(THRESHOLDS)
    for target_variant, feature_group, horizon, model_family in itertools.product(TARGET_VARIANTS, FEATURE_GROUP_ORDER, HORIZONS, MODEL_FAMILIES):
        if feature_group not in feature_groups:
            continue
        params = param_grids.get(model_family, [])
        param_summary: dict[str, Any] = {}
        for param in params:
            for key, value in param.items():
                param_summary.setdefault(key, set()).add(json.dumps(json_safe(value), sort_keys=True))
        param_summary = {
            key: [json.loads(item) for item in sorted(values)]
            for key, values in sorted(param_summary.items())
        }
        rows.append(
            {
                "row_type": "full_grid_spec",
                "grid_id": f"spec__{target_variant}__{feature_group}__h{horizon}__{model_family}",
                "source": "grid_spec",
                "model_family": model_family,
                "model_params": "",
                "model_params_grid_json": compact_json({"parameter_ranges": param_summary}),
                "target_variant": target_variant,
                "feature_group": feature_group,
                "horizon": horizon,
                "threshold_grid_start": THRESHOLDS[0],
                "threshold_grid_end": THRESHOLDS[-1],
                "threshold_grid_step": 0.005,
                "parameter_combo_count": len(params),
                "threshold_combo_count": threshold_count,
                "expanded_candidate_count": len(params) * threshold_count,
                "selected_for_fit": False,
                "stage": "full_grid_enumerated_as_parameter_spec",
            }
        )
    theoretical = int(sum(int(row["expanded_candidate_count"]) for row in rows))
    return pd.DataFrame(rows), theoretical


def selected_param_templates() -> dict[str, list[dict[str, Any]]]:
    return {
        "logistic_regression": [
            {"model_family": "logistic_regression", "penalty": penalty, "C": c, "class_weight": class_weight, "solver": "liblinear", "l1_ratio": None}
            for penalty, c, class_weight in itertools.product(["l1", "l2"], [0.001, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 1], [None, "balanced"])
        ],
        "elasticnet_logistic": [
            {"model_family": "elasticnet_logistic", "penalty": "elasticnet", "C": c, "class_weight": class_weight, "solver": "saga", "l1_ratio": l1_ratio}
            for c, class_weight, l1_ratio in itertools.product([0.001, 0.003, 0.01, 0.03, 0.1, 0.3], [None, "balanced"], [0.25, 0.5, 0.75])
        ],
        "calibrated_logistic": [
            {"model_family": "calibrated_logistic", "penalty": "l2", "C": c, "class_weight": class_weight, "solver": "liblinear", "calibration": "sigmoid_cv3"}
            for c, class_weight in itertools.product([0.003, 0.01, 0.03, 0.1, 0.3, 1], [None, "balanced"])
        ],
        "random_forest": [
            {"model_family": "random_forest", "n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt", "class_weight": class_weight}
            for n, depth, leaf, class_weight in itertools.product([200, 500], [3, 5, 8], [10, 50, 100], [None, "balanced"])
        ],
        "extra_trees": [
            {"model_family": "extra_trees", "n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": "sqrt", "class_weight": class_weight}
            for n, depth, leaf, class_weight in itertools.product([200, 500], [3, 5, 8], [10, 50, 100], [None, "balanced"])
        ],
        "xgboost": [
            {
                "model_family": "xgboost",
                "n_estimators": n,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": reg,
                "min_child_weight": child,
            }
            for n, depth, lr, reg, child in itertools.product([100, 200], [2, 3], [0.01, 0.03], [5, 20], [10, 20])
        ],
        "lightgbm": [
            {
                "model_family": "lightgbm",
                "n_estimators": n,
                "num_leaves": leaves,
                "learning_rate": lr,
                "min_child_samples": child,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": reg,
            }
            for n, leaves, lr, child, reg in itertools.product([100, 200], [7, 15], [0.01, 0.03], [30, 100], [5, 20])
        ],
        "hist_gradient_boosting": [
            {"model_family": "hist_gradient_boosting", "max_iter": max_iter, "max_leaf_nodes": leaves, "learning_rate": lr, "l2_regularization": l2}
            for max_iter, leaves, lr, l2 in itertools.product([100, 200], [7, 15], [0.01, 0.03], [0, 5])
        ],
    }


def round_robin_budget(rows: list[dict[str, Any]], budgets: dict[str, int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    forced = [row for row in rows if row.get("forced")]
    selected.extend(forced)
    forced_keys = {row["grid_id"] for row in forced}

    def pick_rows(family_rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        if not family_rows or budget <= 0:
            return []
        by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in family_rows:
            by_group.setdefault((row["feature_group"], int(row["horizon"])), []).append(row)
        groups = [items for _key, items in sorted(by_group.items())]
        picked: list[dict[str, Any]] = []
        cursor = 0
        while len(picked) < budget and groups:
            next_groups: list[list[dict[str, Any]]] = []
            for group in groups:
                if cursor < len(group) and len(picked) < budget:
                    picked.append(group[cursor])
                if cursor + 1 < len(group):
                    next_groups.append(group)
            groups = next_groups
            cursor += 1
        return picked

    for model_family, budget in budgets.items():
        family_rows = [row for row in rows if row["model_family"] == model_family and row["grid_id"] not in forced_keys]
        if not family_rows or budget <= 0:
            continue
        forced_count = len([row for row in forced if row["model_family"] == model_family])
        remaining_budget = max(0, budget - forced_count)
        per_target = remaining_budget // len(TARGET_VARIANTS)
        remainder = remaining_budget % len(TARGET_VARIANTS)
        for target_idx, target_variant in enumerate(TARGET_VARIANTS):
            target_budget = per_target + (1 if target_idx < remainder else 0)
            target_rows = [row for row in family_rows if row["target_variant"] == target_variant]
            selected.extend(pick_rows(target_rows, target_budget))
    dedup: dict[str, dict[str, Any]] = {}
    for row in selected:
        dedup[row["grid_id"]] = row
    return list(dedup.values())


def build_fit_grid(feature_groups: dict[str, list[str]], budgets: dict[str, int]) -> pd.DataFrame:
    templates = selected_param_templates()
    priority_features = [
        "compact_stable_features",
        "feature_set_C_closest",
        "low_noise_features",
        "compact_stable_plus_index_context",
        "feature_set_C_closest_plus_index_context",
        "feature_set_C_closest_plus_relative_strength",
        "regime_interaction_features",
        "market_context_features",
        "all_stable_features",
        "feature_set_C_closest_plus_volatility_regime",
        "feature_set_C_closest_plus_volume_shock",
    ]
    rows: list[dict[str, Any]] = []
    seq = 0
    for model_family, params_list in templates.items():
        for target_variant, feature_group, horizon, params in itertools.product(TARGET_VARIANTS, priority_features, HORIZONS, params_list):
            if feature_group not in feature_groups:
                continue
            seq += 1
            source = "index_context" if "context" in feature_group or "market" in feature_group or "regime" in feature_group else "new_tuning"
            rows.append(
                {
                    "row_type": "selected_fit_candidate",
                    "grid_id": f"v3grid_{seq:06d}",
                    "source": source,
                    "model_family": model_family,
                    "model_params": compact_json(params),
                    "target_variant": target_variant,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "threshold_grid_start": THRESHOLDS[0],
                    "threshold_grid_end": THRESHOLDS[-1],
                    "threshold_grid_step": 0.005,
                    "selected_for_fit": True,
                    "stage": "cheap_validation_screening",
                    "forced": False,
                }
            )
    forced_specs = [
        (
            "forced_current_champion_l2_h40",
            "historical_replay",
            "logistic_regression",
            {"model_family": "logistic_regression", "penalty": "l2", "C": 0.3, "class_weight": "balanced", "solver": "liblinear", "l1_ratio": None},
            "absolute_direction",
            "feature_set_C_closest",
            40,
        ),
        (
            "forced_exploratory_best_l1_compact_h50",
            "historical_replay",
            "logistic_regression",
            {"model_family": "logistic_regression", "penalty": "l1", "C": 0.003, "class_weight": None, "solver": "liblinear", "l1_ratio": None},
            "absolute_direction",
            "compact_stable_features",
            50,
        ),
        (
            "forced_exploratory_best_l1_compact_h40",
            "historical_replay",
            "logistic_regression",
            {"model_family": "logistic_regression", "penalty": "l1", "C": 0.003, "class_weight": None, "solver": "liblinear", "l1_ratio": None},
            "absolute_direction",
            "compact_stable_features",
            40,
        ),
        (
            "forced_locked_v2_lgbm_h20",
            "historical_replay",
            "lightgbm",
            {"model_family": "lightgbm", "n_estimators": 100, "num_leaves": 7, "learning_rate": 0.01, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 1},
            "absolute_direction",
            "feature_set_C_closest",
            20,
        ),
        (
            "forced_v3_calibrated_compact_h40",
            "historical_replay",
            "calibrated_logistic",
            {"model_family": "calibrated_logistic", "penalty": "l2", "C": 0.003, "class_weight": None, "solver": "liblinear", "calibration": "sigmoid_cv3"},
            "absolute_direction",
            "compact_stable_features",
            40,
        ),
        (
            "forced_v3_calibrated_all_stable_h35",
            "historical_replay",
            "calibrated_logistic",
            {"model_family": "calibrated_logistic", "penalty": "l2", "C": 0.003, "class_weight": None, "solver": "liblinear", "calibration": "sigmoid_cv3"},
            "absolute_direction",
            "all_stable_features",
            35,
        ),
        (
            "forced_v3_calibrated_all_stable_h40",
            "historical_replay",
            "calibrated_logistic",
            {"model_family": "calibrated_logistic", "penalty": "l2", "C": 0.003, "class_weight": None, "solver": "liblinear", "calibration": "sigmoid_cv3"},
            "absolute_direction",
            "all_stable_features",
            40,
        ),
    ]
    for grid_id, source, model_family, params, target_variant, feature_group, horizon in forced_specs:
        rows.append(
            {
                "row_type": "selected_fit_candidate",
                "grid_id": grid_id,
                "source": source,
                "model_family": model_family,
                "model_params": compact_json(params),
                "target_variant": target_variant,
                "feature_group": feature_group,
                "horizon": horizon,
                "threshold_grid_start": THRESHOLDS[0],
                "threshold_grid_end": THRESHOLDS[-1],
                "threshold_grid_step": 0.005,
                "selected_for_fit": True,
                "stage": "forced_historical_or_exploratory_replay",
                "forced": True,
            }
        )
    return pd.DataFrame(round_robin_budget(rows, budgets)).reset_index(drop=True)


def make_model(model_family: str, params: dict[str, Any]) -> Pipeline | None:
    if model_family in {"logistic_regression", "elasticnet_logistic"}:
        solver = str(params.get("solver", "liblinear"))
        model_params = {
            "C": float(params["C"]),
            "penalty": params["penalty"],
            "solver": solver,
            "class_weight": params.get("class_weight"),
            "max_iter": 900 if solver == "saga" else 1000,
            "tol": 1e-3 if solver == "saga" else 1e-4,
            "random_state": SEED,
        }
        if params["penalty"] == "elasticnet":
            model_params["l1_ratio"] = float(params.get("l1_ratio", 0.5))
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", LogisticRegression(**model_params))])
    if model_family == "calibrated_logistic":
        base = LogisticRegression(
            C=float(params["C"]),
            penalty="l2",
            solver=str(params.get("solver", "liblinear")),
            class_weight=params.get("class_weight"),
            max_iter=1000,
            random_state=SEED,
        )
        calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", calibrated)])
    if model_family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params.get("class_weight"),
            random_state=SEED,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params.get("class_weight"),
            random_state=SEED,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "xgboost" and XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "lightgbm" and LGBMClassifier is not None:
        model = LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=SEED,
            verbose=-1,
            n_jobs=2,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            learning_rate=float(params["learning_rate"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=SEED,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    return None


def prediction_frame(
    features: pd.DataFrame,
    idx: pd.Index,
    labels: pd.Series,
    probability: np.ndarray,
    threshold: float,
    candidate_id: str,
    split: str,
    target_variant: str,
) -> pd.DataFrame:
    out = features.loc[idx, ["feature_timestamp", "datetime", "ticker"]].copy()
    target_timestamp = target_timestamp_from_labels(labels).reindex(idx)
    out["target_timestamp"] = target_timestamp.to_numpy()
    out["target_variant"] = target_variant
    out["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    out["y_score_or_probability"] = np.asarray(probability, dtype=float)
    out["threshold"] = float(threshold)
    out["y_pred"] = (out["y_score_or_probability"].to_numpy(dtype=float) >= float(threshold)).astype(int)
    out["correct"] = (out["y_true"].to_numpy(dtype=int) == out["y_pred"].to_numpy(dtype=int)).astype(int)
    out["candidate_id"] = candidate_id
    out["split"] = split
    return out.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def candidate_metrics(
    frame: pd.DataFrame,
    strongest_baseline: dict[str, Any],
    strongest_baseline_frame: pd.DataFrame,
    candidate_id: str,
    split: str,
    target_variant: str,
    simplicity_score: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    q, t = detail_rows(frame, strongest_baseline_frame, split, candidate_id)
    q.insert(2, "target_variant", target_variant)
    t.insert(2, "target_variant", target_variant)
    acc = float(frame["correct"].mean()) if len(frame) else math.nan
    baseline_acc = float(strongest_baseline.get("accuracy", math.nan))
    lift = acc - baseline_acc
    quarter_min = float(q["accuracy"].min()) if len(q) else math.nan
    positive_quarters = int((q["lift"] > 0.0).sum()) if len(q) else 0
    ticker_median = float(t["accuracy"].median()) if len(t) else math.nan
    baseline_ticker_median = float(t["baseline_accuracy"].median()) if len(t) else math.nan
    pred_up = float(frame["y_pred"].astype(int).mean()) if len(frame) else math.nan
    pred_balance = max(0.0, 1.0 - abs(pred_up - 0.525) / 0.525) if math.isfinite(pred_up) else 0.0
    quarter_score = max(0.0, min(1.0, quarter_min)) if math.isfinite(quarter_min) else 0.0
    ticker_score = max(0.0, min(1.0, ticker_median)) if math.isfinite(ticker_median) else 0.0
    row_count_score = max(0.0, min(1.0, len(frame) / 3500.0)) if split == "validation" else math.nan
    composite = (
        0.30 * lift
        + 0.25 * acc
        + 0.15 * quarter_score
        + 0.10 * ticker_score
        + 0.10 * pred_balance
        + 0.05 * (row_count_score if math.isfinite(row_count_score) else 0.0)
        + 0.05 * simplicity_score
    )
    shortlist_pass = bool(
        split == "validation"
        and lift > 0.0
        and len(frame) >= 3500
        and math.isfinite(pred_up)
        and 0.35 <= pred_up <= 0.70
        and math.isfinite(quarter_min)
        and quarter_min >= 0.45
        and positive_quarters >= 2
        and math.isfinite(ticker_median)
        and math.isfinite(baseline_ticker_median)
        and ticker_median >= baseline_ticker_median
    )
    metrics = {
        f"{split}_accuracy": acc,
        f"{split}_lift": lift,
        f"{split}_rows": int(len(frame)),
        "strongest_baseline": strongest_baseline.get("baseline_name", ""),
        f"{split}_strongest_baseline_accuracy": baseline_acc,
        "ticker_median_accuracy": ticker_median,
        "baseline_ticker_median_accuracy": baseline_ticker_median,
        "ticker_median_lift": ticker_median - baseline_ticker_median,
        "quarter_min_accuracy": quarter_min,
        "quarters_positive_lift": positive_quarters,
        "rolling250_min": rolling_min(frame),
        "prediction_up_ratio": pred_up,
        "quarterly_stability": quarter_score,
        "ticker_stability": ticker_score,
        "prediction_balance": pred_balance,
        "row_count_score": row_count_score,
        "simplicity_score": simplicity_score,
        "validation_composite_score": composite if split == "validation" else math.nan,
        "shortlist_pass": shortlist_pass,
    }
    balance = {
        "candidate_id": candidate_id,
        "split": split,
        "target_variant": target_variant,
        "rows": int(len(frame)),
        "prediction_up_ratio": pred_up,
        "prediction_down_ratio": 1.0 - pred_up if math.isfinite(pred_up) else math.nan,
        "passes_35_70_band": bool(math.isfinite(pred_up) and 0.35 <= pred_up <= 0.70),
    }
    return metrics, q, t, balance


def validation_result_row(
    candidate_id: str,
    base_id: str,
    source: str,
    model_family: str,
    model_params: str,
    target_variant: str,
    feature_group: str,
    horizon: int,
    threshold: float,
    frame: pd.DataFrame,
    strongest_baseline: dict[str, Any],
    strongest_baseline_frame: pd.DataFrame,
    simplicity_score: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics, q, t, balance = candidate_metrics(
        frame,
        strongest_baseline,
        strongest_baseline_frame,
        candidate_id,
        "validation",
        target_variant,
        simplicity_score,
    )
    row = {
        "candidate_id": candidate_id,
        "base_id": base_id,
        "source": source,
        "model_family": model_family,
        "model_params": model_params,
        "target_variant": target_variant,
        "feature_group": feature_group,
        "horizon": int(horizon),
        "threshold": float(threshold),
        "status": "ok",
        "selection_source": "validation_only",
        "final_accuracy_used_for_selection": False,
        "claim_label": "diagnostic_only" if metrics["validation_lift"] > 0.0 else "not_claimable",
        **metrics,
    }
    return row, q, t, balance


def fit_validation_candidates(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    fit_grid: pd.DataFrame,
    label_cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]],
    baseline_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for n, grid_row in enumerate(fit_grid.to_dict("records"), start=1):
        model_family = str(grid_row["model_family"])
        if model_family == "xgboost" and XGBClassifier is None:
            continue
        if model_family == "lightgbm" and LGBMClassifier is None:
            continue
        target_variant = str(grid_row["target_variant"])
        feature_group = str(grid_row["feature_group"])
        horizon = int(grid_row["horizon"])
        feature_cols = feature_groups.get(feature_group, [])
        labels, splits = label_cache[(target_variant, horizon)]
        train_idx = splits["train"]
        val_idx = splits["validation"]
        train_y = labels.reindex(train_idx).astype(int)
        if len(train_y) < 100 or train_y.nunique() < 2 or not feature_cols or len(val_idx) < 100:
            continue
        params = json.loads(str(grid_row["model_params"]))
        estimator = make_model(model_family, params)
        base_id = str(grid_row["grid_id"])
        if estimator is None:
            continue
        try:
            estimator.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
            val_prob = predict_probability(estimator, clean_feature_matrix(features.loc[val_idx], feature_cols))
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": f"{base_id}__failed",
                    "base_id": base_id,
                    "source": grid_row["source"],
                    "model_family": model_family,
                    "model_params": str(grid_row["model_params"]),
                    "target_variant": target_variant,
                    "feature_group": feature_group,
                    "horizon": horizon,
                    "status": "failed",
                    "failure_reason": str(exc)[:300],
                    "final_accuracy_used_for_selection": False,
                    "claim_label": "not_claimable",
                }
            )
            continue
        payloads[base_id] = {
            "payload_type": "model",
            "base_id": base_id,
            "source": grid_row["source"],
            "model_family": model_family,
            "model_params": str(grid_row["model_params"]),
            "target_variant": target_variant,
            "feature_group": feature_group,
            "horizon": horizon,
            "model_object": estimator,
            "feature_cols": feature_cols,
            "labels": labels,
            "splits": splits,
            "validation_prob": val_prob,
        }
        _baseline_df, strongest, frames = baseline_cache[(target_variant, horizon, "validation")]
        strongest_frame = frames[str(strongest["baseline_name"])]
        simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY.get(model_family, 5)) / 5.0)
        for threshold in THRESHOLDS:
            candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
            val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation", target_variant)
            row, q, t, balance = validation_result_row(
                candidate_id,
                base_id,
                str(grid_row["source"]),
                model_family,
                str(grid_row["model_params"]),
                target_variant,
                feature_group,
                horizon,
                threshold,
                val_frame,
                strongest,
                strongest_frame,
                simplicity,
            )
            rows.append(row)
            quarter_rows.append(q)
            ticker_rows.append(t)
            balance_rows.append(balance)
        if n % 25 == 0:
            print(f"fit_validation_candidates: completed {n}/{len(fit_grid)} base fits", flush=True)
    return (
        pd.DataFrame(rows),
        payloads,
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
    )


def fit_regime_router(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    feature_cols: list[str],
    group_col: str,
) -> dict[str, Any]:
    train_idx = splits["train"]
    train_y = labels.reindex(train_idx).astype(int)
    global_model = make_model(
        "logistic_regression",
        {"model_family": "logistic_regression", "penalty": "l2", "solver": "liblinear", "C": 0.3, "class_weight": "balanced", "l1_ratio": None},
    )
    if global_model is None:
        raise RuntimeError("regime router global logistic unavailable")
    global_model.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
    group_models: dict[str, Any] = {}
    for group_name, group in features.loc[train_idx].groupby(group_col, sort=True):
        group_idx = group.index
        group_y = labels.reindex(group_idx).astype(int)
        if len(group_y) < 100 or group_y.nunique() < 2:
            continue
        local_model = clone(global_model)
        local_model.fit(clean_feature_matrix(features.loc[group_idx], feature_cols), group_y)
        group_models[str(group_name)] = local_model
    return {"global_model": global_model, "group_models": group_models, "group_col": group_col, "feature_cols": feature_cols}


def predict_regime_router(payload: dict[str, Any], features: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    feature_cols = payload["feature_cols"]
    out = predict_probability(payload["global_model"], clean_feature_matrix(features.loc[idx], feature_cols))
    groups = features.loc[idx, payload["group_col"]].astype(str)
    for group_name, model in payload["group_models"].items():
        mask = groups.eq(str(group_name)).to_numpy()
        if mask.any():
            local_idx = idx[mask]
            out[mask] = predict_probability(model, clean_feature_matrix(features.loc[local_idx], feature_cols))
    return np.clip(out, 0.0, 1.0)


def add_regime_gate_candidates(
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    label_cache: dict[tuple[str, int], tuple[pd.Series, dict[str, pd.Index]]],
    baseline_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]],
    payloads: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    new_payloads: dict[str, dict[str, Any]] = {}
    for target_variant in TARGET_VARIANTS:
        for feature_group in ["feature_set_C_closest", "compact_stable_features", "feature_set_C_closest_plus_index_context"]:
            feature_cols = feature_groups.get(feature_group, [])
            if not feature_cols:
                continue
            for horizon in [40, 50, 60]:
                labels, splits = label_cache[(target_variant, horizon)]
                if len(splits["train"]) < 100 or labels.reindex(splits["train"]).nunique() < 2:
                    continue
                for group_col in ["market_direction_regime", "volatility_regime"]:
                    if group_col not in features.columns:
                        continue
                    base_id = f"regime_gate__{target_variant}__{group_col}__{feature_group}__h{horizon}"
                    try:
                        router = fit_regime_router(features, labels, splits, feature_cols, group_col)
                        val_idx = splits["validation"]
                        val_prob = predict_regime_router(router, features, val_idx)
                    except Exception:
                        continue
                    params = compact_json({"group_col": group_col, "base_model": "logistic_l2_C0.3_balanced"})
                    new_payloads[base_id] = {
                        "payload_type": "regime_gate",
                        "base_id": base_id,
                        "source": "regime_gate",
                        "model_family": "regime_gated_ensemble",
                        "model_params": params,
                        "target_variant": target_variant,
                        "feature_group": feature_group,
                        "horizon": horizon,
                        "router": router,
                        "feature_cols": feature_cols,
                        "labels": labels,
                        "splits": splits,
                        "validation_prob": val_prob,
                    }
                    _baseline_df, strongest, frames = baseline_cache[(target_variant, horizon, "validation")]
                    strongest_frame = frames[str(strongest["baseline_name"])]
                    simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY["regime_gated_ensemble"]) / 5.0)
                    for threshold in THRESHOLDS:
                        candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
                        val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation", target_variant)
                        row, q, t, balance = validation_result_row(
                            candidate_id,
                            base_id,
                            "regime_gate",
                            "regime_gated_ensemble",
                            params,
                            target_variant,
                            feature_group,
                            horizon,
                            threshold,
                            val_frame,
                            strongest,
                            strongest_frame,
                            simplicity,
                        )
                        rows.append(row)
                        quarter_rows.append(q)
                        ticker_rows.append(t)
                        balance_rows.append(balance)
    payloads.update(new_payloads)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
        new_payloads,
    )


def add_soft_vote_candidates(
    features: pd.DataFrame,
    validation_results: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    baseline_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    base_results = validation_results[
        validation_results["status"].eq("ok")
        & validation_results["model_family"].isin(["logistic_regression", "elasticnet_logistic", "random_forest", "extra_trees", "xgboost", "lightgbm", "hist_gradient_boosting"])
    ].copy()
    if base_results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    base_results = base_results.sort_values(
        by=["target_variant", "feature_group", "horizon", "model_family", "validation_composite_score", "validation_accuracy"],
        ascending=[True, True, True, True, False, False],
    )
    top = base_results.groupby(["target_variant", "feature_group", "horizon", "model_family"], as_index=False).head(1)
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    new_payloads: dict[str, dict[str, Any]] = {}
    combos = [
        ("logistic_tree", ["logistic_regression", "random_forest"]),
        ("logistic_extra", ["logistic_regression", "extra_trees"]),
        ("logistic_lightgbm", ["logistic_regression", "lightgbm"]),
        ("boosting_pair", ["xgboost", "lightgbm"]),
    ]
    for (target_variant, feature_group, horizon), subset in top.groupby(["target_variant", "feature_group", "horizon"], sort=True):
        if feature_group not in {"feature_set_C_closest", "compact_stable_features", "feature_set_C_closest_plus_index_context", "low_noise_features"}:
            continue
        by_model = {str(row["model_family"]): row for _, row in subset.iterrows()}
        for combo_name, model_names in combos:
            if not all(name in by_model for name in model_names):
                continue
            base_rows = [by_model[name] for name in model_names]
            base_payloads = [payloads[str(row["base_id"])] for row in base_rows if str(row["base_id"]) in payloads]
            if len(base_payloads) != len(model_names):
                continue
            first = base_payloads[0]
            labels = first["labels"]
            splits = first["splits"]
            val_idx = splits["validation"]
            if not all(payload["splits"]["validation"].equals(val_idx) for payload in base_payloads):
                continue
            weights = np.asarray([max(float(row["validation_lift"]), 0.0001) for row in base_rows], dtype=float)
            weights = weights / weights.sum()
            base_id = f"soft_vote__{target_variant}__{combo_name}__{feature_group}__h{horizon}__validation_weighted"
            val_prob = np.zeros(len(val_idx), dtype=float)
            for weight, payload in zip(weights, base_payloads):
                val_prob += float(weight) * np.asarray(payload["validation_prob"], dtype=float)
            params = compact_json({"base_model_ids": [str(row["base_id"]) for row in base_rows], "weights": {name: float(weight) for name, weight in zip(model_names, weights)}})
            new_payloads[base_id] = {
                "payload_type": "soft_vote",
                "base_id": base_id,
                "source": "ensemble",
                "model_family": "soft_vote_ensemble",
                "model_params": params,
                "target_variant": target_variant,
                "feature_group": feature_group,
                "horizon": int(horizon),
                "labels": labels,
                "splits": splits,
                "base_model_ids": [str(row["base_id"]) for row in base_rows],
                "weights": weights,
                "validation_prob": val_prob,
            }
            _baseline_df, strongest, frames = baseline_cache[(target_variant, int(horizon), "validation")]
            strongest_frame = frames[str(strongest["baseline_name"])]
            simplicity = max(0.0, (6.0 - MODEL_COMPLEXITY["soft_vote_ensemble"]) / 5.0)
            for threshold in THRESHOLDS:
                candidate_id = f"{base_id}__t{threshold:.3f}".replace(".", "p")
                val_frame = prediction_frame(features, val_idx, labels, val_prob, threshold, candidate_id, "validation", target_variant)
                row, q, t, balance = validation_result_row(
                    candidate_id,
                    base_id,
                    "ensemble",
                    "soft_vote_ensemble",
                    params,
                    target_variant,
                    feature_group,
                    int(horizon),
                    threshold,
                    val_frame,
                    strongest,
                    strongest_frame,
                    simplicity,
                )
                rows.append(row)
                quarter_rows.append(q)
                ticker_rows.append(t)
                balance_rows.append(balance)
    payloads.update(new_payloads)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
        new_payloads,
    )


def final_probability_for_payload(payload: dict[str, Any], features: pd.DataFrame, payloads: dict[str, dict[str, Any]], cache: dict[str, np.ndarray]) -> np.ndarray:
    base_id = payload["base_id"]
    if base_id in cache:
        return cache[base_id]
    final_idx = payload["splits"]["final"]
    if payload["payload_type"] == "model":
        out = predict_probability(payload["model_object"], clean_feature_matrix(features.loc[final_idx], payload["feature_cols"]))
    elif payload["payload_type"] == "regime_gate":
        out = predict_regime_router(payload["router"], features, final_idx)
    elif payload["payload_type"] == "soft_vote":
        out = np.zeros(len(final_idx), dtype=float)
        for weight, child_id in zip(payload["weights"], payload["base_model_ids"]):
            out += float(weight) * final_probability_for_payload(payloads[child_id], features, payloads, cache)
    else:
        raise ValueError(f"unknown payload type {payload['payload_type']}")
    cache[base_id] = out
    return out


def final_claim_label(row: dict[str, Any], validation_governed: bool, leaderboard_role: str) -> str:
    final_accuracy = as_float(row.get("final_accuracy"))
    final_lift = as_float(row.get("final_lift"))
    if leaderboard_role == "exploratory_final_rank" and not validation_governed:
        return "exploratory_not_claimable"
    if not validation_governed:
        return "future_blind_required" if final_accuracy > CLAIMABLE_CHAMPION["final_accuracy"] else "exploratory_not_claimable"
    if final_lift <= 0.0:
        return "not_claimable"
    if final_accuracy >= 0.62:
        return "target62_candidate"
    if final_accuracy >= 0.60:
        return "baseline60_candidate"
    return "diagnostic_only"


def reason_not_claimable(row: dict[str, Any], validation_governed: bool, leaderboard_role: str) -> str:
    if not validation_governed:
        return "final-ranked candidate is exploratory and requires re-lock/future-blind confirmation"
    reasons: list[str] = []
    if as_float(row.get("final_lift")) <= 0:
        reasons.append("non-positive final lift versus strongest same-horizon baseline")
    if as_float(row.get("final_accuracy")) <= CLAIMABLE_CHAMPION["final_accuracy"]:
        reasons.append("does not beat 61.61% claimable champion")
    if as_float(row.get("final_lift")) <= CLAIMABLE_CHAMPION["final_lift"]:
        reasons.append("does not beat +10.90pp claimable-champion lift")
    if row.get("target_variant") != "absolute_direction":
        reasons.append("target variant is separate and cannot be mixed with absolute-direction claim")
    if leaderboard_role != "validation_governed":
        reasons.append("not in validation-governed leaderboard")
    return "; ".join(reasons)


def evaluate_final_candidates(
    features: pd.DataFrame,
    validation_results: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    baseline_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[pd.DataFrame] = []
    ticker_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    final_prob_cache: dict[str, np.ndarray] = {}
    valid = validation_results[validation_results["status"].eq("ok")].copy()
    for val_row in valid.to_dict("records"):
        base_id = str(val_row["base_id"])
        if base_id not in payloads:
            continue
        payload = payloads[base_id]
        final_idx = payload["splits"]["final"]
        labels = payload["labels"]
        target_variant = str(payload["target_variant"])
        horizon = int(payload["horizon"])
        final_prob = final_probability_for_payload(payload, features, payloads, final_prob_cache)
        threshold = float(val_row["threshold"])
        candidate_id = str(val_row["candidate_id"])
        final_frame = prediction_frame(features, final_idx, labels, final_prob, threshold, candidate_id, "final", target_variant)
        _baseline_df, strongest, frames = baseline_cache[(target_variant, horizon, "final")]
        strongest_frame = frames[str(strongest["baseline_name"])]
        simplicity = float(val_row.get("simplicity_score", 0.0))
        metrics, q, t, balance = candidate_metrics(final_frame, strongest, strongest_frame, candidate_id, "final", target_variant, simplicity)
        validation_governed = bool(val_row.get("shortlist_pass", False))
        leaderboard_role = "validation_governed" if validation_governed else "exploratory_final_rank"
        row = {
            "candidate_id": candidate_id,
            "source": val_row["source"],
            "model_family": val_row["model_family"],
            "model_params": val_row["model_params"],
            "target_variant": target_variant,
            "feature_group": val_row["feature_group"],
            "horizon": horizon,
            "threshold": threshold,
            "validation_accuracy": float(val_row["validation_accuracy"]),
            "validation_lift": float(val_row["validation_lift"]),
            "validation_rows": int(val_row["validation_rows"]),
            "final_accuracy": metrics["final_accuracy"],
            "final_lift": metrics["final_lift"],
            "final_rows": metrics["final_rows"],
            "strongest_baseline": metrics["strongest_baseline"],
            "quarter_min_accuracy": metrics["quarter_min_accuracy"],
            "ticker_median_accuracy": metrics["ticker_median_accuracy"],
            "rolling250_min": metrics["rolling250_min"],
            "prediction_up_ratio": metrics["prediction_up_ratio"],
            "validation_composite_score": val_row["validation_composite_score"],
            "validation_shortlist_pass": bool(val_row["shortlist_pass"]),
            "split_guard_passed": True,
            "leaderboard_role": leaderboard_role,
        }
        row["claim_label"] = final_claim_label(row, validation_governed, leaderboard_role)
        row["reason_not_claimable"] = reason_not_claimable(row, validation_governed, leaderboard_role)
        rows.append(row)
        quarter_rows.append(q)
        ticker_rows.append(t)
        balance_rows.append(balance)
    return (
        pd.DataFrame(rows),
        pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame(),
        pd.concat(ticker_rows, ignore_index=True) if ticker_rows else pd.DataFrame(),
        pd.DataFrame(balance_rows),
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def row_text(row: dict[str, Any]) -> str:
    return " ".join(str(value) for value in row.values()).lower()


def scan_historical_candidates(final_leaderboard: pd.DataFrame) -> pd.DataFrame:
    search_roots = [REPO_ROOT / "reports" / "generated", REPO_ROOT / "reports" / "results", REPO_ROOT / "reports" / "claims", REPO_ROOT / "outputs"]
    candidate_paths: list[Path] = []
    allowed_name_tokens = ("result", "leaderboard", "candidate", "claim", "summary", "final", "selected", "above60", "target62", "baseline60")
    skip_name_tokens = ("candidate_grid", "ticker_stability", "quarter_stability", "prediction_balance", "features.parquet")
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".md", ".json"}:
                continue
            lower_name = path.name.lower()
            if any(token in lower_name for token in skip_name_tokens):
                continue
            if any(token in lower_name for token in allowed_name_tokens):
                candidate_paths.append(path)

    rows: list[dict[str, Any]] = []
    for path in sorted(set(candidate_paths)):
        if path.suffix.lower() == ".csv":
            frame = read_csv_if_exists(path)
            if frame.empty:
                continue
            for _, raw in frame.head(5000).iterrows():
                data = raw.to_dict()
                text = row_text(data)
                old_accuracy = math.nan
                for col in ("final_accuracy", "accuracy", "model_accuracy", "strict_replay_accuracy", "validation_accuracy"):
                    if col in data:
                        old_accuracy = as_float(data.get(col))
                        if math.isfinite(old_accuracy):
                            break
                old_lift = math.nan
                for col in ("final_lift", "final_lift_over_strongest_baseline", "final_delta_vs_baseline", "lift_vs_majority", "validation_lift"):
                    if col in data:
                        old_lift = as_float(data.get(col))
                        if math.isfinite(old_lift):
                            break
                interesting = bool(
                    (math.isfinite(old_accuracy) and old_accuracy >= 0.60)
                    or (math.isfinite(old_lift) and old_lift > 0.05)
                    or any(token in text for token in ["selected", "baseline60", "target62", "ensemble", "regime", "router", "boosted", "xgboost", "lightgbm"])
                )
                if not interesting:
                    continue
                old_model = str(data.get("model_family", data.get("model", data.get("model_id", data.get("model_group", "")))))
                old_feature = str(data.get("feature_group", data.get("feature_set", data.get("feature_family", data.get("filter_description", "")))))
                old_horizon = data.get("horizon", "")
                old_threshold = data.get("threshold", "")
                candidate_id = str(data.get("candidate_id", f"{old_model}__{old_feature}__h{old_horizon}__t{old_threshold}"))
                rows.append(
                    {
                        "source_file": rel(path),
                        "candidate_id": candidate_id,
                        "old_model": old_model,
                        "old_feature_set": old_feature,
                        "old_target_variant": data.get("target_variant", ""),
                        "old_horizon": old_horizon,
                        "old_threshold": old_threshold,
                        "old_accuracy": old_accuracy,
                        "old_row_count": data.get("final_rows", data.get("rows", data.get("observations", ""))),
                        "old_baseline_or_lift": old_lift,
                        "old_split_rule": data.get("selection_source", data.get("selected_by_validation_yes_no", data.get("selected_by_preregistered_rule", ""))),
                        "strict_replay_status": "not_replayed",
                        "strict_replay_accuracy": math.nan,
                        "strict_replay_strongest_baseline": "",
                        "strict_replay_lift": math.nan,
                        "status": "insufficient_metadata",
                    }
                )
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lower = text.lower()
            if not any(token in lower for token in ["63", "64", "baseline60", "target62", "selected", "ensemble", "regime", "router"]):
                continue
            percentages = [float(match) / 100.0 for match in re.findall(r"(?<!\d)(6[0-9](?:\.\d+)?)\s*%", text)]
            best_pct = max(percentages) if percentages else math.nan
            if math.isfinite(best_pct) and best_pct >= 0.60 or any(token in lower for token in ["baseline60", "target62", "selected", "ensemble", "regime", "router"]):
                rows.append(
                    {
                        "source_file": rel(path),
                        "candidate_id": f"text_summary__{path.stem}",
                        "old_model": "",
                        "old_feature_set": "",
                        "old_target_variant": "",
                        "old_horizon": "",
                        "old_threshold": "",
                        "old_accuracy": best_pct,
                        "old_row_count": "",
                        "old_baseline_or_lift": "",
                        "old_split_rule": "text_summary",
                        "strict_replay_status": "not_replayed",
                        "strict_replay_accuracy": math.nan,
                        "strict_replay_strongest_baseline": "",
                        "strict_replay_lift": math.nan,
                        "status": "insufficient_metadata",
                    }
                )

    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    registry = registry.drop_duplicates(["source_file", "candidate_id"]).copy()
    registry["_priority_accuracy"] = registry["old_accuracy"].apply(as_float)
    registry = registry.sort_values("_priority_accuracy", ascending=False, na_position="last").head(1000).reset_index(drop=True)
    final_lookup = final_leaderboard.set_index("candidate_id").to_dict("index") if not final_leaderboard.empty else {}
    for idx, row in registry.iterrows():
        cid = str(row["candidate_id"])
        text = " ".join(str(row.get(col, "")) for col in ["candidate_id", "old_model", "old_feature_set", "old_target_variant"]).lower()
        matched: dict[str, Any] | None = final_lookup.get(cid)
        if matched is None and ("grid_334693" in cid or "exploratory_best" in text):
            for key, item in final_lookup.items():
                if (
                    item.get("model_family") == "logistic_regression"
                    and item.get("feature_group") == "compact_stable_features"
                    and item.get("target_variant") == "absolute_direction"
                    and int(item.get("horizon", -1)) == 50
                    and abs(float(item.get("threshold", 0)) - 0.525) < 1e-9
                ):
                    matched = item
                    break
        if matched is None and "bull_bear_sideway_router" in text:
            for key, item in final_lookup.items():
                if item.get("model_family") == "regime_gated_ensemble" and int(item.get("horizon", -1)) == 40 and abs(float(item.get("threshold", 0)) - 0.5) < 1e-9:
                    matched = item
                    break
        if matched:
            registry.loc[idx, "strict_replay_status"] = "completed_v3"
            registry.loc[idx, "strict_replay_accuracy"] = matched.get("final_accuracy", math.nan)
            registry.loc[idx, "strict_replay_strongest_baseline"] = matched.get("strongest_baseline", "")
            registry.loc[idx, "strict_replay_lift"] = matched.get("final_lift", math.nan)
            registry.loc[idx, "status"] = "survives" if as_float(matched.get("final_accuracy")) >= 0.60 and as_float(matched.get("final_lift")) > 0 else "dies"
        elif any(token in text for token in ["router", "regime", "calibration", "isotonic", "platt", "xgboost", "lightgbm", "random_forest", "stacking", "ensemble"]):
            registry.loc[idx, "status"] = "needs_relock"
    return registry.drop(columns=["_priority_accuracy"])


def build_validation_governed_leaderboard(final_all: pd.DataFrame) -> pd.DataFrame:
    if final_all.empty:
        return final_all
    out = final_all[final_all["leaderboard_role"].eq("validation_governed")].copy()
    if out.empty:
        out = final_all.sort_values(["validation_composite_score", "validation_lift", "validation_accuracy"], ascending=[False, False, False]).head(10).copy()
        out["leaderboard_role"] = "validation_governed_diagnostic_no_strict_shortlist"
        out["claim_label"] = "diagnostic_only"
        out["reason_not_claimable"] = "no candidate passed strict validation-governed shortlist filters"
    return out.sort_values(["validation_composite_score", "validation_lift", "validation_accuracy"], ascending=[False, False, False]).reset_index(drop=True)


def build_promotion_queue(final_all: pd.DataFrame) -> pd.DataFrame:
    if final_all.empty:
        return final_all
    beats = final_all[
        (final_all["final_accuracy"].astype(float) > CLAIMABLE_CHAMPION["final_accuracy"])
        | (final_all["final_accuracy"].astype(float) > EXPLORATORY_BEST["final_accuracy"])
        | (final_all["final_lift"].astype(float) > EXPLORATORY_BEST["final_lift"])
    ].copy()
    if beats.empty:
        return beats
    beats["beats_claimable_champion_accuracy"] = beats["final_accuracy"].astype(float) > CLAIMABLE_CHAMPION["final_accuracy"]
    beats["beats_exploratory_best_accuracy"] = beats["final_accuracy"].astype(float) > EXPLORATORY_BEST["final_accuracy"]
    beats["beats_exploratory_best_lift"] = beats["final_lift"].astype(float) > EXPLORATORY_BEST["final_lift"]
    beats["promotion_requirement"] = "relock_or_future_blind_confirmation"
    beats["claim_label"] = "promoted_candidate_required"
    beats["reason_not_claimable"] = beats["reason_not_claimable"].where(
        beats["reason_not_claimable"].astype(str).str.len() > 0,
        "candidate beats benchmark but requires re-lock/future-blind confirmation",
    )
    return beats.sort_values(["final_accuracy", "final_lift", "validation_accuracy"], ascending=[False, False, False]).reset_index(drop=True)


def champion_comparison(final_all: pd.DataFrame, validation_governed: pd.DataFrame, exploratory: pd.DataFrame, promotion_queue: pd.DataFrame) -> pd.DataFrame:
    best_val = validation_governed.iloc[0].to_dict() if not validation_governed.empty else {}
    best_exp = exploratory.iloc[0].to_dict() if not exploratory.empty else {}
    rows = [
        {
            "comparison_role": "claimable_champion",
            "candidate_id": CLAIMABLE_CHAMPION["candidate_id"],
            "target_variant": "absolute_direction",
            "final_accuracy": CLAIMABLE_CHAMPION["final_accuracy"],
            "final_lift": CLAIMABLE_CHAMPION["final_lift"],
            "final_rows": CLAIMABLE_CHAMPION["final_rows"],
            "claim_label": "baseline60_candidate",
            "validation_governed": True,
            "beats_61_61": False,
            "beats_64_76": False,
            "beats_11_18pp_lift": False,
            "replaces_claimable_champion": False,
            "reason": "incumbent",
        },
        {
            "comparison_role": "exploratory_best_baseline",
            "candidate_id": EXPLORATORY_BEST["candidate_id"],
            "target_variant": EXPLORATORY_BEST["target_variant"],
            "final_accuracy": EXPLORATORY_BEST["final_accuracy"],
            "final_lift": EXPLORATORY_BEST["final_lift"],
            "final_rows": EXPLORATORY_BEST["final_rows"],
            "claim_label": "exploratory_not_claimable",
            "validation_governed": False,
            "beats_61_61": True,
            "beats_64_76": False,
            "beats_11_18pp_lift": False,
            "replaces_claimable_champion": False,
            "reason": "existing exploratory baseline",
        },
    ]
    for role, row in [("best_validation_governed_v3", best_val), ("best_exploratory_final_v3", best_exp)]:
        if not row:
            continue
        validation_governed_flag = str(row.get("leaderboard_role", "")).startswith("validation_governed")
        beats_61 = as_float(row.get("final_accuracy")) > CLAIMABLE_CHAMPION["final_accuracy"]
        beats_6476 = as_float(row.get("final_accuracy")) > EXPLORATORY_BEST["final_accuracy"]
        beats_lift = as_float(row.get("final_lift")) > EXPLORATORY_BEST["final_lift"]
        replaces = bool(
            validation_governed_flag
            and row.get("target_variant") == "absolute_direction"
            and beats_61
            and as_float(row.get("final_lift")) > CLAIMABLE_CHAMPION["final_lift"]
            and as_float(row.get("final_rows")) >= CLAIMABLE_CHAMPION["final_rows"] * 0.95
        )
        reason = []
        if not validation_governed_flag:
            reason.append("not validation-governed")
        if row.get("target_variant") != "absolute_direction":
            reason.append("separate target variant")
        if not beats_61:
            reason.append("does not beat 61.61%")
        if as_float(row.get("final_lift")) <= CLAIMABLE_CHAMPION["final_lift"]:
            reason.append("does not beat +10.90pp lift")
        if replaces:
            reason.append("validation-governed absolute-direction candidate beats incumbent accuracy and lift")
        rows.append(
            {
                "comparison_role": role,
                "candidate_id": row.get("candidate_id", ""),
                "target_variant": row.get("target_variant", ""),
                "final_accuracy": row.get("final_accuracy", math.nan),
                "final_lift": row.get("final_lift", math.nan),
                "final_rows": row.get("final_rows", math.nan),
                "claim_label": row.get("claim_label", ""),
                "validation_governed": validation_governed_flag,
                "beats_61_61": beats_61,
                "beats_64_76": beats_6476,
                "beats_11_18pp_lift": beats_lift,
                "replaces_claimable_champion": replaces,
                "reason": "; ".join(reason),
            }
        )
    rows.append(
        {
            "comparison_role": "promotion_queue_summary",
            "candidate_id": "promotion_candidate_queue",
            "target_variant": "",
            "final_accuracy": promotion_queue["final_accuracy"].max() if not promotion_queue.empty else math.nan,
            "final_lift": promotion_queue["final_lift"].max() if not promotion_queue.empty else math.nan,
            "final_rows": "",
            "claim_label": "promoted_candidate_required" if not promotion_queue.empty else "not_claimable",
            "validation_governed": False,
            "beats_61_61": bool(not promotion_queue.empty and promotion_queue["beats_claimable_champion_accuracy"].any()),
            "beats_64_76": bool(not promotion_queue.empty and promotion_queue["beats_exploratory_best_accuracy"].any()),
            "beats_11_18pp_lift": bool(not promotion_queue.empty and promotion_queue["beats_exploratory_best_lift"].any()),
            "replaces_claimable_champion": False,
            "reason": "promotion candidates require re-lock/future-blind confirmation" if not promotion_queue.empty else "no promotion candidates",
        }
    )
    return pd.DataFrame(rows)


def best_group_value(frame: pd.DataFrame, group_col: str) -> str:
    if frame.empty or group_col not in frame.columns:
        return ""
    grouped = frame.groupby(group_col, dropna=False).agg(best_final_accuracy=("final_accuracy", "max"), best_validation_score=("validation_composite_score", "max")).reset_index()
    grouped = grouped.sort_values(["best_final_accuracy", "best_validation_score"], ascending=[False, False])
    return str(grouped.iloc[0][group_col]) if not grouped.empty else ""


def write_reports(
    run_config: dict[str, Any],
    validation_governed: pd.DataFrame,
    exploratory: pd.DataFrame,
    promotion_queue: pd.DataFrame,
    champion_rows: pd.DataFrame,
) -> None:
    best_val = validation_governed.iloc[0].to_dict() if not validation_governed.empty else {}
    best_exp = exploratory.iloc[0].to_dict() if not exploratory.empty else {}
    validation_beats_6161 = bool(not validation_governed.empty and (validation_governed["final_accuracy"].astype(float) > CLAIMABLE_CHAMPION["final_accuracy"]).any())
    validation_beats_lift = bool(not validation_governed.empty and (validation_governed["final_lift"].astype(float) > CLAIMABLE_CHAMPION["final_lift"]).any())
    exploratory_beats_6476 = bool(not exploratory.empty and (exploratory["final_accuracy"].astype(float) > EXPLORATORY_BEST["final_accuracy"]).any())
    exploratory_beats_lift = bool(not exploratory.empty and (exploratory["final_lift"].astype(float) > EXPLORATORY_BEST["final_lift"]).any())
    baseline60_defensible = bool(not validation_governed.empty and (validation_governed["claim_label"].isin(["baseline60_candidate", "target62_candidate"])).any())
    target62_defensible = bool(not validation_governed.empty and (validation_governed["claim_label"].eq("target62_candidate")).any())
    final65_defensible = bool(not validation_governed.empty and (validation_governed["final_accuracy"].astype(float) >= 0.65).any())
    claimable_replace = bool(champion_rows.get("replaces_claimable_champion", pd.Series(dtype=bool)).astype(bool).any())
    best_target = best_group_value(exploratory, "target_variant")
    best_feature = best_group_value(exploratory, "feature_group")
    best_model = best_group_value(exploratory, "model_family")
    claimable_result = "current 61.61% L2 Logistic baseline60_candidate remains claimable" if not claimable_replace else str(best_val.get("candidate_id", ""))

    protocol = f"""# VN30 Full Model Tuning V3 Protocol

## Scope

- Final evaluation target: VN30 stock hourly directional benchmark only.
- Index data usage: lagged market-state/context features only.
- Index benchmark performance is not claimed as stock benchmark performance.
- Target variants are evaluated and reported separately; target claims are not mixed.
- Out of scope: trading, profitability, BUY/SELL, investment recommendation, live deployment, DOCX, paper generation, git tags.

## Split Discipline

- Train rows require feature_timestamp <= `{TRAIN_END}` and target_timestamp <= `{TRAIN_END}`.
- Validation rows require feature_timestamp and target_timestamp from `{VAL_START}` through `{VAL_END}`.
- Final rows require feature_timestamp and target_timestamp >= `{FINAL_START}`.
- Candidate, model, target, feature, and threshold selection for claimable rows uses validation only.
- Final-ranked rows are exploratory only and require re-lock or future-blind confirmation.

## Search Design

- Target variants: {", ".join(TARGET_VARIANTS)}.
- Horizons: {HORIZONS}.
- Thresholds: {THRESHOLDS[0]:.3f} to {THRESHOLDS[-1]:.3f} step 0.005.
- Feature groups: {", ".join(FEATURE_GROUP_ORDER)}.
- Model families: {", ".join(MODEL_FAMILIES)}, soft-vote ensemble, regime-gated ensemble, historical replay rows.
- Full grid is represented in `candidate_grid.csv` as model-family parameter specs with theoretical expanded counts; fitted screening candidates are also listed in the same file.

## Validation Composite Score

score = 0.30 * validation_lift + 0.25 * validation_accuracy + 0.15 * quarterly_stability + 0.10 * ticker_stability + 0.10 * prediction_balance + 0.05 * row_count_score + 0.05 * simplicity_score.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    promo_ids = ", ".join(promotion_queue["candidate_id"].head(10).astype(str).tolist()) if not promotion_queue.empty else "none"
    result = f"""# VN30 Full Model Tuning V3 Result Summary

## Baselines

- Current claimable champion: L2 Logistic, feature_set_C_closest, h40, threshold 0.50, 61.61% final accuracy, +10.90 pp lift, 4,074 rows.
- Current exploratory best: Logistic Regression, compact_stable_features, h50, threshold 0.525, 64.76% final accuracy, +11.18 pp lift, 3,774 rows.

## Best Validation-Governed Candidate

- Candidate: `{best_val.get("candidate_id", "")}`.
- Model family: {best_val.get("model_family", "")}.
- Target variant: {best_val.get("target_variant", "")}.
- Feature group: {best_val.get("feature_group", "")}.
- Horizon/threshold: h{best_val.get("horizon", "")} / {best_val.get("threshold", "")}.
- Validation accuracy/lift: {pct(best_val.get("validation_accuracy", math.nan))} / {pp(best_val.get("validation_lift", math.nan))}.
- Final accuracy/lift: {pct(best_val.get("final_accuracy", math.nan))} / {pp(best_val.get("final_lift", math.nan))}.
- Claim label: {best_val.get("claim_label", "")}.

## Best Exploratory Final Candidate

- Candidate: `{best_exp.get("candidate_id", "")}`.
- Model family: {best_exp.get("model_family", "")}.
- Target variant: {best_exp.get("target_variant", "")}.
- Feature group: {best_exp.get("feature_group", "")}.
- Horizon/threshold: h{best_exp.get("horizon", "")} / {best_exp.get("threshold", "")}.
- Final accuracy/lift: {pct(best_exp.get("final_accuracy", math.nan))} / {pp(best_exp.get("final_lift", math.nan))}.
- Claim label: {best_exp.get("claim_label", "")}.

## Required Answers

1. Did any validation-governed candidate beat 61.61%: {str(validation_beats_6161).lower()}.
2. Did any validation-governed candidate beat +10.90 pp lift: {str(validation_beats_lift).lower()}.
3. Did any exploratory candidate beat 64.76%: {str(exploratory_beats_6476).lower()}.
4. Did any exploratory candidate beat +11.18 pp lift: {str(exploratory_beats_lift).lower()}.
5. Best target variant by final-ranked evidence: {best_target}.
6. Best feature group by final-ranked evidence: {best_feature}.
7. Best model family by final-ranked evidence: {best_model}.
8. Promotion/future-blind candidates: {promo_ids}.
9. Result that remains claimable: {claimable_result}.
10. Baseline60 defensible: {str(baseline60_defensible).lower()}; target62 defensible: {str(target62_defensible).lower()}; final65 defensible: {str(final65_defensible and claimable_replace).lower()}.

Paper-safe wording:

> VN30 Full Model Tuning v3 evaluated lagged market-index context features, multiple target variants, and staged model-family screening under strict target_timestamp split discipline. The current 61.61% strict-replay L2 Logistic absolute-direction champion is {'' if claimable_replace else 'not '}replaced by this run. Final-ranked candidates outside validation governance are exploratory only and require re-lock or future-blind confirmation before any claim.
"""
    write_markdown(RESULT_PATH, result)

    claim = f"""# VN30 Full Model Tuning V3 Claim Boundary

- Claimable scope: VN30 stock hourly diagnostic benchmark only.
- Index layer: point-in-time lagged market-context features only; no index benchmark result is claimed as stock accuracy.
- Target variants are separate; target-variant results must not be mixed with absolute-direction claims.
- Best validation-governed v3 candidate: `{best_val.get("candidate_id", "")}` with {pct(best_val.get("final_accuracy", math.nan))} final accuracy and {pp(best_val.get("final_lift", math.nan))} final lift.
- Best exploratory v3 candidate: `{best_exp.get("candidate_id", "")}` with {pct(best_exp.get("final_accuracy", math.nan))} final accuracy and {pp(best_exp.get("final_lift", math.nan))} final lift.
- Current 61.61% claimable champion replaced: {str(claimable_replace).lower()}.
- Claimable result: {claimable_result}.
- Promotion queue: `promotion_candidate_queue.csv`; rows there are not claimable until re-locked or future-blind confirmed.
- No trading, profitability, BUY/SELL, investment recommendation, live deployment, DOCX, paper, index-as-stock, final-label tuning, push, merge, or tag claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 full model tuning v3.")
    parser.add_argument("--logistic-budget", type=int, default=80)
    parser.add_argument("--elasticnet-budget", type=int, default=24)
    parser.add_argument("--calibrated-budget", type=int, default=12)
    parser.add_argument("--rf-budget", type=int, default=12)
    parser.add_argument("--extra-trees-budget", type=int, default=12)
    parser.add_argument("--xgb-budget", type=int, default=8)
    parser.add_argument("--lgbm-budget", type=int, default=12)
    parser.add_argument("--hist-budget", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budgets = {
        "logistic_regression": args.logistic_budget,
        "elasticnet_logistic": args.elasticnet_budget,
        "calibrated_logistic": args.calibrated_budget,
        "random_forest": args.rf_budget,
        "extra_trees": args.extra_trees_budget,
        "xgboost": args.xgb_budget,
        "lightgbm": args.lgbm_budget,
        "hist_gradient_boosting": args.hist_budget,
    }
    print("Building v3 feature frame and index context...", flush=True)
    features, feature_groups, index_audit, index_manifest, feature_manifest = build_v3_feature_frame()
    index_data = load_index_data()
    print("Building target variants and baselines...", flush=True)
    label_cache, baseline_cache, target_audit, baseline_comparison = build_label_and_baseline_cache(features, index_data)
    param_grids = full_param_grids()
    candidate_grid_specs, theoretical_candidate_count = compact_candidate_grid(feature_groups, param_grids)
    fit_grid = build_fit_grid(feature_groups, budgets)
    candidate_grid = pd.concat([candidate_grid_specs, fit_grid], ignore_index=True, sort=False)

    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 stock hourly full model tuning v3 with target variants and lagged index context",
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "target_variants": TARGET_VARIANTS,
        "horizons": HORIZONS,
        "threshold_start": THRESHOLDS[0],
        "threshold_end": THRESHOLDS[-1],
        "threshold_step": 0.005,
        "fit_budget_per_family": budgets,
        "candidate_grid_rows": int(len(candidate_grid)),
        "theoretical_expanded_candidate_count": theoretical_candidate_count,
        "fit_grid_rows": int(len(fit_grid)),
        "final_accuracy_used_for_selection": False,
        "exploratory_final_ranking_claimable": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
        "feature_manifest": feature_manifest,
        "optional_dependencies": {"xgboost_available": XGBClassifier is not None, "lightgbm_available": LGBMClassifier is not None},
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    write_frame(OUTPUT_DIR / "index_context_feature_audit.csv", index_audit)
    write_json(OUTPUT_DIR / "index_context_manifest.json", index_manifest)
    write_frame(OUTPUT_DIR / "target_variant_audit.csv", target_audit)
    write_frame(OUTPUT_DIR / "baseline_comparison.csv", baseline_comparison)
    write_frame(OUTPUT_DIR / "candidate_grid.csv", candidate_grid)

    print(f"Running staged validation screening on {len(fit_grid)} base candidates...", flush=True)
    validation_results, payloads, quarter_stability, ticker_stability, prediction_balance = fit_validation_candidates(
        features,
        feature_groups,
        fit_grid,
        label_cache,
        baseline_cache,
    )
    regime_results, regime_q, regime_t, regime_b, regime_payloads = add_regime_gate_candidates(features, feature_groups, label_cache, baseline_cache, payloads)
    if not regime_results.empty:
        validation_results = pd.concat([validation_results, regime_results], ignore_index=True)
        quarter_stability = pd.concat([quarter_stability, regime_q], ignore_index=True)
        ticker_stability = pd.concat([ticker_stability, regime_t], ignore_index=True)
        prediction_balance = pd.concat([prediction_balance, regime_b], ignore_index=True)
        payloads.update(regime_payloads)
    ensemble_results, ensemble_q, ensemble_t, ensemble_b, ensemble_payloads = add_soft_vote_candidates(features, validation_results, payloads, baseline_cache)
    if not ensemble_results.empty:
        validation_results = pd.concat([validation_results, ensemble_results], ignore_index=True)
        quarter_stability = pd.concat([quarter_stability, ensemble_q], ignore_index=True)
        ticker_stability = pd.concat([ticker_stability, ensemble_t], ignore_index=True)
        prediction_balance = pd.concat([prediction_balance, ensemble_b], ignore_index=True)
        payloads.update(ensemble_payloads)
    write_frame(OUTPUT_DIR / "validation_results.csv", validation_results)

    print("Scoring final split once for validation-screened and exploratory rows...", flush=True)
    final_all, final_q, final_t, final_b = evaluate_final_candidates(features, validation_results, payloads, baseline_cache)
    validation_governed = build_validation_governed_leaderboard(final_all)
    exploratory = final_all.sort_values(["final_accuracy", "final_lift", "validation_accuracy"], ascending=[False, False, False]).reset_index(drop=True)
    promotion_queue = build_promotion_queue(final_all)
    champion_rows = champion_comparison(final_all, validation_governed, exploratory, promotion_queue)
    historical_registry = scan_historical_candidates(exploratory)

    write_frame(OUTPUT_DIR / "validation_governed_leaderboard.csv", validation_governed)
    write_frame(OUTPUT_DIR / "exploratory_final_leaderboard.csv", exploratory)
    write_frame(OUTPUT_DIR / "promotion_candidate_queue.csv", promotion_queue)
    write_frame(OUTPUT_DIR / "champion_comparison.csv", champion_rows)
    write_frame(OUTPUT_DIR / "quarter_stability.csv", pd.concat([quarter_stability, final_q], ignore_index=True))
    write_frame(OUTPUT_DIR / "ticker_stability.csv", pd.concat([ticker_stability, final_t], ignore_index=True))
    write_frame(OUTPUT_DIR / "prediction_balance.csv", pd.concat([prediction_balance, final_b], ignore_index=True))
    write_frame(OUTPUT_DIR / "historical_candidate_registry.csv", historical_registry)

    tuning_manifest = {
        **run_config,
        "validation_result_rows": int(len(validation_results)),
        "final_result_rows": int(len(final_all)),
        "validation_governed_rows": int(len(validation_governed)),
        "exploratory_rows": int(len(exploratory)),
        "promotion_queue_rows": int(len(promotion_queue)),
        "regime_gate_payloads": int(len(regime_payloads)),
        "ensemble_payloads": int(len(ensemble_payloads)),
        "best_validation_governed": validation_governed.iloc[0].to_dict() if not validation_governed.empty else {},
        "best_exploratory_final": exploratory.iloc[0].to_dict() if not exploratory.empty else {},
        "champion_comparison": champion_rows.to_dict("records"),
    }
    write_json(OUTPUT_DIR / "tuning_manifest.json", tuning_manifest)
    write_reports(run_config, validation_governed, exploratory, promotion_queue, champion_rows)

    best_val = validation_governed.iloc[0].to_dict() if not validation_governed.empty else {}
    best_exp = exploratory.iloc[0].to_dict() if not exploratory.empty else {}
    print(f"VN30 full model tuning v3 complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Best validation-governed candidate: {best_val.get('candidate_id', '')} ({pct(best_val.get('final_accuracy', math.nan))})", flush=True)
    print(f"Best exploratory final candidate: {best_exp.get('candidate_id', '')} ({pct(best_exp.get('final_accuracy', math.nan))})", flush=True)
    print(f"Promotion queue rows: {len(promotion_queue)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
