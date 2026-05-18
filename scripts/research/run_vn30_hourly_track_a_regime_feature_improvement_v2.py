"""Run Track A regime-aware feature improvement v2."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    LOCKED_RF_H60,
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    load_index_data,
    load_stock_data,
    rel,
    write_csv,
    write_json,
)
from scripts.research.vn30_hourly_track_a_regime_feature_v2 import FEATURE_SET_NAME, build_regime_feature_v2  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_regime_feature_improvement_v2"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_regime_feature_v2"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
HORIZONS = [40, 60, 80]
BASELINE_LOGISTIC_H40 = 0.6043200785468826
RESULT_COLUMNS = [
    "model",
    "horizon",
    "feature_set",
    "regime_method",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "final_accuracy",
    "final_baseline_accuracy",
    "final_delta_vs_baseline",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "delta_vs_60_43",
    "delta_vs_60_31",
    "pass_60",
    "pass_60_31",
    "pass_60_43",
    "pass_65",
    "selected_on_validation",
    "claim_level",
]


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    return {
        "train": labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index,
        "validation": labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index,
        "final": labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index,
    }


def make_model(model_name: str) -> Any | None:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42)
    if model_name == "l2_logistic":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=42)
    if model_name == "elasticnet_logistic":
        return LogisticRegression(max_iter=2500, solver="saga", penalty="elasticnet", C=0.3, l1_ratio=0.2, class_weight="balanced", random_state=42)
    if model_name == "random_forest_shallow":
        return RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=20, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    if model_name == "lightgbm_shallow" and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=90, max_depth=2, learning_rate=0.04, min_child_samples=50, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=2)
    if model_name == "xgboost_shallow" and XGBClassifier is not None:
        return XGBClassifier(n_estimators=90, max_depth=2, learning_rate=0.04, min_child_weight=20, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric="logloss", n_jobs=2)
    return None


def model_names() -> list[str]:
    return ["logistic_regression", "l2_logistic", "elasticnet_logistic", "random_forest_shallow", "lightgbm_shallow", "xgboost_shallow"]


def fit_pipeline(model_name: str, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline | None:
    model = make_model(model_name)
    if model is None or len(y_train) < 50 or y_train.nunique() < 2:
        return None
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    try:
        pipeline.fit(x_train, y_train.astype(int))
    except Exception:
        return None
    return pipeline


def predict_probability(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    return model.predict(x_data).astype(float)


def accuracy(y_true: pd.Series, pred: np.ndarray | pd.Series) -> float:
    return float((y_true.astype(int).to_numpy() == np.asarray(pred).astype(int)).mean()) if len(y_true) else math.nan


def train_majority_baseline(train_y: pd.Series, y_eval: pd.Series) -> float:
    majority = int(float(train_y.mean()) >= 0.5)
    return accuracy(y_eval, np.full(len(y_eval), majority))


def row_from_predictions(
    model_name: str,
    horizon: int,
    regime_method: str,
    val_y: pd.Series,
    val_prob: np.ndarray,
    final_y: pd.Series,
    final_prob: np.ndarray,
    val_base: float,
    final_base: float,
    active_ticker_count: int,
) -> dict[str, Any]:
    val_pred = (val_prob >= 0.5).astype(int)
    final_pred = (final_prob >= 0.5).astype(int)
    val_acc = accuracy(val_y, val_pred)
    final_acc = accuracy(final_y, final_pred)
    return {
        "model": model_name,
        "horizon": horizon,
        "feature_set": FEATURE_SET_NAME,
        "regime_method": regime_method,
        "validation_accuracy": val_acc,
        "validation_baseline_accuracy": val_base,
        "validation_delta_vs_baseline": val_acc - val_base,
        "final_accuracy": final_acc,
        "final_baseline_accuracy": final_base,
        "final_delta_vs_baseline": final_acc - final_base,
        "final_rows": int(len(final_y)),
        "final_coverage": 1.0,
        "active_ticker_count": active_ticker_count,
        "delta_vs_60_43": final_acc - BASELINE_LOGISTIC_H40,
        "delta_vs_60_31": final_acc - LOCKED_RF_H60,
        "pass_60": final_acc >= 0.60,
        "pass_60_31": final_acc > LOCKED_RF_H60,
        "pass_60_43": final_acc > BASELINE_LOGISTIC_H40,
        "pass_65": final_acc >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory_baseline60" if final_acc >= 0.60 else "exploratory",
    }


def regime_specific_probabilities(
    model_name: str,
    features: pd.DataFrame,
    feature_cols: list[str],
    train_idx: pd.Index,
    val_idx: pd.Index,
    final_idx: pd.Index,
    train_y: pd.Series,
    regime_col: str,
    fallback_model: Pipeline,
) -> tuple[np.ndarray, np.ndarray]:
    val_prob = predict_probability(fallback_model, features.reindex(val_idx)[feature_cols])
    final_prob = predict_probability(fallback_model, features.reindex(final_idx)[feature_cols])
    train_regimes = features.reindex(train_idx)[regime_col].fillna("unknown").astype(str)
    val_regimes = features.reindex(val_idx)[regime_col].fillna("unknown").astype(str)
    final_regimes = features.reindex(final_idx)[regime_col].fillna("unknown").astype(str)
    for regime in sorted(train_regimes.unique()):
        regime_train_idx = train_idx[train_regimes.to_numpy() == regime]
        if len(regime_train_idx) < 100:
            continue
        model = fit_pipeline(model_name, features.reindex(regime_train_idx)[feature_cols], train_y.reindex(regime_train_idx))
        if model is None:
            continue
        val_mask = val_regimes.to_numpy() == regime
        final_mask = final_regimes.to_numpy() == regime
        if val_mask.any():
            val_prob[val_mask] = predict_probability(model, features.reindex(val_idx[val_mask])[feature_cols])
        if final_mask.any():
            final_prob[final_mask] = predict_probability(model, features.reindex(final_idx[final_mask])[feature_cols])
    return val_prob, final_prob


def run_horizon(features: pd.DataFrame, feature_cols: list[str], horizon: int, active_ticker_count: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    labels = add_absolute_labels(features, horizon)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    val_base = train_majority_baseline(train_y, val_y)
    final_base = train_majority_baseline(train_y, final_y)
    x_train = features.reindex(idx["train"])[feature_cols]
    x_val = features.reindex(idx["validation"])[feature_cols]
    x_final = features.reindex(idx["final"])[feature_cols]
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for model_name in model_names():
        global_model = fit_pipeline(model_name, x_train, train_y)
        if global_model is None:
            continue
        val_prob = predict_probability(global_model, x_val)
        final_prob = predict_probability(global_model, x_final)
        row = row_from_predictions(model_name, horizon, "global_v2", val_y, val_prob, final_y, final_prob, val_base, final_base, active_ticker_count)
        rows.append(row)
        predictions[f"{model_name}__h{horizon}__global_v2"] = {"row": row, "val_y": val_y, "final_y": final_y, "val_prob": val_prob, "final_prob": final_prob, "val_idx": idx["validation"], "final_idx": idx["final"]}
        if model_name in {"logistic_regression", "l2_logistic", "elasticnet_logistic", "random_forest_shallow", "lightgbm_shallow", "xgboost_shallow"}:
            val_regime_prob, final_regime_prob = regime_specific_probabilities(model_name, features, feature_cols, idx["train"], idx["validation"], idx["final"], train_y, "market_regime_v2", global_model)
            method = "regime_specific_logistic_v2" if "logistic" in model_name else "regime_specific_shallow_tree_v2"
            row = row_from_predictions(model_name, horizon, method, val_y, val_regime_prob, final_y, final_regime_prob, val_base, final_base, active_ticker_count)
            rows.append(row)
            predictions[f"{model_name}__h{horizon}__{method}"] = {"row": row, "val_y": val_y, "final_y": final_y, "val_prob": val_regime_prob, "final_prob": final_regime_prob, "val_idx": idx["validation"], "final_idx": idx["final"]}
    return rows, predictions


def validation_weighted_regime_router(features: pd.DataFrame, horizon: int, predictions: dict[str, dict[str, Any]], active_ticker_count: int) -> dict[str, Any] | None:
    horizon_predictions = {key: value for key, value in predictions.items() if f"__h{horizon}__" in key}
    if len(horizon_predictions) < 2:
        return None
    first = next(iter(horizon_predictions.values()))
    val_y = first["val_y"]
    final_y = first["final_y"]
    val_idx = first["val_idx"]
    final_idx = first["final_idx"]
    val_regimes = features.reindex(val_idx)["market_regime_v2"].fillna("unknown").astype(str).to_numpy()
    final_regimes = features.reindex(final_idx)["market_regime_v2"].fillna("unknown").astype(str).to_numpy()
    val_prob = np.zeros(len(val_y), dtype=float)
    final_prob = np.zeros(len(final_y), dtype=float)
    val_base = float(next(iter(horizon_predictions.values()))["row"]["validation_baseline_accuracy"])
    final_base = float(next(iter(horizon_predictions.values()))["row"]["final_baseline_accuracy"])
    selected_ids: list[str] = []
    for regime in sorted(set(val_regimes) | set(final_regimes)):
        val_mask = val_regimes == regime
        final_mask = final_regimes == regime
        scored: list[tuple[str, float, float]] = []
        for key, payload in horizon_predictions.items():
            if not val_mask.any():
                continue
            prob = payload["val_prob"][val_mask]
            regime_acc = accuracy(val_y.iloc[val_mask], prob >= 0.5)
            regime_base = float((val_y.iloc[val_mask].astype(int).to_numpy() == int(float(val_y.mean()) >= 0.5)).mean())
            scored.append((key, regime_acc, max(regime_acc - regime_base, 0.0)))
        if not scored:
            continue
        top = sorted(scored, key=lambda item: (item[1], item[2]), reverse=True)[:3]
        weights = np.array([item[2] for item in top], dtype=float)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(top), dtype=float)
        weights = weights / weights.sum()
        selected_ids.extend([item[0] for item in top])
        if val_mask.any():
            val_stack = np.column_stack([horizon_predictions[item[0]]["val_prob"][val_mask] for item in top])
            val_prob[val_mask] = val_stack @ weights
        if final_mask.any():
            final_stack = np.column_stack([horizon_predictions[item[0]]["final_prob"][final_mask] for item in top])
            final_prob[final_mask] = final_stack @ weights
    row = row_from_predictions("validation_weighted_regime_router", horizon, "validation_weighted_regime_router_v2", val_y, val_prob, final_y, final_prob, val_base, final_base, active_ticker_count)
    row["model"] = "validation_weighted_regime_router"
    row["feature_set"] = FEATURE_SET_NAME
    row["regime_method"] = "validation_weighted_regime_router_v2"
    return row


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if math.isfinite(float(row["validation_accuracy"]))
        and int(row["active_ticker_count"]) == 30
        and abs(float(row["final_coverage"]) - 1.0) < 1e-12
    ]
    if not valid:
        return None
    selected = max(valid, key=lambda row: (float(row["validation_accuracy"]), float(row["validation_delta_vs_baseline"]), -abs(int(row["horizon"]) - 40)))
    selected["selected_on_validation"] = True
    return selected


def copy_outputs_to_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_config.json",
        "improvement_manifest.json",
        "validation_candidate_results.csv",
        "final_candidate_results.csv",
        "selected_candidate_summary.csv",
        "above60_candidates.csv",
        "above6043_candidates.csv",
        "above65_candidates.csv",
        "improvement_run_log.md",
    ]:
        source = OUTPUT_DIR / name
        if source.exists():
            (REPORT_DIR / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    features, feature_cols, feature_manifest = build_regime_feature_v2(load_stock_data(tickers), load_index_data())
    rows: list[dict[str, Any]] = []
    predictions_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        horizon_rows, horizon_predictions = run_horizon(features, feature_cols, horizon, len(tickers))
        rows.extend(horizon_rows)
        predictions_by_horizon.update(horizon_predictions)
        router_row = validation_weighted_regime_router(features, horizon, horizon_predictions, len(tickers))
        if router_row is not None:
            rows.append(router_row)
    selected = select_candidate(rows)
    selected_rows = [selected] if selected else []
    above60 = [row for row in rows if bool(row["pass_60"])]
    above6043 = [row for row in rows if bool(row["pass_60_43"])]
    above65 = [row for row in rows if bool(row["pass_65"])]
    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "track": "Track A canonical-like",
            "feature_set": FEATURE_SET_NAME,
            "horizons": HORIZONS,
            "models": model_names(),
            "methods": ["global_v2", "regime_specific_logistic_v2", "regime_specific_shallow_tree_v2", "validation_weighted_regime_router_v2"],
            "selection_rule": "validation_accuracy_then_validation_delta; final scoring only",
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "data_fetch": False,
        },
    )
    write_json(
        OUTPUT_DIR / "improvement_manifest.json",
        {
            "status": "completed",
            "active_ticker_count": len(tickers),
            "candidate_count": len(rows),
            "selected_candidate": selected,
            "selected_on_validation": bool(selected),
            "final_accuracy_used_for_selection": False,
            "feature_manifest": feature_manifest,
            "baseline_logistic_h40": BASELINE_LOGISTIC_H40,
            "locked_rf_h60_reference": LOCKED_RF_H60,
        },
    )
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", selected_rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above60_candidates.csv", above60, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above6043_candidates.csv", above6043, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above65_candidates.csv", above65, RESULT_COLUMNS)
    log = [
        "# Track A Regime Feature Improvement V2 Run Log",
        "",
        "- Status: completed.",
        "- Selection: validation-only.",
        "- Final evaluation: scoring-only.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking: no.",
        f"- Candidate count: {len(rows)}.",
        f"- Selected candidate: `{selected.get('model') if selected else ''}` h={selected.get('horizon') if selected else ''} `{selected.get('regime_method') if selected else ''}`.",
        f"- Selected final accuracy: {selected.get('final_accuracy') if selected else ''}.",
        "",
    ]
    (OUTPUT_DIR / "improvement_run_log.md").write_text("\n".join(log), encoding="utf-8")
    copy_outputs_to_report_dir()
    print(f"regime_feature_v2_status=completed selected={selected.get('model') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
