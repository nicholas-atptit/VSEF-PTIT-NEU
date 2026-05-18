"""Run validation-safe Track A baseline60 improvement v1."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    rel,
    write_csv,
    write_json,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_baseline60_improvement_v1"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_baseline60_improvement_v1"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
PREVIOUS_LOGISTIC_H40 = 0.6043200785468826
THRESHOLDS = [round(0.45 + i * 0.01, 2) for i in range(11)]
HORIZONS = [40, 60]
ACTIVE_TICKER_COUNT = 30
RESULT_COLUMNS = [
    "model",
    "horizon",
    "feature_set",
    "hyperparams_id",
    "threshold",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "final_accuracy",
    "final_baseline_accuracy",
    "final_delta_vs_baseline",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "delta_vs_60_31",
    "pass_60",
    "pass_60_31",
    "pass_65",
    "selected_on_validation",
    "claim_level",
]


def make_candidate_model(name: str, params: dict[str, Any]) -> Any | None:
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=42, **params)
    if name == "lightgbm_shallow" and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=80, max_depth=2, learning_rate=0.04, min_child_samples=50, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1)
    if name == "xgboost_shallow" and XGBClassifier is not None:
        return XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.04, min_child_weight=20, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric="logloss", n_jobs=2)
    return None


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for class_weight in [None, "balanced"]:
            specs.append(
                {
                    "model": "logistic_regression",
                    "hyperparams_id": f"logit_l2_C{c_value}_{'balanced' if class_weight else 'none'}",
                    "params": {"penalty": "l2", "C": c_value, "solver": "liblinear", "class_weight": class_weight},
                }
            )
    for c_value in [0.1, 0.3, 1.0]:
        for l1_ratio in [0.15, 0.5]:
            specs.append(
                {
                    "model": "logistic_regression",
                    "hyperparams_id": f"logit_elastic_C{c_value}_l1_{l1_ratio}",
                    "params": {"penalty": "elasticnet", "C": c_value, "solver": "saga", "l1_ratio": l1_ratio, "class_weight": "balanced"},
                }
            )
    specs.extend(
        [
            {"model": "lightgbm_shallow", "hyperparams_id": "lgbm_shallow_v1", "params": {}},
            {"model": "xgboost_shallow", "hyperparams_id": "xgb_shallow_v1", "params": {}},
        ]
    )
    return specs


def split_xy(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    return {
        "train": labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index,
        "validation": labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index,
        "final": labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index,
    }


def baseline_accuracy(y_train: pd.Series, y_eval: pd.Series) -> tuple[float, int]:
    majority = int(y_train.mean() >= 0.5)
    return float((y_eval.astype(int) == majority).mean()), majority


def score_threshold(y: pd.Series, probability: np.ndarray, threshold: float) -> float:
    pred = (probability >= threshold).astype(int)
    return float((y.astype(int).to_numpy() == pred).mean())


def result_row(spec: dict[str, Any], horizon: int, threshold: float, val_acc: float, val_base: float, final_acc: float, final_base: float, final_rows: int) -> dict[str, Any]:
    return {
        "model": spec["model"],
        "horizon": horizon,
        "feature_set": "feature_set_C_closest",
        "hyperparams_id": spec["hyperparams_id"],
        "threshold": threshold,
        "validation_accuracy": val_acc,
        "validation_baseline_accuracy": val_base,
        "validation_delta_vs_baseline": val_acc - val_base,
        "final_accuracy": final_acc,
        "final_baseline_accuracy": final_base,
        "final_delta_vs_baseline": final_acc - final_base,
        "final_rows": final_rows,
        "final_coverage": 1.0,
        "active_ticker_count": ACTIVE_TICKER_COUNT,
        "delta_vs_60_31": final_acc - LOCKED_RF_H60,
        "pass_60": final_acc >= 0.60,
        "pass_60_31": final_acc > LOCKED_RF_H60,
        "pass_65": final_acc >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory_baseline60" if final_acc >= 0.60 else "exploratory",
    }


def run_base_candidates(features: pd.DataFrame, feature_cols: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_xy(features, labels)
        y_train = labels.reindex(idx["train"]).astype(int)
        y_val = labels.reindex(idx["validation"]).astype(int)
        y_final = labels.reindex(idx["final"]).astype(int)
        val_base, _ = baseline_accuracy(y_train, y_val)
        final_base, _ = baseline_accuracy(y_train, y_final)
        x_train = features.reindex(idx["train"])[feature_cols]
        x_val = features.reindex(idx["validation"])[feature_cols]
        x_final = features.reindex(idx["final"])[feature_cols]
        for spec in candidate_specs():
            model = make_candidate_model(spec["model"], spec["params"])
            if model is None:
                continue
            pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
            try:
                pipeline.fit(x_train, y_train)
                val_prob = pipeline.predict_proba(x_val)[:, 1]
                final_prob = pipeline.predict_proba(x_final)[:, 1]
            except Exception:
                continue
            threshold_scores = [(threshold, score_threshold(y_val, val_prob, threshold)) for threshold in THRESHOLDS]
            best_threshold, best_val_acc = max(threshold_scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))
            final_acc = score_threshold(y_final, final_prob, best_threshold)
            row = result_row(spec, horizon, best_threshold, best_val_acc, val_base, final_acc, final_base, len(y_final))
            rows.append(row)
            cid = f"{spec['model']}__h{horizon}__{spec['hyperparams_id']}__t{best_threshold}"
            predictions[cid] = {
                "row": row,
                "horizon": horizon,
                "validation_y": y_val,
                "final_y": y_final,
                "validation_probability": val_prob,
                "final_probability": final_prob,
                "validation_baseline_accuracy": val_base,
                "final_baseline_accuracy": final_base,
            }
    return rows, predictions


def run_soft_vote(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ensemble_rows: list[dict[str, Any]] = []
    eligible = [key for key, value in predictions.items() if float(value["row"]["validation_accuracy"]) >= 0.50]
    for horizon in HORIZONS:
        horizon_ids = [key for key in eligible if predictions[key]["horizon"] == horizon]
        horizon_ids = sorted(horizon_ids, key=lambda key: float(predictions[key]["row"]["validation_accuracy"]), reverse=True)[:5]
        if len(horizon_ids) < 2:
            continue
        val_y = predictions[horizon_ids[0]]["validation_y"]
        final_y = predictions[horizon_ids[0]]["final_y"]
        weights = np.array([max(float(predictions[key]["row"]["validation_delta_vs_baseline"]), 0.0) for key in horizon_ids])
        if float(weights.sum()) <= 0:
            weights = np.ones(len(horizon_ids))
        weights = weights / weights.sum()
        val_probs = np.column_stack([predictions[key]["validation_probability"] for key in horizon_ids])
        final_probs = np.column_stack([predictions[key]["final_probability"] for key in horizon_ids])
        val_vote = val_probs @ weights
        final_vote = final_probs @ weights
        threshold_scores = [(threshold, score_threshold(val_y, val_vote, threshold)) for threshold in THRESHOLDS]
        best_threshold, best_val_acc = max(threshold_scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))
        final_acc = score_threshold(final_y, final_vote, best_threshold)
        spec = {"model": "validation_weighted_soft_vote", "hyperparams_id": ";".join(horizon_ids)}
        ensemble_rows.append(
            result_row(
                spec,
                horizon,
                best_threshold,
                best_val_acc,
                float(predictions[horizon_ids[0]]["validation_baseline_accuracy"]),
                final_acc,
                float(predictions[horizon_ids[0]]["final_baseline_accuracy"]),
                len(final_y),
            )
        )
    return ensemble_rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if math.isfinite(float(row["validation_accuracy"]))]
    if not valid:
        return None
    selected = max(valid, key=lambda row: (float(row["validation_accuracy"]), float(row["validation_delta_vs_baseline"]), -abs(float(row["threshold"]) - 0.5)))
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
        "above6031_candidates.csv",
        "above65_candidates.csv",
        "improvement_run_log.md",
    ]:
        source = OUTPUT_DIR / name
        if source.exists():
            (REPORT_DIR / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    features, feature_cols = build_feature_set_c(load_stock_data(tickers), load_index_data())
    rows, predictions = run_base_candidates(features, feature_cols)
    rows.extend(run_soft_vote(rows, predictions))
    selected = select_candidate(rows)
    selected_rows = [selected] if selected else []
    above60 = [row for row in rows if bool(row["pass_60"])]
    above6031 = [row for row in rows if bool(row["pass_60_31"])]
    above65 = [row for row in rows if bool(row["pass_65"])]
    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "track": "Track A canonical-like",
            "horizons": HORIZONS,
            "threshold_grid": THRESHOLDS,
            "selection_rule": "validation_accuracy_then_validation_delta; final scoring only",
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "data_fetch": False,
            "feature_set": "feature_set_C_closest",
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
            "previous_logistic_h40_accuracy": PREVIOUS_LOGISTIC_H40,
            "locked_rf_h60_reference": LOCKED_RF_H60,
        },
    )
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", selected_rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above60_candidates.csv", above60, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above6031_candidates.csv", above6031, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "above65_candidates.csv", above65, RESULT_COLUMNS)
    log = [
        "# Track A Baseline60 Improvement V1 Run Log",
        "",
        "- Status: completed.",
        "- Selection: validation-only.",
        "- Final evaluation: scoring-only.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking: no.",
        f"- Candidate count: {len(rows)}.",
        f"- Selected candidate: `{selected.get('model') if selected else ''}` h={selected.get('horizon') if selected else ''} threshold={selected.get('threshold') if selected else ''}.",
        f"- Selected final accuracy: {selected.get('final_accuracy') if selected else ''}.",
        "",
    ]
    (OUTPUT_DIR / "improvement_run_log.md").write_text("\n".join(log), encoding="utf-8")
    copy_outputs_to_report_dir()
    print(f"track_a_improvement_status=completed selected={selected.get('model') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
