"""Run validation-selected joint VN30 stock + index panel training v1."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

from scripts.research.audit_vn30_stock_index_joint_panel_readiness import audit_one
from scripts.research.vn30_stock_index_joint_panel_features import (
    BASELINE_STOCK_RF_H60_REFERENCE,
    FEATURE_SETS,
    HORIZONS,
    OUTPUT_DIR,
    REPORT_DIR,
    SUPPORTED_INDICES,
    VN30_TICKERS,
    add_time_splits,
    baseline_prediction,
    build_model_dataset,
    metric_slice,
    pct,
    split_metrics,
    write_csv,
    write_json,
)


RANDOM_STATE = 42


RESULT_COLUMNS = [
    "candidate_id",
    "model",
    "horizon",
    "feature_set",
    "hyperparams_id",
    "validation_combined_accuracy",
    "validation_stock_only_accuracy",
    "validation_index_only_accuracy",
    "validation_combined_baseline_accuracy",
    "validation_combined_delta_vs_baseline",
    "validation_stability_score",
    "final_combined_accuracy",
    "final_stock_only_accuracy",
    "final_index_only_accuracy",
    "final_combined_baseline_accuracy",
    "final_combined_delta_vs_baseline",
    "final_rows_combined",
    "final_rows_stock_only",
    "final_rows_index_only",
    "final_coverage_combined",
    "stock_instrument_count",
    "index_instrument_count",
    "total_instrument_count",
    "pass_combined_60",
    "pass_combined_65",
    "pass_stock_only_60",
    "pass_stock_only_65",
    "selected_on_validation",
    "claim_level",
]


def readiness_rows() -> list[dict[str, Any]]:
    rows = [audit_one(code, "stock") for code in VN30_TICKERS]
    rows.extend(audit_one(code, "index") for code in SUPPORTED_INDICES)
    return rows


def readiness_passed(rows: list[dict[str, Any]]) -> bool:
    stock = [row for row in rows if row["instrument_type"] == "stock" and row["usable"]]
    index = [row for row in rows if row["instrument_type"] == "index" and row["usable"]]
    return len(stock) == 30 and len(index) == 6


def make_model(name: str) -> Any | None:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    if name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbose=-1,
        )
    return None


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def fit_predict_candidate(data: pd.DataFrame, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    train = data[data["split"].eq("train")]
    if train.empty:
        raise ValueError("No train rows available")
    model = make_model(model_name)
    if model is None:
        raise RuntimeError(f"Model unavailable: {model_name}")
    pipeline = make_pipeline(model)
    pipeline.fit(train[feature_cols], train["target_direction"].astype(int))
    scored = data.copy()
    scored["prediction"] = pipeline.predict(scored[feature_cols]).astype(int)
    return scored


def fit_predict_ensemble(data: pd.DataFrame, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    estimators = []
    for name in ("random_forest", "xgboost", "lightgbm"):
        model = make_model(name)
        if model is not None:
            estimators.append((name, model))
    if len(estimators) < 2:
        return fit_predict_candidate(data, feature_cols, "random_forest")
    train = data[data["split"].eq("train")]
    if model_name == "stacking":
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba",
            cv=3,
            n_jobs=None,
        )
    else:
        model = VotingClassifier(estimators=estimators, voting="soft")
    pipeline = make_pipeline(model)
    pipeline.fit(train[feature_cols], train["target_direction"].astype(int))
    scored = data.copy()
    scored["prediction"] = pipeline.predict(scored[feature_cols]).astype(int)
    return scored


def stability_score(scored: pd.DataFrame) -> float:
    validation = scored[scored["split"].eq("validation")].copy()
    if validation.empty:
        return math.nan
    validation["month"] = pd.to_datetime(validation["datetime"], errors="coerce").dt.to_period("M").astype(str)
    monthly = []
    for _month, group in validation.groupby("month"):
        metric = metric_slice(group)
        if metric["rows"]:
            monthly.append(metric["accuracy"])
    if not monthly:
        return math.nan
    return float(np.nanmean(monthly) - np.nanstd(monthly))


def baseline_combined_accuracy(data: pd.DataFrame, split: str) -> float:
    baseline = data.copy()
    baseline["prediction"] = baseline_prediction(baseline, "majority_class")
    return split_metrics(baseline, split)[f"{split}_combined_accuracy"]


def candidate_row(candidate_id: str, model: str, horizon: int, feature_set: str, data: pd.DataFrame, scored: pd.DataFrame) -> dict[str, Any]:
    validation = split_metrics(scored, "validation")
    final = split_metrics(scored, "final")
    validation_baseline = baseline_combined_accuracy(data, "validation")
    final_baseline = baseline_combined_accuracy(data, "final")
    row = {
        "candidate_id": candidate_id,
        "model": model,
        "horizon": horizon,
        "feature_set": feature_set,
        "hyperparams_id": "v1_default",
        "validation_combined_accuracy": validation["validation_combined_accuracy"],
        "validation_stock_only_accuracy": validation["validation_stock_only_accuracy"],
        "validation_index_only_accuracy": validation["validation_index_only_accuracy"],
        "validation_combined_baseline_accuracy": validation_baseline,
        "validation_combined_delta_vs_baseline": validation["validation_combined_accuracy"] - validation_baseline,
        "validation_stability_score": stability_score(scored),
        "final_combined_accuracy": final["final_combined_accuracy"],
        "final_stock_only_accuracy": final["final_stock_only_accuracy"],
        "final_index_only_accuracy": final["final_index_only_accuracy"],
        "final_combined_baseline_accuracy": final_baseline,
        "final_combined_delta_vs_baseline": final["final_combined_accuracy"] - final_baseline,
        "final_rows_combined": final["final_rows_combined"],
        "final_rows_stock_only": final["final_rows_stock_only"],
        "final_rows_index_only": final["final_rows_index_only"],
        "final_coverage_combined": final["final_coverage_combined"],
        "stock_instrument_count": final["final_stock_instrument_count"],
        "index_instrument_count": final["final_index_instrument_count"],
        "total_instrument_count": final["final_total_instrument_count"],
        "pass_combined_60": final["final_combined_accuracy"] >= 0.60,
        "pass_combined_65": final["final_combined_accuracy"] >= 0.65,
        "pass_stock_only_60": final["final_stock_only_accuracy"] >= 0.60,
        "pass_stock_only_65": final["final_stock_only_accuracy"] >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory",
    }
    return row


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    combined = row.get("validation_combined_accuracy")
    stability = row.get("validation_stability_score")
    delta = row.get("validation_combined_delta_vs_baseline")
    stock = row.get("validation_stock_only_accuracy")
    return (
        -1.0 if pd.isna(combined) else float(combined),
        -1.0 if pd.isna(stability) else float(stability),
        -1.0 if pd.isna(delta) else float(delta),
        -1.0 if pd.isna(stock) else float(stock),
    )


def write_gated_outputs(reason: str, rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "not_run",
        "reason": reason,
        "joint_panel_used": False,
        "stock_instrument_count": 30,
        "index_instrument_count": 6,
        "total_instrument_count": 36,
        "selected_on_validation": False,
        "final_evaluation_scoring_only": True,
        "confidence_abstention": False,
        "instrument_subset": False,
        "topk_ranking_substitution": False,
        "benchmark_run": False,
        "data_fetch": False,
        "model_training": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "run_config.json", payload)
    write_json(OUTPUT_DIR / "joint_training_manifest.json", {**payload, "readiness_rows": rows})
    empty: list[dict[str, Any]] = []
    for name in (
        "validation_candidate_results.csv",
        "final_candidate_results.csv",
        "selected_candidate_summary.csv",
        "combined_60_candidates.csv",
        "combined_65_candidates.csv",
        "stock_only_slice_results.csv",
        "index_only_slice_results.csv",
        "baseline_delta_summary.csv",
    ):
        write_csv(OUTPUT_DIR / name, empty, RESULT_COLUMNS if "candidate" in name or "summary" in name else [])
    log = [
        "# Joint Stock + Index Panel Training V1 Run Log",
        "",
        "- Status: not run.",
        f"- Reason: {reason}.",
        "- Model training: no.",
        "- Data fetch: no.",
        "- Paper/DOCX generated: no.",
        "",
    ]
    (OUTPUT_DIR / "joint_training_v1_run_log.md").write_text("\n".join(log), encoding="utf-8")


def main() -> int:
    readiness = readiness_rows()
    if not readiness_passed(readiness):
        write_gated_outputs("readiness_failed_36_instrument_hourly_panel", readiness)
        print("joint_training_status=not_run reason=readiness_failed_36_instrument_hourly_panel")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "status": "run",
            "models": ["random_forest", "xgboost", "lightgbm", "validation_weighted_soft_voting", "stacking"],
            "horizons": list(HORIZONS),
            "feature_sets": list(FEATURE_SETS),
            "selection_rule": "validation only; final scoring only",
            "confidence_abstention": False,
            "instrument_subset": False,
        },
    )

    rows: list[dict[str, Any]] = []
    scored_predictions: dict[str, pd.DataFrame] = {}
    model_names = ["random_forest", "xgboost", "lightgbm", "validation_weighted_soft_voting", "stacking"]
    for horizon in HORIZONS:
        for feature_set in FEATURE_SETS:
            data, feature_cols = build_model_dataset(horizon, feature_set)
            data = add_time_splits(data)
            for model_name in model_names:
                candidate_id = f"{model_name}__h{horizon}__{feature_set}"
                try:
                    if model_name in {"validation_weighted_soft_voting", "stacking"}:
                        scored = fit_predict_ensemble(data, feature_cols, "stacking" if model_name == "stacking" else "voting")
                    else:
                        scored = fit_predict_candidate(data, feature_cols, model_name)
                except Exception as exc:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "model": model_name,
                            "horizon": horizon,
                            "feature_set": feature_set,
                            "hyperparams_id": "v1_default",
                            "claim_level": f"failed:{exc}",
                        }
                    )
                    continue
                row = candidate_row(candidate_id, model_name, horizon, feature_set, data, scored)
                rows.append(row)
                scored_predictions[candidate_id] = scored

    valid_rows = [row for row in rows if "validation_combined_accuracy" in row and not pd.isna(row["validation_combined_accuracy"])]
    selected = max(valid_rows, key=selection_key) if valid_rows else None
    if selected is not None:
        selected["selected_on_validation"] = True
        selected["claim_level"] = "exploratory"

    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", [selected] if selected else [], RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "combined_60_candidates.csv", [row for row in rows if row.get("pass_combined_60")], RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "combined_65_candidates.csv", [row for row in rows if row.get("pass_combined_65")], RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "stock_only_slice_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "index_only_slice_results.csv", rows, RESULT_COLUMNS)
    write_csv(
        OUTPUT_DIR / "baseline_delta_summary.csv",
        [
            {
                "candidate_id": row.get("candidate_id", ""),
                "horizon": row.get("horizon", ""),
                "feature_set": row.get("feature_set", ""),
                "final_combined_accuracy": row.get("final_combined_accuracy", ""),
                "final_combined_baseline_accuracy": row.get("final_combined_baseline_accuracy", ""),
                "final_combined_delta_vs_baseline": row.get("final_combined_delta_vs_baseline", ""),
                "baseline_stock_rf_h60_reference": BASELINE_STOCK_RF_H60_REFERENCE,
            }
            for row in rows
        ],
    )
    write_json(
        OUTPUT_DIR / "joint_training_manifest.json",
        {
            "status": "run",
            "selected_candidate_id": selected.get("candidate_id") if selected else "",
            "selected_on_validation": bool(selected),
            "final_evaluation_scoring_only": True,
            "confidence_abstention": False,
            "instrument_subset": False,
            "topk_ranking_substitution": False,
            "stock_instrument_count": 30,
            "index_instrument_count": 6,
            "total_instrument_count": 36,
        },
    )
    log = [
        "# Joint Stock + Index Panel Training V1 Run Log",
        "",
        "- Status: run.",
        "- Selection: validation only.",
        "- Final evaluation: scoring only.",
        "- Confidence abstention: no.",
        "- Instrument subset: no.",
        "- Top-k/ranking substitution: no.",
        f"- Selected candidate: `{selected.get('candidate_id') if selected else ''}`.",
        f"- Final combined accuracy: {pct(selected.get('final_combined_accuracy') if selected else math.nan)}.",
        f"- Final stock-only accuracy: {pct(selected.get('final_stock_only_accuracy') if selected else math.nan)}.",
        f"- Final index-only accuracy: {pct(selected.get('final_index_only_accuracy') if selected else math.nan)}.",
        "",
    ]
    (OUTPUT_DIR / "joint_training_v1_run_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"joint_training_status=run selected={selected.get('candidate_id') if selected else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
