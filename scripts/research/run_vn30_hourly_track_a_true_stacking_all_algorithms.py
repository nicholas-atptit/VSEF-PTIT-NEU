"""Run true Track A stacking across all available base algorithms."""

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
    read_json,
    rel,
    write_csv,
    write_json,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_all_algorithm_stacking"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_true_stacking_all_algorithms"
PREVIOUS_LOGISTIC_H40 = 0.6043200785468826
RESULT_COLUMNS = [
    "stacking_method",
    "base_models_used",
    "meta_model",
    "horizon",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "final_accuracy",
    "final_baseline_accuracy",
    "final_delta_vs_baseline",
    "delta_vs_60_31",
    "delta_vs_60_43",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "selected_on_validation",
    "pass_60",
    "pass_60_31",
    "pass_60_43",
    "pass_65",
    "claim_level",
]


def read_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(OUTPUT_DIR / "base_oof_predictions.csv")
    final = pd.read_csv(OUTPUT_DIR / "base_final_predictions.csv")
    scores = pd.read_csv(OUTPUT_DIR / "base_model_scores.csv")
    return validation, final, scores


def matrix_for_split(predictions: pd.DataFrame, horizon: int, model_ids: list[str]) -> pd.DataFrame:
    subset = predictions[(predictions["horizon"] == horizon) & (predictions["base_model_id"].isin(model_ids))]
    probs = subset.pivot_table(index="row_id", columns="base_model_id", values="probability_up", aggfunc="last")
    meta = subset.drop_duplicates("row_id").set_index("row_id")[["target", "ticker"]]
    matrix = meta.join(probs, how="inner")
    return matrix.dropna(subset=model_ids)


def meta_estimator(meta_model: str) -> Any | None:
    if meta_model == "logistic_meta":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))])
    if meta_model == "l2_logistic_meta":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=0.25, random_state=42))])
    if meta_model == "lightgbm_shallow_meta" and LGBMClassifier is not None:
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LGBMClassifier(n_estimators=60, max_depth=2, learning_rate=0.04, min_child_samples=30, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1))])
    if meta_model == "xgboost_shallow_meta" and XGBClassifier is not None:
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.04, min_child_weight=10, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss", n_jobs=2))])
    return None


def accuracy(target: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    return float((target.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean()) if len(target) else math.nan


def majority_baseline(target: pd.Series) -> float:
    majority = int(float(target.mean()) >= 0.5)
    return accuracy(target, np.full(len(target), majority))


def result_row(method: str, base_ids: list[str], meta_model: str, horizon: int, val_acc: float, val_base: float, final_acc: float, final_base: float, final_rows: int, final_coverage: float, active_ticker_count: int) -> dict[str, Any]:
    return {
        "stacking_method": method,
        "base_models_used": ";".join(base_ids),
        "meta_model": meta_model,
        "horizon": horizon,
        "validation_accuracy": val_acc,
        "validation_baseline_accuracy": val_base,
        "validation_delta_vs_baseline": val_acc - val_base,
        "final_accuracy": final_acc,
        "final_baseline_accuracy": final_base,
        "final_delta_vs_baseline": final_acc - final_base,
        "delta_vs_60_31": final_acc - LOCKED_RF_H60,
        "delta_vs_60_43": final_acc - PREVIOUS_LOGISTIC_H40,
        "final_rows": final_rows,
        "final_coverage": final_coverage,
        "active_ticker_count": active_ticker_count,
        "selected_on_validation": False,
        "pass_60": final_acc >= 0.60,
        "pass_60_31": final_acc > LOCKED_RF_H60,
        "pass_60_43": final_acc > PREVIOUS_LOGISTIC_H40,
        "pass_65": final_acc >= 0.65,
        "claim_level": "exploratory_baseline60" if final_acc >= 0.60 else "exploratory",
    }


def fit_stack(method: str, meta_model: str, horizon: int, base_ids: list[str], validation: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any] | None:
    val_matrix = matrix_for_split(validation, horizon, base_ids)
    final_matrix = matrix_for_split(final, horizon, base_ids)
    if len(base_ids) < 2 or len(val_matrix) < 100 or final_matrix.empty:
        return None
    estimator = meta_estimator(meta_model)
    if estimator is None:
        return None
    estimator.fit(val_matrix[base_ids], val_matrix["target"].astype(int))
    val_pred = estimator.predict(val_matrix[base_ids])
    final_pred = estimator.predict(final_matrix[base_ids])
    val_acc = accuracy(val_matrix["target"], val_pred)
    final_acc = accuracy(final_matrix["target"], final_pred)
    return result_row(
        method,
        base_ids,
        meta_model,
        horizon,
        val_acc,
        majority_baseline(val_matrix["target"]),
        final_acc,
        majority_baseline(final_matrix["target"]),
        int(len(final_matrix)),
        1.0,
        int(final_matrix["ticker"].nunique()),
    )


def weighted_soft_vote(method: str, horizon: int, base_ids: list[str], validation: pd.DataFrame, final: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Any] | None:
    val_matrix = matrix_for_split(validation, horizon, base_ids)
    final_matrix = matrix_for_split(final, horizon, base_ids)
    if len(base_ids) < 2 or val_matrix.empty or final_matrix.empty:
        return None
    score_map = scores.set_index("base_model_id")["validation_delta_vs_baseline"].to_dict()
    weights = np.array([max(float(score_map.get(model_id, 0.0)), 0.0) for model_id in base_ids])
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(base_ids), dtype=float)
    weights = weights / weights.sum()
    val_score = val_matrix[base_ids].to_numpy() @ weights
    final_score = final_matrix[base_ids].to_numpy() @ weights
    val_pred = (val_score >= 0.5).astype(int)
    final_pred = (final_score >= 0.5).astype(int)
    return result_row(
        method,
        base_ids,
        "validation_weighted_soft_vote",
        horizon,
        accuracy(val_matrix["target"], val_pred),
        majority_baseline(val_matrix["target"]),
        accuracy(final_matrix["target"], final_pred),
        majority_baseline(final_matrix["target"]),
        int(len(final_matrix)),
        1.0,
        int(final_matrix["ticker"].nunique()),
    )


def diverse_model_ids(horizon_scores: pd.DataFrame) -> list[str]:
    chosen: list[str] = []
    for _family, group in horizon_scores.sort_values("validation_accuracy", ascending=False).groupby("model_family", sort=False):
        chosen.append(str(group.iloc[0]["base_model_id"]))
    return chosen


def run_variants(validation: pd.DataFrame, final: pd.DataFrame, scores: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in sorted(scores["horizon"].astype(int).unique()):
        horizon_scores = scores[scores["horizon"].astype(int) == horizon].sort_values("validation_accuracy", ascending=False)
        all_ids = [str(value) for value in horizon_scores["base_model_id"].tolist()]
        top_ids = all_ids[: min(6, len(all_ids))]
        diverse_ids = diverse_model_ids(horizon_scores)
        specs = [
            ("stack_all_logistic_meta", "logistic_meta", all_ids),
            ("stack_all_l2_logistic_meta", "l2_logistic_meta", all_ids),
            ("stack_all_lightgbm_shallow_meta", "lightgbm_shallow_meta", all_ids),
            ("stack_all_xgboost_shallow_meta", "xgboost_shallow_meta", all_ids),
            ("stack_top_validation_models_logistic_meta", "logistic_meta", top_ids),
            ("stack_diverse_models_logistic_meta", "logistic_meta", diverse_ids),
        ]
        for method, meta_model, model_ids in specs:
            row = fit_stack(method, meta_model, horizon, model_ids, validation, final)
            if row is not None:
                rows.append(row)
        for method, model_ids in [("validation_weighted_soft_vote_all", all_ids), ("validation_weighted_soft_vote_top", top_ids)]:
            row = weighted_soft_vote(method, horizon, model_ids, validation, final, scores)
            if row is not None:
                rows.append(row)
    return rows


def select_stack(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if math.isfinite(float(row["validation_accuracy"]))
        and int(row.get("active_ticker_count", 0)) == 30
        and abs(float(row.get("final_coverage", 0.0)) - 1.0) < 1e-12
    ]
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: (float(row["validation_accuracy"]), float(row["validation_delta_vs_baseline"]), -len(str(row["base_models_used"]).split(";"))))
    selected["selected_on_validation"] = True
    return selected


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def copy_outputs_to_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "stacking_validation_results.csv",
        "stacking_final_results.csv",
        "selected_stack_summary.csv",
        "stacking_run_log.md",
    ]:
        source = OUTPUT_DIR / name
        if source.exists():
            (REPORT_DIR / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    validation, final, scores = read_predictions()
    rows = run_variants(validation, final, scores)
    selected = select_stack(rows)
    selected_rows = [selected] if selected else []
    write_csv(OUTPUT_DIR / "stacking_validation_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "stacking_final_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_stack_summary.csv", selected_rows, RESULT_COLUMNS)
    manifest = read_json(OUTPUT_DIR / "oof_manifest.json")
    write_json(
        OUTPUT_DIR / "stacking_manifest.json",
        {
            "status": "completed",
            "track": "Track A canonical-like",
            "true_stacking_used": True,
            "meta_models_train_on_validation_oof_predictions_only": True,
            "final_labels_used_for_training_or_selection": False,
            "selection_rule": "validation_accuracy_then_validation_delta; final scoring only",
            "stacking_variant_count": len(rows),
            "selected_stack": json_ready(selected),
            "base_oof_manifest": manifest,
        },
    )
    if (OUTPUT_DIR / "stacking_manifest.json").exists():
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "stacking_manifest.json").write_text((OUTPUT_DIR / "stacking_manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    log = [
        "# Track A True Stacking All Algorithms Run Log",
        "",
        "- Status: completed.",
        "- True stacking: yes.",
        "- Base predictions: validation/OOF probabilities and final probabilities.",
        "- Meta-model training: validation/OOF predictions only.",
        "- Final labels used for selection: no.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking: no.",
        f"- Stacking variants evaluated: {len(rows)}.",
        f"- Selected method: `{selected.get('stacking_method') if selected else ''}`.",
        f"- Selected final accuracy: {selected.get('final_accuracy') if selected else ''}.",
        "",
    ]
    (OUTPUT_DIR / "stacking_run_log.md").write_text("\n".join(log), encoding="utf-8")
    copy_outputs_to_report_dir()
    print(f"true_stacking_status=completed selected={selected.get('stacking_method') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
