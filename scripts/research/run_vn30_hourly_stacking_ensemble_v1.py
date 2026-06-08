"""Run VN30 hourly ensemble/stacking v1 from validation-selected candidates."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import BASELINE_STOCK_RF_H60_REFERENCE, write_csv, write_json  # noqa: E402

SCREENING_DIR = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_expanded_model_pool"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_stacking_ensemble_v1"
LOCKED_BASELINE = BASELINE_STOCK_RF_H60_REFERENCE
ACTIVE_TICKER_COUNT = 30

ENSEMBLE_COLUMNS = [
    "ensemble_method",
    "base_models_used",
    "horizon",
    "validation_accuracy",
    "validation_baseline_delta",
    "final_accuracy",
    "final_baseline_delta",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "selected_on_validation",
    "pass_locked_60_31",
    "pass_65",
    "claim_level",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions_path = SCREENING_DIR / "candidate_predictions.csv"
    shortlist_path = REPORT_DIR / "ensemble_shortlist.csv"
    candidate_results_path = SCREENING_DIR / "validation_candidate_results.csv"
    if not predictions_path.exists() or not shortlist_path.exists() or not candidate_results_path.exists():
        raise FileNotFoundError("Required screening/shortlist outputs are missing.")
    predictions = pd.read_csv(predictions_path, low_memory=False)
    shortlist = pd.read_csv(shortlist_path)
    candidate_results = pd.read_csv(candidate_results_path)
    return predictions, shortlist, candidate_results


def add_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["key"] = out["datetime"].astype(str) + "|" + out["ticker"].astype(str) + "|" + out["horizon"].astype(str)
    return out


def baseline_accuracy(predictions: pd.DataFrame, split: str, horizon: int) -> float:
    baseline = predictions[
        predictions["candidate_id"].str.startswith("majority__")
        & predictions["split"].eq(split)
        & (pd.to_numeric(predictions["horizon"], errors="coerce") == horizon)
    ]
    if baseline.empty:
        return math.nan
    return float(pd.to_numeric(baseline["is_correct"], errors="coerce").mean())


def evaluate_rows(scored: pd.DataFrame, split: str) -> dict[str, Any]:
    scoped = scored[scored["split"].eq(split)].dropna(subset=["target_direction", "prediction"])
    rows = len(scoped)
    if rows == 0:
        return {"accuracy": math.nan, "rows": 0, "coverage": 0.0}
    accuracy = float((scoped["target_direction"].astype(int) == scoped["prediction"].astype(int)).mean())
    return {"accuracy": accuracy, "rows": rows, "coverage": rows / len(scored[scored["split"].eq(split)])}


def build_wide(predictions: pd.DataFrame, candidate_ids: list[str], split: str, horizon: int, value_column: str) -> pd.DataFrame:
    scoped = add_key(predictions[
        predictions["candidate_id"].isin(candidate_ids)
        & predictions["split"].eq(split)
        & (pd.to_numeric(predictions["horizon"], errors="coerce") == horizon)
    ])
    return scoped.pivot_table(index="key", columns="candidate_id", values=value_column, aggfunc="first")


def base_truth(predictions: pd.DataFrame, split: str, horizon: int) -> pd.DataFrame:
    scoped = add_key(predictions[
        predictions["split"].eq(split)
        & (pd.to_numeric(predictions["horizon"], errors="coerce") == horizon)
    ])
    truth = scoped.groupby("key", as_index=False).agg(
        target_direction=("target_direction", "first"),
        datetime=("datetime", "first"),
        ticker=("ticker", "first"),
    )
    truth["split"] = split
    return truth


def vote_predictions(pred_wide: pd.DataFrame, prob_wide: pd.DataFrame, method: str, weights: dict[str, float]) -> pd.Series:
    if method == "majority_vote":
        return (pred_wide.mean(axis=1) >= 0.5).astype(int)
    if method == "soft_vote_equal_weight":
        return (prob_wide.mean(axis=1) >= 0.5).astype(int)
    if method == "soft_vote_validation_weighted":
        aligned = prob_wide.copy()
        total = sum(max(weights.get(col, 0.0), 0.0) for col in aligned.columns)
        if total <= 0:
            return (aligned.mean(axis=1) >= 0.5).astype(int)
        score = sum(aligned[col] * max(weights.get(col, 0.0), 0.0) for col in aligned.columns) / total
        return (score >= 0.5).astype(int)
    raise ValueError(method)


def run_vote_method(predictions: pd.DataFrame, candidate_ids: list[str], horizon: int, method: str, weights: dict[str, float]) -> pd.DataFrame:
    scored_parts = []
    for split in ("validation", "final"):
        truth = base_truth(predictions, split, horizon)
        pred_wide = build_wide(predictions, candidate_ids, split, horizon, "prediction")
        prob_wide = build_wide(predictions, candidate_ids, split, horizon, "probability_up")
        common = truth[truth["key"].isin(pred_wide.dropna().index)].copy()
        pred = vote_predictions(pred_wide.loc[common["key"]].astype(float), prob_wide.loc[common["key"]].astype(float), method, weights)
        common["prediction"] = pred.to_numpy()
        scored_parts.append(common)
    return pd.concat(scored_parts, ignore_index=True)


def run_stacking(predictions: pd.DataFrame, candidate_ids: list[str], horizon: int, method: str) -> pd.DataFrame | None:
    train_x = build_wide(predictions, candidate_ids, "validation", horizon, "probability_up").dropna()
    train_truth = base_truth(predictions, "validation", horizon).set_index("key")
    common_train = train_x.index.intersection(train_truth.index)
    if len(common_train) < 100:
        return None
    y = train_truth.loc[common_train, "target_direction"].astype(int)
    if y.nunique() < 2:
        return None
    if method == "stacking_logistic_oof":
        model: Any = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    elif method == "stacking_lightgbm_shallow_oof" and LGBMClassifier is not None:
        model = LGBMClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, min_child_samples=40, random_state=42, verbose=-1)
    else:
        return None
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipeline.fit(train_x.loc[common_train], y)
    scored_parts = []
    for split in ("validation", "final"):
        x = build_wide(predictions, candidate_ids, split, horizon, "probability_up").dropna()
        truth = base_truth(predictions, split, horizon).set_index("key")
        common = x.index.intersection(truth.index)
        if len(common) == 0:
            continue
        scored = truth.loc[common].reset_index()
        scored["prediction"] = pipeline.predict(x.loc[common]).astype(int)
        scored_parts.append(scored)
    if not scored_parts:
        return None
    return pd.concat(scored_parts, ignore_index=True)


def result_row(method: str, candidate_ids: list[str], horizon: int, scored: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    validation = evaluate_rows(scored, "validation")
    final = evaluate_rows(scored, "final")
    val_base = baseline_accuracy(predictions, "validation", horizon)
    final_base = baseline_accuracy(predictions, "final", horizon)
    final_acc = final["accuracy"]
    return {
        "ensemble_method": method,
        "base_models_used": ";".join(candidate_ids),
        "horizon": horizon,
        "validation_accuracy": validation["accuracy"],
        "validation_baseline_delta": validation["accuracy"] - val_base if math.isfinite(validation["accuracy"]) and math.isfinite(val_base) else math.nan,
        "final_accuracy": final_acc,
        "final_baseline_delta": final_acc - final_base if math.isfinite(final_acc) and math.isfinite(final_base) else math.nan,
        "final_rows": final["rows"],
        "final_coverage": final["coverage"],
        "active_ticker_count": ACTIVE_TICKER_COUNT,
        "selected_on_validation": False,
        "pass_locked_60_31": bool(math.isfinite(final_acc) and final_acc > LOCKED_BASELINE),
        "pass_65": bool(math.isfinite(final_acc) and final_acc >= 0.65),
        "claim_level": "exploratory",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions, shortlist, candidate_results = load_inputs()
    candidate_ids = shortlist["candidate_id"].dropna().astype(str).tolist()
    if len(candidate_ids) < 2:
        raise RuntimeError("Need at least two validation-selected base models for ensemble.")
    shortlist["horizon"] = pd.to_numeric(shortlist["horizon"], errors="coerce").astype(int)
    horizons = sorted(shortlist["horizon"].unique().tolist())
    candidate_results["candidate_id"] = candidate_results.apply(lambda row: f"{row['model_name']}__h{int(row['horizon'])}__{row['feature_set']}", axis=1)
    weights = {
        row["candidate_id"]: max(float(row.get("validation_delta_vs_baseline", 0.0) or 0.0), 0.0)
        for _idx, row in candidate_results[candidate_results["candidate_id"].isin(candidate_ids)].iterrows()
    }
    methods = ["majority_vote", "soft_vote_equal_weight", "soft_vote_validation_weighted", "stacking_logistic_oof", "stacking_lightgbm_shallow_oof"]
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon_ids = shortlist[shortlist["horizon"].eq(horizon)]["candidate_id"].astype(str).tolist()
        if len(horizon_ids) < 2:
            continue
        for method in methods:
            if method.startswith("stacking"):
                scored = run_stacking(predictions, horizon_ids, horizon, method)
                if scored is None:
                    continue
            else:
                scored = run_vote_method(predictions, horizon_ids, horizon, method, weights)
            rows.append(result_row(method, horizon_ids, horizon, scored, predictions))

    selected = None
    valid = [row for row in rows if math.isfinite(float(row.get("validation_accuracy") or math.nan))]
    if valid:
        selected = max(valid, key=lambda row: (float(row["validation_accuracy"]), float(row.get("validation_baseline_delta") or -1)))
        selected["selected_on_validation"] = True

    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "status": "completed",
            "input_predictions": rel(SCREENING_DIR / "candidate_predictions.csv"),
            "shortlist": rel(REPORT_DIR / "ensemble_shortlist.csv"),
            "methods": methods,
            "selection_rule": "validation-only",
            "final_labels_used_for_training": False,
            "final_accuracy_used_for_selection": False,
            "confidence_abstention": False,
            "ticker_subset": False,
            "active_ticker_count": ACTIVE_TICKER_COUNT,
        },
    )
    write_json(
        OUTPUT_DIR / "ensemble_manifest.json",
        {
            "status": "completed",
            "selected_ensemble": selected,
            "base_models_used": candidate_ids,
            "selected_on_validation": bool(selected),
            "final_window_scoring_only": True,
            "stacking_training_source": "validation prediction rows only",
            "final_labels_used_for_meta_model_training": False,
            "stock_only": True,
            "active_ticker_count": ACTIVE_TICKER_COUNT,
            "full_coverage": True,
        },
    )
    write_csv(OUTPUT_DIR / "validation_ensemble_results.csv", rows, ENSEMBLE_COLUMNS)
    write_csv(OUTPUT_DIR / "final_ensemble_results.csv", rows, ENSEMBLE_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_ensemble_summary.csv", [selected] if selected else [], ENSEMBLE_COLUMNS)
    write_csv(
        OUTPUT_DIR / "ensemble_baseline_delta_summary.csv",
        [
            {
                "ensemble_method": row["ensemble_method"],
                "horizon": row["horizon"],
                "validation_baseline_delta": row["validation_baseline_delta"],
                "final_baseline_delta": row["final_baseline_delta"],
                "locked_rf_h60_baseline": LOCKED_BASELINE,
                "delta_vs_locked_baseline": row["final_accuracy"] - LOCKED_BASELINE if math.isfinite(float(row["final_accuracy"])) else math.nan,
            }
            for row in rows
        ],
    )
    log = [
        "# VN30 Hourly Stacking Ensemble V1 Run Log",
        "",
        "- Status: completed.",
        "- Selection: validation-only.",
        "- Final labels for meta-model training: no.",
        "- Final accuracy for selection: no.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        f"- Selected ensemble: `{selected.get('ensemble_method') if selected else ''}`.",
        f"- Final accuracy: {pct(selected.get('final_accuracy') if selected else math.nan)}.",
        "",
    ]
    (OUTPUT_DIR / "ensemble_run_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"ensemble_status=completed selected={selected.get('ensemble_method') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

