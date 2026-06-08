"""Build Track A all-algorithm validation/OOF and final predictions for stacking."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    BASE_MODEL_NAMES,
    HORIZONS,
    LOCKED_RF_H60,
    REPO_ROOT,
    active_stock_tickers,
    add_absolute_labels,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    make_model,
    rel,
    write_csv,
    write_json,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_all_algorithm_stacking"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_true_stacking_all_algorithms"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
FEATURE_SET = "feature_set_C_closest"

SCORE_COLUMNS = [
    "base_model_id",
    "model_family",
    "model_name",
    "horizon",
    "feature_set",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "final_accuracy",
    "final_baseline_accuracy",
    "final_delta_vs_baseline",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "claim_level",
]


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    return {
        "train": labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index,
        "validation": labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index,
        "final": labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index,
    }


def majority_from_train(y_train: pd.Series) -> int:
    return int(float(y_train.mean()) >= 0.5)


def accuracy(y_true: pd.Series, pred: np.ndarray | pd.Series) -> float:
    target = y_true.astype(int).to_numpy()
    prediction = np.asarray(pred).astype(int)
    return float((target == prediction).mean()) if len(target) else math.nan


def baseline_signal_probability(name: str, frame: pd.DataFrame, idx: pd.Index, y_train: pd.Series) -> np.ndarray:
    subset = frame.reindex(idx)
    if name == "previous_direction":
        return (subset["return_1"].fillna(0.0).to_numpy() > 0.0).astype(float)
    if name == "moving_average":
        signal = subset.get("roll_ret_20", pd.Series(0.0, index=subset.index)).fillna(0.0).to_numpy()
        return (signal > 0.0).astype(float)
    if name == "majority_train":
        return np.full(len(subset), float(majority_from_train(y_train)))
    if name == "always_up":
        return np.ones(len(subset), dtype=float)
    raise ValueError(f"unknown baseline signal {name}")


def prediction_rows(
    split: str,
    base_model_id: str,
    model_family: str,
    model_name: str,
    horizon: int,
    frame: pd.DataFrame,
    idx: pd.Index,
    y_true: pd.Series,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    pred = (probability >= 0.5).astype(int)
    subset = frame.reindex(idx)
    rows: list[dict[str, Any]] = []
    for row_id, dt_value, ticker, target, prob, hard in zip(idx, subset["datetime"], subset["ticker"], y_true.astype(int), probability, pred):
        rows.append(
            {
                "split": split,
                "row_id": int(row_id),
                "datetime": str(dt_value),
                "ticker": ticker,
                "horizon": horizon,
                "feature_set": FEATURE_SET,
                "base_model_id": base_model_id,
                "model_family": model_family,
                "model_name": model_name,
                "target": int(target),
                "probability_up": float(prob),
                "prediction": int(hard),
            }
        )
    return rows


def score_row(
    base_model_id: str,
    model_family: str,
    model_name: str,
    horizon: int,
    val_y: pd.Series,
    val_prob: np.ndarray,
    final_y: pd.Series,
    final_prob: np.ndarray,
    val_base: float,
    final_base: float,
    active_ticker_count: int,
) -> dict[str, Any]:
    val_acc = accuracy(val_y, val_prob >= 0.5)
    final_acc = accuracy(final_y, final_prob >= 0.5)
    return {
        "base_model_id": base_model_id,
        "model_family": model_family,
        "model_name": model_name,
        "horizon": horizon,
        "feature_set": FEATURE_SET,
        "validation_accuracy": val_acc,
        "validation_baseline_accuracy": val_base,
        "validation_delta_vs_baseline": val_acc - val_base,
        "final_accuracy": final_acc,
        "final_baseline_accuracy": final_base,
        "final_delta_vs_baseline": final_acc - final_base,
        "final_rows": int(len(final_y)),
        "final_coverage": 1.0,
        "active_ticker_count": active_ticker_count,
        "claim_level": "base_model_screening_only",
    }


def pairwise_diagnostics(predictions: pd.DataFrame, scores: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_by_id = {row["base_model_id"]: row for row in scores}
    corr_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    validation = predictions[predictions["split"] == "validation"]
    for horizon, group in validation.groupby("horizon", sort=True):
        errors = group.assign(error=(group["target"].astype(int) != group["prediction"].astype(int)).astype(int))
        error_wide = errors.pivot_table(index="row_id", columns="base_model_id", values="error", aggfunc="last")
        pred_wide = group.pivot_table(index="row_id", columns="base_model_id", values="prediction", aggfunc="last")
        model_ids = list(error_wide.columns)
        for i, model_a in enumerate(model_ids):
            for model_b in model_ids[i + 1 :]:
                pair_errors = error_wide[[model_a, model_b]].dropna()
                pair_preds = pred_wide[[model_a, model_b]].dropna()
                corr = pair_errors[model_a].corr(pair_errors[model_b]) if len(pair_errors) > 1 else math.nan
                disagreement = float((pair_preds[model_a].astype(int) != pair_preds[model_b].astype(int)).mean()) if len(pair_preds) else math.nan
                corr_rows.append(
                    {
                        "horizon": int(horizon),
                        "model_a": model_a,
                        "model_b": model_b,
                        "family_a": score_by_id.get(model_a, {}).get("model_family", ""),
                        "family_b": score_by_id.get(model_b, {}).get("model_family", ""),
                        "validation_error_correlation": corr,
                        "rows": int(len(pair_errors)),
                    }
                )
                disagreement_rows.append(
                    {
                        "horizon": int(horizon),
                        "model_a": model_a,
                        "model_b": model_b,
                        "validation_disagreement_rate": disagreement,
                        "rows": int(len(pair_preds)),
                    }
                )
    return corr_rows, disagreement_rows


def copy_outputs_to_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "base_oof_predictions.csv",
        "base_final_predictions.csv",
        "base_model_scores.csv",
        "base_model_error_correlation.csv",
        "base_model_disagreement.csv",
        "oof_manifest.json",
    ]:
        source = OUTPUT_DIR / name
        if source.exists():
            (REPORT_DIR / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    features, feature_cols = build_feature_set_c(load_stock_data(tickers), load_index_data())
    all_prediction_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    baseline_models = ["previous_direction", "moving_average", "majority_train", "always_up"]

    for horizon in HORIZONS:
        labels = add_absolute_labels(features, horizon)
        idx = split_indices(features, labels)
        train_y = labels.reindex(idx["train"]).astype(int)
        val_y = labels.reindex(idx["validation"]).astype(int)
        final_y = labels.reindex(idx["final"]).astype(int)
        val_majority = majority_from_train(train_y)
        val_base = accuracy(val_y, np.full(len(val_y), val_majority))
        final_base = accuracy(final_y, np.full(len(final_y), val_majority))
        x_train = features.reindex(idx["train"])[feature_cols]
        x_val = features.reindex(idx["validation"])[feature_cols]
        x_final = features.reindex(idx["final"])[feature_cols]

        for model_name in BASE_MODEL_NAMES:
            model = make_model(model_name)
            if model is None:
                continue
            pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
            try:
                pipeline.fit(x_train, train_y)
                val_prob = pipeline.predict_proba(x_val)[:, 1]
                final_prob = pipeline.predict_proba(x_final)[:, 1]
            except Exception:
                continue
            base_model_id = f"{model_name}__h{horizon}__{FEATURE_SET}"
            all_prediction_rows.extend(prediction_rows("validation", base_model_id, model_name, model_name, horizon, features, idx["validation"], val_y, val_prob))
            all_prediction_rows.extend(prediction_rows("final", base_model_id, model_name, model_name, horizon, features, idx["final"], final_y, final_prob))
            final_ticker_count = int(features.reindex(idx["final"])["ticker"].nunique())
            score_rows.append(score_row(base_model_id, model_name, model_name, horizon, val_y, val_prob, final_y, final_prob, val_base, final_base, final_ticker_count))

        for model_name in baseline_models:
            val_prob = baseline_signal_probability(model_name, features, idx["validation"], train_y)
            final_prob = baseline_signal_probability(model_name, features, idx["final"], train_y)
            base_model_id = f"{model_name}__h{horizon}__{FEATURE_SET}"
            all_prediction_rows.extend(prediction_rows("validation", base_model_id, "baseline_signal", model_name, horizon, features, idx["validation"], val_y, val_prob))
            all_prediction_rows.extend(prediction_rows("final", base_model_id, "baseline_signal", model_name, horizon, features, idx["final"], final_y, final_prob))
            final_ticker_count = int(features.reindex(idx["final"])["ticker"].nunique())
            score_rows.append(score_row(base_model_id, "baseline_signal", model_name, horizon, val_y, val_prob, final_y, final_prob, val_base, final_base, final_ticker_count))

    predictions = pd.DataFrame(all_prediction_rows)
    validation_predictions = predictions[predictions["split"] == "validation"].copy()
    final_predictions = predictions[predictions["split"] == "final"].copy()
    validation_predictions.to_csv(OUTPUT_DIR / "base_oof_predictions.csv", index=False)
    final_predictions.to_csv(OUTPUT_DIR / "base_final_predictions.csv", index=False)
    write_csv(OUTPUT_DIR / "base_model_scores.csv", score_rows, SCORE_COLUMNS)
    corr_rows, disagreement_rows = pairwise_diagnostics(predictions, score_rows)
    write_csv(OUTPUT_DIR / "base_model_error_correlation.csv", corr_rows)
    write_csv(OUTPUT_DIR / "base_model_disagreement.csv", disagreement_rows)
    write_json(
        OUTPUT_DIR / "oof_manifest.json",
        {
            "status": "completed",
            "track": "Track A canonical-like",
            "feature_set": FEATURE_SET,
            "horizons": HORIZONS,
            "base_algorithms": BASE_MODEL_NAMES,
            "baseline_signals": baseline_models,
            "active_ticker_count": len(tickers),
            "base_model_count": len(score_rows),
            "validation_prediction_rows": int(len(validation_predictions)),
            "final_prediction_rows": int(len(final_predictions)),
            "base_models_train_only_on_training_fold": True,
            "validation_oof_predictions_used": True,
            "final_labels_used_for_training_or_selection": False,
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "data_fetch": False,
            "locked_rf_h60_reference": LOCKED_RF_H60,
        },
    )
    copy_outputs_to_report_dir()
    print(f"all_algorithm_oof_status=completed base_model_count={len(score_rows)} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
