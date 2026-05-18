"""Run Track A canonical VN30 hourly expanded model comparison."""

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

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import (
    BASE_MODEL_NAMES,
    HISTORICAL_FINAL_ROWS,
    HORIZONS,
    LOCKED_RF_H60,
    REPO_ROOT,
    RESULT_COLUMNS_CANONICAL,
    active_stock_tickers,
    add_absolute_labels,
    build_feature_set_c,
    load_index_data,
    load_stock_data,
    make_model,
    markdown_table,
    pct,
    rel,
    score_accuracy,
    selected_by_validation,
    write_csv,
    write_json,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_canonical_expanded_model_comparison"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_canonical_expanded_model_comparison"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")


def fit_predict(frame: pd.DataFrame, labels: pd.Series, feature_cols: list[str], model_name: str) -> dict[str, Any] | None:
    model = make_model(model_name)
    if model is None:
        return None
    train_idx = frame.index[frame["datetime"].le(TRAIN_END)]
    val_idx = frame.index[frame["datetime"].between(VAL_START, VAL_END)]
    final_idx = frame.index[frame["datetime"].ge(EVAL_START)]
    train_y = labels.reindex(train_idx).dropna().astype(int)
    val_y = labels.reindex(val_idx).dropna().astype(int)
    final_y = labels.reindex(final_idx).dropna().astype(int)
    if len(train_y) < 100 or len(val_y) < 20 or len(final_y) < 20:
        return None
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    pipeline.fit(frame.reindex(train_y.index)[feature_cols], train_y)
    val_pred = pipeline.predict(frame.reindex(val_y.index)[feature_cols])
    final_pred = pipeline.predict(frame.reindex(final_y.index)[feature_cols])
    val_proba = pipeline.predict_proba(frame.reindex(val_y.index)[feature_cols])[:, 1] if hasattr(pipeline, "predict_proba") else val_pred
    final_proba = pipeline.predict_proba(frame.reindex(final_y.index)[feature_cols])[:, 1] if hasattr(pipeline, "predict_proba") else final_pred
    val_acc, val_rows = score_accuracy(val_y, val_pred)
    final_acc, final_rows = score_accuracy(final_y, final_pred)
    return {
        "validation_accuracy": val_acc,
        "validation_rows": val_rows,
        "final_accuracy": final_acc,
        "final_rows": final_rows,
        "final_coverage": 1.0,
        "val_pred": pd.DataFrame({"target": val_y, "prediction": val_pred, "probability_up": val_proba}, index=val_y.index),
        "final_pred": pd.DataFrame({"target": final_y, "prediction": final_pred, "probability_up": final_proba}, index=final_y.index),
    }


def result_row(model_name: str, horizon: int, feature_set: str, metrics: dict[str, Any]) -> dict[str, Any]:
    final_accuracy = float(metrics["final_accuracy"])
    return {
        "track": "canonical",
        "model": model_name,
        "horizon": horizon,
        "feature_set": feature_set,
        "validation_accuracy": metrics["validation_accuracy"],
        "final_accuracy": final_accuracy,
        "final_rows": metrics["final_rows"],
        "final_coverage": metrics["final_coverage"],
        "delta_vs_locked_60_31": final_accuracy - LOCKED_RF_H60,
        "pass_60_31": final_accuracy > LOCKED_RF_H60,
        "pass_65": final_accuracy >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory",
    }


def weighted_vote(base_predictions: dict[str, dict[str, Any]], selected_ids: list[str], split: str, weights: dict[str, float]) -> dict[str, Any] | None:
    pieces = []
    prediction_key = "val_pred" if split == "validation" else "final_pred"
    for cid in selected_ids:
        pred = base_predictions[cid][prediction_key].copy()
        pred = pred.rename(columns={"probability_up": cid})
        pieces.append(pred[["target", cid]])
    if len(pieces) < 2:
        return None
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.join(piece.drop(columns=["target"]), how="inner")
    if wide.empty:
        return None
    total = sum(max(weights.get(cid, 0.0), 0.0) for cid in selected_ids)
    if total <= 0:
        score = wide[selected_ids].mean(axis=1)
    else:
        score = sum(wide[cid] * max(weights.get(cid, 0.0), 0.0) for cid in selected_ids) / total
    pred = (score >= 0.5).astype(int)
    accuracy = float((wide["target"].astype(int) == pred.astype(int)).mean())
    return {"accuracy": accuracy, "rows": len(wide), "coverage": 1.0}


def stacking_logistic(base_predictions: dict[str, dict[str, Any]], selected_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    val_parts = []
    final_parts = []
    for cid in selected_ids:
        val_parts.append(base_predictions[cid]["val_pred"].rename(columns={"probability_up": cid})[["target", cid]])
        final_parts.append(base_predictions[cid]["final_pred"].rename(columns={"probability_up": cid})[["target", cid]])
    val = val_parts[0]
    final = final_parts[0]
    for piece in val_parts[1:]:
        val = val.join(piece.drop(columns=["target"]), how="inner")
    for piece in final_parts[1:]:
        final = final.join(piece.drop(columns=["target"]), how="inner")
    if len(val) < 100 or final.empty:
        return None
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))])
    model.fit(val[selected_ids], val["target"].astype(int))
    val_pred = model.predict(val[selected_ids])
    final_pred = model.predict(final[selected_ids])
    return (
        {"accuracy": float((val["target"].astype(int) == val_pred.astype(int)).mean()), "rows": len(val), "coverage": 1.0},
        {"accuracy": float((final["target"].astype(int) == final_pred.astype(int)).mean()), "rows": len(final), "coverage": 1.0},
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    feature_df, feature_cols = build_feature_set_c(stock_df, index_data)
    rows: list[dict[str, Any]] = []
    base_predictions: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        labels = add_absolute_labels(feature_df, horizon)
        for model_name in BASE_MODEL_NAMES:
            metrics = fit_predict(feature_df, labels, feature_cols, model_name)
            if metrics is None:
                continue
            row = result_row(model_name, horizon, "feature_set_C_closest", metrics)
            rows.append(row)
            base_predictions[f"{model_name}__h{horizon}__feature_set_C_closest"] = metrics
    selected = selected_by_validation(rows)
    if selected:
        selected_id = f"{selected['model']}__h{int(selected['horizon'])}__{selected['feature_set']}"
    candidates_for_ensemble = sorted(rows, key=lambda row: float(row["validation_accuracy"]), reverse=True)[:6]
    selected_ids = [f"{row['model']}__h{int(row['horizon'])}__{row['feature_set']}" for row in candidates_for_ensemble]
    weights = {cid: max(float(row["validation_accuracy"]) - 0.5, 0.0) for cid, row in zip(selected_ids, candidates_for_ensemble)}
    ensemble_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in candidates_for_ensemble}):
        horizon_ids = [cid for cid in selected_ids if f"__h{horizon}__" in cid]
        if len(horizon_ids) < 2:
            continue
        val = weighted_vote(base_predictions, horizon_ids, "validation", weights)
        final = weighted_vote(base_predictions, horizon_ids, "final", weights)
        if val and final:
            metrics = {"validation_accuracy": val["accuracy"], "final_accuracy": final["accuracy"], "final_rows": final["rows"], "final_coverage": final["coverage"]}
            ensemble_rows.append(result_row("validation_weighted_soft_voting", horizon, "feature_set_C_closest", metrics))
        stacked = stacking_logistic(base_predictions, horizon_ids)
        if stacked:
            val_s, final_s = stacked
            metrics = {"validation_accuracy": val_s["accuracy"], "final_accuracy": final_s["accuracy"], "final_rows": final_s["rows"], "final_coverage": final_s["coverage"]}
            ensemble_rows.append(result_row("stacking_logistic_oof", horizon, "feature_set_C_closest", metrics))
    rows.extend(ensemble_rows)
    selected = selected_by_validation(rows)
    selected_summary = [selected] if selected else []
    baseline_rows = [row for row in rows if row["model"] == "random_forest" and int(row["horizon"]) == 60]
    write_json(OUTPUT_DIR / "run_config.json", {"track": "canonical", "models": BASE_MODEL_NAMES, "horizons": HORIZONS, "feature_set": "feature_set_C_closest", "train_end": str(TRAIN_END), "validation_window": f"{VAL_START} to {VAL_END}", "final_start": str(EVAL_START), "data_fetch": False, "confidence_abstention": False})
    write_json(OUTPUT_DIR / "canonical_comparison_manifest.json", {"track": "canonical", "status": "completed", "active_ticker_count": len(tickers), "feature_count": len(feature_cols), "historical_locked_accuracy": LOCKED_RF_H60, "historical_final_rows": HISTORICAL_FINAL_ROWS, "rf_h60_reproduced": any(abs(float(row["final_accuracy"]) - LOCKED_RF_H60) <= 0.001 and int(row["final_rows"]) == HISTORICAL_FINAL_ROWS for row in baseline_rows), "selected_on_validation": bool(selected)})
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS_CANONICAL)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS_CANONICAL)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", selected_summary, RESULT_COLUMNS_CANONICAL)
    write_csv(OUTPUT_DIR / "baseline_reproduction_row.csv", baseline_rows, RESULT_COLUMNS_CANONICAL)
    write_csv(OUTPUT_DIR / "model_comparison_summary.csv", rows, RESULT_COLUMNS_CANONICAL)
    write_csv(OUTPUT_DIR / "ensemble_summary.csv", ensemble_rows, RESULT_COLUMNS_CANONICAL)
    for name in ["run_config.json", "canonical_comparison_manifest.json"]:
        source = OUTPUT_DIR / name
        target = REPORT_DIR / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ["validation_candidate_results.csv", "final_candidate_results.csv", "selected_candidate_summary.csv", "baseline_reproduction_row.csv", "model_comparison_summary.csv", "ensemble_summary.csv"]:
        source = OUTPUT_DIR / name
        target = REPORT_DIR / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    best_final = max(rows, key=lambda row: float(row["final_accuracy"])) if rows else {}
    report_rows = [
        {"metric": "rf_h60_reproduced", "value": str(any(abs(float(row["final_accuracy"]) - LOCKED_RF_H60) <= 0.001 and int(row["final_rows"]) == HISTORICAL_FINAL_ROWS for row in baseline_rows)).lower()},
        {"metric": "selected_model", "value": selected.get("model", "") if selected else ""},
        {"metric": "selected_final_accuracy", "value": pct(selected.get("final_accuracy", math.nan)) if selected else ""},
        {"metric": "best_final_model", "value": best_final.get("model", "")},
        {"metric": "best_final_accuracy", "value": pct(best_final.get("final_accuracy", math.nan))},
    ]
    (REPORT_DIR / "canonical_comparison_report.md").write_text("# Track A Canonical Expanded Model Comparison\n\n" + markdown_table(["metric", "value"], report_rows) + "\n", encoding="utf-8")
    print(f"canonical_comparison_status=completed selected={selected.get('model') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
