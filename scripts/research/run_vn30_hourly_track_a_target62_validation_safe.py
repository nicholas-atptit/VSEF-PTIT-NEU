"""Run Track A target62 validation-safe linear rule."""

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
from scripts.research.vn30_hourly_track_a_regime_feature_v2 import build_regime_feature_v2  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_target62_validation_safe"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
HORIZONS = [40, 60, 80]
THRESHOLDS = [0.45, 0.50, 0.55]
BASELINE_LOGISTIC_H40 = 0.6043200785468826
TARGET62 = 0.62
TOLERANCE = 0.0075
MODEL_RANK = {
    "l2_logistic": 0,
    "balanced_l2_logistic": 1,
    "elasticnet_logistic": 2,
    "ridge_logistic_c1": 3,
}
FEATURE_RANK = {
    "regime_feature_v2": 0,
    "feature_set_C_closest": 1,
    "stock_lagged_rolling_plus_index_context": 2,
}
RESULT_COLUMNS = [
    "model",
    "horizon",
    "feature_set",
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
    "delta_vs_60_43",
    "delta_vs_60_31",
    "pass_60",
    "pass_60_43",
    "pass_62",
    "pass_65",
    "selected_by_preregistered_rule",
    "claim_level",
]


def split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    return {
        "train": labels.reindex(features.index[features["datetime"].le(TRAIN_END)]).dropna().index,
        "validation": labels.reindex(features.index[features["datetime"].between(VAL_START, VAL_END)]).dropna().index,
        "final": labels.reindex(features.index[features["datetime"].ge(EVAL_START)]).dropna().index,
    }


def candidate_model(model_name: str) -> LogisticRegression:
    if model_name == "l2_logistic":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=42)
    if model_name == "balanced_l2_logistic":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, class_weight="balanced", random_state=42)
    if model_name == "elasticnet_logistic":
        return LogisticRegression(max_iter=2500, solver="saga", penalty="elasticnet", C=0.3, l1_ratio=0.2, class_weight="balanced", random_state=42)
    if model_name == "ridge_logistic_c1":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, class_weight=None, random_state=42)
    raise ValueError(f"unknown model {model_name}")


def model_names() -> list[str]:
    return ["l2_logistic", "balanced_l2_logistic", "elasticnet_logistic", "ridge_logistic_c1"]


def build_feature_sets() -> dict[str, tuple[pd.DataFrame, list[str], dict[str, Any]]]:
    tickers = active_stock_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    regime_frame, regime_cols, regime_manifest = build_regime_feature_v2(stock_df, index_data)
    base_frame, base_cols = build_feature_set_c(stock_df, index_data)
    context_cols = [
        col
        for col in base_cols
        if col.startswith(("return_", "rolling_", "close_sma_", "momentum_", "lag_ret_", "roll_", "volume_", "high_low_", "open_close_", "close_position_", "rsi_", "macd", "vnindex", "vn30", "hnx", "upcom"))
        or col in {"day_of_week", "day_of_month", "month", "quarter", "hour", "minute", "market_minus_stock_ret"}
    ]
    return {
        "regime_feature_v2": (regime_frame, regime_cols, regime_manifest),
        "feature_set_C_closest": (
            base_frame,
            base_cols,
            {
                "feature_set": "feature_set_C_closest",
                "feature_count": len(base_cols),
                "leakage_safe": True,
                "future_return_features": False,
                "future_regime_features": False,
                "final_label_derived_features": False,
                "final_period_manual_filters": False,
            },
        ),
        "stock_lagged_rolling_plus_index_context": (
            base_frame,
            context_cols,
            {
                "feature_set": "stock_lagged_rolling_plus_index_context",
                "feature_count": len(context_cols),
                "track_a_compatible": True,
                "leakage_safe": True,
                "future_return_features": False,
                "future_regime_features": False,
                "final_label_derived_features": False,
                "final_period_manual_filters": False,
            },
        ),
    }


def accuracy(y_true: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    return float((y_true.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean()) if len(y_true) else math.nan


def train_majority(train_y: pd.Series) -> int:
    return int(float(train_y.mean()) >= 0.5)


def result_row(
    model_name: str,
    horizon: int,
    feature_set: str,
    threshold: float,
    val_y: pd.Series,
    val_prob: np.ndarray,
    final_y: pd.Series,
    final_prob: np.ndarray,
    val_base: float,
    final_base: float,
    active_ticker_count: int,
) -> dict[str, Any]:
    val_pred = (val_prob >= threshold).astype(int)
    final_pred = (final_prob >= threshold).astype(int)
    val_acc = accuracy(val_y, val_pred)
    final_acc = accuracy(final_y, final_pred)
    return {
        "model": model_name,
        "horizon": horizon,
        "feature_set": feature_set,
        "threshold": threshold,
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
        "pass_60_43": final_acc > BASELINE_LOGISTIC_H40,
        "pass_62": final_acc >= TARGET62,
        "pass_65": final_acc >= 0.65,
        "selected_by_preregistered_rule": False,
        "claim_level": "target62_exploratory" if final_acc >= TARGET62 else ("exploratory_baseline60" if final_acc >= 0.60 else "exploratory"),
    }


def run_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_sets = build_feature_sets()
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    active_ticker_count = len(active_stock_tickers())
    for feature_set, (features, feature_cols, manifest) in feature_sets.items():
        manifests[feature_set] = manifest
        for horizon in HORIZONS:
            labels = add_absolute_labels(features, horizon)
            idx = split_indices(features, labels)
            train_y = labels.reindex(idx["train"]).astype(int)
            val_y = labels.reindex(idx["validation"]).astype(int)
            final_y = labels.reindex(idx["final"]).astype(int)
            majority = train_majority(train_y)
            val_base = accuracy(val_y, np.full(len(val_y), majority))
            final_base = accuracy(final_y, np.full(len(final_y), majority))
            x_train = features.reindex(idx["train"])[feature_cols]
            x_val = features.reindex(idx["validation"])[feature_cols]
            x_final = features.reindex(idx["final"])[feature_cols]
            for model_name in model_names():
                pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", candidate_model(model_name))])
                try:
                    pipeline.fit(x_train, train_y)
                    val_prob = pipeline.predict_proba(x_val)[:, 1]
                    final_prob = pipeline.predict_proba(x_final)[:, 1]
                except Exception:
                    continue
                for threshold in THRESHOLDS:
                    rows.append(result_row(model_name, horizon, feature_set, threshold, val_y, val_prob, final_y, final_prob, val_base, final_base, active_ticker_count))
    return rows, manifests


def select_by_preregistered_rule(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if math.isfinite(float(row["validation_accuracy"]))
        and float(row["validation_delta_vs_baseline"]) > 0
        and int(row["active_ticker_count"]) == 30
        and abs(float(row["final_coverage"]) - 1.0) < 1e-12
    ]
    if not valid:
        return None
    best_validation = max(float(row["validation_accuracy"]) for row in valid)
    eligible = [row for row in valid if float(row["validation_accuracy"]) >= best_validation - TOLERANCE]
    selected = min(
        eligible,
        key=lambda row: (
            0 if int(row["horizon"]) == 40 else (1 if int(row["horizon"]) == 60 else 2),
            MODEL_RANK.get(str(row["model"]), 99),
            abs(float(row["threshold"]) - 0.50),
            FEATURE_RANK.get(str(row["feature_set"]), 99),
            -float(row["validation_accuracy"]),
        ),
    )
    selected["selected_by_preregistered_rule"] = True
    return selected


def copy_outputs_to_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_config.json",
        "validation_candidate_results.csv",
        "final_candidate_results.csv",
        "selected_candidate_summary.csv",
        "target62_run_log.md",
    ]:
        source = OUTPUT_DIR / name
        if source.exists():
            (REPORT_DIR / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows, feature_manifests = run_candidates()
    selected = select_by_preregistered_rule(rows)
    selected_rows = [selected] if selected else []
    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "track": "Track A canonical-like",
            "target": "target62",
            "baseline_logistic_h40": BASELINE_LOGISTIC_H40,
            "historical_rf_h60": LOCKED_RF_H60,
            "horizons": HORIZONS,
            "thresholds": THRESHOLDS,
            "models": model_names(),
            "feature_sets": list(feature_manifests),
            "feature_manifests": feature_manifests,
            "selection_rule": {
                "validation_lift_positive": True,
                "within_best_validation_accuracy": TOLERANCE,
                "prefer_h40": True,
                "model_order": MODEL_RANK,
                "prefer_threshold_closest_to": 0.50,
                "feature_order": FEATURE_RANK,
                "final_accuracy_used_for_selection": False,
            },
            "confidence_abstention": False,
            "ticker_subset": False,
            "topk": False,
            "data_fetch": False,
            "paper_docx": False,
        },
    )
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", selected_rows, RESULT_COLUMNS)
    log = [
        "# Track A Target62 Validation-Safe Run Log",
        "",
        "- Status: completed.",
        "- Selection: pre-registered validation-only rule.",
        "- Final evaluation: scoring-only.",
        "- Confidence abstention: no.",
        "- Ticker subset: no.",
        "- Top-k/ranking: no.",
        f"- Candidate count: {len(rows)}.",
        f"- Selected candidate: `{selected.get('model') if selected else ''}` h={selected.get('horizon') if selected else ''} `{selected.get('feature_set') if selected else ''}` threshold={selected.get('threshold') if selected else ''}.",
        f"- Selected final accuracy: {selected.get('final_accuracy') if selected else ''}.",
        "",
    ]
    (OUTPUT_DIR / "target62_run_log.md").write_text("\n".join(log), encoding="utf-8")
    copy_outputs_to_report_dir()
    print(f"target62_status=completed selected={selected.get('model') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
