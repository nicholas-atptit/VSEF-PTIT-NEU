"""Audit Track A target62 validation-safe selected candidate."""

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
    LOCKED_RF_H60,
    REPO_ROOT,
    add_absolute_labels,
    markdown_table,
    pct,
    read_json,
    write_csv,
)
from scripts.research.run_vn30_hourly_track_a_target62_validation_safe import (  # noqa: E402
    BASELINE_LOGISTIC_H40,
    OUTPUT_DIR,
    REPORT_DIR,
    TARGET62,
    build_feature_sets,
    candidate_model,
    split_indices,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")


def accuracy(y_true: pd.Series, prediction: np.ndarray | pd.Series) -> float:
    return float((y_true.astype(int).to_numpy() == np.asarray(prediction).astype(int)).mean()) if len(y_true) else math.nan


def majority_prediction(train_y: pd.Series) -> int:
    return int(float(train_y.mean()) >= 0.5)


def bool_row(item: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"audit_item": item, "passed": bool(passed), "detail": detail}


def overfit_risk(validation_accuracy: float, final_accuracy: float) -> str:
    gap = validation_accuracy - final_accuracy
    if gap >= 0.08:
        return "high"
    if gap >= 0.04:
        return "medium"
    return "low"


def mismatch_label(validation_accuracy: float, final_accuracy: float) -> str:
    gap = final_accuracy - validation_accuracy
    if gap >= 0.08:
        return "high_positive_final_gap"
    if gap <= -0.08:
        return "high_negative_final_gap"
    if abs(gap) >= 0.04:
        return "medium_validation_final_gap"
    return "low_validation_final_gap"


def summarize_group(frame: pd.DataFrame, group_cols: list[str], majority: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values)}
        acc = float(group["is_correct"].mean())
        majority_acc = float((group["target_direction"].astype(int) == majority).mean())
        row.update(
            {
                "slice_type": "+".join(group_cols),
                "rows": int(len(group)),
                "accuracy": acc,
                "majority_baseline_accuracy": majority_acc,
                "delta_vs_majority_baseline": acc - majority_acc,
                "target_up_rate": float(group["target_direction"].mean()),
                "prediction_up_rate": float(group["prediction"].mean()),
                "correct": int(group["is_correct"].sum()),
            }
        )
        rows.append(row)
    return rows


def stability_label(by_ticker: list[dict[str, Any]], by_month: list[dict[str, Any]], by_quarter: list[dict[str, Any]], by_regime: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    tickers_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in by_ticker)
    tickers_above_62 = sum(float(row["accuracy"]) >= TARGET62 for row in by_ticker)
    months_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in by_month)
    months_above_62 = sum(float(row["accuracy"]) >= TARGET62 for row in by_month)
    quarters_above_60 = sum(float(row["accuracy"]) >= 0.60 for row in by_quarter)
    quarters_above_62 = sum(float(row["accuracy"]) >= TARGET62 for row in by_quarter)
    regimes_positive = sum(float(row["delta_vs_majority_baseline"]) > 0 for row in by_regime)
    broad = tickers_above_62 >= 18 and months_above_60 >= max(1, math.ceil(len(by_month) * 0.50)) and quarters_above_60 >= max(1, math.ceil(len(by_quarter) * 0.50)) and regimes_positive >= max(1, math.ceil(len(by_regime) * 0.50))
    details = {
        "tickers_above_60": tickers_above_60,
        "tickers_above_62": tickers_above_62,
        "months_above_60": months_above_60,
        "months_above_62": months_above_62,
        "quarters_above_60": quarters_above_60,
        "quarters_above_62": quarters_above_62,
        "regimes_positive_lift": regimes_positive,
        "regime_count": len(by_regime),
    }
    return ("broad_based" if broad else "concentrated_or_mixed"), details


def selected_candidate() -> dict[str, Any]:
    selected_path = OUTPUT_DIR / "selected_candidate_summary.csv"
    if not selected_path.exists():
        selected_path = REPORT_DIR / "selected_candidate_summary.csv"
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()


def replay_selected(selected: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_sets = build_feature_sets()
    feature_set = str(selected["feature_set"])
    features, feature_cols, feature_manifest = feature_sets[feature_set]
    horizon = int(selected["horizon"])
    threshold = float(selected["threshold"])
    labels = add_absolute_labels(features, horizon)
    idx = split_indices(features, labels)
    train_y = labels.reindex(idx["train"]).astype(int)
    val_y = labels.reindex(idx["validation"]).astype(int)
    final_y = labels.reindex(idx["final"]).astype(int)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", candidate_model(str(selected["model"])))])
    pipeline.fit(features.reindex(idx["train"])[feature_cols], train_y)
    val_prob = pipeline.predict_proba(features.reindex(idx["validation"])[feature_cols])[:, 1]
    final_prob = pipeline.predict_proba(features.reindex(idx["final"])[feature_cols])[:, 1]
    val_pred = (val_prob >= threshold).astype(int)
    final_pred = (final_prob >= threshold).astype(int)
    final = features.reindex(idx["final"])[["datetime", "ticker"]].copy()
    final["target_direction"] = final_y.astype(int)
    final["prediction"] = final_pred
    final["probability_up"] = final_prob
    final["is_correct"] = (final["target_direction"] == final["prediction"]).astype(int)
    final["month"] = final["datetime"].dt.to_period("M").astype(str)
    final["quarter"] = final["datetime"].dt.to_period("Q").astype(str)
    if "market_regime_v2" in features.columns:
        final["market_regime_v2"] = features.reindex(idx["final"])["market_regime_v2"].fillna("unknown").astype(str)
    else:
        regime_features = feature_sets["regime_feature_v2"][0][["datetime", "ticker", "market_regime_v2"]]
        final = final.merge(regime_features, on=["datetime", "ticker"], how="left")
        final["market_regime_v2"] = final["market_regime_v2"].fillna("unknown").astype(str)
    majority = majority_prediction(train_y)
    metrics = {
        "validation_accuracy": accuracy(val_y, val_pred),
        "final_accuracy": accuracy(final_y, final_pred),
        "validation_baseline_accuracy": accuracy(val_y, np.full(len(val_y), majority)),
        "final_baseline_accuracy": accuracy(final_y, np.full(len(final_y), majority)),
        "majority_prediction": majority,
        "feature_manifest": feature_manifest,
        "validation_rows": int(len(val_y)),
        "final_rows": int(len(final_y)),
    }
    return final, metrics


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected = selected_candidate()
    final, metrics = replay_selected(selected)
    majority = int(metrics["majority_prediction"])
    by_ticker = summarize_group(final, ["ticker"], majority)
    by_month = summarize_group(final, ["month"], majority)
    by_quarter = summarize_group(final, ["quarter"], majority)
    by_regime = summarize_group(final, ["market_regime_v2"], majority)
    for row in by_month:
        row["time_grain"] = "month"
    for row in by_quarter:
        row["time_grain"] = "quarter"
    stability, stability_details = stability_label(by_ticker, by_month, by_quarter, by_regime)
    validation_accuracy = float(metrics["validation_accuracy"])
    final_accuracy = float(metrics["final_accuracy"])
    overfit = overfit_risk(validation_accuracy, final_accuracy)
    mismatch = mismatch_label(validation_accuracy, final_accuracy)
    validation_final_gap = final_accuracy - validation_accuracy
    final_lift = final_accuracy - float(metrics["final_baseline_accuracy"])
    run_config = read_json(OUTPUT_DIR / "run_config.json")
    feature_manifest = metrics["feature_manifest"]
    audit_rows = [
        bool_row("validation_only_selection", bool(selected.get("selected_by_preregistered_rule", False)), selected.get("selected_by_preregistered_rule", "")),
        bool_row("stock_30_present", int(selected.get("active_ticker_count", 0)) == 30, selected.get("active_ticker_count", "")),
        bool_row("full_coverage", abs(float(selected.get("final_coverage", 0.0)) - 1.0) < 1e-12, selected.get("final_coverage", "")),
        bool_row("no_confidence_abstention", not bool(run_config.get("confidence_abstention")), run_config.get("confidence_abstention")),
        bool_row("no_ticker_subset", not bool(run_config.get("ticker_subset")), run_config.get("ticker_subset")),
        bool_row("no_topk_ranking", not bool(run_config.get("topk")), run_config.get("topk")),
        bool_row("no_final_label_selection", not bool(run_config.get("selection_rule", {}).get("final_accuracy_used_for_selection")), run_config.get("selection_rule", {}).get("final_accuracy_used_for_selection")),
        bool_row("leakage_audit_passed", bool(feature_manifest.get("leakage_safe")) and not bool(feature_manifest.get("future_return_features")) and not bool(feature_manifest.get("future_regime_features")) and not bool(feature_manifest.get("final_label_derived_features")), feature_manifest),
        bool_row("lift_over_majority_baseline", final_lift > 0, final_lift),
        bool_row("pass_60", bool(selected.get("pass_60", False)), final_accuracy),
        bool_row("pass_60_43", bool(selected.get("pass_60_43", False)), final_accuracy - BASELINE_LOGISTIC_H40),
        bool_row("pass_62", bool(selected.get("pass_62", False)), final_accuracy),
        bool_row("pass_65", bool(selected.get("pass_65", False)), final_accuracy),
        bool_row("overfit_risk_recorded", overfit in {"low", "medium", "high"}, overfit),
        bool_row("validation_final_mismatch_recorded", mismatch != "", mismatch),
        bool_row("stability_audit_recorded", stability in {"broad_based", "concentrated_or_mixed"}, stability),
    ]
    mismatch_rows = [
        {
            "validation_accuracy": validation_accuracy,
            "final_accuracy": final_accuracy,
            "validation_final_gap": validation_final_gap,
            "overfit_risk": overfit,
            "validation_final_mismatch": mismatch,
            "validation_baseline_accuracy": metrics["validation_baseline_accuracy"],
            "final_baseline_accuracy": metrics["final_baseline_accuracy"],
            "final_delta_vs_baseline": final_lift,
        }
    ]
    write_csv(REPORT_DIR / "target62_audit.csv", audit_rows, ["audit_item", "passed", "detail"])
    write_csv(REPORT_DIR / "target62_by_ticker.csv", by_ticker)
    write_csv(REPORT_DIR / "target62_by_time.csv", by_month + by_quarter)
    write_csv(REPORT_DIR / "target62_by_regime.csv", by_regime)
    write_csv(REPORT_DIR / "target62_validation_mismatch.csv", mismatch_rows)
    worst_tickers = sorted(by_ticker, key=lambda row: float(row["accuracy"]))[:10]
    worst_time = sorted(by_month + by_quarter, key=lambda row: float(row["accuracy"]))[:12]
    report = [
        "# Track A Target62 Validation-Safe Audit",
        "",
        f"- Selected candidate: `{selected.get('model', '')}` h={selected.get('horizon', '')} `{selected.get('feature_set', '')}` threshold={selected.get('threshold', '')}.",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Final majority/simple baseline: {pct(metrics['final_baseline_accuracy'])}.",
        f"- Lift over majority/simple baseline: {pct(final_lift)}.",
        f"- Delta vs 60.43: {pct(final_accuracy - BASELINE_LOGISTIC_H40)}.",
        f"- Delta vs 60.31: {pct(final_accuracy - LOCKED_RF_H60)}.",
        f"- Validation-final gap: {pct(validation_final_gap)} (`{mismatch}`).",
        f"- Overfit risk: `{overfit}`.",
        f"- Stability: `{stability}` `{stability_details}`.",
        "",
        markdown_table(["audit_item", "passed", "detail"], audit_rows),
        "",
        "## Regime Stability",
        "",
        markdown_table(["market_regime_v2", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], by_regime),
        "",
        "## Worst Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], worst_tickers),
        "",
        "## Worst Time Slices",
        "",
        markdown_table(["time_grain", "month", "quarter", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline", "target_up_rate", "prediction_up_rate"], worst_time),
        "",
    ]
    (REPORT_DIR / "target62_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"target62_audit_status=completed final_accuracy={final_accuracy:.6f} mismatch={mismatch} stability={stability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
