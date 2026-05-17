"""Postmortem audit for VN30 daily 2015 target60 failures.

This script does not run a tuning sweep. It reads saved daily benchmark,
target60 v1, and target60 v2 outputs, then reproduces two already recorded
candidates only to produce row-level diagnostic breakdowns:

- v1 recorded final best: LightGBM daily_cross h=40, threshold 0.500.
- v2 recorded final best: LightGBM volatility_normalized h=50, threshold 0.525.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_vn30_daily_2015_target60_v2 import (  # noqa: E402
    EVAL_START,
    TRAIN_END,
    build_features_cross,
    build_features_vol_normalized,
    compute_future_returns,
    load_stock_data,
    load_universe_tickers,
    train_model,
)
from scripts.research.vn30_hourly_2015_canonical_eval import EVALUATOR_VERSION  # noqa: E402

BENCHMARK_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_benchmark"
V1_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_target60_optimization"
V2_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_target60_v2"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "daily_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015_target60_postmortem"

BEST_COMPARISON_CSV = REPORT_DIR / "daily_best_candidate_comparison.csv"
TICKER_DRAG_CSV = REPORT_DIR / "daily_ticker_drag.csv"
TIME_DRAG_CSV = REPORT_DIR / "daily_time_drag.csv"
CLASS_BALANCE_CSV = REPORT_DIR / "daily_class_balance.csv"
BASELINE_COMPARISON_CSV = REPORT_DIR / "daily_baseline_comparison.csv"
VALIDATION_FINAL_MISMATCH_CSV = REPORT_DIR / "daily_validation_final_mismatch.csv"
POSTMORTEM_MD = REPORT_DIR / "daily_target60_failure_postmortem.md"

TARGET_ACCURACY = 0.60
NAIVE_BASELINE_ACCURACY = 0.50
LGBM_NL20_D3_LR002_N700 = {
    "num_leaves": 20,
    "max_depth": 3,
    "learning_rate": 0.02,
    "n_estimators": 700,
    "min_child_samples": 25,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "random_state": 42,
    "verbose": -1,
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_pct(value: Any) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def fmt_pp(value: Any) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100:+.2f}pp"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def row_to_dict(row: pd.Series | dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def best_by(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.notna().sum() == 0:
        return {}
    return df.loc[numeric.idxmax()].to_dict()


def candidate_id_from_row(source: str, row: dict[str, Any]) -> str:
    if str(row.get("candidate_id", "")).strip():
        return str(row["candidate_id"]).strip()
    model = str(row.get("model", "unknown"))
    horizon = to_int(row.get("horizon"))
    feature_set = str(row.get("feature_set", "unknown"))
    hyperparams_id = str(row.get("hyperparams_id", "default"))
    threshold = row.get("decision_threshold", "")
    suffix = f"_t{int(round(to_float(threshold) * 1000))}" if str(threshold) not in {"", "nan"} else ""
    return f"{source}_{model}_h{horizon}_{feature_set}_{hyperparams_id}{suffix}"


def load_context() -> dict[str, Any]:
    return {
        "benchmark_summary": read_json(BENCHMARK_DIR / "daily" / "benchmark_summary.json"),
        "benchmark_accuracy": read_frame(BENCHMARK_DIR / "daily" / "accuracy_summary.csv"),
        "benchmark_baseline": read_frame(BENCHMARK_DIR / "daily" / "baseline_summary.csv"),
        "benchmark_baseline_delta": read_frame(BENCHMARK_DIR / "daily" / "baseline_delta_summary.csv"),
        "v1_manifest": read_json(V1_DIR / "daily_target60_manifest.json"),
        "v1_final": read_frame(V1_DIR / "daily" / "final_candidate_results.csv"),
        "v1_best": read_frame(V1_DIR / "daily" / "best_daily_candidates.csv"),
        "v2_manifest": read_json(V2_DIR / "daily_target60_v2_manifest.json"),
        "v2_final": read_frame(V2_DIR / "daily" / "final_candidate_results.csv"),
        "v2_selection": read_frame(V2_DIR / "daily" / "candidate_selection_scores.csv"),
    }


def add_comparison_row(
    rows: list[dict[str, Any]],
    source: str,
    selection_basis: str,
    row: dict[str, Any],
    validation_col: str,
    canonical_accuracy: float | None = None,
) -> None:
    if not row:
        return
    final_accuracy = to_float(row.get("final_accuracy", row.get("best_final_accuracy")))
    threshold = row.get("decision_threshold", "")
    comparison = {
        "source": source,
        "selection_basis": selection_basis,
        "candidate_id": candidate_id_from_row(source, row),
        "model": row.get("model", row.get("best_model", row.get("best_final_model", ""))),
        "horizon": row.get("horizon", row.get("best_horizon", row.get("best_final_horizon", ""))),
        "feature_set": row.get("feature_set", row.get("best_final_feature_set", "")),
        "hyperparams_id": row.get("hyperparams_id", ""),
        "decision_threshold": threshold,
        "validation_metric_name": validation_col,
        "validation_metric": row.get(validation_col, ""),
        "rolling_validation_min_accuracy": row.get("rolling_validation_min_accuracy", ""),
        "rolling_validation_std": row.get("rolling_validation_std", ""),
        "stability_score": row.get("stability_score", ""),
        "final_accuracy": final_accuracy,
        "final_rows": to_int(row.get("final_rows", row.get("best_final_rows"))),
        "gap_to_60_pp": (TARGET_ACCURACY - final_accuracy) * 100 if math.isfinite(final_accuracy) else "",
        "pass_60": bool(final_accuracy >= TARGET_ACCURACY) if math.isfinite(final_accuracy) else False,
        "selected_on_validation": row.get("selected_on_validation", ""),
        "claim_level": row.get("claim_level", "failed"),
        "evaluator_version": row.get("evaluator_version", EVALUATOR_VERSION),
    }
    if canonical_accuracy is not None and math.isfinite(final_accuracy):
        comparison["delta_vs_canonical_pp"] = (final_accuracy - canonical_accuracy) * 100
    else:
        comparison["delta_vs_canonical_pp"] = ""
    rows.append(comparison)


def build_best_candidate_comparison(context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_accuracy = context["benchmark_accuracy"]
    v1_final = context["v1_final"]
    v2_final = context["v2_final"]

    benchmark_best = best_by(benchmark_accuracy, "final_accuracy")
    v1_best_final = best_by(v1_final, "final_accuracy")
    v1_best_validation = best_by(v1_final, "validation_accuracy")
    v2_best_final = best_by(v2_final, "final_accuracy")
    v2_best_stability = best_by(v2_final, "stability_score")
    v2_best_validation_mean = best_by(v2_final, "rolling_validation_mean_accuracy")
    canonical_accuracy = to_float(v1_best_final.get("final_accuracy"))

    rows: list[dict[str, Any]] = []
    add_comparison_row(rows, "benchmark", "best_saved_final_accuracy", benchmark_best, "validation_accuracy", canonical_accuracy)
    add_comparison_row(rows, "v1", "best_saved_validation_accuracy", v1_best_validation, "validation_accuracy", canonical_accuracy)
    add_comparison_row(rows, "v1", "best_saved_final_accuracy", v1_best_final, "validation_accuracy", canonical_accuracy)
    add_comparison_row(rows, "v2", "best_saved_stability_score", v2_best_stability, "stability_score", canonical_accuracy)
    add_comparison_row(rows, "v2", "best_saved_rolling_validation_mean", v2_best_validation_mean, "rolling_validation_mean_accuracy", canonical_accuracy)
    add_comparison_row(rows, "v2", "best_saved_final_accuracy", v2_best_final, "rolling_validation_mean_accuracy", canonical_accuracy)

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison.to_csv(BEST_COMPARISON_CSV, index=False)
    anchors = {
        "benchmark_best": benchmark_best,
        "v1_best_final": v1_best_final,
        "v1_best_validation": v1_best_validation,
        "v2_best_final": v2_best_final,
        "v2_best_stability": v2_best_stability,
        "v2_best_validation_mean": v2_best_validation_mean,
        "canonical_accuracy": canonical_accuracy,
    }
    return comparison, anchors


def reproduce_candidate(
    stock_df: pd.DataFrame,
    candidate_id: str,
    source: str,
    model: str,
    feature_set: str,
    horizon: int,
    threshold: float,
    params: dict[str, Any],
) -> pd.DataFrame:
    if feature_set == "daily_cross":
        feature_df, feature_cols = build_features_cross(stock_df)
    elif feature_set == "volatility_normalized":
        feature_df, feature_cols = build_features_vol_normalized(stock_df)
    else:
        raise ValueError(f"Unsupported diagnostic feature set: {feature_set}")

    future_returns = compute_future_returns(feature_df, horizon)
    labels = (future_returns > 0).astype(int)
    all_idx = future_returns.index[future_returns.notna()]
    train_idx = all_idx[
        (feature_df.loc[all_idx, "datetime"] <= TRAIN_END)
        & future_returns.reindex(all_idx).notna()
    ]
    eval_idx = all_idx[
        (feature_df.loc[all_idx, "datetime"] >= EVAL_START)
        & future_returns.reindex(all_idx).notna()
    ]
    feature_cols_present = [c for c in feature_cols if c in feature_df.columns]
    train_x = feature_df.reindex(train_idx)[feature_cols_present].fillna(0)
    train_y = labels.reindex(train_idx)
    eval_x = feature_df.reindex(eval_idx)[feature_cols_present].fillna(0)
    eval_y = labels.reindex(eval_idx)

    fitted = train_model(model, params, train_x, train_y)
    prob = fitted.predict_proba(eval_x)[:, 1]
    pred = (prob >= threshold).astype(int)
    diagnostic = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "source": source,
            "model": model,
            "feature_set": feature_set,
            "horizon": horizon,
            "decision_threshold": threshold,
            "datetime": feature_df.reindex(eval_idx)["datetime"].values,
            "ticker": feature_df.reindex(eval_idx)["ticker"].values,
            "y_true": eval_y.values.astype(int),
            "y_pred": pred.astype(int),
            "y_prob": prob,
        }
    )
    diagnostic["is_correct"] = (diagnostic["y_true"] == diagnostic["y_pred"]).astype(int)
    diagnostic["year"] = diagnostic["datetime"].dt.year
    diagnostic["month"] = diagnostic["datetime"].dt.month
    diagnostic["quarter"] = diagnostic["datetime"].dt.quarter
    diagnostic["month_id"] = diagnostic["datetime"].dt.strftime("%Y-%m")
    diagnostic["quarter_id"] = diagnostic["year"].astype(str) + "-Q" + diagnostic["quarter"].astype(str)
    return diagnostic


def aggregate_prediction_rows(
    df: pd.DataFrame,
    group_cols: list[str],
    scope: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["candidate_id", "source", "model", "feature_set", "horizon", "decision_threshold"] + group_cols, sort=True)
    out = grouped.agg(
        n_rows=("is_correct", "count"),
        n_correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        positive_rate=("y_true", "mean"),
        predicted_positive_rate=("y_pred", "mean"),
    ).reset_index()
    out["scope"] = scope
    out["n_errors"] = out["n_rows"] - out["n_correct"]
    out["majority_baseline_accuracy"] = out["positive_rate"].apply(lambda x: max(x, 1.0 - x))
    out["model_minus_majority_pp"] = (out["accuracy"] - out["majority_baseline_accuracy"]) * 100
    return out


def add_drag_columns(agg: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    out = agg.copy()
    overall_cols = overall[["candidate_id", "accuracy", "n_errors"]].rename(
        columns={"accuracy": "candidate_overall_accuracy", "n_errors": "candidate_total_errors"}
    )
    out = out.merge(overall_cols, on="candidate_id", how="left")
    out["delta_to_candidate_accuracy_pp"] = (out["accuracy"] - out["candidate_overall_accuracy"]) * 100
    out["error_share"] = out["n_errors"] / out["candidate_total_errors"].replace(0, np.nan)
    out["expected_errors_at_overall_accuracy"] = out["n_rows"] * (1.0 - out["candidate_overall_accuracy"])
    out["excess_errors_vs_overall"] = out["n_errors"] - out["expected_errors_at_overall_accuracy"]
    return out


def confusion_values(df: pd.DataFrame) -> dict[str, int]:
    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)
    return {
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def build_class_balance_rows(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [
        ("overall", []),
        ("ticker", ["ticker"]),
        ("month", ["month_id"]),
        ("quarter", ["quarter_id"]),
    ]
    identity_cols = ["candidate_id", "source", "model", "feature_set", "horizon", "decision_threshold"]
    for candidate_id, candidate_df in diagnostic_df.groupby("candidate_id", sort=True):
        identity = {col: candidate_df[col].iloc[0] for col in identity_cols}
        for scope, group_cols in group_specs:
            if group_cols:
                iterator = candidate_df.groupby(group_cols, sort=True)
            else:
                iterator = [("all", candidate_df)]
            for group_key, group_df in iterator:
                if isinstance(group_key, tuple):
                    group_value = "-".join(str(x) for x in group_key)
                else:
                    group_value = str(group_key)
                n_rows = len(group_df)
                positive_rate = float(group_df["y_true"].mean()) if n_rows else math.nan
                predicted_positive_rate = float(group_df["y_pred"].mean()) if n_rows else math.nan
                accuracy = float(group_df["is_correct"].mean()) if n_rows else math.nan
                majority = max(positive_rate, 1.0 - positive_rate) if math.isfinite(positive_rate) else math.nan
                row = {
                    **identity,
                    "scope": scope,
                    "group": group_value,
                    "n_rows": n_rows,
                    "positive_rate": positive_rate,
                    "negative_rate": 1.0 - positive_rate if math.isfinite(positive_rate) else math.nan,
                    "predicted_positive_rate": predicted_positive_rate,
                    "model_accuracy": accuracy,
                    "majority_baseline_accuracy": majority,
                    "model_minus_majority_pp": (accuracy - majority) * 100 if math.isfinite(accuracy) and math.isfinite(majority) else math.nan,
                }
                row.update(confusion_values(group_df))
                rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    if len(tickers) != 30:
        raise RuntimeError(f"Expected frozen universe of 30 tickers, got {len(tickers)}")
    if stock_df.empty:
        raise RuntimeError(f"No daily stock data loaded from {rel(CACHE_ROOT)}")
    if stock_df["ticker"].nunique() != 30:
        raise RuntimeError(f"Expected 30 daily usable tickers, got {stock_df['ticker'].nunique()}")

    candidates = [
        {
            "candidate_id": "canonical_v1_lightgbm_h40_daily_cross_nl20_d3_lr0.02_n700_t500",
            "source": "v1_best_final_reproduced",
            "model": "lightgbm",
            "feature_set": "daily_cross",
            "horizon": 40,
            "threshold": 0.500,
            "params": LGBM_NL20_D3_LR002_N700,
        },
        {
            "candidate_id": "v2_best_final_lightgbm_h50_volatility_normalized_nl20_d3_lr0.02_n700_t525",
            "source": "v2_best_final_reproduced",
            "model": "lightgbm",
            "feature_set": "volatility_normalized",
            "horizon": 50,
            "threshold": 0.525,
            "params": LGBM_NL20_D3_LR002_N700,
        },
    ]
    diagnostics = [
        reproduce_candidate(
            stock_df=stock_df,
            candidate_id=c["candidate_id"],
            source=c["source"],
            model=c["model"],
            feature_set=c["feature_set"],
            horizon=c["horizon"],
            threshold=c["threshold"],
            params=c["params"],
        )
        for c in candidates
    ]
    diagnostic_df = pd.concat(diagnostics, ignore_index=True)

    overall = aggregate_prediction_rows(diagnostic_df, [], "overall")
    ticker = aggregate_prediction_rows(diagnostic_df, ["ticker"], "ticker")
    ticker = add_drag_columns(ticker, overall)
    ticker = ticker.sort_values(["candidate_id", "accuracy", "excess_errors_vs_overall"], ascending=[True, True, False])
    ticker.to_csv(TICKER_DRAG_CSV, index=False)

    month = aggregate_prediction_rows(diagnostic_df, ["month_id"], "month")
    quarter = aggregate_prediction_rows(diagnostic_df, ["quarter_id"], "quarter")
    time_drag = pd.concat([month, quarter], ignore_index=True)
    time_drag = add_drag_columns(time_drag, overall)
    time_drag = time_drag.sort_values(["candidate_id", "scope", "accuracy"], ascending=[True, True, True])
    time_drag.to_csv(TIME_DRAG_CSV, index=False)

    class_balance = build_class_balance_rows(diagnostic_df)
    class_balance.to_csv(CLASS_BALANCE_CSV, index=False)
    return diagnostic_df, overall, ticker, time_drag


def build_baseline_comparison(overall: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark_baseline = context["benchmark_baseline"]
    benchmark_naive = NAIVE_BASELINE_ACCURACY
    if not benchmark_baseline.empty and "final_accuracy" in benchmark_baseline.columns:
        benchmark_naive = to_float(benchmark_baseline["final_accuracy"].iloc[0], NAIVE_BASELINE_ACCURACY)

    for _, row in overall.iterrows():
        model_accuracy = to_float(row["accuracy"])
        majority = to_float(row["majority_baseline_accuracy"])
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "model": row["model"],
                "feature_set": row["feature_set"],
                "horizon": to_int(row["horizon"]),
                "decision_threshold": row["decision_threshold"],
                "model_accuracy": model_accuracy,
                "naive_50_baseline_accuracy": NAIVE_BASELINE_ACCURACY,
                "benchmark_file_baseline_accuracy": benchmark_naive,
                "final_majority_class_baseline_accuracy": majority,
                "model_minus_naive_50_pp": (model_accuracy - NAIVE_BASELINE_ACCURACY) * 100,
                "model_minus_final_majority_pp": (model_accuracy - majority) * 100,
                "gap_to_60_pp": (TARGET_ACCURACY - model_accuracy) * 100,
                "positive_rate": row["positive_rate"],
                "predicted_positive_rate": row["predicted_positive_rate"],
                "n_rows": to_int(row["n_rows"]),
            }
        )
    baseline = pd.DataFrame(rows)
    baseline.to_csv(BASELINE_COMPARISON_CSV, index=False)
    return baseline


def rank_value(df: pd.DataFrame, target_index: Any, rank_col: str) -> int:
    if df.empty or rank_col not in df.columns:
        return 0
    ranks = df[rank_col].rank(method="min", ascending=False)
    try:
        return int(ranks.loc[target_index])
    except KeyError:
        return 0


def validation_final_mismatch_for(df: pd.DataFrame, source: str, validation_col: str) -> dict[str, Any]:
    if df.empty or validation_col not in df.columns or "final_accuracy" not in df.columns:
        return {
            "source": source,
            "validation_metric_name": validation_col,
            "n_candidates": 0,
            "status": "missing",
        }
    work = df.copy()
    work[validation_col] = pd.to_numeric(work[validation_col], errors="coerce")
    work["final_accuracy"] = pd.to_numeric(work["final_accuracy"], errors="coerce")
    work = work.dropna(subset=[validation_col, "final_accuracy"])
    if work.empty:
        return {
            "source": source,
            "validation_metric_name": validation_col,
            "n_candidates": 0,
            "status": "empty_after_numeric_filter",
        }

    validation_best = work.loc[work[validation_col].idxmax()]
    final_best = work.loc[work["final_accuracy"].idxmax()]
    validation_best_final = to_float(validation_best["final_accuracy"])
    final_best_final = to_float(final_best["final_accuracy"])
    pearson = work[validation_col].corr(work["final_accuracy"], method="pearson") if len(work) > 1 else math.nan
    spearman = work[validation_col].corr(work["final_accuracy"], method="spearman") if len(work) > 1 else math.nan
    final_best_validation_rank = rank_value(work, final_best.name, validation_col)
    validation_best_final_rank = rank_value(work, validation_best.name, "final_accuracy")

    return {
        "source": source,
        "validation_metric_name": validation_col,
        "n_candidates": len(work),
        "status": "ok",
        "pearson_corr_validation_final": pearson,
        "spearman_corr_validation_final": spearman,
        "validation_best_candidate_id": candidate_id_from_row(source, validation_best.to_dict()),
        "validation_best_metric": to_float(validation_best[validation_col]),
        "validation_best_final_accuracy": validation_best_final,
        "validation_best_final_rank": validation_best_final_rank,
        "final_best_candidate_id": candidate_id_from_row(source, final_best.to_dict()),
        "final_best_metric": to_float(final_best[validation_col]),
        "final_best_final_accuracy": final_best_final,
        "final_best_validation_rank": final_best_validation_rank,
        "selected_final_gap_to_final_best_pp": (final_best_final - validation_best_final) * 100,
    }


def build_validation_final_mismatch(context: dict[str, Any]) -> pd.DataFrame:
    rows = [
        validation_final_mismatch_for(context["v1_final"], "v1", "validation_accuracy"),
        validation_final_mismatch_for(context["v2_final"], "v2", "rolling_validation_mean_accuracy"),
        validation_final_mismatch_for(context["v2_final"], "v2", "stability_score"),
    ]
    mismatch = pd.DataFrame(rows)
    mismatch.to_csv(VALIDATION_FINAL_MISMATCH_CSV, index=False)
    return mismatch


def simple_records(df: pd.DataFrame, columns: list[str], formatters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    formatters = formatters or {}
    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        row: dict[str, Any] = {}
        for col in columns:
            value = raw.get(col, "")
            formatter = formatters.get(col)
            row[col] = formatter(value) if formatter else value
        rows.append(row)
    return rows


def first_candidate(overall: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    rows = overall[overall["candidate_id"] == candidate_id]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def build_postmortem_report(
    context: dict[str, Any],
    comparison: pd.DataFrame,
    anchors: dict[str, Any],
    overall: pd.DataFrame,
    ticker_drag: pd.DataFrame,
    time_drag: pd.DataFrame,
    baseline: pd.DataFrame,
    mismatch: pd.DataFrame,
) -> None:
    canonical_id = "canonical_v1_lightgbm_h40_daily_cross_nl20_d3_lr0.02_n700_t500"
    v2_id = "v2_best_final_lightgbm_h50_volatility_normalized_nl20_d3_lr0.02_n700_t525"
    canonical = first_candidate(overall, canonical_id)
    v2_diag = first_candidate(overall, v2_id)
    canonical_accuracy = to_float(anchors.get("canonical_accuracy"))
    v2_best_final_accuracy = to_float(anchors.get("v2_best_final", {}).get("final_accuracy"))
    v2_delta = v2_best_final_accuracy - canonical_accuracy
    v2_improved = bool(v2_delta > 0)
    h50_val_mean = to_float(anchors.get("v2_best_final", {}).get("rolling_validation_mean_accuracy"))
    h50_final_minus_val = v2_best_final_accuracy - h50_val_mean

    canonical_ticker = ticker_drag[ticker_drag["candidate_id"] == canonical_id].sort_values("accuracy").head(8)
    canonical_months = time_drag[
        (time_drag["candidate_id"] == canonical_id) & (time_drag["scope"] == "month")
    ].sort_values("accuracy").head(6)
    canonical_quarters = time_drag[
        (time_drag["candidate_id"] == canonical_id) & (time_drag["scope"] == "quarter")
    ].sort_values("accuracy").head(4)
    canonical_baseline = baseline[baseline["candidate_id"] == canonical_id]
    v2_baseline = baseline[baseline["candidate_id"] == v2_id]
    v2_stability_mismatch = mismatch[
        (mismatch["source"] == "v2") & (mismatch["validation_metric_name"] == "stability_score")
    ]
    v1_mismatch = mismatch[mismatch["source"] == "v1"]

    model_minus_majority_pp = to_float(canonical_baseline["model_minus_final_majority_pp"].iloc[0]) if not canonical_baseline.empty else math.nan
    model_minus_naive_pp = to_float(canonical_baseline["model_minus_naive_50_pp"].iloc[0]) if not canonical_baseline.empty else math.nan
    majority_baseline = to_float(canonical_baseline["final_majority_class_baseline_accuracy"].iloc[0]) if not canonical_baseline.empty else math.nan
    positive_rate = to_float(canonical_baseline["positive_rate"].iloc[0]) if not canonical_baseline.empty else math.nan

    daily_v3_recommended = False
    v3_reason = (
        "V2 did not improve over the h=40 record, the h=40 model is only "
        f"{model_minus_majority_pp:.2f}pp above the final-period majority-class diagnostic baseline, "
        "and validation rankings do not provide a clear final-period fix."
    )

    comparison_rows = simple_records(
        comparison,
        [
            "source",
            "selection_basis",
            "model",
            "horizon",
            "feature_set",
            "decision_threshold",
            "validation_metric",
            "stability_score",
            "final_accuracy",
            "gap_to_60_pp",
            "delta_vs_canonical_pp",
        ],
        {
            "validation_metric": fmt_pct,
            "stability_score": fmt_pct,
            "final_accuracy": fmt_pct,
            "gap_to_60_pp": lambda v: f"{to_float(v) if math.isfinite(to_float(v)) else 0:.2f}pp",
            "delta_vs_canonical_pp": lambda v: f"{to_float(v):+.2f}pp" if math.isfinite(to_float(v)) else "",
        },
    )
    ticker_rows = simple_records(
        canonical_ticker,
        ["ticker", "n_rows", "n_errors", "accuracy", "positive_rate", "majority_baseline_accuracy", "delta_to_candidate_accuracy_pp"],
        {
            "accuracy": fmt_pct,
            "positive_rate": fmt_pct,
            "majority_baseline_accuracy": fmt_pct,
            "delta_to_candidate_accuracy_pp": lambda v: f"{to_float(v):+.2f}pp",
        },
    )
    month_rows = simple_records(
        canonical_months,
        ["month_id", "n_rows", "n_errors", "accuracy", "positive_rate", "majority_baseline_accuracy", "delta_to_candidate_accuracy_pp"],
        {
            "accuracy": fmt_pct,
            "positive_rate": fmt_pct,
            "majority_baseline_accuracy": fmt_pct,
            "delta_to_candidate_accuracy_pp": lambda v: f"{to_float(v):+.2f}pp",
        },
    )
    quarter_rows = simple_records(
        canonical_quarters,
        ["quarter_id", "n_rows", "n_errors", "accuracy", "positive_rate", "majority_baseline_accuracy", "delta_to_candidate_accuracy_pp"],
        {
            "accuracy": fmt_pct,
            "positive_rate": fmt_pct,
            "majority_baseline_accuracy": fmt_pct,
            "delta_to_candidate_accuracy_pp": lambda v: f"{to_float(v):+.2f}pp",
        },
    )
    baseline_rows = simple_records(
        baseline,
        [
            "candidate_id",
            "model_accuracy",
            "naive_50_baseline_accuracy",
            "final_majority_class_baseline_accuracy",
            "model_minus_naive_50_pp",
            "model_minus_final_majority_pp",
            "gap_to_60_pp",
        ],
        {
            "model_accuracy": fmt_pct,
            "naive_50_baseline_accuracy": fmt_pct,
            "final_majority_class_baseline_accuracy": fmt_pct,
            "model_minus_naive_50_pp": lambda v: f"{to_float(v):+.2f}pp",
            "model_minus_final_majority_pp": lambda v: f"{to_float(v):+.2f}pp",
            "gap_to_60_pp": lambda v: f"{to_float(v):.2f}pp",
        },
    )
    mismatch_rows = simple_records(
        mismatch,
        [
            "source",
            "validation_metric_name",
            "n_candidates",
            "pearson_corr_validation_final",
            "spearman_corr_validation_final",
            "validation_best_final_accuracy",
            "final_best_final_accuracy",
            "selected_final_gap_to_final_best_pp",
            "final_best_validation_rank",
        ],
        {
            "pearson_corr_validation_final": lambda v: f"{to_float(v):.3f}" if math.isfinite(to_float(v)) else "",
            "spearman_corr_validation_final": lambda v: f"{to_float(v):.3f}" if math.isfinite(to_float(v)) else "",
            "validation_best_final_accuracy": fmt_pct,
            "final_best_final_accuracy": fmt_pct,
            "selected_final_gap_to_final_best_pp": lambda v: f"{to_float(v):.2f}pp",
        },
    )

    lines = [
        "# VN30 Daily 2015 Target60 Failure Postmortem",
        "",
        f"- Created at UTC: `{now_utc()}`.",
        f"- Inputs: `{rel(BENCHMARK_DIR)}`, `{rel(V1_DIR)}`, `{rel(V2_DIR)}`, `{rel(CACHE_ROOT)}`, `{rel(UNIVERSE_PATH)}`.",
        "- Scope: daily-only; no hourly data; no daily-to-hourly resampling; no data fetch.",
        "- Selection boundary: saved candidates and thresholds were selected from validation artifacts; final labels are used here only for postmortem scoring diagnostics.",
        "- Reproduced candidates: existing h=40 v1 final-best and h=50 v2 final-best only; no new tuning sweep.",
        "",
        "## Executive Answer",
        "",
        f"- Canonical best daily result: LightGBM `daily_cross` h=40, final accuracy {fmt_pct(canonical_accuracy)}, final rows 8,880.",
        f"- Did v2 improve over v1: {'yes' if v2_improved else 'no'} ({fmt_pp(v2_delta)} versus h=40).",
        f"- Target60 passed: no; h=40 gap to 60 is {(TARGET_ACCURACY - canonical_accuracy) * 100:.2f}pp.",
        f"- h=40 remains the canonical best recorded daily candidate: {'yes' if canonical_accuracy >= v2_best_final_accuracy else 'no'}.",
        f"- h=50 `volatility_normalized`: final accuracy {fmt_pct(v2_best_final_accuracy)}, validation mean {fmt_pct(h50_val_mean)}, final-minus-validation {fmt_pp(h50_final_minus_val)}.",
        "- Interpretation of h=50: it did not underperform its validation mean on final; it underperformed the h=40 record and remained below 60%.",
        f"- Daily v3 recommended: {'yes' if daily_v3_recommended else 'no'}. {v3_reason}",
        "",
        "## Why V2 Failed To Improve",
        "",
        "- V2 broadened horizons, thresholds, and feature sets, but its best final result was 57.45%, below the 57.58% h=40 record.",
        "- V2's validation-stability-selected candidate was not the final-best candidate, which indicates validation-final ranking mismatch rather than a clear new signal.",
        f"- The h=40 diagnostic model is only {model_minus_majority_pp:.2f}pp above the final-period majority-class baseline ({fmt_pct(majority_baseline)}), so the exploitable daily signal above a simple class baseline is small.",
        "- Errors are concentrated by ticker and final-period regime; the allowed scope does not permit fixing this by ticker exclusion, universe change, hourly data, or final-label selection.",
        "",
        "## Best Candidate Comparison",
        "",
        markdown_table(
            [
                "source",
                "selection_basis",
                "model",
                "horizon",
                "feature_set",
                "decision_threshold",
                "validation_metric",
                "stability_score",
                "final_accuracy",
                "gap_to_60_pp",
                "delta_vs_canonical_pp",
            ],
            comparison_rows,
        ),
        "",
        "## Ticker Drag",
        "",
        "Worst h=40 ticker diagnostics:",
        "",
        markdown_table(
            [
                "ticker",
                "n_rows",
                "n_errors",
                "accuracy",
                "positive_rate",
                "majority_baseline_accuracy",
                "delta_to_candidate_accuracy_pp",
            ],
            ticker_rows,
        ),
        "",
        "## Time Drag",
        "",
        "Worst h=40 monthly diagnostics:",
        "",
        markdown_table(
            [
                "month_id",
                "n_rows",
                "n_errors",
                "accuracy",
                "positive_rate",
                "majority_baseline_accuracy",
                "delta_to_candidate_accuracy_pp",
            ],
            month_rows,
        ),
        "",
        "Worst h=40 quarterly diagnostics:",
        "",
        markdown_table(
            [
                "quarter_id",
                "n_rows",
                "n_errors",
                "accuracy",
                "positive_rate",
                "majority_baseline_accuracy",
                "delta_to_candidate_accuracy_pp",
            ],
            quarter_rows,
        ),
        "",
        "## Class Imbalance",
        "",
        f"- h=40 final positive rate: {fmt_pct(positive_rate)}.",
        f"- h=40 final majority-class diagnostic baseline: {fmt_pct(majority_baseline)}.",
        f"- h=40 model lift over final majority baseline: {model_minus_majority_pp:+.2f}pp.",
        "- Class imbalance is not the only problem because weak months include both positive-light and positive-heavy regimes, but the majority baseline is close enough to constrain the daily claim.",
        "",
        "## Baseline Comparison",
        "",
        markdown_table(
            [
                "candidate_id",
                "model_accuracy",
                "naive_50_baseline_accuracy",
                "final_majority_class_baseline_accuracy",
                "model_minus_naive_50_pp",
                "model_minus_final_majority_pp",
                "gap_to_60_pp",
            ],
            baseline_rows,
        ),
        "",
        f"- Against the fixed 50% baseline, h=40 is {model_minus_naive_pp:+.2f}pp higher.",
        f"- Against the final majority-class diagnostic baseline, h=40 is only {model_minus_majority_pp:+.2f}pp higher.",
        "",
        "## Validation-Final Mismatch",
        "",
        markdown_table(
            [
                "source",
                "validation_metric_name",
                "n_candidates",
                "pearson_corr_validation_final",
                "spearman_corr_validation_final",
                "validation_best_final_accuracy",
                "final_best_final_accuracy",
                "selected_final_gap_to_final_best_pp",
                "final_best_validation_rank",
            ],
            mismatch_rows,
        ),
        "",
        "Interpretation:",
        "",
    ]
    if not v1_mismatch.empty:
        v1_gap = to_float(v1_mismatch["selected_final_gap_to_final_best_pp"].iloc[0])
        lines.append(f"- V1 validation-best selection trailed the v1 final-best result by {v1_gap:.2f}pp.")
    if not v2_stability_mismatch.empty:
        v2_gap = to_float(v2_stability_mismatch["selected_final_gap_to_final_best_pp"].iloc[0])
        lines.append(f"- V2 stability-best selection trailed the v2 final-best result by {v2_gap:.2f}pp.")
    lines.extend(
        [
            "- This means final-period top results should be treated as postmortem evidence, not as proof that validation can reliably select a stronger daily candidate.",
            "",
            "## Daily V3 Decision",
            "",
            "- Daily v3 is not justified now.",
            "- A broad v3 tuning sweep would be weakly motivated because v2 already widened the search without improving the h=40 record.",
            "- A narrow v3 would require a pre-registered validation-only fix for class/regime calibration or validation-final ranking; the current diagnostics do not identify one with enough specificity.",
            "",
            "## Current Daily Claim Boundary",
            "",
            "- Daily target60 failed under the frozen 30/30 VN30 daily universe.",
            f"- The current daily benchmark boundary is best recorded final accuracy {fmt_pct(canonical_accuracy)} for LightGBM `daily_cross` h=40.",
            "- The daily result may be used only as robustness context for separate hourly available-window evidence.",
            "- No trading-readiness claim.",
            "- No profitability claim.",
            "- No live-deployment claim.",
            "- No hourly claim made from daily data.",
            "- No paper or DOCX generated.",
            "",
            "## Generated Files",
            "",
            f"- `{rel(BEST_COMPARISON_CSV)}`",
            f"- `{rel(TICKER_DRAG_CSV)}`",
            f"- `{rel(TIME_DRAG_CSV)}`",
            f"- `{rel(CLASS_BALANCE_CSV)}`",
            f"- `{rel(BASELINE_COMPARISON_CSV)}`",
            f"- `{rel(VALIDATION_FINAL_MISMATCH_CSV)}`",
        ]
    )
    POSTMORTEM_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_required_inputs() -> None:
    required_paths = [
        BENCHMARK_DIR,
        V1_DIR,
        V2_DIR,
        CACHE_ROOT,
        UNIVERSE_PATH,
        BENCHMARK_DIR / "daily" / "accuracy_summary.csv",
        BENCHMARK_DIR / "daily" / "baseline_summary.csv",
        V1_DIR / "daily" / "final_candidate_results.csv",
        V2_DIR / "daily" / "final_candidate_results.csv",
    ]
    missing = [rel(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required postmortem input(s): " + ", ".join(missing))
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        tickers = [str(row.get("ticker", "")).strip().upper() for row in csv.DictReader(f) if str(row.get("ticker", "")).strip()]
    if len(tickers) != 30:
        raise RuntimeError(f"Frozen VN30 universe must contain 30 tickers, found {len(tickers)}")


def main() -> int:
    print("=" * 72)
    print("VN30 Daily 2015 Target60 Failure Postmortem Audit")
    print("=" * 72)
    validate_required_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    context = load_context()
    comparison, anchors = build_best_candidate_comparison(context)
    print(f"Best candidate comparison: {rel(BEST_COMPARISON_CSV)}")

    _, overall, ticker_drag, time_drag = build_prediction_diagnostics()
    print(f"Ticker drag: {rel(TICKER_DRAG_CSV)}")
    print(f"Time drag: {rel(TIME_DRAG_CSV)}")
    print(f"Class balance: {rel(CLASS_BALANCE_CSV)}")

    baseline = build_baseline_comparison(overall, context)
    mismatch = build_validation_final_mismatch(context)
    print(f"Baseline comparison: {rel(BASELINE_COMPARISON_CSV)}")
    print(f"Validation-final mismatch: {rel(VALIDATION_FINAL_MISMATCH_CSV)}")

    build_postmortem_report(context, comparison, anchors, overall, ticker_drag, time_drag, baseline, mismatch)
    print(f"Postmortem report: {rel(POSTMORTEM_MD)}")

    canonical_accuracy = anchors["canonical_accuracy"]
    v2_best_final = to_float(anchors["v2_best_final"].get("final_accuracy"))
    print(f"Canonical best daily result: LightGBM daily_cross h=40 = {fmt_pct(canonical_accuracy)}")
    print(f"V2 improved over v1: {'yes' if v2_best_final > canonical_accuracy else 'no'}")
    print("Daily-only audit complete. No hourly data used. No daily-to-hourly resampling used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
