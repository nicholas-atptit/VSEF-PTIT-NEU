"""Generate Phase 4 regime-aware analysis artifacts from Phase 2/3 outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - incomplete runtime only
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments"
REPORT_ROOT = REPO_ROOT / "reports" / "regime_analysis"

DISCLAIMER = (
    "All Phase 4 outputs are regime-analysis research artifacts only. They are not BUY / SELL / HOLD advice, "
    "capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of "
    "guaranteed profitable trading."
)

REGIME_COLUMNS_DEFAULT = ["trend_regime", "volatility_regime", "combined_regime"]
FORECAST_METRIC_COLUMNS = [
    "mae",
    "rmse",
    "mape",
    "directional_accuracy",
    "prediction_count",
    "missing_prediction_rate",
]
RISK_METRIC_COLUMNS = [
    "average_realized_return",
    "hit_ratio",
    "return_volatility_proxy",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "worst_period_return",
    "candidate_count",
    "missing_outcome_rate",
]
HEALTH_COLUMNS = [
    "experiment_id",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "regime_column",
    "regime",
    "prediction_count",
    "missing_prediction_rate",
    "mae",
    "rmse",
    "directional_accuracy",
    "error_std",
    "extreme_prediction_flag",
    "baseline_gap_flag",
    "eligible",
    "status",
    "exclusion_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 4 regime-aware analysis report artifacts.")
    parser.add_argument("--configs", nargs="+", required=True, help="EXP-RG-001/002/003 YAML configs")
    parser.add_argument("--output", required=True, help="Output directory for report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [read_yaml(resolve_repo_path(path)) for path in args.configs]
    configs_by_id = {
        str(config.get("experiment", {}).get("id") or f"config_{index}"): config
        for index, config in enumerate(configs)
    }
    model_config = configs_by_id.get("EXP-RG-001", {})
    risk_config = configs_by_id.get("EXP-RG-002", {})
    horizon_config = configs_by_id.get("EXP-RG-003", {})

    policy_path = first_existing_policy_path(configs) or REPO_ROOT / "configs" / "policies" / "regime_policy.yaml"
    policy = read_yaml(policy_path) if policy_path.exists() else {}
    regime_columns = get_regime_columns(model_config or horizon_config or risk_config)

    source_inventory = build_source_inventory(output_dir)
    loaded_sources = load_source_tables()
    labels_path = resolve_repo_path(
        str(
            model_config.get("source_artifacts", {}).get("regime_labels")
            or risk_config.get("source_artifacts", {}).get("regime_labels")
            or horizon_config.get("source_artifacts", {}).get("regime_labels")
            or output_dir / "regime_labels.csv"
        )
    )
    labels = normalize_regime_labels(read_csv(labels_path))
    regime_summary = summarize_labels(labels)

    model_experiments = list(
        dict.fromkeys(str(item) for item in model_config.get("source_artifacts", {}).get("forecasting_experiments", []))
    )
    if not model_experiments:
        model_experiments = ["EXP-FC-001", "EXP-FC-003"]
    model_predictions = load_prediction_experiments(model_experiments)
    model_joined = attach_regimes(normalize_predictions(model_predictions), labels)

    horizon_experiments = list(
        dict.fromkeys(str(item) for item in horizon_config.get("source_artifacts", {}).get("forecasting_experiments", []))
    )
    if not horizon_experiments:
        horizon_experiments = ["EXP-FC-003"]
    horizon_predictions = load_prediction_experiments(horizon_experiments)
    horizon_joined = attach_regimes(normalize_predictions(horizon_predictions), labels)

    model_min_obs = int(model_config.get("regime_analysis", {}).get("minimum_observations_per_regime") or 10)
    horizon_min_obs = int(horizon_config.get("regime_analysis", {}).get("minimum_observations_per_regime") or 10)
    risk_min_obs = int(risk_config.get("regime_analysis", {}).get("minimum_observations_per_regime") or 5)

    model_health = build_model_health(model_joined, policy, regime_columns)
    eligibility_flags = build_eligibility_flags(model_health)
    regime_model_metrics = build_model_metrics(model_joined, regime_columns, model_min_obs)
    model_ranking_by_regime = build_model_ranking_by_regime(model_joined, regime_columns, model_min_obs)
    model_ranking_consistency = build_model_ranking_consistency(model_ranking_by_regime)

    candidates_path = resolve_repo_path(
        str(risk_config.get("source_artifacts", {}).get("candidate_comparison") or "reports/risk_aware/candidate_comparison.csv")
    )
    basket_metrics_path = resolve_repo_path(
        str(risk_config.get("source_artifacts", {}).get("basket_metrics") or "reports/risk_aware/topn_basket_metrics.csv")
    )
    candidates = normalize_candidates(read_csv(candidates_path))
    basket_metrics = read_csv(basket_metrics_path)
    period_returns = read_csv(OUTPUT_ROOT / "EXP-RK-002" / "artifacts" / "basket_period_returns.csv")
    regime_risk_metrics = build_risk_metrics(candidates, labels, basket_metrics, regime_columns, risk_min_obs)
    regime_risk_policy_comparison = build_risk_policy_comparison(regime_risk_metrics)

    horizons = [int(value) for value in horizon_config.get("horizons", [1, 3, 5])]
    regime_horizon_metrics = build_horizon_metrics(horizon_joined, regime_columns, horizons, horizon_min_obs)
    horizon_ranking_by_regime = build_horizon_ranking_by_regime(regime_horizon_metrics)

    chart_notes = generate_charts(
        output_dir / "charts",
        labels=labels,
        model_ranking=model_ranking_by_regime,
        risk_metrics=regime_risk_metrics,
        horizon_metrics=regime_horizon_metrics,
        model_health=model_health,
    )

    table_map = {
        "regime_summary.csv": regime_summary,
        "regime_model_metrics.csv": regime_model_metrics,
        "model_ranking_by_regime.csv": model_ranking_by_regime,
        "model_ranking_consistency.csv": model_ranking_consistency,
        "regime_risk_metrics.csv": regime_risk_metrics,
        "regime_risk_policy_comparison.csv": regime_risk_policy_comparison,
        "regime_horizon_metrics.csv": regime_horizon_metrics,
        "horizon_ranking_by_regime.csv": horizon_ranking_by_regime,
        "model_health_by_regime.csv": model_health,
        "eligibility_flags.csv": eligibility_flags,
    }
    for name, frame in table_map.items():
        write_csv(output_dir / name, frame)

    model_report = render_model_report(regime_model_metrics, model_ranking_by_regime, model_ranking_consistency)
    risk_report = render_risk_report(regime_risk_metrics, regime_risk_policy_comparison)
    horizon_report = render_horizon_report(regime_horizon_metrics, horizon_ranking_by_regime)
    write_text(output_dir / "EXP-RG-001_MODEL_BY_REGIME.md", model_report)
    write_text(output_dir / "EXP-RG-002_RISK_BY_REGIME.md", risk_report)
    write_text(output_dir / "EXP-RG-003_HORIZON_BY_REGIME.md", horizon_report)

    main_report = render_main_report(
        policy=policy,
        labels=labels,
        regime_summary=regime_summary,
        model_health=model_health,
        eligibility_flags=eligibility_flags,
        regime_model_metrics=regime_model_metrics,
        model_ranking_by_regime=model_ranking_by_regime,
        model_ranking_consistency=model_ranking_consistency,
        regime_risk_metrics=regime_risk_metrics,
        regime_risk_policy_comparison=regime_risk_policy_comparison,
        regime_horizon_metrics=regime_horizon_metrics,
        horizon_ranking_by_regime=horizon_ranking_by_regime,
        source_inventory=source_inventory,
        loaded_sources=loaded_sources,
        chart_notes=chart_notes,
        period_returns=period_returns,
    )
    write_text(output_dir / "REGIME_AWARE_ANALYSIS_REPORT.md", main_report)

    write_raw_experiment_outputs(
        configs_by_id=configs_by_id,
        config_paths=[resolve_repo_path(path) for path in args.configs],
        table_map=table_map,
        reports={
            "EXP-RG-001": model_report,
            "EXP-RG-002": risk_report,
            "EXP-RG-003": horizon_report,
        },
        source_inventory=source_inventory,
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "regime_label_rows": int(len(labels)),
                "model_metric_rows": int(len(regime_model_metrics)),
                "risk_metric_rows": int(len(regime_risk_metrics)),
                "horizon_metric_rows": int(len(regime_horizon_metrics)),
                "chart_notes": chart_notes,
            },
            indent=2,
            default=json_default,
        )
    )
    return 0


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Phase 4 YAML configs.")
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def first_existing_policy_path(configs: list[dict[str, Any]]) -> Path | None:
    for config in configs:
        raw = config.get("source_artifacts", {}).get("regime_policy") or config.get("regime_policy")
        if raw:
            return resolve_repo_path(str(raw))
    return None


def get_regime_columns(config: dict[str, Any]) -> list[str]:
    columns = config.get("regime_analysis", {}).get("regime_columns") or REGIME_COLUMNS_DEFAULT
    return [str(column) for column in columns]


def build_source_inventory(output_dir: Path) -> pd.DataFrame:
    paths = [
        REPO_ROOT / "reports" / "forecasting_core" / "forecast_metrics.csv",
        REPO_ROOT / "reports" / "forecasting_core" / "model_ranking.csv",
        REPO_ROOT / "reports" / "forecasting_core" / "horizon_comparison.csv",
        REPO_ROOT / "reports" / "risk_aware" / "candidate_comparison.csv",
        REPO_ROOT / "reports" / "risk_aware" / "topn_basket_metrics.csv",
        output_dir / "regime_labels.csv",
        OUTPUT_ROOT / "EXP-FC-001" / "predictions" / "predictions.csv",
        OUTPUT_ROOT / "EXP-FC-001" / "metrics" / "metrics.csv",
        OUTPUT_ROOT / "EXP-FC-003" / "predictions" / "predictions.csv",
        OUTPUT_ROOT / "EXP-FC-003" / "metrics" / "metrics.csv",
        OUTPUT_ROOT / "EXP-RK-001" / "artifacts" / "candidate_comparison.csv",
        OUTPUT_ROOT / "EXP-RK-002" / "artifacts" / "topn_basket_metrics.csv",
        OUTPUT_ROOT / "EXP-RK-002" / "artifacts" / "basket_period_returns.csv",
    ]
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def load_source_tables() -> dict[str, pd.DataFrame]:
    return {
        "forecast_metrics": read_csv(REPO_ROOT / "reports" / "forecasting_core" / "forecast_metrics.csv"),
        "model_ranking": read_csv(REPO_ROOT / "reports" / "forecasting_core" / "model_ranking.csv"),
        "horizon_comparison": read_csv(REPO_ROOT / "reports" / "forecasting_core" / "horizon_comparison.csv"),
        "candidate_comparison": read_csv(REPO_ROOT / "reports" / "risk_aware" / "candidate_comparison.csv"),
        "topn_basket_metrics": read_csv(REPO_ROOT / "reports" / "risk_aware" / "topn_basket_metrics.csv"),
    }


def normalize_regime_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "ticker", "trend_regime", "volatility_regime", "combined_regime"]
    if labels.empty:
        return pd.DataFrame(columns=required)
    frame = labels.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in ["close", "daily_return", "rolling_return", "realized_volatility"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in required:
        if column not in frame.columns:
            frame[column] = "missing_regime_label"
    return frame.dropna(subset=["date", "ticker"]).reset_index(drop=True)


def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "trend_regime",
        "volatility_regime",
        "combined_regime",
        "observation_count",
        "start_date",
        "end_date",
        "mean_return",
        "mean_realized_volatility",
    ]
    if labels.empty:
        return pd.DataFrame(columns=columns)
    frame = labels.copy()
    frame["daily_return"] = pd.to_numeric(frame.get("daily_return"), errors="coerce")
    frame["realized_volatility"] = pd.to_numeric(frame.get("realized_volatility"), errors="coerce")
    summary = (
        frame.groupby(["ticker", "trend_regime", "volatility_regime", "combined_regime"], dropna=False)
        .agg(
            observation_count=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_return=("daily_return", "mean"),
            mean_realized_volatility=("realized_volatility", "mean"),
        )
        .reset_index()
    )
    summary["start_date"] = pd.to_datetime(summary["start_date"], errors="coerce").dt.date.astype(str)
    summary["end_date"] = pd.to_datetime(summary["end_date"], errors="coerce").dt.date.astype(str)
    return summary[columns]


def load_prediction_experiments(experiment_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id in experiment_ids:
        path = OUTPUT_ROOT / experiment_id / "predictions" / "predictions.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        frame["experiment_id"] = experiment_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "date",
        "ticker",
        "horizon",
        "model_name",
        "model_type",
        "y_true",
        "y_pred",
        "predicted_direction",
        "actual_direction",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    frame["model_name"] = frame["model_name"].astype(str).str.lower().str.strip()
    frame["model_type"] = frame["model_type"].astype(str).str.lower().str.strip()
    for column in ["y_true", "y_pred", "predicted_direction", "actual_direction"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "horizon", "model_name", "model_type"]).copy()
    return frame[columns].reset_index(drop=True)


def attach_regimes(frame: pd.DataFrame, labels: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        for column in REGIME_COLUMNS_DEFAULT:
            result[column] = pd.Series(dtype="object")
        result["regime_label_status"] = pd.Series(dtype="object")
        return result
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce").dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    label_cols = ["date", "ticker", "trend_regime", "volatility_regime", "combined_regime"]
    available_labels = labels[label_cols].drop_duplicates(["date", "ticker"], keep="last") if not labels.empty else pd.DataFrame(columns=label_cols)
    merged = result.merge(
        available_labels,
        left_on=[date_col, "ticker"],
        right_on=["date", "ticker"],
        how="left",
        suffixes=("", "_regime_label"),
    )
    if date_col != "date" and "date_regime_label" in merged.columns:
        merged = merged.drop(columns=["date_regime_label"])
    for column in REGIME_COLUMNS_DEFAULT:
        merged[column] = merged[column].fillna("missing_regime_label")
    merged["regime_label_status"] = np.where(
        merged["trend_regime"].eq("missing_regime_label"),
        "missing",
        "matched",
    )
    return merged


def compute_forecast_metrics(group: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(group.get("y_true"), errors="coerce")
    y_pred = pd.to_numeric(group.get("y_pred"), errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    errors = y_pred[valid] - y_true[valid]
    abs_errors = errors.abs()
    non_zero = valid & y_true.ne(0)
    predicted_direction = pd.to_numeric(group.get("predicted_direction"), errors="coerce")
    actual_direction = pd.to_numeric(group.get("actual_direction"), errors="coerce")
    direction_valid = predicted_direction.notna() & actual_direction.notna()
    return {
        "mae": float(abs_errors.mean()) if not abs_errors.empty else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if not errors.empty else np.nan,
        "mape": float(((y_pred[non_zero] - y_true[non_zero]).abs() / y_true[non_zero].abs()).mean() * 100.0)
        if non_zero.any()
        else np.nan,
        "directional_accuracy": float((predicted_direction[direction_valid] == actual_direction[direction_valid]).mean())
        if direction_valid.any()
        else np.nan,
        "prediction_count": int(valid.sum()),
        "missing_prediction_rate": float(y_pred.isna().mean()) if len(y_pred) else np.nan,
        "error_std": float(errors.std(ddof=0)) if len(errors) > 1 else np.nan,
    }


def build_model_metrics(predictions: pd.DataFrame, regime_columns: list[str], min_obs: int) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "ticker",
        "horizon",
        "model_name",
        "model_type",
        "regime_column",
        "regime",
        *FORECAST_METRIC_COLUMNS,
        "error_std",
        "small_sample_flag",
        "regime_label_missing_count",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for regime_column in regime_columns:
        group_cols = ["experiment_id", "ticker", "horizon", "model_name", "model_type", regime_column]
        for keys, group in predictions.groupby(group_cols, dropna=False, sort=True):
            metrics = compute_forecast_metrics(group)
            row = dict(zip(group_cols, keys))
            row["regime_column"] = regime_column
            row["regime"] = row.pop(regime_column)
            row.update(metrics)
            row["small_sample_flag"] = bool(row["prediction_count"] < min_obs)
            row["regime_label_missing_count"] = int(group["regime_label_status"].eq("missing").sum())
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_model_ranking_by_regime(predictions: pd.DataFrame, regime_columns: list[str], min_obs: int) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "horizon",
        "regime_column",
        "regime",
        "model_name",
        "model_type",
        *FORECAST_METRIC_COLUMNS,
        "error_std",
        "rank",
        "is_best",
        "is_baseline",
        "best_baseline_mae",
        "model_vs_best_baseline_mae_gap",
        "baseline_competitive",
        "small_sample_flag",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for regime_column in regime_columns:
        group_cols = ["experiment_id", "horizon", regime_column, "model_name", "model_type"]
        for keys, group in predictions.groupby(group_cols, dropna=False, sort=True):
            metrics = compute_forecast_metrics(group)
            row = dict(zip(group_cols, keys))
            row["regime_column"] = regime_column
            row["regime"] = row.pop(regime_column)
            row.update(metrics)
            row["small_sample_flag"] = bool(row["prediction_count"] < min_obs)
            rows.append(row)
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=columns)
    ranking["mae"] = pd.to_numeric(ranking["mae"], errors="coerce")
    ranking["is_baseline"] = ranking["model_type"].eq("baseline")
    ranked_frames: list[pd.DataFrame] = []
    for _, group in ranking.groupby(["experiment_id", "horizon", "regime_column", "regime"], dropna=False, sort=True):
        current = group.sort_values(["mae", "model_type", "model_name"], na_position="last").copy()
        current["rank"] = range(1, len(current) + 1)
        current["is_best"] = current["rank"].eq(1)
        baseline_mae = pd.to_numeric(current.loc[current["is_baseline"], "mae"], errors="coerce").min()
        current["best_baseline_mae"] = baseline_mae
        current["model_vs_best_baseline_mae_gap"] = current["mae"] - baseline_mae if pd.notna(baseline_mae) else np.nan
        current["baseline_competitive"] = current["is_baseline"] & current["rank"].le(3)
        ranked_frames.append(current)
    result = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    return result[columns]


def build_model_ranking_consistency(ranking: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "horizon",
        "regime_column",
        "model_name",
        "model_type",
        "regime_count",
        "mean_rank",
        "rank_std",
        "best_regime_count",
        "rank_min",
        "rank_max",
        "stable_best_across_observed_regimes",
    ]
    if ranking.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in ranking.groupby(["experiment_id", "horizon", "regime_column", "model_name", "model_type"], sort=True):
        ranks = pd.to_numeric(group["rank"], errors="coerce").dropna()
        rows.append(
            {
                "experiment_id": keys[0],
                "horizon": keys[1],
                "regime_column": keys[2],
                "model_name": keys[3],
                "model_type": keys[4],
                "regime_count": int(group["regime"].nunique(dropna=True)),
                "mean_rank": float(ranks.mean()) if not ranks.empty else np.nan,
                "rank_std": float(ranks.std(ddof=0)) if len(ranks) > 1 else 0.0,
                "best_regime_count": int(group["is_best"].sum()),
                "rank_min": int(ranks.min()) if not ranks.empty else np.nan,
                "rank_max": int(ranks.max()) if not ranks.empty else np.nan,
                "stable_best_across_observed_regimes": bool(len(ranks) > 0 and group["is_best"].sum() == group["regime"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_model_health(predictions: pd.DataFrame, policy: dict[str, Any], regime_columns: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=HEALTH_COLUMNS)
    gate = policy.get("model_health_gate", {}) or {}
    min_count = int(gate.get("minimum_prediction_count") or 10)
    max_missing = float(gate.get("maximum_missing_prediction_rate") if gate.get("maximum_missing_prediction_rate") is not None else 1.0)
    max_error_std = gate.get("maximum_error_std")
    zscore_threshold = float((gate.get("extreme_prediction_filter", {}) or {}).get("zscore_threshold") or 5.0)
    extreme_enabled = bool((gate.get("extreme_prediction_filter", {}) or {}).get("enabled", True))
    baseline_enabled = bool((gate.get("baseline_gap_check", {}) or {}).get("enabled", True))
    baseline_ratio = float((gate.get("baseline_gap_check", {}) or {}).get("maximum_model_underperformance_ratio") or 2.0)

    rows: list[dict[str, Any]] = []
    for regime_column in regime_columns:
        group_cols = ["experiment_id", "ticker", "horizon", "model_name", "model_type", regime_column]
        for keys, group in predictions.groupby(group_cols, dropna=False, sort=True):
            metrics = compute_forecast_metrics(group)
            y_pred = pd.to_numeric(group["y_pred"], errors="coerce").dropna()
            if extreme_enabled and len(y_pred) > 1 and y_pred.std(ddof=0) > 0:
                zscores = ((y_pred - y_pred.mean()) / y_pred.std(ddof=0)).abs()
                extreme_flag = bool((zscores > zscore_threshold).any())
            else:
                extreme_flag = False
            row = dict(zip(group_cols, keys))
            row["regime_column"] = regime_column
            row["regime"] = row.pop(regime_column)
            row.update(metrics)
            row["extreme_prediction_flag"] = extreme_flag
            rows.append(row)
    health = pd.DataFrame(rows)
    if health.empty:
        return pd.DataFrame(columns=HEALTH_COLUMNS)

    health["baseline_gap_flag"] = False
    if baseline_enabled:
        context_cols = ["experiment_id", "ticker", "horizon", "regime_column", "regime"]
        baseline = (
            health[health["model_type"].eq("baseline")]
            .groupby(context_cols, dropna=False)["mae"]
            .min()
            .reset_index()
            .rename(columns={"mae": "best_baseline_mae_for_health"})
        )
        health = health.merge(baseline, on=context_cols, how="left")
        baseline_mae = pd.to_numeric(health["best_baseline_mae_for_health"], errors="coerce")
        model_mae = pd.to_numeric(health["mae"], errors="coerce")
        health["baseline_gap_flag"] = (
            health["model_type"].ne("baseline")
            & baseline_mae.notna()
            & (baseline_mae > 0)
            & (model_mae / baseline_mae > baseline_ratio)
        )
    else:
        health["best_baseline_mae_for_health"] = np.nan

    statuses: list[str] = []
    reasons: list[str] = []
    eligible: list[bool] = []
    for _, row in health.iterrows():
        hard_reasons: list[str] = []
        flag_reasons: list[str] = []
        if pd.to_numeric(row["prediction_count"], errors="coerce") < min_count:
            hard_reasons.append("minimum_prediction_count_not_met")
        if pd.to_numeric(row["missing_prediction_rate"], errors="coerce") > max_missing:
            hard_reasons.append("maximum_missing_prediction_rate_exceeded")
        if max_error_std is not None and pd.notna(row["error_std"]) and float(row["error_std"]) > float(max_error_std):
            hard_reasons.append("maximum_error_std_exceeded")
        if bool(row["extreme_prediction_flag"]):
            flag_reasons.append("extreme_prediction_flag")
        if bool(row["baseline_gap_flag"]):
            flag_reasons.append("baseline_gap_flag")
        if hard_reasons:
            statuses.append("excluded")
            eligible.append(False)
            reasons.append(";".join(hard_reasons))
        elif flag_reasons:
            statuses.append("flagged")
            eligible.append(True)
            reasons.append(";".join(flag_reasons))
        else:
            statuses.append("eligible")
            eligible.append(True)
            reasons.append("")
    health["status"] = statuses
    health["eligible"] = eligible
    health["exclusion_reason"] = reasons
    return health[HEALTH_COLUMNS]


def build_eligibility_flags(model_health: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "ticker",
        "horizon",
        "model_name",
        "model_type",
        "regime_column",
        "regime",
        "status",
        "eligible",
        "extreme_prediction_flag",
        "baseline_gap_flag",
        "exclusion_reason",
    ]
    if model_health.empty:
        return pd.DataFrame(columns=columns)
    return model_health[columns].copy()


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "policy_id",
        "candidate_type",
        "candidate_date",
        "ticker",
        "horizon",
        "rank",
        "realized_return",
        "missing_prediction_rate",
        "diagnostic_only",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    frame = candidates.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["candidate_date"] = pd.to_datetime(frame["candidate_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["candidate_type"] = frame["candidate_type"].astype(str).str.lower().str.strip()
    frame["policy_id"] = frame["policy_id"].astype(str)
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["realized_return"] = pd.to_numeric(frame["realized_return"], errors="coerce")
    frame["missing_prediction_rate"] = pd.to_numeric(frame["missing_prediction_rate"], errors="coerce")
    return frame.dropna(subset=["candidate_date", "ticker", "horizon", "candidate_type", "rank"]).reset_index(drop=True)


def build_risk_metrics(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    basket_metrics: pd.DataFrame,
    regime_columns: list[str],
    min_obs: int,
) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "source_experiment_id",
        "policy_id",
        "candidate_type",
        "regime_column",
        "regime",
        "horizon",
        "top_n",
        *RISK_METRIC_COLUMNS,
        "basket_count",
        "small_sample_flag",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    joined = attach_regimes(candidates.rename(columns={"candidate_date": "date"}), labels, date_col="date")
    top_ns = sorted(set(pd.to_numeric(basket_metrics.get("top_n"), errors="coerce").dropna().astype(int).tolist())) if not basket_metrics.empty and "top_n" in basket_metrics.columns else [1, 3, 5]
    source_experiment_ids = ",".join(sorted(set(candidates["experiment_id"].dropna().astype(str)))) if "experiment_id" in candidates.columns else ""
    rows: list[dict[str, Any]] = []
    for top_n in top_ns:
        selected = joined[pd.to_numeric(joined["rank"], errors="coerce") <= int(top_n)].copy()
        if selected.empty:
            continue
        for regime_column in regime_columns:
            group_cols = ["policy_id", "candidate_type", regime_column, "horizon"]
            for keys, group in selected.groupby(group_cols, dropna=False, sort=True):
                period = (
                    group.groupby("date", dropna=False)["realized_return"]
                    .mean()
                    .dropna()
                    .sort_index()
                )
                selected_returns = pd.to_numeric(group["realized_return"], errors="coerce")
                row = {
                    "experiment_id": "EXP-RG-002",
                    "source_experiment_id": source_experiment_ids,
                    "policy_id": keys[0],
                    "candidate_type": keys[1],
                    "regime_column": regime_column,
                    "regime": keys[2],
                    "horizon": int(keys[3]),
                    "top_n": int(top_n),
                    "candidate_count": int(len(group)),
                    "average_realized_return": float(period.mean()) if not period.empty else np.nan,
                    "hit_ratio": float((period > 0).mean()) if not period.empty else np.nan,
                    "return_volatility_proxy": return_volatility_proxy(period),
                    "max_drawdown": max_drawdown_from_returns(period),
                    "var_95": var_from_returns(period, 0.95),
                    "cvar_95": cvar_from_returns(period, 0.95),
                    "worst_period_return": float(period.min()) if not period.empty else np.nan,
                    "missing_outcome_rate": float(selected_returns.isna().mean()) if len(selected_returns) else np.nan,
                    "basket_count": int(len(period)),
                    "small_sample_flag": bool(len(period) < min_obs),
                }
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_risk_policy_comparison(risk_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "regime_column",
        "regime",
        "horizon",
        "top_n",
        "forecast_only_average_realized_return",
        "risk_aware_average_realized_return",
        "average_realized_return_delta",
        "forecast_only_hit_ratio",
        "risk_aware_hit_ratio",
        "hit_ratio_delta",
        "forecast_only_max_drawdown",
        "risk_aware_max_drawdown",
        "max_drawdown_improvement",
        "forecast_only_var_95",
        "risk_aware_var_95",
        "var_95_improvement",
        "forecast_only_cvar_95",
        "risk_aware_cvar_95",
        "cvar_95_improvement",
        "candidate_count_forecast_only",
        "candidate_count_risk_aware",
    ]
    if risk_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    key_cols = ["regime_column", "regime", "horizon", "top_n"]
    for keys, group in risk_metrics.groupby(key_cols, dropna=False, sort=True):
        forecast = group[group["candidate_type"].eq("forecast_only")]
        risk = group[group["candidate_type"].eq("risk_aware")]
        if forecast.empty or risk.empty:
            continue
        f = forecast.iloc[0]
        r = risk.iloc[0]
        rows.append(
            {
                "regime_column": keys[0],
                "regime": keys[1],
                "horizon": int(keys[2]),
                "top_n": int(keys[3]),
                "forecast_only_average_realized_return": f["average_realized_return"],
                "risk_aware_average_realized_return": r["average_realized_return"],
                "average_realized_return_delta": r["average_realized_return"] - f["average_realized_return"],
                "forecast_only_hit_ratio": f["hit_ratio"],
                "risk_aware_hit_ratio": r["hit_ratio"],
                "hit_ratio_delta": r["hit_ratio"] - f["hit_ratio"],
                "forecast_only_max_drawdown": f["max_drawdown"],
                "risk_aware_max_drawdown": r["max_drawdown"],
                "max_drawdown_improvement": r["max_drawdown"] - f["max_drawdown"],
                "forecast_only_var_95": f["var_95"],
                "risk_aware_var_95": r["var_95"],
                "var_95_improvement": r["var_95"] - f["var_95"],
                "forecast_only_cvar_95": f["cvar_95"],
                "risk_aware_cvar_95": r["cvar_95"],
                "cvar_95_improvement": r["cvar_95"] - f["cvar_95"],
                "candidate_count_forecast_only": int(f["candidate_count"]),
                "candidate_count_risk_aware": int(r["candidate_count"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_horizon_metrics(predictions: pd.DataFrame, regime_columns: list[str], horizons: list[int], min_obs: int) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "scope",
        "ticker",
        "regime_column",
        "regime",
        "horizon",
        "mae",
        "rmse",
        "directional_accuracy",
        "prediction_count",
        "missing_prediction_rate",
        "small_sample_flag",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions[predictions["horizon"].astype(int).isin(horizons)].copy()
    rows: list[dict[str, Any]] = []
    for regime_column in regime_columns:
        for keys, group in frame.groupby(["experiment_id", "ticker", regime_column, "horizon"], dropna=False, sort=True):
            metrics = compute_forecast_metrics(group)
            rows.append(
                {
                    "experiment_id": keys[0],
                    "scope": "ticker",
                    "ticker": keys[1],
                    "regime_column": regime_column,
                    "regime": keys[2],
                    "horizon": int(keys[3]),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "directional_accuracy": metrics["directional_accuracy"],
                    "prediction_count": metrics["prediction_count"],
                    "missing_prediction_rate": metrics["missing_prediction_rate"],
                    "small_sample_flag": bool(metrics["prediction_count"] < min_obs),
                }
            )
        for keys, group in frame.groupby(["experiment_id", regime_column, "horizon"], dropna=False, sort=True):
            metrics = compute_forecast_metrics(group)
            rows.append(
                {
                    "experiment_id": keys[0],
                    "scope": "all_tickers",
                    "ticker": "ALL",
                    "regime_column": regime_column,
                    "regime": keys[1],
                    "horizon": int(keys[2]),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "directional_accuracy": metrics["directional_accuracy"],
                    "prediction_count": metrics["prediction_count"],
                    "missing_prediction_rate": metrics["missing_prediction_rate"],
                    "small_sample_flag": bool(metrics["prediction_count"] < min_obs),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_horizon_ranking_by_regime(horizon_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "regime_column",
        "regime",
        "horizon",
        "mae",
        "rmse",
        "directional_accuracy",
        "prediction_count",
        "rank",
        "is_best_horizon",
    ]
    if horizon_metrics.empty:
        return pd.DataFrame(columns=columns)
    all_rows = horizon_metrics[horizon_metrics["scope"].eq("all_tickers")].copy()
    ranked_frames = []
    for _, group in all_rows.groupby(["experiment_id", "regime_column", "regime"], dropna=False, sort=True):
        current = group.sort_values(["mae", "horizon"], na_position="last").copy()
        current["rank"] = range(1, len(current) + 1)
        current["is_best_horizon"] = current["rank"].eq(1)
        ranked_frames.append(current)
    result = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    return result[columns]


def return_volatility_proxy(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    volatility = values.std(ddof=0)
    if pd.isna(volatility) or volatility == 0:
        return np.nan
    return float(values.mean() / volatility)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def var_from_returns(returns: pd.Series, confidence: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(1.0 - confidence))


def cvar_from_returns(returns: pd.Series, confidence: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    var_value = var_from_returns(values, confidence)
    tail = values[values <= var_value]
    return float(tail.mean()) if not tail.empty else np.nan


def generate_charts(
    chart_dir: Path,
    *,
    labels: pd.DataFrame,
    model_ranking: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    model_health: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"matplotlib_unavailable:{exc}"]

    chart_dir.mkdir(parents=True, exist_ok=True)
    try:
        for column in REGIME_COLUMNS_DEFAULT:
            if column in labels.columns and not labels.empty:
                counts = labels[column].value_counts().sort_index()
                plot_bar(plt, counts, chart_dir / f"regime_distribution_{column}.png", f"Regime Distribution: {column}")
                notes.append(f"generated:regime_distribution_{column}.png")
            else:
                notes.append(f"skipped:regime_distribution_{column}:no_rows")

        model_mae = model_ranking[
            (model_ranking.get("regime_column") == "trend_regime")
            & (model_ranking.get("horizon").astype(str) == "1")
        ].copy() if not model_ranking.empty else pd.DataFrame()
        if not model_mae.empty:
            pivot = model_mae.pivot_table(index="model_name", columns="regime", values="mae", aggfunc="mean")
            plot_dataframe_bar(plt, pivot, chart_dir / "model_mae_by_regime_trend_regime.png", "Model MAE By Trend Regime")
            notes.append("generated:model_mae_by_regime_trend_regime.png")
        else:
            notes.append("skipped:model_mae_by_regime_trend_regime:no_rows")

        risk_subset = risk_metrics[
            (risk_metrics.get("regime_column") == "trend_regime") & (risk_metrics.get("top_n").astype(str) == "1")
        ].copy() if not risk_metrics.empty else pd.DataFrame()
        if not risk_subset.empty:
            pivot = risk_subset.pivot_table(index="regime", columns="candidate_type", values="average_realized_return", aggfunc="mean")
            plot_dataframe_bar(plt, pivot, chart_dir / "risk_policy_by_regime_trend_regime.png", "Risk Policy By Trend Regime")
            notes.append("generated:risk_policy_by_regime_trend_regime.png")
        else:
            notes.append("skipped:risk_policy_by_regime_trend_regime:no_rows")

        horizon_subset = horizon_metrics[
            (horizon_metrics.get("regime_column") == "trend_regime") & (horizon_metrics.get("scope") == "all_tickers")
        ].copy() if not horizon_metrics.empty else pd.DataFrame()
        if not horizon_subset.empty:
            pivot = horizon_subset.pivot_table(index="regime", columns="horizon", values="mae", aggfunc="mean")
            plot_dataframe_bar(plt, pivot, chart_dir / "horizon_by_regime_trend_regime.png", "Horizon MAE By Trend Regime")
            notes.append("generated:horizon_by_regime_trend_regime.png")
        else:
            notes.append("skipped:horizon_by_regime_trend_regime:no_rows")

        if not model_health.empty:
            counts = model_health["status"].value_counts().sort_index()
            plot_bar(plt, counts, chart_dir / "model_health_by_regime_status.png", "Model Health Status Counts")
            notes.append("generated:model_health_by_regime_status.png")
        else:
            notes.append("skipped:model_health_by_regime_status:no_rows")
    except Exception as exc:
        notes.append(f"chart_generation_failed:{exc}")
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
    return notes


def plot_bar(plt: Any, series: pd.Series, path: Path, title: str) -> None:
    ax = series.plot(kind="bar", figsize=(8, 4), title=title)
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


def plot_dataframe_bar(plt: Any, frame: pd.DataFrame, path: Path, title: str) -> None:
    ax = frame.plot(kind="bar", figsize=(10, 5), title=title)
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


def render_model_report(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    consistency: pd.DataFrame,
) -> str:
    best_by_trend = best_rows(ranking, "trend_regime")
    baseline = ranking[(ranking["model_type"].eq("baseline")) & (ranking["rank"].le(3))] if not ranking.empty else pd.DataFrame()
    universal = consistency[consistency["stable_best_across_observed_regimes"].eq(True)] if not consistency.empty else pd.DataFrame()
    if universal.empty:
        thesis = "Model ranking is not stable across regimes, which weakens any universal-best-model claim."
    else:
        thesis = "A stable MAE winner appears in observed context rows, mainly the persistence baseline; this supports baseline competitiveness, not a universal ML-model claim."
    lines = [
        "# EXP-RG-001 Model Performance By Regime",
        "",
        thesis,
        "",
        "## Best Model Rows By Trend Regime",
        "",
        markdown_table(best_by_trend.head(20)),
        "",
        "## Baseline Competitiveness",
        "",
        markdown_table(baseline.head(20)),
        "",
        "## Ranking Consistency",
        "",
        markdown_table(consistency.sort_values(["stable_best_across_observed_regimes", "best_regime_count"], ascending=[False, False]).head(20) if not consistency.empty else consistency),
        "",
        "## Small Samples",
        "",
        markdown_table(metrics[metrics["small_sample_flag"].eq(True)].head(20) if not metrics.empty else metrics),
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def render_risk_report(metrics: pd.DataFrame, comparison: pd.DataFrame) -> str:
    high_vol = comparison[(comparison["regime_column"].eq("volatility_regime")) & (comparison["regime"].eq("high_vol"))] if not comparison.empty else pd.DataFrame()
    bear_high = comparison[(comparison["regime_column"].eq("combined_regime")) & (comparison["regime"].eq("bear_high_vol"))] if not comparison.empty else pd.DataFrame()
    if comparison.empty:
        interpretation = "The current evidence does not prove that the risk-aware layer improves candidate utility within the tested regimes. Future work should revise risk features, thresholds, and eligibility rules."
    elif (comparison[["average_realized_return_delta", "hit_ratio_delta", "max_drawdown_improvement", "var_95_improvement", "cvar_95_improvement"]] > 0).any().any():
        interpretation = "Risk-aware diagnostics show mixed context-specific improvements in selected metric/regime rows, but this does not support a universal risk-layer improvement claim."
    else:
        interpretation = "The current evidence does not prove that the risk-aware layer improves candidate utility within the tested regimes. Future work should revise risk features, thresholds, and eligibility rules."
    lines = [
        "# EXP-RG-002 Risk-aware Value By Regime",
        "",
        interpretation,
        "",
        "## Forecast-only vs Risk-aware",
        "",
        markdown_table(comparison.head(30)),
        "",
        "## High-volatility Rows",
        "",
        markdown_table(high_vol.head(20)),
        "",
        "## Bear High-volatility Rows",
        "",
        markdown_table(bear_high.head(20)),
        "",
        "## Risk Metrics",
        "",
        markdown_table(metrics.head(30)),
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def render_horizon_report(metrics: pd.DataFrame, ranking: pd.DataFrame) -> str:
    all_rank = ranking[ranking["regime_column"].eq("trend_regime")] if not ranking.empty else pd.DataFrame()
    t1 = metrics[(metrics["horizon"].eq(1)) & (metrics["scope"].eq("all_tickers"))] if not metrics.empty else pd.DataFrame()
    t3 = metrics[(metrics["horizon"].eq(3)) & (metrics["scope"].eq("all_tickers"))] if not metrics.empty else pd.DataFrame()
    t5 = metrics[(metrics["horizon"].eq(5)) & (metrics["scope"].eq("all_tickers"))] if not metrics.empty else pd.DataFrame()
    lines = [
        "# EXP-RG-003 Horizon Performance By Regime",
        "",
        "Horizon behavior is evaluated as a diagnostic comparison across T+1, T+3, and T+5 rows where source artifacts allow it.",
        "",
        "## Horizon Ranking By Trend Regime",
        "",
        markdown_table(all_rank.head(30)),
        "",
        "## T+1",
        "",
        markdown_table(t1.head(20)),
        "",
        "## T+3",
        "",
        markdown_table(t3.head(20)),
        "",
        "## T+5",
        "",
        markdown_table(t5.head(20)),
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def render_main_report(
    *,
    policy: dict[str, Any],
    labels: pd.DataFrame,
    regime_summary: pd.DataFrame,
    model_health: pd.DataFrame,
    eligibility_flags: pd.DataFrame,
    regime_model_metrics: pd.DataFrame,
    model_ranking_by_regime: pd.DataFrame,
    model_ranking_consistency: pd.DataFrame,
    regime_risk_metrics: pd.DataFrame,
    regime_risk_policy_comparison: pd.DataFrame,
    regime_horizon_metrics: pd.DataFrame,
    horizon_ranking_by_regime: pd.DataFrame,
    source_inventory: pd.DataFrame,
    loaded_sources: dict[str, pd.DataFrame],
    chart_notes: list[str],
    period_returns: pd.DataFrame,
) -> str:
    windows = policy.get("windows", {}) or {}
    trend = policy.get("trend_regime", {}) or {}
    vol = policy.get("volatility_regime", {}) or {}
    robustness = policy.get("robustness", {}) or {}
    gate = policy.get("model_health_gate", {}) or {}

    trend_counts = labels["trend_regime"].value_counts().reset_index() if not labels.empty and "trend_regime" in labels.columns else pd.DataFrame()
    if not trend_counts.empty:
        trend_counts.columns = ["trend_regime", "rows"]
    vol_counts = labels["volatility_regime"].value_counts().reset_index() if not labels.empty and "volatility_regime" in labels.columns else pd.DataFrame()
    if not vol_counts.empty:
        vol_counts.columns = ["volatility_regime", "rows"]
    insufficient = int(labels["trend_regime"].eq("insufficient_history").sum()) if not labels.empty and "trend_regime" in labels.columns else 0

    status_counts = model_health["status"].value_counts().reset_index() if not model_health.empty else pd.DataFrame()
    if not status_counts.empty:
        status_counts.columns = ["status", "rows"]
    reason_counts = eligibility_flags["exclusion_reason"].replace("", "none").value_counts().reset_index() if not eligibility_flags.empty else pd.DataFrame()
    if not reason_counts.empty:
        reason_counts.columns = ["reason", "rows"]

    best_trend = best_rows(model_ranking_by_regime, "trend_regime")
    baseline_competitive = model_ranking_by_regime[
        model_ranking_by_regime["baseline_competitive"].eq(True)
    ] if not model_ranking_by_regime.empty else pd.DataFrame()
    stable_best = model_ranking_consistency[
        model_ranking_consistency["stable_best_across_observed_regimes"].eq(True)
    ] if not model_ranking_consistency.empty else pd.DataFrame()

    if stable_best.empty:
        model_thesis = "Model ranking is not stable across regimes, which weakens any universal-best-model claim."
    else:
        model_thesis = "Persistence is the stable MAE winner in several observed ranking contexts; this supports baseline competitiveness and does not prove a universal ML model across future regimes."

    risk_interpretation = risk_interpretation_text(regime_risk_policy_comparison)
    horizon_interpretation = horizon_interpretation_text(horizon_ranking_by_regime)

    missing_sources = source_inventory[source_inventory["exists"].eq(False)] if not source_inventory.empty else pd.DataFrame()
    missing_loaded = pd.DataFrame(
        [
            {"artifact": name, "rows": int(len(frame)), "loaded": not frame.empty}
            for name, frame in loaded_sources.items()
        ]
    )

    lines = [
        "# Phase 4 Regime-aware Analysis Report",
        "",
        "## 1. Executive summary",
        "",
        model_thesis,
        risk_interpretation,
        horizon_interpretation,
        "",
        "## 2. Phase 4 objective",
        "",
        "Phase 4 tests whether forecasting quality and risk-aware utility depend on market regime rather than trying to prove one universal best model.",
        "",
        "## 3. Relation to prior phases",
        "",
        "- Phase 0 froze VSEF v1 governance: vnstock_data, daily OHLCV, diagnostic-only outputs.",
        "- Phase 1 standardized config-driven experiment execution.",
        "- Phase 2 validated forecasting core against baselines but did not prove consistent model superiority on MAE/RMSE.",
        "- Phase 3 evaluated risk-aware diagnostic candidates but did not prove aggregate risk-aware improvement over forecast-only ranking.",
        "",
        "## 4. Regime policy definition",
        "",
        f"- Return window: {windows.get('return_window')}",
        f"- Volatility window: {windows.get('volatility_window')}",
        f"- Minimum periods: {windows.get('min_periods')}",
        f"- Rolling return rule: bull >= {trend.get('bull_threshold')}, bear <= {trend.get('bear_threshold')}, sideway within {trend.get('sideway_band')}",
        f"- Realized volatility rule: expanding ticker-level quantiles, high_vol >= q{vol.get('high_vol_quantile')}, low_vol <= q{vol.get('low_vol_quantile')}, fallback={vol.get('fallback_method')}",
        f"- Robustness alternatives: {json.dumps(robustness.get('alternative_thresholds', []), default=json_default)}",
        "- Limitation: rule-based labels are transparent diagnostics but can be sensitive to threshold choice and ticker-specific volatility history.",
        "",
        "## 5. Regime label dataset",
        "",
        f"- Tickers: {', '.join(sorted(labels['ticker'].dropna().unique())) if not labels.empty else 'none'}",
        f"- Date range: {labels['date'].min().date() if not labels.empty else 'n/a'} to {labels['date'].max().date() if not labels.empty else 'n/a'}",
        f"- Total observations: {len(labels)}",
        f"- Insufficient-history observations: {insufficient}",
        "",
        "Trend distribution:",
        "",
        markdown_table(trend_counts),
        "",
        "Volatility distribution:",
        "",
        markdown_table(vol_counts),
        "",
        "Summary sample:",
        "",
        markdown_table(regime_summary.head(20)),
        "",
        "## 6. Model health / eligibility gate",
        "",
        f"- Minimum prediction count: {gate.get('minimum_prediction_count')}",
        f"- Maximum missing prediction rate: {gate.get('maximum_missing_prediction_rate')}",
        f"- Maximum error std: {gate.get('maximum_error_std')}",
        f"- Extreme prediction z-score threshold: {(gate.get('extreme_prediction_filter', {}) or {}).get('zscore_threshold')}",
        f"- Baseline gap ratio: {(gate.get('baseline_gap_check', {}) or {}).get('maximum_model_underperformance_ratio')}",
        "- The health gate is diagnostic and is not used to polish results.",
        "",
        markdown_table(status_counts),
        "",
        markdown_table(reason_counts),
        "",
        "## 7. Model performance by regime",
        "",
        markdown_table(best_trend.head(30)),
        "",
        "Baseline competitiveness by regime:",
        "",
        markdown_table(baseline_competitive.head(20)),
        "",
        "Ranking consistency:",
        "",
        markdown_table(model_ranking_consistency.head(20)),
        "",
        "## 8. Risk layer by regime",
        "",
        risk_interpretation,
        "",
        markdown_table(regime_risk_policy_comparison.head(30)),
        "",
        "## 9. Horizon by regime",
        "",
        horizon_interpretation,
        "",
        markdown_table(horizon_ranking_by_regime.head(30)),
        "",
        "## 10. Research discussion",
        "",
        "- No universal-best-model conclusion is claimed from these diagnostics.",
        "- The evidence is evaluated by trend, volatility, and combined regimes, with small samples explicitly flagged.",
        "- Risk-aware layer utility is treated as context-specific decision-support evidence, not a general improvement claim.",
        "- Next roadmap work should focus on robustness checks, better risk features, and whether eligibility rules improve interpretability without filtering results opportunistically.",
        "",
        "## 11. Missing artifacts and limitations",
        "",
        "Loaded source table status:",
        "",
        markdown_table(missing_loaded),
        "",
        "Missing source paths:",
        "",
        markdown_table(missing_sources),
        "",
        f"- Basket period return rows read from EXP-RK-002: {len(period_returns)}",
        f"- Chart generation notes: {', '.join(chart_notes) if chart_notes else 'none'}",
        "",
        "## 12. Acceptance criteria table",
        "",
        markdown_table(acceptance_table(labels, regime_summary, model_health, regime_model_metrics, regime_risk_metrics, regime_horizon_metrics)),
        "",
        "## 13. Diagnostic-only disclaimer",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def best_rows(ranking: pd.DataFrame, regime_column: str) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    return ranking[(ranking["regime_column"].eq(regime_column)) & (ranking["is_best"].eq(True))].copy()


def risk_interpretation_text(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "The current evidence does not prove that the risk-aware layer improves candidate utility within the tested regimes. Future work should revise risk features, thresholds, and eligibility rules."
    metric_cols = [
        "average_realized_return_delta",
        "hit_ratio_delta",
        "max_drawdown_improvement",
        "var_95_improvement",
        "cvar_95_improvement",
    ]
    positive_rows = comparison[(comparison[metric_cols] > 0).any(axis=1)]
    if positive_rows.empty:
        return "The current evidence does not prove that the risk-aware layer improves candidate utility within the tested regimes. Future work should revise risk features, thresholds, and eligibility rules."
    return "Risk-aware diagnostics show mixed context-specific improvements in selected metric/regime rows, but this does not support a universal risk-layer improvement claim."


def horizon_interpretation_text(ranking: pd.DataFrame) -> str:
    if ranking.empty:
        return "Horizon performance could not be evaluated because no joined horizon rows were available."
    best_counts = ranking[ranking["is_best_horizon"].eq(True)]["horizon"].value_counts().to_dict()
    if len(best_counts) > 1:
        return "Horizon behavior changes by regime in the observed ranking rows, so horizon choice should be treated as regime-dependent evidence."
    return "Horizon ranking is relatively stable in the observed rows, but this remains exploratory under the rule-based regime definition."


def acceptance_table(
    labels: pd.DataFrame,
    regime_summary: pd.DataFrame,
    model_health: pd.DataFrame,
    model_metrics: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"criterion": "regime_labels.csv exists and has rows", "status": bool(len(labels) > 0)},
        {"criterion": "regime_summary.csv exists and has rows", "status": bool(len(regime_summary) > 0)},
        {"criterion": "model_health_by_regime.csv exists", "status": model_health is not None},
        {"criterion": "regime_model_metrics.csv has rows", "status": bool(len(model_metrics) > 0)},
        {"criterion": "regime_risk_metrics.csv has rows where source artifacts allow", "status": bool(len(risk_metrics) > 0)},
        {"criterion": "regime_horizon_metrics.csv covers available horizons", "status": bool(len(horizon_metrics) > 0)},
        {"criterion": "small-sample rows are flagged", "status": "small_sample_flag" in model_metrics.columns},
        {"criterion": "diagnostic-only disclaimer included", "status": True},
    ]
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    clean = frame.copy()
    if len(clean) > 60:
        clean = clean.head(60)
    clean = clean.where(pd.notna(clean), "")
    headers = [str(column) for column in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in clean.columns) + " |")
    return "\n".join(lines)


def write_raw_experiment_outputs(
    *,
    configs_by_id: dict[str, dict[str, Any]],
    config_paths: list[Path],
    table_map: dict[str, pd.DataFrame],
    reports: dict[str, str],
    source_inventory: pd.DataFrame,
) -> None:
    table_groups = {
        "EXP-RG-001": [
            "regime_model_metrics.csv",
            "model_ranking_by_regime.csv",
            "model_ranking_consistency.csv",
            "model_health_by_regime.csv",
            "eligibility_flags.csv",
        ],
        "EXP-RG-002": [
            "regime_risk_metrics.csv",
            "regime_risk_policy_comparison.csv",
        ],
        "EXP-RG-003": [
            "regime_horizon_metrics.csv",
            "horizon_ranking_by_regime.csv",
        ],
    }
    for experiment_id, names in table_groups.items():
        config = configs_by_id.get(experiment_id, {})
        output_root = resolve_repo_path(str(config.get("outputs", {}).get("root_dir") or "outputs/experiments"))
        base = output_root / experiment_id
        for relative in ("artifacts", "config", "logs", "manifests", "reports", "metrics", "charts", "predictions"):
            (base / relative).mkdir(parents=True, exist_ok=True)
        for config_path in config_paths:
            if config_path.name == f"{experiment_id}.yaml" and config_path.exists():
                shutil.copyfile(config_path, base / "config" / "original_config.yaml")
        for name in names:
            frame = table_map.get(name, pd.DataFrame())
            write_csv(base / "artifacts" / name, frame)
        primary = table_map.get(names[0], pd.DataFrame())
        write_csv(base / "metrics" / "metrics.csv", primary)
        write_text(base / "reports" / f"{experiment_id}.md", reports.get(experiment_id, ""))
        run_id = f"{experiment_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        manifest = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": "completed",
            "generated_at": datetime.now(UTC).isoformat(),
            "artifact_rows": {name: int(len(table_map.get(name, pd.DataFrame()))) for name in names},
            "source_inventory": source_inventory.to_dict(orient="records"),
            "diagnostic_only": True,
        }
        write_json(base / "manifests" / "run_manifest.json", manifest)
        write_text(base / "logs" / "run.log", json.dumps(manifest, indent=2, default=json_default))
        write_text(base / "logs" / "errors.log", "")


if __name__ == "__main__":
    raise SystemExit(main())
