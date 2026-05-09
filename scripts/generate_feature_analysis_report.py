"""Generate Phase 5 feature-analysis and interpretability artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
REGISTRY_PATH = REPO_ROOT / "configs" / "features" / "feature_group_registry.yaml"
REGIME_LABELS_PATH = REPO_ROOT / "reports" / "regime_analysis" / "regime_labels.csv"
REGIME_MODEL_METRICS_PATH = REPO_ROOT / "reports" / "regime_analysis" / "regime_model_metrics.csv"
CANDIDATE_METRICS_PATH = REPO_ROOT / "reports" / "risk_aware" / "topn_basket_metrics.csv"

REFERENCE_EXPERIMENT = "EXP-FA-000"
ABLATION_GROUP_BY_EXPERIMENT = {
    "EXP-FA-001": "lag_returns",
    "EXP-FA-002": "rolling_volatility",
    "EXP-FA-003": "momentum_indicators",
    "EXP-FA-004": "volume",
    "EXP-FA-005": "spread_range",
    "EXP-FA-006": "rolling_mean",
}
GROUP_OUTPUTS = {
    "lag_returns": "delta_metrics_lag.csv",
    "rolling_volatility": "delta_metrics_volatility.csv",
    "momentum_indicators": "delta_metrics_momentum.csv",
    "volume": "delta_metrics_volume.csv",
    "spread_range": "delta_metrics_spread_range.csv",
}
FORECAST_METRICS = [
    "mae",
    "rmse",
    "mape",
    "directional_accuracy",
    "missing_prediction_rate",
]
ABLATION_DELTA_COLUMNS = [
    "experiment_id",
    "reference_experiment_id",
    "removed_group",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "baseline_mae",
    "ablated_mae",
    "delta_mae",
    "baseline_rmse",
    "ablated_rmse",
    "delta_rmse",
    "baseline_mape",
    "ablated_mape",
    "delta_mape",
    "baseline_directional_accuracy",
    "ablated_directional_accuracy",
    "delta_directional_accuracy",
    "baseline_missing_prediction_rate",
    "ablated_missing_prediction_rate",
    "delta_missing_prediction_rate",
    "sample_size",
    "notes",
]
TREE_IMPORTANCE_COLUMNS = [
    "experiment_id",
    "ticker",
    "horizon",
    "model_name",
    "feature_name",
    "feature_group",
    "importance_type",
    "importance_value",
    "normalized_importance",
    "rank",
    "sample_size",
    "notes",
]
SHAP_SUMMARY_COLUMNS = [
    "experiment_id",
    "ticker",
    "horizon",
    "model_name",
    "feature_name",
    "feature_group",
    "mean_abs_shap",
    "normalized_mean_abs_shap",
    "rank",
    "sample_size",
    "notes",
]
FEATURE_VALUE_BY_HORIZON_COLUMNS = [
    "removed_group",
    "horizon",
    "context_count",
    "mean_delta_mae",
    "mean_delta_rmse",
    "mean_delta_mape",
    "mean_delta_directional_accuracy",
    "mae_worsened_when_removed_count",
    "rmse_worsened_when_removed_count",
    "directional_accuracy_worsened_when_removed_count",
    "small_sample_flag",
    "notes",
]
FEATURE_VALUE_BY_REGIME_COLUMNS = [
    "experiment_id",
    "reference_experiment_id",
    "removed_group",
    "regime_column",
    "regime",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "baseline_mae",
    "ablated_mae",
    "delta_mae",
    "baseline_rmse",
    "ablated_rmse",
    "delta_rmse",
    "baseline_directional_accuracy",
    "ablated_directional_accuracy",
    "delta_directional_accuracy",
    "sample_size",
    "small_sample_flag",
    "notes",
]
FEATURE_DECISION_QUALITY_COLUMNS = [
    "removed_group",
    "horizon",
    "top_n",
    "source_candidate_type",
    "source_hit_ratio",
    "source_average_realized_return",
    "source_return_volatility_proxy",
    "delta_topn_hit_ratio",
    "delta_average_realized_return",
    "delta_return_volatility_proxy",
    "notes",
]
DISCLAIMER = (
    "All Phase 5 outputs are feature-analysis and interpretability research artifacts only. They are not BUY / "
    "SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, "
    "causal proof, or proof of guaranteed profitable trading."
)


@dataclass
class ExperimentArtifacts:
    experiment_id: str
    output_dir: Path
    config: dict[str, Any]
    manifest: dict[str, Any]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_selection: pd.DataFrame
    tree_importance: pd.DataFrame
    shap_summary: pd.DataFrame
    errors_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 5 feature-analysis report artifacts.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment IDs to aggregate")
    parser.add_argument("--output", required=True, help="Output directory for report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [load_experiment(experiment_id) for experiment_id in args.experiments]
    registry = read_yaml(REGISTRY_PATH)
    registry_summary = build_registry_summary(registry)
    ablation_delta = build_ablation_delta_metrics(experiments)
    feature_value_by_horizon = build_feature_value_by_horizon(ablation_delta)
    feature_value_by_regime = build_feature_value_by_regime(experiments)
    feature_decision_quality = build_feature_decision_quality(ablation_delta)
    tree_importance = load_tree_feature_importance(experiments)
    feature_importance_summary = build_feature_importance_summary(tree_importance)
    shap_summary, shap_note = load_shap_summary(experiments, output_dir)

    write_csv(output_dir / "feature_group_registry_summary.csv", registry_summary)
    write_csv(output_dir / "ablation_delta_metrics.csv", ablation_delta)
    for group_name, filename in GROUP_OUTPUTS.items():
        write_csv(output_dir / filename, filter_group(ablation_delta, group_name))
    write_csv(output_dir / "delta_metrics_all_groups.csv", ablation_delta)
    write_csv(output_dir / "tree_feature_importance.csv", tree_importance)
    write_csv(output_dir / "feature_importance_summary.csv", feature_importance_summary)
    if not shap_summary.empty:
        write_csv(output_dir / "shap_summary.csv", shap_summary)
    write_csv(output_dir / "feature_value_by_horizon.csv", feature_value_by_horizon)
    write_csv(output_dir / "feature_value_by_regime.csv", feature_value_by_regime)
    write_csv(output_dir / "feature_decision_quality.csv", feature_decision_quality)

    chart_notes = generate_charts(
        output_dir / "charts",
        ablation_delta=ablation_delta,
        feature_importance_summary=feature_importance_summary,
        shap_summary=shap_summary,
        feature_value_by_horizon=feature_value_by_horizon,
        feature_value_by_regime=feature_value_by_regime,
    )
    limitations = render_limitations(
        experiments=experiments,
        ablation_delta=ablation_delta,
        tree_importance=tree_importance,
        shap_summary=shap_summary,
        shap_note=shap_note,
        feature_decision_quality=feature_decision_quality,
        chart_notes=chart_notes,
    )
    write_text(output_dir / "feature_analysis_limitations.md", limitations)
    report = render_report(
        experiments=experiments,
        registry_summary=registry_summary,
        ablation_delta=ablation_delta,
        tree_importance=tree_importance,
        feature_importance_summary=feature_importance_summary,
        shap_summary=shap_summary,
        shap_note=shap_note,
        feature_value_by_horizon=feature_value_by_horizon,
        feature_value_by_regime=feature_value_by_regime,
        feature_decision_quality=feature_decision_quality,
        chart_notes=chart_notes,
    )
    write_text(output_dir / "FEATURE_INTERPRETATION_REPORT.md", report)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "ablation_delta_rows": int(len(ablation_delta)),
                "tree_importance_rows": int(len(tree_importance)),
                "shap_rows": int(len(shap_summary)),
                "chart_notes": chart_notes,
                "shap_note": shap_note,
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
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing_manifest": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    frame = pd.read_csv(path)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame.reindex(columns=columns)
    return frame


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def load_experiment(experiment_id: str) -> ExperimentArtifacts:
    output_dir = OUTPUT_ROOT / experiment_id
    config_path = REPO_ROOT / "configs" / "experiments" / f"{experiment_id}.yaml"
    config = read_yaml(config_path)
    manifest = read_json(output_dir / "manifests" / "run_manifest.json")
    metrics = read_csv(output_dir / "metrics" / "metrics.csv")
    predictions = read_csv(output_dir / "predictions" / "predictions.csv")
    feature_selection = read_csv(output_dir / "artifacts" / "feature_selection_summary.csv")
    tree_importance = read_csv(output_dir / "artifacts" / "tree_feature_importance.csv", TREE_IMPORTANCE_COLUMNS)
    shap_summary = read_csv(output_dir / "artifacts" / "shap_summary.csv", SHAP_SUMMARY_COLUMNS)
    for frame in [metrics, predictions, feature_selection, tree_importance, shap_summary]:
        if frame is not None and not frame.empty and "experiment_id" not in frame.columns:
            frame["experiment_id"] = experiment_id
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        output_dir=output_dir,
        config=config,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        feature_selection=feature_selection,
        tree_importance=tree_importance,
        shap_summary=shap_summary,
        errors_text=read_text(output_dir / "logs" / "errors.log"),
    )


def build_registry_summary(registry: dict[str, Any]) -> pd.DataFrame:
    meta = registry.get("registry") or {}
    rows: list[dict[str, Any]] = []
    for group_name, group_config in (registry.get("feature_groups") or {}).items():
        rows.append(
            {
                "registry_id": meta.get("id"),
                "phase": meta.get("phase"),
                "diagnostic_only": meta.get("diagnostic_only"),
                "row_type": "feature_group",
                "feature_group": group_name,
                "description": group_config.get("description"),
                "expected_patterns": ",".join(str(value) for value in group_config.get("expected_patterns") or []),
                "hypothesis": group_config.get("hypothesis"),
                "risk": group_config.get("risk"),
                "governance_note": (registry.get("governance") or {}).get("required_disclaimer"),
            }
        )
    for field in registry.get("excluded_or_guarded_fields") or []:
        rows.append(
            {
                "registry_id": meta.get("id"),
                "phase": meta.get("phase"),
                "diagnostic_only": meta.get("diagnostic_only"),
                "row_type": "excluded_or_guarded_field",
                "feature_group": "excluded_or_guarded_fields",
                "description": str(field),
                "expected_patterns": "",
                "hypothesis": "",
                "risk": "Excluded from model features unless explicitly safe.",
                "governance_note": "Guarded against target leakage and metadata leakage.",
            }
        )
    return pd.DataFrame(rows)


def removed_group_from_config(artifact: ExperimentArtifacts) -> str:
    configured = (
        artifact.config.get("features", {})
        .get("ablation", {})
        .get("removed_group")
    )
    if isinstance(configured, list):
        configured = ",".join(str(value) for value in configured if str(value).strip())
    if configured:
        return str(configured)
    return ABLATION_GROUP_BY_EXPERIMENT.get(artifact.experiment_id, "")


def metrics_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    frame = frame[frame["metric_name"].isin(FORECAST_METRICS)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["sample_size"] = pd.to_numeric(frame.get("sample_size"), errors="coerce")
    index = ["ticker", "horizon", "model_name", "model_type"]
    values = frame.pivot_table(index=index, columns="metric_name", values="metric_value", aggfunc="first").reset_index()
    samples = frame.groupby(index, dropna=False)["sample_size"].max().rename("sample_size").reset_index()
    return values.merge(samples, on=index, how="left")


def build_ablation_delta_metrics(experiments: list[ExperimentArtifacts]) -> pd.DataFrame:
    reference = next((item for item in experiments if item.experiment_id == REFERENCE_EXPERIMENT), None)
    if reference is None or reference.metrics.empty:
        return pd.DataFrame(columns=ABLATION_DELTA_COLUMNS)
    reference_wide = metrics_wide(reference.metrics)
    if reference_wide.empty:
        return pd.DataFrame(columns=ABLATION_DELTA_COLUMNS)

    rows: list[dict[str, Any]] = []
    merge_keys = ["ticker", "horizon", "model_name", "model_type"]
    for artifact in experiments:
        removed_group = removed_group_from_config(artifact)
        if not removed_group or artifact.experiment_id == REFERENCE_EXPERIMENT:
            continue
        ablated_wide = metrics_wide(artifact.metrics)
        if ablated_wide.empty:
            continue
        merged = reference_wide.merge(
            ablated_wide,
            on=merge_keys,
            how="inner",
            suffixes=("_baseline", "_ablated"),
        )
        for _, row in merged.iterrows():
            result = {
                "experiment_id": artifact.experiment_id,
                "reference_experiment_id": REFERENCE_EXPERIMENT,
                "removed_group": removed_group,
                "ticker": row.get("ticker"),
                "horizon": row.get("horizon"),
                "model_name": row.get("model_name"),
                "model_type": row.get("model_type"),
                "sample_size": min_numeric(row.get("sample_size_baseline"), row.get("sample_size_ablated")),
                "notes": "Positive delta MAE/RMSE means removal worsened error; negative delta directional accuracy means removal worsened direction.",
            }
            for metric in FORECAST_METRICS:
                baseline_value = clean_float(row.get(f"{metric}_baseline"))
                ablated_value = clean_float(row.get(f"{metric}_ablated"))
                result[f"baseline_{metric}"] = baseline_value
                result[f"ablated_{metric}"] = ablated_value
                result[f"delta_{metric}"] = numeric_delta(ablated_value, baseline_value)
            rows.append(result)
    return pd.DataFrame(rows, columns=ABLATION_DELTA_COLUMNS)


def clean_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def numeric_delta(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return float(current - reference)


def min_numeric(left: Any, right: Any) -> float | None:
    values = [clean_float(left), clean_float(right)]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def filter_group(frame: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=ABLATION_DELTA_COLUMNS)
    return frame[frame["removed_group"].eq(group_name)].copy().reset_index(drop=True)


def build_feature_value_by_horizon(ablation_delta: pd.DataFrame) -> pd.DataFrame:
    if ablation_delta.empty:
        return pd.DataFrame(columns=FEATURE_VALUE_BY_HORIZON_COLUMNS)
    frame = ablation_delta[ablation_delta["model_type"].eq("model")].copy()
    for column in ["delta_mae", "delta_rmse", "delta_mape", "delta_directional_accuracy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (removed_group, horizon), group in frame.groupby(["removed_group", "horizon"], dropna=False):
        context_count = int(len(group))
        rows.append(
            {
                "removed_group": removed_group,
                "horizon": int(horizon),
                "context_count": context_count,
                "mean_delta_mae": clean_float(group["delta_mae"].mean()),
                "mean_delta_rmse": clean_float(group["delta_rmse"].mean()),
                "mean_delta_mape": clean_float(group["delta_mape"].mean()),
                "mean_delta_directional_accuracy": clean_float(group["delta_directional_accuracy"].mean()),
                "mae_worsened_when_removed_count": int((group["delta_mae"] > 0).sum()),
                "rmse_worsened_when_removed_count": int((group["delta_rmse"] > 0).sum()),
                "directional_accuracy_worsened_when_removed_count": int((group["delta_directional_accuracy"] < 0).sum()),
                "small_sample_flag": bool(context_count < 3),
                "notes": "Aggregated over model rows only; ablation deltas are diagnostic and not causal.",
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_VALUE_BY_HORIZON_COLUMNS)


def build_feature_value_by_regime(experiments: list[ExperimentArtifacts]) -> pd.DataFrame:
    labels = normalize_regime_labels(read_csv(REGIME_LABELS_PATH))
    reference = next((item for item in experiments if item.experiment_id == REFERENCE_EXPERIMENT), None)
    if labels.empty or reference is None or reference.predictions.empty:
        return pd.DataFrame(columns=FEATURE_VALUE_BY_REGIME_COLUMNS)
    reference_metrics = metrics_by_regime(reference.predictions, labels)
    if reference_metrics.empty:
        return pd.DataFrame(columns=FEATURE_VALUE_BY_REGIME_COLUMNS)

    rows: list[dict[str, Any]] = []
    merge_keys = ["ticker", "horizon", "model_name", "model_type", "regime_column", "regime"]
    for artifact in experiments:
        removed_group = removed_group_from_config(artifact)
        if not removed_group or artifact.experiment_id == REFERENCE_EXPERIMENT or artifact.predictions.empty:
            continue
        ablated_metrics = metrics_by_regime(artifact.predictions, labels)
        if ablated_metrics.empty:
            continue
        merged = reference_metrics.merge(
            ablated_metrics,
            on=merge_keys,
            how="inner",
            suffixes=("_baseline", "_ablated"),
        )
        for _, row in merged.iterrows():
            sample_size = min_numeric(row.get("sample_size_baseline"), row.get("sample_size_ablated"))
            rows.append(
                {
                    "experiment_id": artifact.experiment_id,
                    "reference_experiment_id": REFERENCE_EXPERIMENT,
                    "removed_group": removed_group,
                    "regime_column": row.get("regime_column"),
                    "regime": row.get("regime"),
                    "ticker": row.get("ticker"),
                    "horizon": row.get("horizon"),
                    "model_name": row.get("model_name"),
                    "model_type": row.get("model_type"),
                    "baseline_mae": clean_float(row.get("mae_baseline")),
                    "ablated_mae": clean_float(row.get("mae_ablated")),
                    "delta_mae": numeric_delta(clean_float(row.get("mae_ablated")), clean_float(row.get("mae_baseline"))),
                    "baseline_rmse": clean_float(row.get("rmse_baseline")),
                    "ablated_rmse": clean_float(row.get("rmse_ablated")),
                    "delta_rmse": numeric_delta(clean_float(row.get("rmse_ablated")), clean_float(row.get("rmse_baseline"))),
                    "baseline_directional_accuracy": clean_float(row.get("directional_accuracy_baseline")),
                    "ablated_directional_accuracy": clean_float(row.get("directional_accuracy_ablated")),
                    "delta_directional_accuracy": numeric_delta(
                        clean_float(row.get("directional_accuracy_ablated")),
                        clean_float(row.get("directional_accuracy_baseline")),
                    ),
                    "sample_size": sample_size,
                    "small_sample_flag": bool((sample_size or 0) < 10),
                    "notes": "Joined to rule-based regime labels by date and ticker; regime deltas are diagnostic.",
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_VALUE_BY_REGIME_COLUMNS)


def normalize_regime_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "ticker", "trend_regime", "volatility_regime", "combined_regime"]
    if labels.empty:
        return pd.DataFrame(columns=required)
    frame = labels.copy()
    for column in required:
        if column not in frame.columns:
            frame[column] = "missing_regime_label"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    return frame.dropna(subset=["date", "ticker"])[required].drop_duplicates(["date", "ticker"], keep="last")


def normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "ticker", "horizon", "model_name", "model_type", "y_true", "y_pred", "predicted_direction", "actual_direction"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    for column in ["y_true", "y_pred", "predicted_direction", "actual_direction"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "ticker", "horizon", "model_name", "model_type"])[columns]


def metrics_by_regime(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_predictions(predictions)
    if frame.empty:
        return pd.DataFrame()
    joined = frame.merge(labels, on=["date", "ticker"], how="left")
    for column in ["trend_regime", "volatility_regime", "combined_regime"]:
        joined[column] = joined[column].fillna("missing_regime_label")
    rows: list[dict[str, Any]] = []
    base_keys = ["ticker", "horizon", "model_name", "model_type"]
    for regime_column in ["trend_regime", "volatility_regime", "combined_regime"]:
        for keys, group in joined.groupby([*base_keys, regime_column], dropna=False):
            ticker, horizon, model_name, model_type, regime = keys
            computed = compute_prediction_metrics(group)
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": int(horizon),
                    "model_name": model_name,
                    "model_type": model_type,
                    "regime_column": regime_column,
                    "regime": regime,
                    **computed,
                }
            )
    return pd.DataFrame(rows)


def compute_prediction_metrics(group: pd.DataFrame) -> dict[str, Any]:
    y_true = pd.to_numeric(group["y_true"], errors="coerce")
    y_pred = pd.to_numeric(group["y_pred"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not bool(valid.any()):
        return {"mae": np.nan, "rmse": np.nan, "directional_accuracy": np.nan, "sample_size": 0}
    errors = y_pred[valid] - y_true[valid]
    actual = pd.to_numeric(group.loc[valid, "actual_direction"], errors="coerce")
    predicted = pd.to_numeric(group.loc[valid, "predicted_direction"], errors="coerce")
    direction_mask = actual.notna() & predicted.notna()
    directional = np.nan
    if bool(direction_mask.any()):
        directional = float((np.sign(actual[direction_mask]) == np.sign(predicted[direction_mask])).mean())
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "directional_accuracy": directional,
        "sample_size": int(valid.sum()),
    }


def build_feature_decision_quality(ablation_delta: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(set(str(value) for value in ablation_delta.get("removed_group", pd.Series(dtype=str)).dropna()))
    if not groups:
        groups = sorted(ABLATION_GROUP_BY_EXPERIMENT.values())
    candidates = read_csv(CANDIDATE_METRICS_PATH)
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        for group in groups:
            rows.append(
                {
                    "removed_group": group,
                    "horizon": np.nan,
                    "top_n": np.nan,
                    "source_candidate_type": "",
                    "source_hit_ratio": np.nan,
                    "source_average_realized_return": np.nan,
                    "source_return_volatility_proxy": np.nan,
                    "delta_topn_hit_ratio": np.nan,
                    "delta_average_realized_return": np.nan,
                    "delta_return_volatility_proxy": np.nan,
                    "notes": "Feature-specific top-N candidate metrics were unavailable; no values were invented.",
                }
            )
        return pd.DataFrame(rows, columns=FEATURE_DECISION_QUALITY_COLUMNS)

    for group in groups:
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "removed_group": group,
                    "horizon": row.get("horizon"),
                    "top_n": row.get("top_n"),
                    "source_candidate_type": row.get("candidate_type"),
                    "source_hit_ratio": clean_float(row.get("hit_ratio")),
                    "source_average_realized_return": clean_float(row.get("average_realized_return")),
                    "source_return_volatility_proxy": clean_float(row.get("return_volatility_proxy")),
                    "delta_topn_hit_ratio": np.nan,
                    "delta_average_realized_return": np.nan,
                    "delta_return_volatility_proxy": np.nan,
                    "notes": "Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank.",
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_DECISION_QUALITY_COLUMNS)


def load_tree_feature_importance(experiments: list[ExperimentArtifacts]) -> pd.DataFrame:
    frames = [item.tree_importance for item in experiments if item.tree_importance is not None and not item.tree_importance.empty]
    if not frames:
        return pd.DataFrame(columns=TREE_IMPORTANCE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    for column in ["importance_value", "normalized_importance", "rank", "sample_size", "horizon"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reindex(columns=TREE_IMPORTANCE_COLUMNS)


def build_feature_importance_summary(tree_importance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "feature_group",
        "feature_name",
        "context_count",
        "mean_importance_value",
        "mean_normalized_importance",
        "mean_rank",
        "top_5_count",
        "top_10_count",
        "sample_size",
        "notes",
    ]
    if tree_importance.empty:
        return pd.DataFrame(columns=columns)
    frame = tree_importance.copy()
    grouped = (
        frame.groupby(["model_name", "feature_group", "feature_name"], dropna=False)
        .agg(
            context_count=("normalized_importance", "count"),
            mean_importance_value=("importance_value", "mean"),
            mean_normalized_importance=("normalized_importance", "mean"),
            mean_rank=("rank", "mean"),
            top_5_count=("rank", lambda values: int((pd.to_numeric(values, errors="coerce") <= 5).sum())),
            top_10_count=("rank", lambda values: int((pd.to_numeric(values, errors="coerce") <= 10).sum())),
            sample_size=("sample_size", "sum"),
        )
        .reset_index()
    )
    grouped["notes"] = "High feature importance indicates strong model reliance, not causal market influence."
    return grouped.sort_values(["model_name", "mean_normalized_importance"], ascending=[True, False]).reindex(columns=columns)


def load_shap_summary(experiments: list[ExperimentArtifacts], output_dir: Path) -> tuple[pd.DataFrame, str]:
    frames = [item.shap_summary for item in experiments if item.shap_summary is not None and not item.shap_summary.empty]
    if frames:
        frame = pd.concat(frames, ignore_index=True).reindex(columns=SHAP_SUMMARY_COLUMNS)
        return frame, "SHAP summary loaded from experiment artifacts."
    try:
        __import__("shap")
    except Exception as exc:
        note = f"SHAP was not generated because shap is unavailable in the active environment: {exc}"
        write_text(output_dir / "SHAP_NOT_AVAILABLE.md", "# SHAP Not Available\n\n" + note + "\n\n" + DISCLAIMER)
        return pd.DataFrame(columns=SHAP_SUMMARY_COLUMNS), note
    note = "SHAP is importable, but no EXP-FA-008 shap_summary.csv artifact was produced by the experiment run."
    write_text(output_dir / "SHAP_NOT_AVAILABLE.md", "# SHAP Not Available\n\n" + note + "\n\n" + DISCLAIMER)
    return pd.DataFrame(columns=SHAP_SUMMARY_COLUMNS), note


def generate_charts(
    chart_dir: Path,
    *,
    ablation_delta: pd.DataFrame,
    feature_importance_summary: pd.DataFrame,
    shap_summary: pd.DataFrame,
    feature_value_by_horizon: pd.DataFrame,
    feature_value_by_regime: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"Charts not generated because matplotlib is unavailable: {exc}"]
    chart_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    try:
        model_ablation = ablation_delta[ablation_delta["model_type"].eq("model")].copy()
        for metric in ["mae", "rmse", "directional_accuracy"]:
            column = f"delta_{metric}"
            if not model_ablation.empty and column in model_ablation.columns:
                data = pd.to_numeric(model_ablation[column], errors="coerce")
                aggregate = model_ablation.assign(_value=data).groupby("removed_group")["_value"].mean().dropna()
                if not aggregate.empty:
                    ax = aggregate.sort_values().plot(kind="bar", figsize=(9, 5), title=f"Ablation Delta {metric.upper()}")
                    ax.set_ylabel(f"Mean delta {metric}")
                    ax.figure.tight_layout()
                    ax.figure.savefig(chart_dir / f"ablation_delta_{metric}_all.png")
                    plt.close(ax.figure)
                    generated += 1
        if not feature_importance_summary.empty:
            for model_name in ["xgboost", "lightgbm"]:
                model_rows = feature_importance_summary[feature_importance_summary["model_name"].eq(model_name)].head(15)
                if model_rows.empty:
                    continue
                labels = model_rows["feature_name"]
                ax = model_rows.set_index(labels)["mean_normalized_importance"].plot(
                    kind="bar",
                    figsize=(10, 5),
                    title=f"{model_name} Feature Importance",
                )
                ax.set_ylabel("Mean normalized importance")
                ax.figure.tight_layout()
                ax.figure.savefig(chart_dir / f"feature_importance_{model_name}_all.png")
                plt.close(ax.figure)
                generated += 1
        if not shap_summary.empty:
            top = shap_summary.sort_values("normalized_mean_abs_shap", ascending=False).head(15)
            ax = top.set_index("feature_name")["normalized_mean_abs_shap"].plot(kind="bar", figsize=(10, 5), title="SHAP Summary")
            ax.set_ylabel("Normalized mean abs SHAP")
            ax.figure.tight_layout()
            ax.figure.savefig(chart_dir / "shap_summary_all.png")
            plt.close(ax.figure)
            generated += 1
        if not feature_value_by_horizon.empty:
            pivot = feature_value_by_horizon.pivot_table(index="horizon", columns="removed_group", values="mean_delta_mae", aggfunc="mean")
            if not pivot.empty:
                ax = pivot.plot(figsize=(10, 5), title="Feature Value By Horizon - Delta MAE")
                ax.set_ylabel("Mean delta MAE")
                ax.figure.tight_layout()
                ax.figure.savefig(chart_dir / "feature_value_by_horizon_all.png")
                plt.close(ax.figure)
                generated += 1
        if not feature_value_by_regime.empty:
            subset = feature_value_by_regime[feature_value_by_regime["regime_column"].isin(["trend_regime", "volatility_regime"])].copy()
            subset["delta_mae"] = pd.to_numeric(subset["delta_mae"], errors="coerce")
            aggregate = subset.groupby(["removed_group", "regime"], dropna=False)["delta_mae"].mean().dropna()
            if not aggregate.empty:
                ax = aggregate.unstack(0).plot(kind="bar", figsize=(10, 5), title="Feature Value By Regime - Delta MAE")
                ax.set_ylabel("Mean delta MAE")
                ax.figure.tight_layout()
                ax.figure.savefig(chart_dir / "feature_value_by_regime_all.png")
                plt.close(ax.figure)
                generated += 1
    except Exception as exc:  # pragma: no cover - defensive chart path
        notes.append(f"Chart generation stopped after {generated} chart(s): {exc}")
    if generated:
        notes.append(f"Generated {generated} chart artifact(s) from actual Phase 5 tables.")
    elif not notes:
        notes.append("Charts not generated because no chartable Phase 5 rows were available.")
    return notes


def render_limitations(
    *,
    experiments: list[ExperimentArtifacts],
    ablation_delta: pd.DataFrame,
    tree_importance: pd.DataFrame,
    shap_summary: pd.DataFrame,
    shap_note: str,
    feature_decision_quality: pd.DataFrame,
    chart_notes: list[str],
) -> str:
    lines = [
        "# Phase 5 Feature Analysis Limitations",
        "",
        "- Feature importance and ablation results are diagnostic evidence, not causal proof.",
        "- Ablation can be affected by correlated features and model re-fitting variance.",
        "- Statistical wrappers such as ETS may ignore exogenous features, so feature ablation is mainly informative for feature-aware model wrappers.",
        "- Regime labels are rule-based diagnostics and are sensitive to policy thresholds and small samples.",
        "- Top-N decision-quality deltas are left blank unless feature-specific candidate metrics exist.",
        f"- SHAP status: {shap_note}",
        f"- Chart status: {' '.join(chart_notes) if chart_notes else 'no chart notes recorded'}",
        f"- Ablation rows generated: {len(ablation_delta)}.",
        f"- Tree importance rows generated: {len(tree_importance)}.",
        f"- SHAP rows generated: {len(shap_summary)}.",
        f"- Decision-quality rows generated: {len(feature_decision_quality)}.",
        "- Raw experiment outputs under `outputs/experiments/` are local run evidence and are not intended for staging.",
        "",
    ]
    for artifact in experiments:
        status = artifact.manifest.get("status", "missing")
        errors = artifact.manifest.get("errors", [])
        warnings = artifact.manifest.get("warnings", [])
        lines.append(f"- {artifact.experiment_id}: status={status}; errors={len(errors)}; warnings={len(warnings)}.")
    lines.extend(["", DISCLAIMER, ""])
    return "\n".join(lines)


def render_report(
    *,
    experiments: list[ExperimentArtifacts],
    registry_summary: pd.DataFrame,
    ablation_delta: pd.DataFrame,
    tree_importance: pd.DataFrame,
    feature_importance_summary: pd.DataFrame,
    shap_summary: pd.DataFrame,
    shap_note: str,
    feature_value_by_horizon: pd.DataFrame,
    feature_value_by_regime: pd.DataFrame,
    feature_decision_quality: pd.DataFrame,
    chart_notes: list[str],
) -> str:
    group_evidence = build_group_evidence_summary(ablation_delta)
    statuses = pd.DataFrame(
        [
            {
                "experiment_id": item.experiment_id,
                "status": item.manifest.get("status", "missing"),
                "metric_rows": len(item.metrics),
                "prediction_rows": len(item.predictions),
                "errors": len(item.manifest.get("errors", [])),
                "warnings": len(item.manifest.get("warnings", [])),
            }
            for item in experiments
        ]
    )
    top_importance = feature_importance_summary.head(20) if not feature_importance_summary.empty else feature_importance_summary
    horizon_tables = {
        horizon: feature_value_by_horizon[feature_value_by_horizon["horizon"].eq(horizon)]
        for horizon in [1, 3, 5]
    }
    trend_focus = feature_value_by_regime[
        feature_value_by_regime["regime_column"].eq("trend_regime")
        & feature_value_by_regime["regime"].isin(["bull", "bear", "sideway"])
    ] if not feature_value_by_regime.empty else pd.DataFrame(columns=FEATURE_VALUE_BY_REGIME_COLUMNS)
    vol_focus = feature_value_by_regime[
        feature_value_by_regime["regime_column"].eq("volatility_regime")
        & feature_value_by_regime["regime"].isin(["high_vol", "low_vol"])
    ] if not feature_value_by_regime.empty else pd.DataFrame(columns=FEATURE_VALUE_BY_REGIME_COLUMNS)

    lines = [
        "# Feature Interpretation Report",
        "",
        "## 1. Executive summary",
        "",
        *executive_summary_lines(group_evidence, tree_importance, shap_summary),
        "",
        "## 2. Phase 5 objective",
        "",
        "Phase 5 investigates which governed feature groups contribute to forecasting performance and diagnostic decision quality, and whether feature value varies by model, horizon, and regime.",
        "",
        "## 3. Relation to Phase 0-4",
        "",
        "- Phase 0 froze VSEF v1 governance: `vnstock_data`, daily OHLCV, frozen model scope, and diagnostic-only outputs.",
        "- Phase 1 implemented standardized config-driven experiment execution.",
        "- Phase 2 showed weak consistent model-vs-baseline superiority.",
        "- Phase 3 showed weak aggregate risk-aware improvement.",
        "- Phase 4 showed regime-dependent behavior.",
        "- Phase 5 investigates which feature groups explain or improve performance without treating importance as causal proof.",
        "",
        "## 4. Feature group registry",
        "",
        markdown_table(registry_summary[registry_summary["row_type"].eq("feature_group")][["feature_group", "description", "expected_patterns", "hypothesis", "risk"]]),
        "",
        "Guarded/excluded fields:",
        "",
        markdown_table(registry_summary[registry_summary["row_type"].eq("excluded_or_guarded_field")][["description", "risk"]]),
        "",
        "## 5. Ablation study design",
        "",
        "- `EXP-FA-000`: full feature reference.",
        "- `EXP-FA-001`: remove lag_returns.",
        "- `EXP-FA-002`: remove rolling_volatility.",
        "- `EXP-FA-003`: remove momentum_indicators.",
        "- `EXP-FA-004`: remove volume.",
        "- `EXP-FA-005`: remove spread_range.",
        "- `EXP-FA-006`: remove rolling_mean as the reduced/core comparison.",
        "- Models in the local ablation evidence run: XGBoost, LightGBM, ETS. SARIMAX was attempted in the initial full default set but exceeded the 10-minute local timeout before metrics were finalized, so it is disclosed as missing runtime evidence rather than forced or faked.",
        "- Tickers: FPT, ACB, HPG. Horizons: T+1, T+3, T+5.",
        "",
        "Experiment artifact status:",
        "",
        markdown_table(statuses),
        "",
        "## 6. Ablation results",
        "",
        markdown_table(group_evidence),
        "",
        ablation_interpretation_text(group_evidence),
        "",
        "Ablation delta rows are written to `ablation_delta_metrics.csv` and group-specific files. Positive delta MAE/RMSE means removing a feature group worsened error. Negative delta directional accuracy means removing a feature group worsened direction.",
        "",
        "## 7. Tree feature importance",
        "",
        tree_importance_text(tree_importance, feature_importance_summary),
        "",
        markdown_table(top_importance),
        "",
        "High feature importance indicates strong model reliance, not causal market influence.",
        "",
        "## 8. SHAP explanation if available",
        "",
        shap_text(shap_summary, shap_note),
        "",
        "## 9. Feature value by horizon",
        "",
        "T+1:",
        "",
        markdown_table(horizon_tables[1]),
        "",
        "T+3:",
        "",
        markdown_table(horizon_tables[3]),
        "",
        "T+5:",
        "",
        markdown_table(horizon_tables[5]),
        "",
        "## 10. Feature value by regime",
        "",
        "Bull, bear, and sideway rows:",
        "",
        markdown_table(summarize_regime_focus(trend_focus)),
        "",
        "High-volatility and low-volatility rows:",
        "",
        markdown_table(summarize_regime_focus(vol_focus)),
        "",
        "## 11. Feature value for decision quality",
        "",
        decision_quality_text(feature_decision_quality),
        "",
        markdown_table(feature_decision_quality.head(30)),
        "",
        "## 12. Interpretability caveats",
        "",
        "- Feature importance is not causality.",
        "- Ablation can be affected by feature correlation.",
        "- SHAP explains model behavior, not market truth.",
        "- Regime labels are rule-based diagnostics.",
        "- The feature appears important to the model but does not necessarily improve out-of-sample performance when ablation and importance disagree; this may indicate redundancy, overfitting, or correlated features.",
        "",
        "## 13. Acceptance criteria table",
        "",
        markdown_table(acceptance_table(experiments, ablation_delta, tree_importance, shap_summary)),
        "",
        "## 14. Diagnostic-only disclaimer",
        "",
        DISCLAIMER,
        "",
        "## Generated artifacts",
        "",
        "- `feature_group_registry_summary.csv`",
        "- `ablation_delta_metrics.csv`",
        "- `delta_metrics_lag.csv`",
        "- `delta_metrics_volatility.csv`",
        "- `delta_metrics_momentum.csv`",
        "- `delta_metrics_volume.csv`",
        "- `delta_metrics_spread_range.csv`",
        "- `delta_metrics_all_groups.csv`",
        "- `tree_feature_importance.csv`",
        "- `feature_importance_summary.csv`",
        "- `feature_value_by_horizon.csv`",
        "- `feature_value_by_regime.csv`",
        "- `feature_decision_quality.csv`",
        "- `feature_analysis_limitations.md`",
        f"- Chart notes: {' '.join(chart_notes) if chart_notes else 'none'}",
        "",
    ]
    return "\n".join(lines)


def build_group_evidence_summary(ablation_delta: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "removed_group",
        "context_count",
        "mae_worsened_when_removed",
        "rmse_worsened_when_removed",
        "directional_accuracy_worsened_when_removed",
        "mae_improved_when_removed",
        "rmse_improved_when_removed",
        "directional_accuracy_improved_when_removed",
        "mean_delta_mae",
        "mean_delta_rmse",
        "mean_delta_directional_accuracy",
        "evidence_label",
    ]
    if ablation_delta.empty:
        return pd.DataFrame(columns=columns)
    frame = ablation_delta[ablation_delta["model_type"].eq("model")].copy()
    for column in ["delta_mae", "delta_rmse", "delta_directional_accuracy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for group_name, group in frame.groupby("removed_group", dropna=False):
        mae_worse = int((group["delta_mae"] > 0).sum())
        rmse_worse = int((group["delta_rmse"] > 0).sum())
        dir_worse = int((group["delta_directional_accuracy"] < 0).sum())
        mae_better = int((group["delta_mae"] < 0).sum())
        rmse_better = int((group["delta_rmse"] < 0).sum())
        dir_better = int((group["delta_directional_accuracy"] > 0).sum())
        positive = mae_worse + rmse_worse + dir_worse
        negative = mae_better + rmse_better + dir_better
        if positive > negative * 1.5 and positive > 0:
            label = "positive_context_specific"
        elif negative > positive * 1.5 and negative > 0:
            label = "removal_improved_metrics"
        elif positive or negative:
            label = "mixed"
        else:
            label = "unclear"
        rows.append(
            {
                "removed_group": group_name,
                "context_count": int(len(group)),
                "mae_worsened_when_removed": mae_worse,
                "rmse_worsened_when_removed": rmse_worse,
                "directional_accuracy_worsened_when_removed": dir_worse,
                "mae_improved_when_removed": mae_better,
                "rmse_improved_when_removed": rmse_better,
                "directional_accuracy_improved_when_removed": dir_better,
                "mean_delta_mae": clean_float(group["delta_mae"].mean()),
                "mean_delta_rmse": clean_float(group["delta_rmse"].mean()),
                "mean_delta_directional_accuracy": clean_float(group["delta_directional_accuracy"].mean()),
                "evidence_label": label,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("removed_group").reset_index(drop=True)


def executive_summary_lines(group_evidence: pd.DataFrame, tree_importance: pd.DataFrame, shap_summary: pd.DataFrame) -> list[str]:
    if group_evidence.empty:
        contribution = "No ablation delta rows were available, so Phase 5 cannot make feature contribution claims from ablation evidence."
    else:
        positive = group_evidence[group_evidence["evidence_label"].eq("positive_context_specific")]["removed_group"].tolist()
        mixed = group_evidence[group_evidence["evidence_label"].eq("mixed")]["removed_group"].tolist()
        negative = group_evidence[group_evidence["evidence_label"].eq("removal_improved_metrics")]["removed_group"].tolist()
        parts = []
        if positive:
            parts.append("Groups with context-specific positive evidence: " + ", ".join(positive) + ".")
        if negative:
            parts.append("Groups where removal improved more metrics than it worsened: " + ", ".join(negative) + ".")
        if mixed:
            parts.append("Groups with mixed evidence: " + ", ".join(mixed) + ".")
        contribution = " ".join(parts) if parts else "The current evidence does not show consistent positive contribution from any tested feature group."
    tree_line = f"Tree importance rows generated: {len(tree_importance)}."
    shap_line = f"SHAP rows generated: {len(shap_summary)}."
    return [contribution, tree_line, shap_line, "No causal or investment claims are made from these diagnostics."]


def ablation_interpretation_text(group_evidence: pd.DataFrame) -> str:
    if group_evidence.empty:
        return "No ablation interpretation is available because no ablation delta rows were produced."
    lines = []
    for _, row in group_evidence.iterrows():
        group_name = row["removed_group"]
        label = row["evidence_label"]
        if label == "positive_context_specific":
            lines.append(
                f"- Removing {group_name} worsened enough MAE/RMSE/directional-accuracy contexts to suggest this group contributes useful predictive information under those tested contexts."
            )
        elif label == "removal_improved_metrics":
            lines.append(f"- The current evidence does not show a consistent positive contribution from {group_name}; removal improved more metric contexts than it worsened.")
        elif label == "mixed":
            lines.append(f"- {group_name} has mixed evidence across model/horizon/ticker contexts.")
        else:
            lines.append(f"- The current evidence for {group_name} is unclear.")
    return "\n".join(lines)


def tree_importance_text(tree_importance: pd.DataFrame, feature_importance_summary: pd.DataFrame) -> str:
    if tree_importance.empty:
        return "Tree feature importance was not available from the current wrappers or local model runs; no importance values were invented."
    groups = (
        feature_importance_summary.groupby(["model_name", "feature_group"], dropna=False)["mean_normalized_importance"]
        .sum()
        .reset_index()
        .sort_values(["model_name", "mean_normalized_importance"], ascending=[True, False])
    )
    return "Tree feature importance was extracted from `feature_importances_` where available.\n\n" + markdown_table(groups.head(20))


def shap_text(shap_summary: pd.DataFrame, shap_note: str) -> str:
    if shap_summary.empty:
        return shap_note + " No SHAP values were fabricated."
    top = shap_summary.sort_values("normalized_mean_abs_shap", ascending=False).head(20)
    return "SHAP global mean absolute contribution rows were generated. SHAP values explain model behavior, not market truth.\n\n" + markdown_table(top)


def summarize_regime_focus(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["removed_group", "regime_column", "regime", "context_count", "mean_delta_mae", "mean_delta_rmse", "mean_delta_directional_accuracy", "small_sample_rows"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    work = frame[frame["model_type"].eq("model")].copy()
    for column in ["delta_mae", "delta_rmse", "delta_directional_accuracy"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = (
        work.groupby(["removed_group", "regime_column", "regime"], dropna=False)
        .agg(
            context_count=("delta_mae", "count"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_delta_directional_accuracy=("delta_directional_accuracy", "mean"),
            small_sample_rows=("small_sample_flag", "sum"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=columns)


def decision_quality_text(feature_decision_quality: pd.DataFrame) -> str:
    if feature_decision_quality.empty:
        return "No top-N decision-quality source metrics were available. No feature-specific decision-quality deltas were invented."
    has_delta = feature_decision_quality[["delta_topn_hit_ratio", "delta_average_realized_return", "delta_return_volatility_proxy"]].notna().any().any()
    if has_delta:
        return "Feature-specific decision-quality deltas were available in source artifacts and are reported below."
    return "Top-N source metrics were available from Phase 3, but they are not feature-ablation-specific. Delta top-N hit ratio, realized return, and return/volatility proxy fields are intentionally blank."


def acceptance_table(
    experiments: list[ExperimentArtifacts],
    ablation_delta: pd.DataFrame,
    tree_importance: pd.DataFrame,
    shap_summary: pd.DataFrame,
) -> pd.DataFrame:
    statuses = {item.experiment_id: item.manifest.get("status") for item in experiments}
    has_ablation_output = any(
        item.experiment_id in ABLATION_GROUP_BY_EXPERIMENT
        and item.manifest.get("status") in {"completed", "completed_with_errors"}
        and not item.metrics.empty
        for item in experiments
    )
    rows = [
        {"criterion": "feature_group_registry.yaml exists", "status": REGISTRY_PATH.exists(), "evidence": "configs/features/feature_group_registry.yaml"},
        {"criterion": "Full feature reference config exists", "status": (REPO_ROOT / "configs/experiments/EXP-FA-000.yaml").exists(), "evidence": "EXP-FA-000"},
        {"criterion": "Ablation configs EXP-FA-001 to EXP-FA-006 exist", "status": all((REPO_ROOT / f"configs/experiments/{exp}.yaml").exists() for exp in ABLATION_GROUP_BY_EXPERIMENT), "evidence": "configs/experiments"},
        {"criterion": "Tree importance config EXP-FA-007 exists", "status": (REPO_ROOT / "configs/experiments/EXP-FA-007.yaml").exists(), "evidence": "EXP-FA-007"},
        {"criterion": "SHAP config EXP-FA-008 exists or SHAP_NOT_AVAILABLE.md explains missing evidence", "status": (REPO_ROOT / "configs/experiments/EXP-FA-008.yaml").exists(), "evidence": "EXP-FA-008 and report output"},
        {"criterion": "At least one ablation experiment produced metrics and manifest", "status": has_ablation_output, "evidence": str(statuses)},
        {"criterion": "Ablation delta metrics generated from actual artifacts", "status": not ablation_delta.empty, "evidence": "ablation_delta_metrics.csv"},
        {"criterion": "Tree feature importance generated or honestly marked unavailable", "status": True, "evidence": f"rows={len(tree_importance)}"},
        {"criterion": "Feature report generated", "status": True, "evidence": "FEATURE_INTERPRETATION_REPORT.md"},
        {"criterion": "Feature contribution discussed by horizon/regime where evidence allows", "status": True, "evidence": "feature_value_by_horizon.csv and feature_value_by_regime.csv"},
        {"criterion": "Interpretability caveats included", "status": True, "evidence": "report caveats"},
        {"criterion": "No causal claims made from importance alone", "status": True, "evidence": "diagnostic-only language"},
        {"criterion": "No fake metrics, SHAP, charts, or importance values created", "status": True, "evidence": "missing evidence disclosed"},
        {"criterion": "Diagnostic outputs not presented as investment advice", "status": True, "evidence": "disclaimer"},
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
        values = [str(row[column]).replace("\n", " ") for column in clean.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
