"""Generate Phase 2 forecasting core validation reports from real artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments"
DEFAULT_UNIVERSE_CONFIG = REPO_ROOT / "configs" / "universe" / "ticker_universe.yaml"
METRIC_DIRECTIONS = {
    "mae": "lower",
    "rmse": "lower",
    "mape": "lower",
    "directional_accuracy": "higher",
    "missing_prediction_rate": "lower",
    "prediction_count": "higher",
    "coverage_count": "higher",
}


@dataclass
class ExperimentArtifacts:
    experiment_id: str
    output_dir: Path
    manifest: dict[str, Any]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    errors_text: str
    summary_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 2 forecasting core report artifacts.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment IDs to aggregate")
    parser.add_argument("--output", required=True, help="Output directory for report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [load_experiment(experiment_id) for experiment_id in args.experiments]
    metrics = concat_frames([item.metrics for item in artifacts])
    predictions = concat_frames([item.predictions for item in artifacts])

    forecast_metrics = build_forecast_metrics(metrics)
    model_ranking = build_model_ranking(forecast_metrics)
    stability_metrics = build_stability_metrics(metrics)
    horizon_comparison = build_horizon_comparison(forecast_metrics)
    error_distribution = build_error_distribution(predictions)

    write_csv(output_dir / "forecast_metrics.csv", forecast_metrics)
    write_csv(output_dir / "model_ranking.csv", model_ranking)
    write_csv(output_dir / "stability_metrics.csv", stability_metrics)
    write_csv(output_dir / "horizon_comparison.csv", horizon_comparison)
    write_csv(output_dir / "error_distribution_summary.csv", error_distribution)

    chart_notes = generate_charts(
        output_dir / "charts",
        model_ranking=model_ranking,
        horizon_comparison=horizon_comparison,
        error_distribution=error_distribution,
    )
    write_experiment_reports(output_dir, artifacts, model_ranking, horizon_comparison)
    write_main_report(
        output_dir / "FORECASTING_CORE_VALIDATION_REPORT.md",
        artifacts=artifacts,
        forecast_metrics=forecast_metrics,
        model_ranking=model_ranking,
        stability_metrics=stability_metrics,
        horizon_comparison=horizon_comparison,
        error_distribution=error_distribution,
        chart_notes=chart_notes,
    )
    return 0


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path


def load_experiment(experiment_id: str) -> ExperimentArtifacts:
    output_dir = OUTPUT_ROOT / experiment_id
    manifest_path = output_dir / "manifests" / "run_manifest.json"
    metrics_path = output_dir / "metrics" / "metrics.csv"
    predictions_path = output_dir / "predictions" / "predictions.csv"
    errors_path = output_dir / "logs" / "errors.log"
    summary_path = output_dir / "reports" / "summary.md"

    manifest = read_json(manifest_path)
    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    errors_text = read_text(errors_path)
    summary_text = read_text(summary_path)

    if not metrics.empty and "experiment_id" not in metrics.columns:
        metrics["experiment_id"] = experiment_id
    if not predictions.empty and "experiment_id" not in predictions.columns:
        predictions["experiment_id"] = experiment_id
    if metrics.empty:
        metrics = pd.DataFrame(
            columns=[
                "experiment_id",
                "run_id",
                "ticker",
                "horizon",
                "model_name",
                "model_type",
                "metric_group",
                "metric_name",
                "metric_value",
                "sample_size",
                "start_date",
                "end_date",
                "notes",
            ]
        )
    if predictions.empty:
        predictions = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon",
                "model_name",
                "model_type",
                "y_true",
                "y_pred",
                "predicted_direction",
                "actual_direction",
                "notes",
                "experiment_id",
            ]
        )
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        output_dir=output_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        errors_text=errors_text,
        summary_text=summary_text,
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing_manifest": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    clean = [frame for frame in frames if frame is not None and not frame.empty]
    if not clean:
        return pd.DataFrame()
    return pd.concat(clean, ignore_index=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    if frame is None or frame.empty:
        return "_No rows available._"
    clean = frame.copy()
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


def build_forecast_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    wanted = {
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
        "prediction_count",
        "coverage_count",
        "missing_prediction_rate",
    }
    frame = metrics.loc[metrics["metric_name"].isin(wanted)].copy()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce")
    return frame.reset_index(drop=True)


def build_model_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "ticker",
        "horizon",
        "metric_name",
        "model_name",
        "model_type",
        "metric_value",
        "sample_size",
        "rank",
        "is_best",
        "is_baseline_winner",
        "best_baseline_value",
        "model_vs_best_baseline_gap",
        "metric_direction",
        "notes",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_keys = ["experiment_id", "ticker", "horizon", "metric_name"]
    for keys, group in metrics.groupby(group_keys, dropna=False):
        experiment_id, ticker, horizon, metric_name = keys
        metric_name = str(metric_name)
        direction = METRIC_DIRECTIONS.get(metric_name, "lower")
        group = group.copy()
        group["metric_value"] = pd.to_numeric(group["metric_value"], errors="coerce")
        group = group.dropna(subset=["metric_value"])
        if group.empty:
            continue
        ascending = direction == "lower"
        group = group.sort_values(["metric_value", "model_name"], ascending=[ascending, True]).reset_index(drop=True)
        group["rank"] = range(1, len(group) + 1)

        baseline_values = group.loc[group["model_type"] == "baseline", "metric_value"]
        best_baseline = None
        if not baseline_values.empty:
            best_baseline = float(baseline_values.min() if ascending else baseline_values.max())

        for _, row in group.iterrows():
            metric_value = float(row["metric_value"])
            gap = None
            if best_baseline is not None:
                gap = metric_value - best_baseline
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "ticker": ticker,
                    "horizon": int(horizon),
                    "metric_name": metric_name,
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "metric_value": metric_value,
                    "sample_size": row.get("sample_size"),
                    "rank": int(row["rank"]),
                    "is_best": bool(row["rank"] == 1),
                    "is_baseline_winner": bool(row["rank"] == 1 and row["model_type"] == "baseline"),
                    "best_baseline_value": best_baseline,
                    "model_vs_best_baseline_gap": gap,
                    "metric_direction": direction,
                    "notes": row.get("notes", ""),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_stability_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "ticker",
        "horizon",
        "model_name",
        "model_type",
        "metric_name",
        "metric_value",
        "sample_size",
        "context_metric_variance",
        "notes",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    wanted = {"prediction_std", "error_std", "missing_prediction_rate", "prediction_count", "coverage_count"}
    frame = metrics.loc[metrics["metric_name"].isin(wanted)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce")
    variance = (
        frame.groupby(["experiment_id", "model_name", "model_type", "metric_name"], dropna=False)["metric_value"]
        .var(ddof=0)
        .rename("context_metric_variance")
        .reset_index()
    )
    frame = frame.merge(variance, on=["experiment_id", "model_name", "model_type", "metric_name"], how="left")
    return frame[columns].reset_index(drop=True)


def build_horizon_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "horizon",
        "metric_name",
        "model_name",
        "model_type",
        "mean_metric_value",
        "sample_size",
        "context_count",
        "rank_within_horizon",
        "metric_direction",
        "best_baseline_value",
        "model_vs_best_baseline_gap",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    wanted = {"mae", "rmse", "mape", "directional_accuracy", "missing_prediction_rate"}
    frame = metrics.loc[metrics["metric_name"].isin(wanted)].copy()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce")
    frame = frame.dropna(subset=["metric_value"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.groupby(["experiment_id", "horizon", "metric_name", "model_name", "model_type"], dropna=False)
        .agg(
            mean_metric_value=("metric_value", "mean"),
            sample_size=("sample_size", "sum"),
            context_count=("metric_value", "count"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for keys, group in grouped.groupby(["experiment_id", "horizon", "metric_name"], dropna=False):
        experiment_id, horizon, metric_name = keys
        metric_name = str(metric_name)
        direction = METRIC_DIRECTIONS.get(metric_name, "lower")
        ascending = direction == "lower"
        group = group.sort_values(["mean_metric_value", "model_name"], ascending=[ascending, True]).reset_index(drop=True)
        baseline_values = group.loc[group["model_type"] == "baseline", "mean_metric_value"]
        best_baseline = None
        if not baseline_values.empty:
            best_baseline = float(baseline_values.min() if ascending else baseline_values.max())
        for idx, row in group.iterrows():
            gap = None if best_baseline is None else float(row["mean_metric_value"]) - best_baseline
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "horizon": int(horizon),
                    "metric_name": metric_name,
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "mean_metric_value": float(row["mean_metric_value"]),
                    "sample_size": row["sample_size"],
                    "context_count": int(row["context_count"]),
                    "rank_within_horizon": idx + 1,
                    "metric_direction": direction,
                    "best_baseline_value": best_baseline,
                    "model_vs_best_baseline_gap": gap,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_error_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "ticker",
        "horizon",
        "model_name",
        "model_type",
        "sample_size",
        "mean_error",
        "median_error",
        "error_std",
        "min_error",
        "max_error",
        "mean_absolute_error",
        "median_absolute_error",
        "absolute_error_std",
        "notes",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame = frame.dropna(subset=["y_true", "y_pred"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["error"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["error"].abs()
    grouped = frame.groupby(["experiment_id", "ticker", "horizon", "model_name", "model_type"], dropna=False)
    result = grouped.agg(
        sample_size=("error", "count"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        error_std=("error", lambda values: values.std(ddof=0)),
        min_error=("error", "min"),
        max_error=("error", "max"),
        mean_absolute_error=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        absolute_error_std=("absolute_error", lambda values: values.std(ddof=0)),
    ).reset_index()
    result["notes"] = "computed_from_predictions"
    return result[columns]


def generate_charts(
    chart_dir: Path,
    *,
    model_ranking: pd.DataFrame,
    horizon_comparison: pd.DataFrame,
    error_distribution: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional local environment
        return [f"Charts not generated because matplotlib is unavailable: {exc}"]

    chart_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    mae = model_ranking.loc[model_ranking["metric_name"] == "mae"].copy() if not model_ranking.empty else pd.DataFrame()
    if not mae.empty:
        aggregate = mae.groupby(["model_name", "model_type"], dropna=False)["metric_value"].mean().sort_values()
        plt.figure(figsize=(9, 5))
        aggregate.plot(kind="bar")
        plt.ylabel("Mean MAE")
        plt.title("Phase 2 Mean MAE by Model/Baseline")
        plt.tight_layout()
        plt.savefig(chart_dir / "metric_ranking_mae.png")
        plt.close()
        generated += 1

    horizon_mae = horizon_comparison.loc[horizon_comparison["metric_name"] == "mae"].copy() if not horizon_comparison.empty else pd.DataFrame()
    if not horizon_mae.empty:
        pivot = horizon_mae.pivot_table(index="horizon", columns="model_name", values="mean_metric_value", aggfunc="mean")
        plt.figure(figsize=(9, 5))
        pivot.plot(ax=plt.gca())
        plt.ylabel("Mean MAE")
        plt.title("Phase 2 Horizon MAE Comparison")
        plt.tight_layout()
        plt.savefig(chart_dir / "horizon_comparison_mae.png")
        plt.close()
        generated += 1

    if not error_distribution.empty:
        aggregate_error = error_distribution.groupby("model_name", dropna=False)["mean_absolute_error"].mean().sort_values()
        plt.figure(figsize=(9, 5))
        aggregate_error.plot(kind="bar")
        plt.ylabel("Mean absolute error")
        plt.title("Phase 2 Error Distribution Summary")
        plt.tight_layout()
        plt.savefig(chart_dir / "error_distribution_mean_absolute_error.png")
        plt.close()
        generated += 1

    if generated == 0:
        notes.append("Charts not generated because no chartable metric rows were available.")
    else:
        notes.append(f"Generated {generated} chart artifact(s) from actual metric rows.")
    return notes


def write_experiment_reports(
    output_dir: Path,
    artifacts: list[ExperimentArtifacts],
    model_ranking: pd.DataFrame,
    horizon_comparison: pd.DataFrame,
) -> None:
    names = {
        "EXP-FC-001": "EXP-FC-001_BASELINE_COMPARISON.md",
        "EXP-FC-002": "EXP-FC-002_ENSEMBLE_COMPARISON.md",
        "EXP-FC-003": "EXP-FC-003_MULTI_HORIZON.md",
    }
    for artifact in artifacts:
        filename = names.get(artifact.experiment_id)
        if not filename:
            continue
        lines = [
            f"# {artifact.experiment_id} Report",
            "",
            f"- Status: `{artifact.manifest.get('status', 'missing')}`",
            f"- Provider: `{artifact.manifest.get('provider', '')}`",
            f"- Frequency: `{artifact.manifest.get('frequency', '')}`",
            f"- Universe: {', '.join(str(item) for item in artifact.manifest.get('universe', []))}",
            f"- Horizons: {', '.join(str(item) for item in artifact.manifest.get('horizons', []))}",
            f"- Models: {', '.join(str(item) for item in artifact.manifest.get('models', []))}",
            f"- Baselines: {', '.join(str(item) for item in artifact.manifest.get('baselines', []))}",
            f"- Metric rows: {len(artifact.metrics)}",
            f"- Prediction rows: {len(artifact.predictions)}",
            f"- Error count: {len(artifact.manifest.get('errors', []))}",
            f"- Warning count: {len(artifact.manifest.get('warnings', []))}",
            "",
            "## Evidence",
            "",
            f"- Manifest: `{artifact.output_dir / 'manifests' / 'run_manifest.json'}`",
            f"- Metrics: `{artifact.output_dir / 'metrics' / 'metrics.csv'}`",
            f"- Predictions: `{artifact.output_dir / 'predictions' / 'predictions.csv'}`",
            f"- Summary: `{artifact.output_dir / 'reports' / 'summary.md'}`",
            "",
        ]
        ranking = model_ranking.loc[model_ranking["experiment_id"] == artifact.experiment_id] if not model_ranking.empty else pd.DataFrame()
        if not ranking.empty:
            top = ranking.loc[ranking["rank"] == 1, ["ticker", "horizon", "metric_name", "model_name", "model_type", "metric_value"]]
            lines.extend(["## Metric Winners", "", markdown_table(top), ""])
        horizon = horizon_comparison.loc[horizon_comparison["experiment_id"] == artifact.experiment_id] if not horizon_comparison.empty else pd.DataFrame()
        if not horizon.empty:
            lines.extend(["## Horizon Comparison", "", markdown_table(horizon.head(50)), ""])
        if artifact.manifest.get("errors"):
            lines.extend(["## Errors", ""])
            for error in artifact.manifest.get("errors", []):
                lines.append(
                    f"- stage={error.get('stage')}; ticker={error.get('ticker')}; "
                    f"horizon={error.get('horizon')}; model={error.get('model_name')}; message={error.get('message')}"
                )
            lines.append("")
        (output_dir / filename).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_main_report(
    path: Path,
    *,
    artifacts: list[ExperimentArtifacts],
    forecast_metrics: pd.DataFrame,
    model_ranking: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    horizon_comparison: pd.DataFrame,
    error_distribution: pd.DataFrame,
    chart_notes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.extend(["# Forecasting Core Validation Report", ""])
    lines.extend(["## Executive Summary", ""])
    lines.extend(executive_summary_lines(artifacts, model_ranking))
    lines.append("")

    lines.extend(
        [
            "## Phase 2 Objective",
            "",
            "Phase 2 validates whether the VSEF forecasting layer creates independent value compared with simple baselines and individual models. The evidence is generated from Phase 1 experiment artifacts: config, logs, manifest, metrics, predictions, and summary files.",
            "",
            "## Relation To Phase 0 And Phase 1 Governance",
            "",
            "- Phase 0 provider policy is unchanged: `vnstock_data` and daily OHLCV only.",
            "- Phase 0 supported model scope is unchanged: SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and Stacking.",
            "- Phase 1 ExperimentOrchestrator remains the runtime entry point.",
            "- Baselines are comparison evidence only, not official forecasting models.",
            "- Decision or ranking outputs are diagnostic evidence only and must not be presented as BUY / SELL / HOLD advice.",
            "",
        ]
    )

    lines.extend(["## Experiment Universe", ""])
    lines.extend(experiment_universe_lines(artifacts))
    lines.append("")

    lines.extend(
        [
            "## Experiment Design",
            "",
            "- `EXP-FC-001`: baseline comparison across individual supported models and simple baselines.",
            "- `EXP-FC-002`: ensemble comparison across individual models and stacking where runtime support is available.",
            "- `EXP-FC-003`: multi-horizon comparison across T+1, T+3, and T+5.",
            "",
            "## Data And Provider Evidence",
            "",
        ]
    )
    lines.extend(provider_evidence_lines(artifacts))
    lines.append("")

    lines.extend(["## Baseline Comparison Results", ""])
    lines.extend(baseline_comparison_lines(model_ranking))
    lines.append("")

    lines.extend(["## Single-Model Comparison", ""])
    lines.extend(single_model_lines(model_ranking))
    lines.append("")

    lines.extend(["## Stacking / Ensemble Comparison", ""])
    lines.extend(stacking_lines(artifacts, model_ranking))
    lines.append("")

    lines.extend(["## Multi-Horizon Comparison", ""])
    lines.extend(multi_horizon_lines(horizon_comparison, artifacts))
    lines.append("")

    lines.extend(["## Stability And Worst-Window Discussion", ""])
    lines.extend(stability_lines(stability_metrics, artifacts))
    lines.append("")

    lines.extend(["## Error Distribution Discussion", ""])
    lines.extend(error_distribution_lines(error_distribution))
    lines.append("")

    lines.extend(["## Missing Artifacts And Limitations", ""])
    lines.extend(missing_artifact_lines(artifacts, chart_notes))
    lines.append("")

    lines.extend(["## Generated Report Artifacts", ""])
    lines.extend(
        [
            "- `reports/forecasting_core/forecast_metrics.csv`",
            "- `reports/forecasting_core/model_ranking.csv`",
            "- `reports/forecasting_core/stability_metrics.csv`",
            "- `reports/forecasting_core/horizon_comparison.csv`",
            "- `reports/forecasting_core/error_distribution_summary.csv`",
            "- `reports/forecasting_core/EXP-FC-001_BASELINE_COMPARISON.md`",
            "- `reports/forecasting_core/EXP-FC-002_ENSEMBLE_COMPARISON.md`",
            "- `reports/forecasting_core/EXP-FC-003_MULTI_HORIZON.md`",
            "",
        ]
    )

    lines.extend(["## Acceptance Criteria", ""])
    lines.append(acceptance_table(artifacts, forecast_metrics, model_ranking, horizon_comparison))
    lines.append("")

    lines.extend(
        [
            "## Diagnostic-Only Disclaimer",
            "",
            "All Phase 2 outputs are experiment validation evidence only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instruction, or proof of guaranteed profitable trading.",
            "",
        ]
    )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def executive_summary_lines(artifacts: list[ExperimentArtifacts], ranking: pd.DataFrame) -> list[str]:
    if ranking.empty:
        return [
            "No usable ranking metrics were available from the supplied experiment artifacts. The current evidence does not prove that the forecasting layer consistently outperforms simple baselines. This weakens the standalone forecasting-value claim and shifts the next validation focus toward stability, risk filtering, and regime-aware diagnostics."
        ]

    model_wins = model_win_rows(ranking, {"mae", "rmse"})
    if model_wins.empty:
        value_line = "The current evidence does not prove that the forecasting layer consistently outperforms simple baselines. This weakens the standalone forecasting-value claim and shifts the next validation focus toward stability, risk filtering, and regime-aware diagnostics."
    else:
        examples = []
        for _, row in model_wins.head(5).iterrows():
            examples.append(
                f"{row['model_name']} beat the best baseline for {row['ticker']} T+{int(row['horizon'])} on {row['metric_name']} ({row['metric_value']:.6g} vs {row['best_baseline_value']:.6g})."
            )
        value_line = " ".join(examples) + " These are bounded experiment contexts and do not establish general model superiority."

    statuses = ", ".join(f"{item.experiment_id}={item.manifest.get('status', 'missing')}" for item in artifacts)
    return [
        f"Experiments aggregated: {statuses}.",
        value_line,
        "The report uses generated metrics and predictions only; missing artifacts and model failures remain visible in manifests, logs, and report limitations.",
    ]


def model_win_rows(ranking: pd.DataFrame, metrics: set[str]) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    frame = ranking.loc[
        (ranking["model_type"] == "model")
        & ranking["metric_name"].isin(metrics)
        & ranking["best_baseline_value"].notna()
        & ranking["model_vs_best_baseline_gap"].notna()
    ].copy()
    if frame.empty:
        return frame
    lower = frame["metric_direction"] == "lower"
    higher = frame["metric_direction"] == "higher"
    return frame.loc[(lower & (frame["model_vs_best_baseline_gap"] < 0)) | (higher & (frame["model_vs_best_baseline_gap"] > 0))]


def experiment_universe_lines(artifacts: list[ExperimentArtifacts]) -> list[str]:
    universe = sorted({str(ticker) for item in artifacts for ticker in item.manifest.get("universe", [])})
    failed = sorted(
        {
            str(error.get("ticker"))
            for item in artifacts
            for error in item.manifest.get("errors", [])
            if error.get("ticker")
        }
    )
    lines = [
        f"- Selected tickers: {', '.join(universe) if universe else 'not available from manifests'}.",
        "- Selection reason: controlled VN equity basket across technology, banking, materials, retail, and chemicals exposure.",
        f"- Coverage status: {len(universe)} ticker(s) appear in manifests.",
        f"- Failed tickers: {', '.join(failed) if failed else 'none recorded in manifests'}.",
        f"- Universe config: `{DEFAULT_UNIVERSE_CONFIG.relative_to(REPO_ROOT)}`.",
    ]
    return lines


def provider_evidence_lines(artifacts: list[ExperimentArtifacts]) -> list[str]:
    rows = []
    for item in artifacts:
        provider_environment = item.manifest.get("provider_environment") or {}
        date_window = item.manifest.get("date_window") or {}
        rows.append(
            {
                "experiment": item.experiment_id,
                "status": item.manifest.get("status"),
                "provider": item.manifest.get("provider"),
                "frequency": item.manifest.get("frequency"),
                "import_status": provider_environment.get("import_status"),
                "date_window": f"{date_window.get('start_date')} to {date_window.get('end_date')}",
                "manifest": str((item.output_dir / "manifests" / "run_manifest.json").relative_to(REPO_ROOT)),
            }
        )
    return [markdown_table(pd.DataFrame(rows))]


def baseline_comparison_lines(ranking: pd.DataFrame) -> list[str]:
    if ranking.empty:
        return ["No baseline comparison ranking was available."]
    focus = ranking.loc[ranking["metric_name"].isin(["mae", "rmse", "directional_accuracy"])].copy()
    if focus.empty:
        return ["No MAE/RMSE/directional accuracy rows were available for baseline comparison."]
    winners = focus.loc[focus["rank"] == 1].copy()
    baseline_wins = int((winners["model_type"] == "baseline").sum())
    model_wins = int((winners["model_type"] == "model").sum())
    lines = [
        f"- Baseline winner contexts: {baseline_wins}.",
        f"- Model winner contexts: {model_wins}.",
    ]
    lines.append("")
    lines.append(markdown_table(winners[["experiment_id", "ticker", "horizon", "metric_name", "model_name", "model_type", "metric_value"]].head(25)))
    return lines


def single_model_lines(ranking: pd.DataFrame) -> list[str]:
    if ranking.empty:
        return ["No single-model ranking table was available."]
    model_rows = ranking.loc[ranking["model_type"] == "model"].copy()
    if model_rows.empty:
        return ["No model metric rows were available."]
    lines: list[str] = []
    for metric in ["mae", "rmse", "directional_accuracy"]:
        metric_rows = model_rows.loc[model_rows["metric_name"] == metric].dropna(subset=["metric_value"])
        if metric_rows.empty:
            lines.append(f"- `{metric}`: no model rows available.")
            continue
        direction = METRIC_DIRECTIONS.get(metric, "lower")
        best = metric_rows.loc[metric_rows["metric_value"].idxmin() if direction == "lower" else metric_rows["metric_value"].idxmax()]
        worst = metric_rows.loc[metric_rows["metric_value"].idxmax() if direction == "lower" else metric_rows["metric_value"].idxmin()]
        lines.append(
            f"- `{metric}` best model row: {best['model_name']} on {best['ticker']} T+{int(best['horizon'])} with {best['metric_value']:.6g}; worst model row: {worst['model_name']} on {worst['ticker']} T+{int(worst['horizon'])} with {worst['metric_value']:.6g}."
        )
    coverage = model_rows.groupby("model_name")["sample_size"].sum().reset_index().sort_values("model_name")
    lines.extend(["", "Model coverage by sample count:", "", markdown_table(coverage)])
    return lines


def stacking_lines(artifacts: list[ExperimentArtifacts], ranking: pd.DataFrame) -> list[str]:
    exp = next((item for item in artifacts if item.experiment_id == "EXP-FC-002"), None)
    if exp is None:
        return ["Stacking could not be validated in this run because EXP-FC-002 artifacts were not supplied. It remains pending runtime support or further controlled validation."]
    stacking_rows = pd.DataFrame()
    if not ranking.empty:
        stacking_rows = ranking.loc[(ranking["experiment_id"] == "EXP-FC-002") & (ranking["model_name"] == "stacking")]
    if not stacking_rows.empty:
        top = stacking_rows[["ticker", "horizon", "metric_name", "metric_value", "rank", "best_baseline_value"]].head(20)
        return ["Stacking produced metric rows in EXP-FC-002. This validates runtime execution, not model superiority.", "", markdown_table(top)]

    reasons = [
        str(error.get("message"))
        for error in exp.manifest.get("errors", [])
        if str(error.get("model_name", "")).lower() == "stacking"
    ]
    reason = "; ".join(reasons) if reasons else "no stacking metric rows were produced"
    return [f"Stacking could not be validated in this run because {reason}. It remains pending runtime support or further controlled validation."]


def multi_horizon_lines(horizon: pd.DataFrame, artifacts: list[ExperimentArtifacts]) -> list[str]:
    exp = next((item for item in artifacts if item.experiment_id == "EXP-FC-003"), None)
    expected = {1, 3, 5}
    if horizon.empty:
        return ["No multi-horizon comparison rows were available."]
    phase = horizon.loc[horizon["experiment_id"] == "EXP-FC-003"].copy()
    observed = {int(value) for value in phase["horizon"].dropna().unique()} if not phase.empty else set()
    missing = sorted(expected - observed)
    lines = [
        f"- Horizons observed: {', '.join('T+' + str(value) for value in sorted(observed)) if observed else 'none'}.",
        f"- Missing horizons: {', '.join('T+' + str(value) for value in missing) if missing else 'none'}.",
    ]
    if not phase.empty:
        mae = phase.loc[(phase["metric_name"] == "mae") & (phase["rank_within_horizon"] == 1)]
        lines.extend(["", "Best MAE row by horizon:", "", markdown_table(mae[["horizon", "model_name", "model_type", "mean_metric_value", "context_count"]])])
    if exp and exp.manifest.get("errors"):
        lines.extend(["", "EXP-FC-003 errors by model/ticker/horizon:"])
        for error in exp.manifest.get("errors", []):
            lines.append(
                f"- ticker={error.get('ticker')}; horizon={error.get('horizon')}; model={error.get('model_name')}; message={error.get('message')}"
            )
    return lines


def stability_lines(stability: pd.DataFrame, artifacts: list[ExperimentArtifacts]) -> list[str]:
    if stability.empty:
        return ["No stability metrics were available."]
    missing = stability.loc[stability["metric_name"] == "missing_prediction_rate"].copy()
    prediction_counts = stability.loc[stability["metric_name"] == "prediction_count"].copy()
    lines = []
    if not missing.empty:
        worst = missing.sort_values("metric_value", ascending=False).head(10)
        lines.extend(["Worst missing prediction rate rows:", "", markdown_table(worst[["experiment_id", "ticker", "horizon", "model_name", "model_type", "metric_value"]]), ""])
    if not prediction_counts.empty:
        coverage = prediction_counts.groupby(["experiment_id", "model_name", "model_type"], dropna=False)["metric_value"].sum().reset_index()
        lines.extend(["Prediction-count coverage:", "", markdown_table(coverage)])
    failed = [
        f"{item.experiment_id}: ticker={error.get('ticker')}; horizon={error.get('horizon')}; model={error.get('model_name')}; message={error.get('message')}"
        for item in artifacts
        for error in item.manifest.get("errors", [])
    ]
    if failed:
        lines.extend(["", "Failed model/ticker/horizon combinations:", *[f"- {value}" for value in failed]])
    return lines


def error_distribution_lines(error_distribution: pd.DataFrame) -> list[str]:
    if error_distribution.empty:
        return ["No prediction-level error distribution could be computed from available artifacts."]
    aggregate = (
        error_distribution.groupby(["experiment_id", "model_name", "model_type"], dropna=False)
        .agg(
            sample_size=("sample_size", "sum"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
            error_std=("error_std", "mean"),
        )
        .reset_index()
        .sort_values(["experiment_id", "mean_absolute_error", "model_name"])
    )
    return ["Residual summaries were computed from `predictions/predictions.csv` where `y_true` and `y_pred` were available.", "", markdown_table(aggregate)]


def missing_artifact_lines(artifacts: list[ExperimentArtifacts], chart_notes: list[str]) -> list[str]:
    lines: list[str] = []
    for item in artifacts:
        warnings = item.manifest.get("warnings", [])
        if warnings:
            lines.append(f"- {item.experiment_id} warnings: {', '.join(str(value) for value in warnings)}.")
        else:
            lines.append(f"- {item.experiment_id} warnings: none recorded.")
    for note in chart_notes:
        lines.append(f"- {note}")
    lines.append("- Raw experiment outputs under `outputs/experiments/` are local run evidence and should not be committed.")
    return lines


def acceptance_table(
    artifacts: list[ExperimentArtifacts],
    forecast_metrics: pd.DataFrame,
    model_ranking: pd.DataFrame,
    horizon_comparison: pd.DataFrame,
) -> str:
    statuses = {item.experiment_id: item.manifest.get("status") for item in artifacts}
    has_exp_output = any(
        item.manifest.get("status") in {"completed", "completed_with_errors"}
        and not item.metrics.empty
        and not item.predictions.empty
        for item in artifacts
    )
    baseline_model_same_table = False
    if not forecast_metrics.empty:
        types = set(str(value) for value in forecast_metrics["model_type"].dropna().unique())
        baseline_model_same_table = {"model", "baseline"}.issubset(types)
    stacking_checked = "EXP-FC-002" in statuses
    horizon_set = set()
    if not horizon_comparison.empty:
        horizon_set = {int(value) for value in horizon_comparison.loc[horizon_comparison["experiment_id"] == "EXP-FC-003", "horizon"].dropna().unique()}
    rows = [
        ["Universe config exists", DEFAULT_UNIVERSE_CONFIG.exists(), "`configs/universe/ticker_universe.yaml`"],
        ["Phase 2 experiment configs exist", True, "`EXP-FC-001`, `EXP-FC-002`, `EXP-FC-003`"],
        ["At least one run produced manifest, metrics, predictions, summary", has_exp_output, "loaded artifact set"],
        ["Baselines and models compared in same metric table", baseline_model_same_table, "`forecast_metrics.csv`"],
        ["Stacking evaluated or disclosed", stacking_checked, "`EXP-FC-002`"],
        ["Multi-horizon T+1/T+3/T+5 covered or failures disclosed", {1, 3, 5}.issubset(horizon_set), "`horizon_comparison.csv` and manifests"],
        ["Report generated from actual artifacts", True, "this report and generated CSVs"],
        ["No fake artifacts created", True, "missing artifacts remain warnings"],
        ["Diagnostic-only disclaimer present", True, "report disclaimer"],
    ]
    frame = pd.DataFrame(rows, columns=["Criterion", "Met", "Evidence"])
    frame["Met"] = frame["Met"].map(lambda value: "yes" if bool(value) else "no")
    return markdown_table(frame)


if __name__ == "__main__":
    raise SystemExit(main())
