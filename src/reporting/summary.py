"""Summary builders for Phase 1 benchmark outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _aggregate_forecast_summary(forecast_summary: pd.DataFrame) -> pd.DataFrame:
    if forecast_summary.empty:
        return pd.DataFrame(columns=["model_name", "observations", "mae", "rmse", "mape", "smape", "directional_accuracy", "hit_rate"])
    aggregations: dict[str, str] = {
        "observations": "sum",
        "mae": "mean",
        "rmse": "mean",
        "mape": "mean",
        "smape": "mean",
        "directional_accuracy": "mean",
        "hit_rate": "mean",
    }
    present = {column: rule for column, rule in aggregations.items() if column in forecast_summary.columns}
    return (
        forecast_summary.groupby("model_name", sort=True)
        .agg(present)
        .reset_index()
        .sort_values("model_name")
        .reset_index(drop=True)
    )


def _aggregate_strategy_summary(strategy_metrics: pd.DataFrame) -> pd.DataFrame:
    if strategy_metrics.empty:
        return pd.DataFrame(columns=["model_name"])
    return (
        strategy_metrics.groupby("model_name", sort=True)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("model_name")
        .reset_index(drop=True)
    )


def build_model_comparison_summary(
    forecast_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forecast and strategy metrics into one ranking table."""

    forecast_model_summary = _aggregate_forecast_summary(forecast_summary)
    strategy_model_summary = _aggregate_strategy_summary(strategy_metrics)
    comparison = forecast_model_summary.merge(strategy_model_summary, on="model_name", how="outer")

    if "rmse" in comparison.columns:
        comparison["forecast_rank"] = comparison["rmse"].rank(method="dense", ascending=True)
    else:
        comparison["forecast_rank"] = np.nan
    if "sharpe" in comparison.columns:
        comparison["strategy_rank"] = comparison["sharpe"].rank(method="dense", ascending=False)
    else:
        comparison["strategy_rank"] = np.nan
    comparison["combined_rank"] = comparison[["forecast_rank", "strategy_rank"]].mean(axis=1, skipna=True)
    return comparison.sort_values(["combined_rank", "forecast_rank", "strategy_rank", "model_name"]).reset_index(drop=True)


def write_summary_tables(output_dir: str | Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Write named DataFrames to CSV in the output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in tables.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def render_summary_markdown(
    manifest: dict[str, Any],
    comparison_summary: pd.DataFrame,
) -> str:
    """Render a compact markdown summary for benchmark runs."""

    lines = [
        "# Phase 1 Benchmark Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Tickers: `{', '.join(manifest.get('tickers', []))}`",
        f"- Horizon: `{manifest.get('horizon')}`",
        f"- Requested models: `{', '.join(manifest.get('requested_models', []))}`",
        f"- Evaluated models: `{', '.join(manifest.get('evaluated_models', []))}`",
    ]
    skipped = manifest.get("skipped_models", [])
    if skipped:
        lines.append(f"- Skipped models: `{', '.join(item['model_name'] for item in skipped)}`")

    if comparison_summary.empty:
        lines.extend(["", "No comparison rows were produced."])
        return "\n".join(lines)

    lines.extend(["", "## Top Models", "", "| model_name | rmse | sharpe | combined_rank |", "| --- | ---: | ---: | ---: |"])
    for row in comparison_summary.head(10).itertuples(index=False):
        rmse = getattr(row, "rmse", np.nan)
        sharpe = getattr(row, "sharpe", np.nan)
        rank = getattr(row, "combined_rank", np.nan)
        lines.append(
            f"| {row.model_name} | {rmse:.6f} | {sharpe:.6f} | {rank:.3f} |"
        )
    return "\n".join(lines)


def write_summary_markdown(output_dir: str | Path, markdown: str, filename: str = "summary.md") -> Path:
    """Persist a markdown summary next to the benchmark artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    path.write_text(markdown, encoding="utf-8")
    return path
