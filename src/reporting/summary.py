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


def build_conditioning_mode_summary(
    strategy_metrics: pd.DataFrame,
    *,
    baseline_variant: str = "forecast_only",
) -> pd.DataFrame:
    """Aggregate strategy metrics by conditioning mode and compute deltas vs baseline."""

    if strategy_metrics.empty or "strategy_variant" not in strategy_metrics.columns:
        return pd.DataFrame(columns=["strategy_variant"])
    summary = (
        strategy_metrics.groupby("strategy_variant", sort=True)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("strategy_variant")
        .reset_index(drop=True)
    )
    baseline = summary[summary["strategy_variant"] == baseline_variant]
    if baseline.empty:
        return summary
    baseline_row = baseline.iloc[0]
    for column in ("cagr", "sharpe", "sortino", "max_drawdown", "turnover", "win_rate", "total_return"):
        if column in summary.columns:
            summary[f"delta_{column}_vs_{baseline_variant}"] = summary[column] - float(baseline_row[column])
    return summary


def build_phase2_conditioning_summary(
    forecast_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    *,
    baseline_variant: str = "forecast_only",
) -> pd.DataFrame:
    """Merge forecast metrics with per-variant strategy metrics for Phase 2 runs."""

    forecast_model_summary = _aggregate_forecast_summary(forecast_summary)
    if strategy_metrics.empty or "strategy_variant" not in strategy_metrics.columns:
        return build_model_comparison_summary(forecast_summary, strategy_metrics)

    strategy_model_summary = (
        strategy_metrics.groupby(["strategy_variant", "model_name"], sort=True)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["strategy_variant", "model_name"])
        .reset_index(drop=True)
    )
    comparison = strategy_model_summary.merge(forecast_model_summary, on="model_name", how="left")

    baseline = comparison[comparison["strategy_variant"] == baseline_variant][
        ["model_name", "cagr", "sharpe", "sortino", "max_drawdown", "turnover", "win_rate", "total_return"]
    ].rename(
        columns={
            "cagr": f"baseline_cagr_{baseline_variant}",
            "sharpe": f"baseline_sharpe_{baseline_variant}",
            "sortino": f"baseline_sortino_{baseline_variant}",
            "max_drawdown": f"baseline_max_drawdown_{baseline_variant}",
            "turnover": f"baseline_turnover_{baseline_variant}",
            "win_rate": f"baseline_win_rate_{baseline_variant}",
            "total_return": f"baseline_total_return_{baseline_variant}",
        }
    )
    comparison = comparison.merge(baseline, on="model_name", how="left")
    for column in ("cagr", "sharpe", "sortino", "max_drawdown", "turnover", "win_rate", "total_return"):
        baseline_column = f"baseline_{column}_{baseline_variant}"
        if baseline_column in comparison.columns:
            comparison[f"delta_{column}_vs_{baseline_variant}"] = comparison[column] - comparison[baseline_column]

    comparison["strategy_rank_within_variant"] = comparison.groupby("strategy_variant")["sharpe"].rank(
        method="dense",
        ascending=False,
    )
    return comparison.sort_values(["strategy_variant", "strategy_rank_within_variant", "rmse", "model_name"]).reset_index(drop=True)


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


def render_phase2_summary_markdown(
    manifest: dict[str, Any],
    mode_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> str:
    """Render a markdown summary for Phase 2 conditioning comparisons."""

    runtime = manifest.get("runtime", {})
    lines = [
        "# Phase 2 Benchmark Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Horizon: `{manifest.get('horizon')}`",
        f"- Evaluated models: `{', '.join(manifest.get('evaluated_models', []))}`",
        f"- Benchmark modes: `{', '.join(manifest.get('benchmark_modes', []))}`",
        f"- Regime model: `{manifest.get('regime', {}).get('model_name')}`",
        f"- Risk model: `{manifest.get('risk', {}).get('model_name')}`",
        f"- Threshold policy: `{manifest.get('strategy', {}).get('threshold_policy')}`",
        f"- Sizing policy: `{manifest.get('strategy', {}).get('sizing_policy')}`",
    ]
    skipped = manifest.get("skipped_models", [])
    if skipped:
        lines.append(f"- Skipped models: `{', '.join(item['model_name'] for item in skipped)}`")

    if not mode_summary.empty:
        lines.extend(
            [
                "",
                "## Conditioning Modes",
                "",
                "| strategy_variant | cagr | sharpe | max_drawdown | turnover |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in mode_summary.itertuples(index=False):
            lines.append(
                f"| {row.strategy_variant} | {getattr(row, 'cagr', np.nan):.6f} | "
                f"{getattr(row, 'sharpe', np.nan):.6f} | {getattr(row, 'max_drawdown', np.nan):.6f} | "
                f"{getattr(row, 'turnover', np.nan):.6f} |"
            )

    if comparison_summary.empty:
        lines.extend(["", "No Phase 2 comparison rows were produced."])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## Top Model Rows",
            "",
            "| strategy_variant | model_name | rmse | sharpe | max_drawdown | delta_sharpe_vs_forecast_only |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_summary.head(12).itertuples(index=False):
        lines.append(
            f"| {getattr(row, 'strategy_variant', 'n/a')} | {row.model_name} | "
            f"{getattr(row, 'rmse', np.nan):.6f} | {getattr(row, 'sharpe', np.nan):.6f} | "
            f"{getattr(row, 'max_drawdown', np.nan):.6f} | "
            f"{getattr(row, 'delta_sharpe_vs_forecast_only', np.nan):.6f} |"
        )
    return "\n".join(lines)


def write_summary_markdown(output_dir: str | Path, markdown: str, filename: str = "summary.md") -> Path:
    """Persist a markdown summary next to the benchmark artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    path.write_text(markdown, encoding="utf-8")
    return path
