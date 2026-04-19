"""Reporting helpers for quant-core orchestration runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _csv_list(values: list[str] | tuple[str, ...] | None) -> str:
    return ", ".join(str(value) for value in (values or []))


def render_quant_core_summary_markdown(
    manifest: dict[str, Any],
    scenario_frame: pd.DataFrame,
    governance_frame: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> str:
    """Render a compact markdown summary for quant-core runs."""

    run_mode = str(manifest.get("run_mode"))
    git = manifest.get("git", {})
    matrix = manifest.get("matrix", {})
    lines = [
        "# Quant Core Summary",
        "",
        f"- Branch: `{git.get('branch')}`",
        f"- Commit: `{git.get('commit_hash')}`",
        f"- Run mode: `{run_mode}`",
        f"- Preset: `{matrix.get('preset')}`",
        f"- Requested models: `{_csv_list(manifest.get('requested_models', []))}`",
        f"- Evaluated models: `{_csv_list(manifest.get('evaluated_models', []))}`",
        f"- Requested roles: `{_csv_list(manifest.get('requested_model_roles', []))}`",
        f"- Scenario count: `{len(scenario_frame)}`",
        f"- Forecast rows: `{manifest.get('run_counts', {}).get('forecast_rows', 0)}`",
        f"- Strategy rows: `{manifest.get('run_counts', {}).get('strategy_rows', 0)}`",
    ]

    if not governance_frame.empty:
        lines.extend(
            [
                "",
                "## Model Roles",
                "",
                "| model_name | role | status | family | decision_core |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in governance_frame.itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {row.role} | {row.status} | {row.family} | {int(bool(row.enabled_for_decision_core))} |"
            )

    if not forecast_summary.empty:
        lines.extend(
            [
                "",
                "## Forecast Summary",
                "",
                "| model_name | rmse | directional_accuracy | observations |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        preview = (
            forecast_summary.groupby("model_name", sort=True)[["rmse", "directional_accuracy", "observations"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["rmse", "directional_accuracy"], ascending=[True, False])
        )
        for row in preview.head(12).itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {getattr(row, 'rmse', np.nan):.6f} | "
                f"{getattr(row, 'directional_accuracy', np.nan):.6f} | {int(getattr(row, 'observations', 0))} |"
            )

    if not strategy_metrics.empty:
        lines.extend(
            [
                "",
                "## Policy Summary",
                "",
                "| model_name | sharpe | cagr | max_drawdown | trade_count |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        preview = (
            strategy_metrics.groupby("model_name", sort=True)[["sharpe", "cagr", "max_drawdown", "trade_count"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["sharpe", "cagr"], ascending=[False, False])
        )
        for row in preview.head(12).itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {getattr(row, 'sharpe', np.nan):.6f} | "
                f"{getattr(row, 'cagr', np.nan):.6f} | {getattr(row, 'max_drawdown', np.nan):.6f} | "
                f"{int(getattr(row, 'trade_count', 0))} |"
            )

    return "\n".join(lines)
