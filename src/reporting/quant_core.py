"""Reporting helpers for quant-core orchestration runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.model_governance import get_run_mode_spec
from src.reporting.manifests import build_batch_manifest


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


def build_quant_core_manifest(
    *,
    git_metadata: dict[str, Any],
    runtime: dict[str, Any],
    dependency_versions: dict[str, str | None],
    command: str,
    requested_models: list[str],
    evaluated_models: list[str],
    skipped_models: list[dict[str, Any]],
    seed: int,
    matrix_config: dict[str, Any],
    run_counts: dict[str, int],
    artifact_paths: dict[str, str],
    started_at: str,
    completed_at: str,
    run_mode: str,
    requested_model_roles: list[str] | None,
    governance_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build the quant-core manifest with explicit governance metadata."""

    manifest = build_batch_manifest(
        git_metadata=git_metadata,
        runtime=runtime,
        dependency_versions=dependency_versions,
        command=command,
        requested_models=requested_models,
        evaluated_models=evaluated_models,
        skipped_models=skipped_models,
        target_type="quant_core_multi_target",
        seed=seed,
        matrix_config=matrix_config,
        run_counts=run_counts,
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
        manifest_type="quant_core_run_manifest_v1",
    )
    manifest["run_mode"] = run_mode
    manifest["run_mode_spec"] = get_run_mode_spec(run_mode).to_dict()
    manifest["requested_model_roles"] = list(requested_model_roles or [])
    manifest["governance_output"] = {
        "model_count": int(len(governance_frame)),
        "artifact_path": artifact_paths.get("model_governance"),
    }
    manifest["artifact_paths"] = dict(artifact_paths)
    return manifest
