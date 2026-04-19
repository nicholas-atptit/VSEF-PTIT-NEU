"""Model health and drift-style summaries for quant-core outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.consensus import scenario_group_columns
from src.evaluation.walkforward import summarize_forecasts


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def _safe_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.std(ddof=0)) if not clean.empty else float("nan")


def _safe_share(mask: pd.Series) -> float:
    clean = pd.Series(mask).dropna()
    return float(clean.astype(bool).mean()) if not clean.empty else float("nan")


def _health_status(
    *,
    run_success_rate: float,
    mean_directional_accuracy: float,
    positive_slice_frequency: float,
    positive_policy_frequency: float,
    window_directional_accuracy_dispersion: float,
    directional_accuracy_drift: float,
) -> str:
    if pd.notna(run_success_rate) and run_success_rate < 0.5:
        return "failing"
    if pd.notna(mean_directional_accuracy) and mean_directional_accuracy < 0.48:
        return "weak"
    if pd.notna(positive_slice_frequency) and positive_slice_frequency < 0.35:
        return "weak"
    if (
        (pd.notna(window_directional_accuracy_dispersion) and window_directional_accuracy_dispersion > 0.08)
        or (pd.notna(directional_accuracy_drift) and abs(directional_accuracy_drift) > 0.05)
        or (pd.notna(positive_policy_frequency) and positive_policy_frequency < 0.40)
    ):
        return "brittle"
    return "healthy"


def build_model_health_summary(
    model_execution_log: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    strategy_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate execution, forecast, and policy stability into one health table."""

    if model_execution_log.empty and forecasts_df.empty and strategy_metrics_df.empty:
        return pd.DataFrame(columns=["model_name", "health_status"])

    forecast_summary = summarize_forecasts(
        forecasts_df,
        group_columns=[
            *scenario_group_columns(forecasts_df),
            *[column for column in ["model_name"] if column in forecasts_df.columns],
        ],
    ) if not forecasts_df.empty else pd.DataFrame()

    window_summary = summarize_forecasts(
        forecasts_df,
        group_columns=[
            *scenario_group_columns(forecasts_df),
            *[column for column in ["model_name", "window_id"] if column in forecasts_df.columns],
        ],
    ) if not forecasts_df.empty else pd.DataFrame()

    known_models = sorted(
        set(model_execution_log.get("model_name", pd.Series(dtype="object")).astype(str))
        | set(forecast_summary.get("model_name", pd.Series(dtype="object")).astype(str))
        | set(strategy_metrics_df.get("model_name", pd.Series(dtype="object")).astype(str))
    )
    rows: list[dict[str, Any]] = []
    for model_name in known_models:
        execution = model_execution_log[model_execution_log.get("model_name", pd.Series(dtype="object")).astype(str) == model_name].copy()
        forecast = forecast_summary[forecast_summary.get("model_name", pd.Series(dtype="object")).astype(str) == model_name].copy()
        window = window_summary[window_summary.get("model_name", pd.Series(dtype="object")).astype(str) == model_name].copy()
        policy = strategy_metrics_df[strategy_metrics_df.get("model_name", pd.Series(dtype="object")).astype(str) == model_name].copy()

        context_row = None
        for source in (execution, forecast, policy):
            if not source.empty:
                context_row = source.iloc[0].to_dict()
                break
        context_row = context_row or {}

        ordered_window = window.sort_values(["core_run_id", "window_id"]).reset_index(drop=True) if not window.empty else window
        directional_accuracy_drift = (
            float(ordered_window["directional_accuracy"].iloc[-1] - ordered_window["directional_accuracy"].iloc[0])
            if len(ordered_window) >= 2 and "directional_accuracy" in ordered_window.columns
            else float("nan")
        )
        rmse_drift = (
            float(ordered_window["rmse"].iloc[-1] - ordered_window["rmse"].iloc[0])
            if len(ordered_window) >= 2 and "rmse" in ordered_window.columns
            else float("nan")
        )
        run_success_rate = _safe_share(execution.get("run_success", pd.Series(dtype=bool)))
        mean_directional_accuracy = _safe_mean(forecast.get("directional_accuracy", pd.Series(dtype=float)))
        positive_slice_frequency = _safe_share(
            pd.to_numeric(forecast.get("directional_accuracy", pd.Series(dtype=float)), errors="coerce") > 0.50
        )
        positive_policy_frequency = _safe_share(
            pd.to_numeric(policy.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
        )
        window_directional_accuracy_dispersion = _safe_std(window.get("directional_accuracy", pd.Series(dtype=float)))
        health_status = _health_status(
            run_success_rate=run_success_rate,
            mean_directional_accuracy=mean_directional_accuracy,
            positive_slice_frequency=positive_slice_frequency,
            positive_policy_frequency=positive_policy_frequency,
            window_directional_accuracy_dispersion=window_directional_accuracy_dispersion,
            directional_accuracy_drift=directional_accuracy_drift,
        )

        rows.append(
            {
                "model_name": model_name,
                "model_family": context_row.get("model_family"),
                "model_role": context_row.get("model_role"),
                "model_status": context_row.get("model_status"),
                "run_success_count": int(pd.to_numeric(execution.get("run_success", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
                "run_failure_count": int(len(execution) - pd.to_numeric(execution.get("run_success", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
                "run_success_rate": run_success_rate,
                "warning_count_total": int(pd.to_numeric(execution.get("warning_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
                "missing_output_count_total": int(pd.to_numeric(execution.get("missing_output_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
                "failure_reasons": "|".join(sorted(set(execution.get("failure_reason", pd.Series(dtype="object")).astype(str)) - {""})),
                "scenario_count": int(forecast.get("core_run_id", pd.Series(dtype="object")).astype(str).nunique()) if "core_run_id" in forecast.columns else int(len(forecast)),
                "forecast_observations_total": int(pd.to_numeric(forecast.get("observations", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
                "mean_rmse": _safe_mean(forecast.get("rmse", pd.Series(dtype=float))),
                "median_rmse": _safe_median(forecast.get("rmse", pd.Series(dtype=float))),
                "rmse_dispersion": _safe_std(forecast.get("rmse", pd.Series(dtype=float))),
                "mean_directional_accuracy": mean_directional_accuracy,
                "median_directional_accuracy": _safe_median(forecast.get("directional_accuracy", pd.Series(dtype=float))),
                "directional_accuracy_dispersion": _safe_std(forecast.get("directional_accuracy", pd.Series(dtype=float))),
                "positive_slice_frequency": positive_slice_frequency,
                "strong_slice_frequency": _safe_share(
                    pd.to_numeric(forecast.get("directional_accuracy", pd.Series(dtype=float)), errors="coerce") >= 0.55
                ),
                "window_count": int(ordered_window.get("window_id", pd.Series(dtype="object")).astype(str).nunique()) if not ordered_window.empty and "window_id" in ordered_window.columns else int(len(ordered_window)),
                "window_rmse_dispersion": _safe_std(window.get("rmse", pd.Series(dtype=float))),
                "window_directional_accuracy_dispersion": window_directional_accuracy_dispersion,
                "directional_accuracy_drift": directional_accuracy_drift,
                "rmse_drift": rmse_drift,
                "policy_eval_count": int(len(policy)),
                "mean_sharpe": _safe_mean(policy.get("sharpe", pd.Series(dtype=float))),
                "median_sharpe": _safe_median(policy.get("sharpe", pd.Series(dtype=float))),
                "positive_policy_frequency": positive_policy_frequency,
                "mean_cagr": _safe_mean(policy.get("cagr", pd.Series(dtype=float))),
                "mean_max_drawdown": _safe_mean(policy.get("max_drawdown", pd.Series(dtype=float))),
                "health_status": health_status,
                "drift_flag": bool(
                    (pd.notna(directional_accuracy_drift) and abs(directional_accuracy_drift) > 0.05)
                    or (pd.notna(rmse_drift) and abs(rmse_drift) > 0.01)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "model_name"]).reset_index(drop=True)
