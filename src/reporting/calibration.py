"""Reporting helpers for Phase 2.6 policy calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


POLICY_METRICS = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "turnover", "win_rate", "total_return", "trade_count"]
POLICY_ID_COLUMNS = ["policy_variant", "policy_label", "policy_family", "strategy_variant", "sizing_profile", "sizing_label"]


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.median())


def _safe_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0))


def _safe_share(mask: pd.Series) -> float:
    clean = pd.Series(mask).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.astype(bool).mean())


def _present(columns: list[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _policy_group_columns(frame: pd.DataFrame) -> list[str]:
    return _present(POLICY_ID_COLUMNS, frame)


def _run_match_columns(frame: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    base = ["core_run_id", "group_name", "horizon", "threshold", "cost_mode"]
    if extra:
        base.extend(extra)
    return _present(base, frame)


def _merge_policy_baseline(
    frame: pd.DataFrame,
    *,
    baseline_policy: str,
    key_columns: list[str],
    suffix: str,
    baseline_sizing_profile: str | None = None,
) -> pd.DataFrame:
    if frame.empty or "policy_variant" not in frame.columns:
        return frame.copy()

    baseline = frame[frame["policy_variant"] == baseline_policy].copy()
    if baseline_sizing_profile is not None and "sizing_profile" in baseline.columns:
        baseline = baseline[baseline["sizing_profile"] == baseline_sizing_profile].copy()
    if baseline.empty:
        return frame.copy()

    baseline_keep = key_columns + [column for column in POLICY_METRICS if column in baseline.columns]
    baseline = baseline[baseline_keep].rename(columns={column: f"{suffix}_{column}" for column in POLICY_METRICS if column in baseline_keep})
    merged = frame.merge(baseline, on=key_columns, how="left")
    for column in POLICY_METRICS:
        baseline_column = f"{suffix}_{column}"
        if column in merged.columns and baseline_column in merged.columns:
            merged[f"delta_{column}_vs_{suffix}"] = merged[column] - merged[baseline_column]
    return merged


def build_policy_run_summary(strategy_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one policy run across models."""

    if strategy_metrics.empty:
        return pd.DataFrame(columns=POLICY_ID_COLUMNS)
    group_columns = _policy_group_columns(strategy_metrics)
    numeric_columns = [column for column in POLICY_METRICS if column in strategy_metrics.columns]
    summary = (
        strategy_metrics.groupby(group_columns, sort=True)[numeric_columns]
        .mean()
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    counts = (
        strategy_metrics.groupby(group_columns, sort=True)
        .size()
        .reset_index(name="model_count")
    )
    return summary.merge(counts, on=group_columns, how="left")


def build_policy_ablation_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize policy quality and robustness across the calibration sweep."""

    if aggregate_results.empty:
        return pd.DataFrame(columns=POLICY_ID_COLUMNS)
    match_columns = _run_match_columns(aggregate_results)
    enriched = _merge_policy_baseline(
        aggregate_results,
        baseline_policy="fixed_threshold_fixed_fraction",
        key_columns=match_columns,
        suffix="simple_baseline",
        baseline_sizing_profile="fixed_fraction_full",
    )
    enriched = _merge_policy_baseline(
        enriched,
        baseline_policy="forecast_only_current",
        key_columns=match_columns,
        suffix="forecast_only_current",
        baseline_sizing_profile="adaptive_current",
    )

    rows: list[dict[str, Any]] = []
    group_columns = _policy_group_columns(enriched)
    for keys, group in enriched.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["run_count"] = int(len(group))
        for metric in POLICY_METRICS:
            if metric in group.columns:
                row[f"mean_{metric}"] = _safe_mean(group[metric])
                row[f"median_{metric}"] = _safe_median(group[metric])
                row[f"{metric}_dispersion"] = _safe_std(group[metric])
        for baseline_suffix in ("simple_baseline", "forecast_only_current"):
            delta_sharpe = group.get(f"delta_sharpe_vs_{baseline_suffix}", pd.Series(dtype=float))
            delta_drawdown = group.get(f"delta_max_drawdown_vs_{baseline_suffix}", pd.Series(dtype=float))
            delta_turnover = group.get(f"delta_turnover_vs_{baseline_suffix}", pd.Series(dtype=float))
            delta_cagr = group.get(f"delta_cagr_vs_{baseline_suffix}", pd.Series(dtype=float))
            row[f"share_sharpe_improved_vs_{baseline_suffix}"] = _safe_share(pd.to_numeric(delta_sharpe, errors="coerce") > 0.0)
            row[f"share_drawdown_improved_vs_{baseline_suffix}"] = _safe_share(pd.to_numeric(delta_drawdown, errors="coerce") > 0.0)
            row[f"share_turnover_reduced_vs_{baseline_suffix}"] = _safe_share(pd.to_numeric(delta_turnover, errors="coerce") < 0.0)
            row[f"share_cagr_hurt_vs_{baseline_suffix}"] = _safe_share(pd.to_numeric(delta_cagr, errors="coerce") < 0.0)
            row[f"median_delta_sharpe_vs_{baseline_suffix}"] = _safe_median(delta_sharpe)
            row[f"median_delta_max_drawdown_vs_{baseline_suffix}"] = _safe_median(delta_drawdown)
            row[f"median_delta_turnover_vs_{baseline_suffix}"] = _safe_median(delta_turnover)
            row[f"median_delta_cagr_vs_{baseline_suffix}"] = _safe_median(delta_cagr)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_threshold_calibration_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold behavior by horizon and policy."""

    if aggregate_results.empty:
        return pd.DataFrame(columns=["horizon", "threshold"])
    group_columns = _present(["horizon", "threshold", *POLICY_ID_COLUMNS], aggregate_results)
    metrics = [column for column in POLICY_METRICS if column in aggregate_results.columns]
    summary = (
        aggregate_results.groupby(group_columns, sort=True)[metrics]
        .agg(["mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    counts = aggregate_results.groupby(group_columns, sort=True).size().reset_index(name="run_count")
    return counts.merge(summary, on=group_columns, how="left").sort_values(group_columns).reset_index(drop=True)


def build_sizing_calibration_summary(
    aggregate_results: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sizing-variant tradeoffs and compare them to adaptive_current."""

    if aggregate_results.empty:
        return pd.DataFrame(columns=["policy_variant", "sizing_profile"])

    result_group_columns = _present(["policy_variant", "policy_label", "strategy_variant", "sizing_profile", "sizing_label"], aggregate_results)
    metrics = [column for column in POLICY_METRICS if column in aggregate_results.columns]
    result_summary = (
        aggregate_results.groupby(result_group_columns, sort=True)[metrics]
        .agg(["mean", "median"])
        .reset_index()
    )
    result_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in result_summary.columns
    ]
    result_counts = aggregate_results.groupby(result_group_columns, sort=True).size().reset_index(name="run_count")
    result_summary = result_counts.merge(result_summary, on=result_group_columns, how="left")

    if not positions_df.empty:
        active = positions_df[pd.to_numeric(positions_df.get("signal"), errors="coerce").fillna(0.0) != 0.0].copy()
        if "size_multiplier" not in active.columns and "position_size" in active.columns:
            scale = pd.to_numeric(active.get("configured_max_position_size"), errors="coerce").fillna(1.0).replace(0.0, 1.0)
            active["size_multiplier"] = pd.to_numeric(active["position_size"], errors="coerce").fillna(0.0) / scale
        sizing_group_columns = _present(["policy_variant", "sizing_profile"], active)
        if sizing_group_columns:
            sizing_summary = (
                active.groupby(sizing_group_columns, sort=True)
                .agg(
                    observations=("signal", "size"),
                    avg_size_multiplier=("size_multiplier", "mean"),
                    median_size_multiplier=("size_multiplier", "median"),
                    exposure_reduction_share=("size_multiplier", lambda series: (pd.to_numeric(series, errors="coerce") < 0.999).mean()),
                    mean_vol_forecast=("vol_forecast", lambda series: _safe_mean(pd.Series(series))),
                    mean_drawdown_haircut_strength=("drawdown_haircut_strength", lambda series: _safe_mean(pd.Series(series))),
                    mean_volatility_target_scale=("volatility_target_scale", lambda series: _safe_mean(pd.Series(series))),
                )
                .reset_index()
            )
            result_summary = result_summary.merge(sizing_summary, on=sizing_group_columns, how="left")

    key_columns = _run_match_columns(aggregate_results, extra=["policy_variant"])
    adaptive_only = aggregate_results[aggregate_results["sizing_profile"].astype(str) != "fixed_fraction_full"].copy()
    baseline = adaptive_only[adaptive_only["sizing_profile"] == "adaptive_current"].copy()
    baseline_keep = key_columns + [column for column in POLICY_METRICS if column in baseline.columns]
    baseline = baseline[baseline_keep].rename(columns={column: f"adaptive_current_{column}" for column in POLICY_METRICS if column in baseline_keep})
    compared = adaptive_only.merge(baseline, on=key_columns, how="left")
    for column in POLICY_METRICS:
        baseline_column = f"adaptive_current_{column}"
        if column in compared.columns and baseline_column in compared.columns:
            compared[f"delta_{column}_vs_adaptive_current"] = compared[column] - compared[baseline_column]

    delta_rows: list[dict[str, Any]] = []
    delta_group_columns = _present(["policy_variant", "sizing_profile"], compared)
    for keys, group in compared.groupby(delta_group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(delta_group_columns, keys))
        row["share_sharpe_improved_vs_adaptive_current"] = _safe_share(
            pd.to_numeric(group.get("delta_sharpe_vs_adaptive_current", pd.Series(dtype=float)), errors="coerce") > 0.0
        )
        row["share_drawdown_improved_vs_adaptive_current"] = _safe_share(
            pd.to_numeric(group.get("delta_max_drawdown_vs_adaptive_current", pd.Series(dtype=float)), errors="coerce") > 0.0
        )
        row["share_cagr_hurt_vs_adaptive_current"] = _safe_share(
            pd.to_numeric(group.get("delta_cagr_vs_adaptive_current", pd.Series(dtype=float)), errors="coerce") < 0.0
        )
        row["median_delta_sharpe_vs_adaptive_current"] = _safe_median(group.get("delta_sharpe_vs_adaptive_current", pd.Series(dtype=float)))
        row["median_delta_max_drawdown_vs_adaptive_current"] = _safe_median(group.get("delta_max_drawdown_vs_adaptive_current", pd.Series(dtype=float)))
        row["median_delta_cagr_vs_adaptive_current"] = _safe_median(group.get("delta_cagr_vs_adaptive_current", pd.Series(dtype=float)))
        delta_rows.append(row)
    delta_summary = pd.DataFrame(delta_rows)
    return result_summary.merge(delta_summary, on=_present(["policy_variant", "sizing_profile"], result_summary), how="left").sort_values(result_group_columns).reset_index(drop=True)


def _build_regime_comparison_rows(
    frame: pd.DataFrame,
    *,
    comparison_name: str,
    segment_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    if not segment_columns:
        segment_iterable = [("all", frame)]
        segment_name = "overall"
    else:
        segment_iterable = frame.groupby(segment_columns, sort=True)
        segment_name = "+".join(segment_columns)
    for segment_value, group in segment_iterable:
        if not isinstance(segment_value, tuple):
            segment_value = (segment_value,)
        label = "all" if segment_name == "overall" else "|".join(str(value) for value in segment_value)
        row: dict[str, Any] = {
            "comparison_name": comparison_name,
            "segment_type": segment_name,
            "segment_value": label,
            "run_count": int(len(group)),
            "share_sharpe_improved": _safe_share(pd.to_numeric(group["delta_sharpe"], errors="coerce") > 0.0),
            "share_drawdown_improved": _safe_share(pd.to_numeric(group["delta_max_drawdown"], errors="coerce") > 0.0),
            "share_turnover_reduced": _safe_share(pd.to_numeric(group["delta_turnover"], errors="coerce") < 0.0),
            "share_cagr_hurt": _safe_share(pd.to_numeric(group["delta_cagr"], errors="coerce") < 0.0),
            "mean_delta_sharpe": _safe_mean(group["delta_sharpe"]),
            "median_delta_sharpe": _safe_median(group["delta_sharpe"]),
            "mean_delta_max_drawdown": _safe_mean(group["delta_max_drawdown"]),
            "median_delta_max_drawdown": _safe_median(group["delta_max_drawdown"]),
            "mean_delta_turnover": _safe_mean(group["delta_turnover"]),
            "median_delta_turnover": _safe_median(group["delta_turnover"]),
            "mean_delta_cagr": _safe_mean(group["delta_cagr"]),
            "median_delta_cagr": _safe_median(group["delta_cagr"]),
        }
        rows.append(row)
    return rows


def build_regime_value_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """Compare regime-conditioned execution against non-regime baselines."""

    if aggregate_results.empty:
        return pd.DataFrame(columns=["comparison_name", "segment_type", "segment_value"])

    rows: list[dict[str, Any]] = []

    paired_configs = [
        {
            "comparison_name": "regime_plus_risk_vs_risk_only",
            "left_policy": "regime_threshold_adaptive_drawdown",
            "right_policy": "risk_only_no_regime",
            "key_columns": _run_match_columns(aggregate_results, extra=["sizing_profile"]),
        },
        {
            "comparison_name": "regime_only_vs_simple_fixed",
            "left_policy": "regime_threshold_fixed_fraction",
            "right_policy": "fixed_threshold_fixed_fraction",
            "key_columns": _run_match_columns(aggregate_results),
        },
    ]

    for config in paired_configs:
        left = aggregate_results[aggregate_results["policy_variant"] == config["left_policy"]].copy()
        right = aggregate_results[aggregate_results["policy_variant"] == config["right_policy"]].copy()
        if left.empty or right.empty:
            continue
        right = right[config["key_columns"] + [column for column in ("sharpe", "max_drawdown", "turnover", "cagr") if column in right.columns]].rename(
            columns={
                "sharpe": "baseline_sharpe",
                "max_drawdown": "baseline_max_drawdown",
                "turnover": "baseline_turnover",
                "cagr": "baseline_cagr",
            }
        )
        merged = left.merge(right, on=config["key_columns"], how="left")
        merged["delta_sharpe"] = merged["sharpe"] - merged["baseline_sharpe"]
        merged["delta_max_drawdown"] = merged["max_drawdown"] - merged["baseline_max_drawdown"]
        merged["delta_turnover"] = merged["turnover"] - merged["baseline_turnover"]
        merged["delta_cagr"] = merged["cagr"] - merged["baseline_cagr"]

        rows.extend(_build_regime_comparison_rows(merged, comparison_name=config["comparison_name"], segment_columns=[]))
        for segment in ("horizon", "group_name", "sizing_profile"):
            if segment in merged.columns:
                rows.extend(_build_regime_comparison_rows(merged, comparison_name=config["comparison_name"], segment_columns=[segment]))
    return pd.DataFrame(rows).sort_values(["comparison_name", "segment_type", "segment_value"]).reset_index(drop=True)


def build_policy_cost_sensitivity_summary(aggregate_results: pd.DataFrame) -> pd.DataFrame:
    """Compare policy robustness under higher explicit costs."""

    if aggregate_results.empty or "cost_mode" not in aggregate_results.columns:
        return pd.DataFrame(columns=["cost_mode", "policy_variant", "sizing_profile"])

    key_columns = _run_match_columns(aggregate_results, extra=["policy_variant", "sizing_profile"])
    baseline = aggregate_results[aggregate_results["cost_mode"] == "baseline"][key_columns + [column for column in POLICY_METRICS if column in aggregate_results.columns]].rename(
        columns={column: f"baseline_{column}" for column in POLICY_METRICS if column in aggregate_results.columns}
    )
    merged = aggregate_results.merge(baseline, on=key_columns, how="left")
    for column in POLICY_METRICS:
        baseline_column = f"baseline_{column}"
        if column in merged.columns and baseline_column in merged.columns:
            merged[f"delta_{column}_vs_baseline_cost"] = merged[column] - merged[baseline_column]

    group_columns = _present(["cost_mode", *POLICY_ID_COLUMNS], merged)
    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["run_count"] = int(len(group))
        for metric in POLICY_METRICS:
            if metric in group.columns:
                row[f"mean_{metric}"] = _safe_mean(group[metric])
                row[f"median_{metric}"] = _safe_median(group[metric])
            delta_column = f"delta_{metric}_vs_baseline_cost"
            if delta_column in group.columns:
                row[f"mean_delta_{metric}_vs_baseline_cost"] = _safe_mean(group[delta_column])
                row[f"median_delta_{metric}_vs_baseline_cost"] = _safe_median(group[delta_column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_forecast_vs_policy_summary(
    aggregate_forecast_summary: pd.DataFrame,
    aggregate_model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forecast quality with monetization quality by model and policy."""

    if aggregate_forecast_summary.empty or aggregate_model_metrics.empty:
        return pd.DataFrame(columns=["group_name", "horizon", "policy_variant", "model_name"])

    forecast_group_columns = _present(["group_name", "horizon", "model_name"], aggregate_forecast_summary)
    forecast_metrics = _present(["mae", "rmse", "mape", "smape", "directional_accuracy", "hit_rate", "observations"], aggregate_forecast_summary)
    forecast_summary = (
        aggregate_forecast_summary.groupby(forecast_group_columns, sort=True)[forecast_metrics]
        .mean()
        .reset_index()
    )

    strategy_group_columns = _present(["group_name", "horizon", "policy_variant", "sizing_profile", "strategy_variant", "model_name"], aggregate_model_metrics)
    strategy_metrics = _present(["cagr", "sharpe", "max_drawdown", "turnover", "trade_count", "total_return"], aggregate_model_metrics)
    strategy_summary = (
        aggregate_model_metrics.groupby(strategy_group_columns, sort=True)[strategy_metrics]
        .mean()
        .reset_index()
    )

    merged = strategy_summary.merge(
        forecast_summary,
        on=_present(["group_name", "horizon", "model_name"], strategy_summary),
        how="left",
    )
    if "rmse" in merged.columns:
        merged["forecast_rank"] = merged.groupby(["group_name", "horizon"])["rmse"].rank(method="dense", ascending=True)
    else:
        merged["forecast_rank"] = np.nan
    if "sharpe" in merged.columns:
        merged["strategy_rank"] = merged.groupby(["group_name", "horizon", "policy_variant", "sizing_profile"])["sharpe"].rank(
            method="dense",
            ascending=False,
        )
    else:
        merged["strategy_rank"] = np.nan
    merged["monetization_gap"] = merged["strategy_rank"] - merged["forecast_rank"]

    counts = merged.groupby(["group_name", "horizon", "policy_variant", "sizing_profile"], sort=True)["model_name"].transform("size")
    midpoint = counts / 2.0
    merged["edge_but_not_monetized"] = (merged["forecast_rank"] <= midpoint) & (merged["strategy_rank"] > midpoint)
    return merged.sort_values(["group_name", "horizon", "policy_variant", "sizing_profile", "strategy_rank", "model_name"]).reset_index(drop=True)


def build_phase26_assessment(
    policy_ablation_summary: pd.DataFrame,
    regime_value_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Recommend the next calibration action and default candidate."""

    if policy_ablation_summary.empty:
        return {
            "recommendation": "recalibrate adaptive sizing and continue hardening",
            "default_policy_candidate": None,
            "phase3_blocked": True,
        }

    sortable = policy_ablation_summary.copy()
    for column, ascending in (
        ("median_sharpe", False),
        ("median_cagr", False),
        ("median_max_drawdown", False),
        ("median_turnover", True),
    ):
        sortable[f"rank_{column}"] = sortable[column].rank(method="dense", ascending=ascending)
    sortable["balanced_policy_rank"] = sortable[
        ["rank_median_sharpe", "rank_median_cagr", "rank_median_max_drawdown", "rank_median_turnover"]
    ].mean(axis=1)
    sortable = sortable.sort_values(
        ["balanced_policy_rank", "median_sharpe", "median_cagr", "median_max_drawdown"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    best_row = sortable.iloc[0]
    default_policy_candidate = f"{best_row['policy_variant']}::{best_row['sizing_profile']}"

    current_mask = (
        (policy_ablation_summary["policy_variant"] == "regime_threshold_adaptive_drawdown")
        & (policy_ablation_summary["sizing_profile"] == "adaptive_current")
    )
    current_row = policy_ablation_summary[current_mask].iloc[0] if current_mask.any() else None
    simple_mask = (
        (policy_ablation_summary["policy_variant"] == "fixed_threshold_fixed_fraction")
        & (policy_ablation_summary["sizing_profile"] == "fixed_fraction_full")
    )
    simple_row = policy_ablation_summary[simple_mask].iloc[0] if simple_mask.any() else None

    regime_overall = regime_value_summary[
        (regime_value_summary["comparison_name"] == "regime_plus_risk_vs_risk_only")
        & (regime_value_summary["segment_type"] == "overall")
    ]
    regime_share_sharpe = _safe_mean(regime_overall.get("share_sharpe_improved", pd.Series(dtype=float)))
    regime_share_drawdown = _safe_mean(regime_overall.get("share_drawdown_improved", pd.Series(dtype=float)))

    elevated = cost_sensitivity_summary[cost_sensitivity_summary.get("cost_mode") == "elevated"].copy()
    elevated_best = (
        elevated.sort_values(["median_sharpe", "median_max_drawdown"], ascending=[False, False]).head(1)
        if not elevated.empty and {"median_sharpe", "median_max_drawdown"} <= set(elevated.columns)
        else pd.DataFrame()
    )
    elevated_best_candidate = (
        None
        if elevated_best.empty
        else f"{elevated_best.iloc[0]['policy_variant']}::{elevated_best.iloc[0]['sizing_profile']}"
    )

    edge_not_monetized_share = _safe_share(
        forecast_vs_policy_summary.get("edge_but_not_monetized", pd.Series(dtype=bool))
    )

    recommendation = "recalibrate adaptive sizing and continue hardening"
    if (
        current_row is not None
        and simple_row is not None
        and float(simple_row.get("median_sharpe", float("-inf"))) >= float(current_row.get("median_sharpe", float("-inf")))
        and float(simple_row.get("median_cagr", float("-inf"))) >= float(current_row.get("median_cagr", float("-inf")))
        and pd.notna(regime_share_sharpe)
        and regime_share_sharpe < 0.50
    ):
        recommendation = "switch to a simpler default policy"
    elif pd.notna(regime_share_sharpe) and regime_share_sharpe < 0.45 and pd.notna(regime_share_drawdown) and regime_share_drawdown < 0.55:
        recommendation = "keep regime only as context, not execution conditioning"
    elif (
        float(best_row.get("median_sharpe", float("-inf"))) < 0.0
        and float(best_row.get("median_cagr", float("-inf"))) < 0.0
        and pd.notna(edge_not_monetized_share)
        and edge_not_monetized_share < 0.35
    ):
        recommendation = "stop policy work and revisit forecast layer"
    elif current_row is not None and default_policy_candidate != "regime_threshold_adaptive_drawdown::adaptive_current":
        recommendation = "recalibrate adaptive sizing and continue hardening"
    else:
        recommendation = "keep current policy"

    return {
        "recommendation": recommendation,
        "default_policy_candidate": default_policy_candidate,
        "current_policy_candidate": None if current_row is None else "regime_threshold_adaptive_drawdown::adaptive_current",
        "best_elevated_cost_candidate": elevated_best_candidate,
        "regime_sharpe_improvement_share_vs_risk_only": regime_share_sharpe,
        "regime_drawdown_improvement_share_vs_risk_only": regime_share_drawdown,
        "edge_but_not_monetized_share": edge_not_monetized_share,
        "phase3_blocked": True,
    }


def render_phase26_summary_markdown(
    manifest: dict[str, Any],
    policy_ablation_summary: pd.DataFrame,
    threshold_calibration_summary: pd.DataFrame,
    sizing_calibration_summary: pd.DataFrame,
    regime_value_summary: pd.DataFrame,
    assessment: dict[str, Any],
) -> str:
    """Render a compact markdown summary for Phase 2.6 calibration."""

    runtime = manifest.get("runtime", {})
    matrix = manifest.get("matrix", {})
    lines = [
        "# Phase 2.6 Calibration Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Preset: `{matrix.get('preset')}`",
        f"- Horizons: `{', '.join(str(value) for value in matrix.get('horizons', []))}`",
        f"- Threshold grid: `{', '.join(str(value) for value in matrix.get('thresholds', []))}`",
        f"- Cost modes: `{', '.join(item['cost_mode'] for item in matrix.get('cost_modes', []))}`",
        f"- Recommendation: `{assessment.get('recommendation')}`",
        f"- Default candidate: `{assessment.get('default_policy_candidate')}`",
    ]

    if not policy_ablation_summary.empty:
        lines.extend(
            [
                "",
                "## Policy Ablations",
                "",
                "| policy_variant | sizing_profile | median_sharpe | median_cagr | median_max_drawdown | median_turnover |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in policy_ablation_summary.sort_values("median_sharpe", ascending=False).head(10).itertuples(index=False):
            lines.append(
                f"| {row.policy_variant} | {row.sizing_profile} | {getattr(row, 'median_sharpe', np.nan):.6f} | "
                f"{getattr(row, 'median_cagr', np.nan):.6f} | {getattr(row, 'median_max_drawdown', np.nan):.6f} | "
                f"{getattr(row, 'median_turnover', np.nan):.6f} |"
            )

    if not threshold_calibration_summary.empty:
        threshold_overall = (
            threshold_calibration_summary.groupby("threshold", sort=True)["sharpe_median"]
            .mean()
            .reset_index()
            .sort_values("sharpe_median", ascending=False)
        )
        lines.extend(["", "## Thresholds", ""])
        for row in threshold_overall.itertuples(index=False):
            lines.append(f"- threshold={row.threshold:.3f}, average median sharpe={row.sharpe_median:.6f}")

    if not regime_value_summary.empty:
        overall = regime_value_summary[
            (regime_value_summary["comparison_name"] == "regime_plus_risk_vs_risk_only")
            & (regime_value_summary["segment_type"] == "overall")
        ]
        if not overall.empty:
            row = overall.iloc[0]
            lines.extend(
                [
                    "",
                    "## Regime Value",
                    "",
                    f"- Sharpe improvement share vs risk-only: `{row['share_sharpe_improved']:.2%}`",
                    f"- Drawdown improvement share vs risk-only: `{row['share_drawdown_improved']:.2%}`",
                    f"- CAGR hurt share vs risk-only: `{row['share_cagr_hurt']:.2%}`",
                ]
            )

    if not sizing_calibration_summary.empty:
        top = sizing_calibration_summary.sort_values("sharpe_median", ascending=False).head(5)
        lines.extend(["", "## Sizing Variants", ""])
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.policy_variant}/{row.sizing_profile}: median sharpe={getattr(row, 'sharpe_median', np.nan):.6f}, avg size multiplier={getattr(row, 'avg_size_multiplier', np.nan):.6f}"
            )
    return "\n".join(lines)


def build_phase26_report(
    manifest: dict[str, Any],
    policy_ablation_summary: pd.DataFrame,
    threshold_calibration_summary: pd.DataFrame,
    sizing_calibration_summary: pd.DataFrame,
    regime_value_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    assessment: dict[str, Any],
) -> str:
    """Render the direct Phase 2.6 technical report."""

    runtime = manifest.get("runtime", {})
    dependencies = manifest.get("dependency_versions", {})
    best_policy = policy_ablation_summary.sort_values("median_sharpe", ascending=False).head(1) if not policy_ablation_summary.empty else pd.DataFrame()
    best_threshold = (
        threshold_calibration_summary.groupby("threshold", sort=True)["sharpe_median"].mean().reset_index().sort_values("sharpe_median", ascending=False).head(1)
        if not threshold_calibration_summary.empty and "sharpe_median" in threshold_calibration_summary.columns
        else pd.DataFrame()
    )
    regime_overall = regime_value_summary[
        (regime_value_summary["comparison_name"] == "regime_plus_risk_vs_risk_only")
        & (regime_value_summary["segment_type"] == "overall")
    ]
    edge_share = assessment.get("edge_but_not_monetized_share", float("nan"))
    lines = [
        "# Phase 2.6 Calibration Report",
        "",
        "## Scope",
        "",
        "- Reused the Phase 2 forecast, regime, risk, and backtest backbone.",
        "- Expanded the decision layer into explicit policy ablations instead of one bundled conditioning path.",
        "- Tested threshold, sizing, regime-execution, and cost interactions on a bounded matrix.",
        "",
        "## Runtime",
        "",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- statsmodels: `{dependencies.get('statsmodels')}`",
        f"- arch: `{dependencies.get('arch')}`",
        "",
        "## Answers",
        "",
    ]

    if best_threshold.empty:
        lines.append("1. Threshold settings: no usable threshold ranking was produced.")
    else:
        row = best_threshold.iloc[0]
        lines.append(f"1. Least-bad threshold setting across the bounded sweep was `{row['threshold']:.3f}` by average median Sharpe.")

    fixed_row = policy_ablation_summary[
        (policy_ablation_summary["policy_variant"] == "fixed_threshold_fixed_fraction")
        & (policy_ablation_summary["sizing_profile"] == "fixed_fraction_full")
    ]
    adaptive_row = policy_ablation_summary[
        (policy_ablation_summary["policy_variant"] == "risk_only_no_regime")
        & (policy_ablation_summary["sizing_profile"] == "adaptive_current")
    ]
    if not fixed_row.empty and not adaptive_row.empty:
        lines.append(
            "2. Fixed fraction vs current adaptive: "
            f"`median_sharpe {fixed_row.iloc[0]['median_sharpe']:.6f}` versus `{adaptive_row.iloc[0]['median_sharpe']:.6f}`; "
            f"`median_cagr {fixed_row.iloc[0]['median_cagr']:.6f}` versus `{adaptive_row.iloc[0]['median_cagr']:.6f}`."
        )
    else:
        lines.append("2. Fixed fraction versus current adaptive could not be compared cleanly from the bounded run.")

    capped_row = policy_ablation_summary[
        (policy_ablation_summary["policy_variant"] == "risk_only_no_regime")
        & (policy_ablation_summary["sizing_profile"] == "adaptive_capped_floor")
    ]
    lighter_vol_row = policy_ablation_summary[
        (policy_ablation_summary["policy_variant"] == "risk_only_no_regime")
        & (policy_ablation_summary["sizing_profile"] == "adaptive_lighter_vol")
    ]
    lighter_dd_row = policy_ablation_summary[
        (policy_ablation_summary["policy_variant"] == "risk_only_no_regime")
        & (policy_ablation_summary["sizing_profile"] == "adaptive_lighter_drawdown")
    ]
    if not capped_row.empty:
        lines.append(
            "3. Capped/floored adaptive sizing: "
            f"`median_sharpe {capped_row.iloc[0]['median_sharpe']:.6f}`, "
            f"`median_max_drawdown {capped_row.iloc[0]['median_max_drawdown']:.6f}`."
        )
    if not lighter_vol_row.empty or not lighter_dd_row.empty:
        lines.append(
            "4. Drawdown and volatility penalty recalibration: "
            f"`lighter_vol median_sharpe {lighter_vol_row.iloc[0]['median_sharpe']:.6f}`; "
            f"`lighter_drawdown median_sharpe {lighter_dd_row.iloc[0]['median_sharpe']:.6f}`."
            if (not lighter_vol_row.empty and not lighter_dd_row.empty)
            else "4. Only part of the adaptive recalibration grid produced comparable rows."
        )
    else:
        lines.append("4. Drawdown control aggressiveness could not be ranked from the bounded run.")

    if not regime_overall.empty:
        row = regime_overall.iloc[0]
        lines.append(
            "5. Regime execution value: "
            f"Sharpe improved in `{row['share_sharpe_improved']:.2%}` of matched risk-only runs, "
            f"drawdown improved in `{row['share_drawdown_improved']:.2%}`, "
            f"CAGR was hurt in `{row['share_cagr_hurt']:.2%}`."
        )
    else:
        lines.append("5. Regime execution value could not be paired cleanly against risk-only runs.")

    lines.append(
        "6. Forecast edge versus policy failure: "
        f"`edge_but_not_monetized_share={edge_share:.2%}` across the merged forecast-vs-policy table."
    )
    candidate_label = assessment.get("default_policy_candidate")
    candidate_row = pd.DataFrame()
    if candidate_label and not policy_ablation_summary.empty and "::" in str(candidate_label):
        policy_variant, sizing_profile = str(candidate_label).split("::", 1)
        candidate_row = policy_ablation_summary[
            (policy_ablation_summary["policy_variant"] == policy_variant)
            & (policy_ablation_summary["sizing_profile"] == sizing_profile)
        ].head(1)
    if candidate_row.empty:
        candidate_row = best_policy

    if not candidate_row.empty:
        row = candidate_row.iloc[0]
        lines.append(
            "7. New default candidate: "
            f"`{row['policy_variant']}::{row['sizing_profile']}` with median Sharpe `{row['median_sharpe']:.6f}`, "
            f"median CAGR `{row['median_cagr']:.6f}`, median max drawdown `{row['median_max_drawdown']:.6f}`."
        )
    else:
        lines.append("7. No policy row was strong enough to nominate a default candidate.")

    elevated = cost_sensitivity_summary[cost_sensitivity_summary.get("cost_mode") == "elevated"]
    if not elevated.empty:
        resilient = elevated.sort_values(["median_sharpe", "median_max_drawdown"], ascending=[False, False]).head(1).iloc[0]
        lines.extend(
            [
                "",
                "## Cost Sensitivity",
                "",
                f"- Under elevated costs, the least-bad candidate was `{resilient['policy_variant']}::{resilient['sizing_profile']}` with median Sharpe `{resilient['median_sharpe']:.6f}` and median max drawdown `{resilient['median_max_drawdown']:.6f}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Direct recommendation: `{assessment.get('recommendation')}`",
            f"- Default policy candidate: `{assessment.get('default_policy_candidate')}`",
            f"- Phase 3 blocked: `{assessment.get('phase3_blocked')}`",
        ]
    )
    return "\n".join(lines)
