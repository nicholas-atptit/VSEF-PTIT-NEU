"""Reporting helpers for Phase 2.5 hardening sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.contracts import validate_regime_frame


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


def build_grouped_metric_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate key strategy metrics by the requested grouping."""

    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "run_count"])
    metrics = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "turnover", "win_rate", "total_return"]
    present = [column for column in metrics if column in frame.columns]
    if not present:
        return pd.DataFrame(columns=[*group_columns, "run_count"])
    summary = (
        frame.groupby(group_columns, sort=True)[present]
        .mean()
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    counts = (
        frame.groupby(group_columns, sort=True)
        .size()
        .reset_index(name="run_count")
    )
    return summary.merge(counts, on=group_columns, how="left")


def build_phase25_stability_summary(
    aggregate_results: pd.DataFrame,
    *,
    baseline_variant: str = "forecast_only",
) -> pd.DataFrame:
    """Summarize stability by conditioning mode across the full sweep."""

    if aggregate_results.empty or "strategy_variant" not in aggregate_results.columns:
        return pd.DataFrame(columns=["strategy_variant", "run_count"])

    rows: list[dict[str, Any]] = []
    for strategy_variant, group in aggregate_results.groupby("strategy_variant", sort=True):
        row: dict[str, Any] = {
            "strategy_variant": strategy_variant,
            "run_count": int(len(group)),
            "mean_sharpe": _safe_mean(group.get("sharpe", pd.Series(dtype=float))),
            "median_sharpe": _safe_median(group.get("sharpe", pd.Series(dtype=float))),
            "sharpe_dispersion": _safe_std(group.get("sharpe", pd.Series(dtype=float))),
            "mean_cagr": _safe_mean(group.get("cagr", pd.Series(dtype=float))),
            "median_cagr": _safe_median(group.get("cagr", pd.Series(dtype=float))),
            "mean_max_drawdown": _safe_mean(group.get("max_drawdown", pd.Series(dtype=float))),
            "median_max_drawdown": _safe_median(group.get("max_drawdown", pd.Series(dtype=float))),
            "mean_turnover": _safe_mean(group.get("turnover", pd.Series(dtype=float))),
            "median_turnover": _safe_median(group.get("turnover", pd.Series(dtype=float))),
            "return_dispersion": _safe_std(group.get("total_return", pd.Series(dtype=float))),
        }
        if strategy_variant != baseline_variant:
            row["share_sharpe_improved_vs_forecast_only"] = _safe_share(
                pd.to_numeric(group.get(f"delta_sharpe_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
            )
            row["share_drawdown_improved_vs_forecast_only"] = _safe_share(
                pd.to_numeric(group.get(f"delta_max_drawdown_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
            )
            row["share_turnover_reduced_vs_forecast_only"] = _safe_share(
                pd.to_numeric(group.get(f"delta_turnover_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") < 0.0
            )
            row["share_cagr_hurt_vs_forecast_only"] = _safe_share(
                pd.to_numeric(group.get(f"delta_cagr_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") < 0.0
            )
            row["median_cagr_delta_vs_forecast_only"] = _safe_median(
                group.get(f"delta_cagr_vs_{baseline_variant}", pd.Series(dtype=float))
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy_variant").reset_index(drop=True)


def build_regime_stability_summary(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify regime switching behavior, fallback use, and probability concentration."""

    if regime_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "window_id",
                "observations",
                "switch_count",
                "switch_rate",
                "average_regime_duration",
                "fallback_observation_share",
                "mean_max_probability",
                "extreme_probability_share",
                "uncertain_probability_share",
            ]
        )

    validated = validate_regime_frame(regime_df)
    group_columns = [
        column
        for column in ["core_run_id", "group_name", "horizon", "ticker", "window_id"]
        if column in validated.columns
    ]
    if not group_columns:
        group_columns = ["ticker", "window_id"]

    probability_columns = ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]
    rows: list[dict[str, Any]] = []
    for keys, group in validated.sort_values("timestamp").groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        labels = group["regime_label"].astype(str)
        switch_mask = labels.ne(labels.shift())
        switch_count = int(max(int(switch_mask.sum()) - 1, 0))
        segments = switch_mask.cumsum()
        segment_lengths = group.groupby(segments).size()
        max_probability = group[probability_columns].max(axis=1)
        fallback_mask = group["source_model"].astype(str).str.contains("fallback", case=False, na=False)
        row.update(
            {
                "observations": int(len(group)),
                "bull_count": int((labels == "bull").sum()),
                "bear_count": int((labels == "bear").sum()),
                "sideway_count": int((labels == "sideway").sum()),
                "switch_count": switch_count,
                "switch_rate": float(switch_count / max(len(group) - 1, 1)),
                "average_regime_duration": float(segment_lengths.mean()) if not segment_lengths.empty else float("nan"),
                "fallback_observation_share": _safe_share(fallback_mask),
                "mean_max_probability": _safe_mean(max_probability),
                "std_max_probability": _safe_std(max_probability),
                "extreme_probability_share": _safe_share(max_probability >= 0.80),
                "uncertain_probability_share": _safe_share(max_probability < 0.55),
                "mean_prob_bull": _safe_mean(group["regime_prob_bull"]),
                "mean_prob_bear": _safe_mean(group["regime_prob_bear"]),
                "mean_prob_sideway": _safe_mean(group["regime_prob_sideway"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_risk_stability_summary(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize volatility-aware sizing behavior across sweep runs."""

    if positions_df.empty:
        return pd.DataFrame(columns=["strategy_variant", "run_count"])

    group_columns = [
        column
        for column in ["run_id", "core_run_id", "group_name", "horizon", "threshold", "cost_mode", "sizing_mode", "strategy_variant"]
        if column in positions_df.columns
    ]
    active = positions_df[pd.to_numeric(positions_df.get("signal"), errors="coerce").fillna(0.0) != 0.0].copy()
    if active.empty:
        return pd.DataFrame(columns=[*group_columns, "observations"])

    if "size_multiplier" not in active.columns:
        max_position = pd.to_numeric(active.get("configured_max_position_size"), errors="coerce").fillna(1.0).replace(0.0, 1.0)
        active["size_multiplier"] = pd.to_numeric(active["position_size"], errors="coerce").fillna(0.0) / max_position

    rows: list[dict[str, Any]] = []
    for keys, group in active.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        vol_series = pd.to_numeric(group.get("vol_forecast", pd.Series(dtype=float)), errors="coerce")
        position_series = pd.to_numeric(group.get("position_size", pd.Series(dtype=float)), errors="coerce")
        size_multiplier = pd.to_numeric(group.get("size_multiplier", pd.Series(dtype=float)), errors="coerce")
        drawdown_state = group.get("drawdown_state", pd.Series(dtype="object")).astype(str).str.lower()
        row.update(
            {
                "observations": int(len(group)),
                "avg_position_size": _safe_mean(position_series),
                "median_position_size": _safe_median(position_series),
                "avg_size_multiplier": _safe_mean(size_multiplier),
                "median_size_multiplier": _safe_median(size_multiplier),
                "exposure_reduction_share": _safe_share(size_multiplier < 0.999),
                "mean_vol_forecast": _safe_mean(vol_series),
                "median_vol_forecast": _safe_median(vol_series),
                "vol_forecast_dispersion": _safe_std(vol_series),
                "elevated_drawdown_share": _safe_share(drawdown_state.isin(["elevated", "severe"])),
                "severe_drawdown_share": _safe_share(drawdown_state == "severe"),
            }
        )
        valid_corr = pd.DataFrame({"vol": vol_series, "size": position_series}).dropna()
        if len(valid_corr) >= 2 and valid_corr["vol"].nunique() > 1 and valid_corr["size"].nunique() > 1:
            row["vol_size_correlation"] = float(valid_corr["vol"].corr(valid_corr["size"]))
        else:
            row["vol_size_correlation"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_cost_sensitivity_summary(
    aggregate_results: pd.DataFrame,
    *,
    baseline_cost_mode: str = "baseline",
) -> pd.DataFrame:
    """Compare strategy variants under baseline and stressed cost assumptions."""

    if aggregate_results.empty or "cost_mode" not in aggregate_results.columns:
        return pd.DataFrame(columns=["cost_mode", "strategy_variant", "run_count"])

    metrics = ["cagr", "sharpe", "max_drawdown", "turnover", "total_return"]
    key_columns = [
        column
        for column in ["core_run_id", "threshold", "sizing_mode", "strategy_variant"]
        if column in aggregate_results.columns
    ]
    baseline = aggregate_results[aggregate_results["cost_mode"] == baseline_cost_mode][key_columns + metrics].rename(
        columns={metric: f"baseline_{metric}" for metric in metrics}
    )
    merged = aggregate_results.merge(baseline, on=key_columns, how="left")
    for metric in metrics:
        merged[f"delta_{metric}_vs_{baseline_cost_mode}_cost"] = merged[metric] - merged[f"baseline_{metric}"]

    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(["cost_mode", "strategy_variant"], sort=True):
        cost_mode, strategy_variant = keys
        row: dict[str, Any] = {
            "cost_mode": cost_mode,
            "strategy_variant": strategy_variant,
            "run_count": int(len(group)),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = _safe_mean(group[metric])
            row[f"median_{metric}"] = _safe_median(group[metric])
            row[f"mean_delta_{metric}_vs_{baseline_cost_mode}_cost"] = _safe_mean(
                group[f"delta_{metric}_vs_{baseline_cost_mode}_cost"]
            )
            row[f"median_delta_{metric}_vs_{baseline_cost_mode}_cost"] = _safe_median(
                group[f"delta_{metric}_vs_{baseline_cost_mode}_cost"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cost_mode", "strategy_variant"]).reset_index(drop=True)


def build_phase3_readiness_assessment(
    aggregate_results: pd.DataFrame,
    regime_stability_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    *,
    baseline_variant: str = "forecast_only",
) -> dict[str, Any]:
    """Form a conservative Phase 3 readiness recommendation from the hardening sweep."""

    risk_rows = aggregate_results[aggregate_results["strategy_variant"] == "forecast_plus_risk"].copy()
    regime_rows = aggregate_results[aggregate_results["strategy_variant"] == "forecast_plus_risk_and_regime"].copy()

    risk_drawdown_improvement = _safe_share(
        pd.to_numeric(risk_rows.get(f"delta_max_drawdown_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
    )
    regime_sharpe_improvement = _safe_share(
        pd.to_numeric(regime_rows.get(f"delta_sharpe_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
    )
    regime_cagr_hurt = _safe_share(
        pd.to_numeric(regime_rows.get(f"delta_cagr_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") < 0.0
    )
    turnover_reduction = _safe_share(
        pd.to_numeric(regime_rows.get(f"delta_turnover_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") < 0.0
    )
    fallback_share = _safe_mean(regime_stability_summary.get("fallback_observation_share", pd.Series(dtype=float)))
    elevated_cost = cost_sensitivity_summary[
        (cost_sensitivity_summary.get("cost_mode") == "elevated")
        & (cost_sensitivity_summary.get("strategy_variant") == "forecast_plus_risk_and_regime")
    ]
    elevated_cost_sharpe_delta = _safe_mean(
        elevated_cost.get("mean_delta_sharpe_vs_baseline_cost", pd.Series(dtype=float))
    )

    recommendation = "NO-GO, harden more first"
    if (
        pd.notna(risk_drawdown_improvement)
        and risk_drawdown_improvement >= 0.60
        and pd.notna(regime_sharpe_improvement)
        and regime_sharpe_improvement >= 0.50
        and (pd.isna(fallback_share) or fallback_share <= 0.50)
        and (pd.isna(elevated_cost_sharpe_delta) or elevated_cost_sharpe_delta >= -0.35)
    ):
        recommendation = "GO Phase 3 with caveats"
    if (
        pd.notna(risk_drawdown_improvement)
        and risk_drawdown_improvement >= 0.70
        and pd.notna(regime_sharpe_improvement)
        and regime_sharpe_improvement >= 0.60
        and pd.notna(regime_cagr_hurt)
        and regime_cagr_hurt <= 0.50
        and (pd.isna(fallback_share) or fallback_share <= 0.30)
        and (pd.isna(elevated_cost_sharpe_delta) or elevated_cost_sharpe_delta >= -0.20)
    ):
        recommendation = "GO Phase 3"

    return {
        "recommendation": recommendation,
        "risk_drawdown_improvement_share": risk_drawdown_improvement,
        "regime_sharpe_improvement_share": regime_sharpe_improvement,
        "regime_cagr_hurt_share": regime_cagr_hurt,
        "turnover_reduction_share": turnover_reduction,
        "mean_regime_fallback_share": fallback_share,
        "elevated_cost_sharpe_delta": elevated_cost_sharpe_delta,
    }


def render_phase25_summary_markdown(
    manifest: dict[str, Any],
    aggregate_results: pd.DataFrame,
    stability_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    assessment: dict[str, Any],
) -> str:
    """Render a compact markdown summary for the hardening sweep."""

    matrix = manifest.get("matrix", {})
    runtime = manifest.get("runtime", {})
    lines = [
        "# Phase 2.5 Hardening Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Preset: `{matrix.get('preset')}`",
        f"- Groups: `{', '.join(group['group_name'] for group in matrix.get('ticker_groups', []))}`",
        f"- Horizons: `{', '.join(str(value) for value in matrix.get('horizons', []))}`",
        f"- Thresholds: `{', '.join(str(value) for value in matrix.get('thresholds', []))}`",
        f"- Cost modes: `{', '.join(item['cost_mode'] for item in matrix.get('cost_modes', []))}`",
        f"- Sizing modes: `{', '.join(item['sizing_mode'] for item in matrix.get('sizing_modes', []))}`",
        f"- Runs: `{len(aggregate_results)}` conditioning rows",
        f"- Recommendation: `{assessment.get('recommendation')}`",
    ]

    if not stability_summary.empty:
        lines.extend(
            [
                "",
                "## Stability By Conditioning Mode",
                "",
                "| strategy_variant | mean_sharpe | median_sharpe | mean_max_drawdown | mean_turnover |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stability_summary.itertuples(index=False):
            lines.append(
                f"| {row.strategy_variant} | {getattr(row, 'mean_sharpe', np.nan):.6f} | "
                f"{getattr(row, 'median_sharpe', np.nan):.6f} | {getattr(row, 'mean_max_drawdown', np.nan):.6f} | "
                f"{getattr(row, 'mean_turnover', np.nan):.6f} |"
            )

    if not cost_sensitivity_summary.empty:
        lines.extend(
            [
                "",
                "## Cost Sensitivity",
                "",
                "| cost_mode | strategy_variant | mean_sharpe | mean_delta_sharpe_vs_baseline_cost | mean_turnover |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in cost_sensitivity_summary.itertuples(index=False):
            lines.append(
                f"| {row.cost_mode} | {row.strategy_variant} | {getattr(row, 'mean_sharpe', np.nan):.6f} | "
                f"{getattr(row, 'mean_delta_sharpe_vs_baseline_cost', np.nan):.6f} | {getattr(row, 'mean_turnover', np.nan):.6f} |"
            )
    return "\n".join(lines)


def build_phase25_report(
    manifest: dict[str, Any],
    aggregate_results: pd.DataFrame,
    stability_summary: pd.DataFrame,
    regime_stability_summary: pd.DataFrame,
    risk_stability_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    assessment: dict[str, Any],
    *,
    baseline_variant: str = "forecast_only",
) -> str:
    """Render a direct Phase 2.5 technical report with go/no-go framing."""

    runtime = manifest.get("runtime", {})
    dependencies = manifest.get("dependency_versions", {})
    risk_rows = aggregate_results[aggregate_results["strategy_variant"] == "forecast_plus_risk"]
    regime_rows = aggregate_results[aggregate_results["strategy_variant"] == "forecast_plus_risk_and_regime"]
    risk_drawdown_share = _safe_share(
        pd.to_numeric(risk_rows.get(f"delta_max_drawdown_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
    )
    regime_sharpe_share = _safe_share(
        pd.to_numeric(regime_rows.get(f"delta_sharpe_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") > 0.0
    )
    regime_cagr_median = _safe_median(regime_rows.get(f"delta_cagr_vs_{baseline_variant}", pd.Series(dtype=float)))
    turnover_reduction_share = _safe_share(
        pd.to_numeric(regime_rows.get(f"delta_turnover_vs_{baseline_variant}", pd.Series(dtype=float)), errors="coerce") < 0.0
    )

    robust_rows = regime_rows[
        pd.to_numeric(
            regime_rows.get(f"delta_sharpe_vs_{baseline_variant}", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        > 0.0
    ]
    fragile_rows = regime_rows[
        pd.to_numeric(
            regime_rows.get(f"delta_sharpe_vs_{baseline_variant}", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        <= 0.0
    ]
    robust_settings = (
        robust_rows.groupby(["horizon", "cost_mode", "sizing_mode"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "horizon"], ascending=[False, True])
        .head(5)
    )
    fragile_settings = (
        fragile_rows.groupby(["horizon", "cost_mode", "sizing_mode"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "horizon"], ascending=[False, True])
        .head(5)
    )
    elevated_cost = cost_sensitivity_summary[cost_sensitivity_summary.get("cost_mode") == "elevated"].copy()
    most_cost_resilient = (
        elevated_cost.sort_values("mean_delta_sharpe_vs_baseline_cost", ascending=False).head(1)
        if not elevated_cost.empty and "mean_delta_sharpe_vs_baseline_cost" in elevated_cost.columns
        else pd.DataFrame()
    )

    lines = [
        "# Phase 2.5 Hardening Report",
        "",
        "## Scope",
        "",
        "- Reused the existing Phase 2 walk-forward, regime, GARCH risk, and strategy stack.",
        "- Expanded the benchmark into a bounded matrix across ticker groups, horizons, thresholds, cost modes, and sizing modes.",
        "- Measured stability, not just peak numbers.",
        "",
        "## Reused From Phase 2",
        "",
        "- Shared forecast contracts, model registry, and weighted ensemble output schema.",
        "- Leakage-safe walk-forward evaluation and explicit post-forecast backtesting.",
        "- Markov-switching regime generation with deterministic fallback behavior.",
        "- GARCH volatility, VaR/CVaR, and drawdown-aware sizing inputs.",
        "",
        "## Validated",
        "",
        f"- Runtime: `{runtime.get('python_executable')}`",
        f"- statsmodels: `{dependencies.get('statsmodels')}`",
        f"- arch: `{dependencies.get('arch')}`",
        "",
        "## Findings",
        "",
        f"1. Risk conditioning improved drawdown in `{risk_drawdown_share:.2%}` of comparable runs.",
        f"2. Regime conditioning improved Sharpe in `{regime_sharpe_share:.2%}` of comparable runs.",
        f"3. Median CAGR delta for forecast+risk+regime versus forecast_only was `{regime_cagr_median:.6f}`.",
        f"4. Regime-conditioned turnover was lower than forecast_only in `{turnover_reduction_share:.2%}` of comparable runs.",
        f"5. Mean regime fallback share was `{assessment.get('mean_regime_fallback_share', float('nan')):.2%}`.",
        f"6. Recommendation: `{assessment.get('recommendation')}`.",
        "",
        "## Regime Layer",
        "",
        f"- Mean switch rate: `{_safe_mean(regime_stability_summary.get('switch_rate', pd.Series(dtype=float))):.6f}`",
        f"- Mean average regime duration: `{_safe_mean(regime_stability_summary.get('average_regime_duration', pd.Series(dtype=float))):.6f}`",
        f"- Mean max regime probability: `{_safe_mean(regime_stability_summary.get('mean_max_probability', pd.Series(dtype=float))):.6f}`",
        f"- Mean fallback observation share: `{_safe_mean(regime_stability_summary.get('fallback_observation_share', pd.Series(dtype=float))):.2%}`",
        "",
        "## Risk Layer",
        "",
        f"- Mean size multiplier: `{_safe_mean(risk_stability_summary.get('avg_size_multiplier', pd.Series(dtype=float))):.6f}`",
        f"- Mean exposure reduction share: `{_safe_mean(risk_stability_summary.get('exposure_reduction_share', pd.Series(dtype=float))):.2%}`",
        f"- Mean volatility forecast: `{_safe_mean(risk_stability_summary.get('mean_vol_forecast', pd.Series(dtype=float))):.6f}`",
        "",
        "## Robust Settings",
        "",
    ]
    if robust_settings.empty:
        lines.append("- No consistently robust regime-conditioned setting stood out in this bounded sweep.")
    else:
        for row in robust_settings.itertuples(index=False):
            lines.append(
                f"- horizon={row.horizon}, cost_mode={row.cost_mode}, sizing_mode={row.sizing_mode}, positive_sharpe_rows={row.count}"
            )

    lines.extend(["", "## Fragile Settings", ""])
    if fragile_settings.empty:
        lines.append("- No clearly fragile regime-conditioned cluster dominated the bounded sweep.")
    else:
        for row in fragile_settings.itertuples(index=False):
            lines.append(
                f"- horizon={row.horizon}, cost_mode={row.cost_mode}, sizing_mode={row.sizing_mode}, non_improving_sharpe_rows={row.count}"
            )

    lines.extend(["", "## Cost Robustness", ""])
    if most_cost_resilient.empty:
        lines.append("- No elevated-cost winner stood out cleanly in this bounded sweep.")
    else:
        row = most_cost_resilient.iloc[0]
        lines.append(
            f"- Most cost-resilient mode under elevated costs was `{row['strategy_variant']}` with mean Sharpe delta `{row['mean_delta_sharpe_vs_baseline_cost']:.6f}`."
        )

    lines.extend(
        [
            "",
            "## Phase 3 Readiness",
            "",
            f"- Regime-aware forecast routing: `{assessment.get('recommendation')}`",
            f"- Stacking/meta-model layer: `{assessment.get('recommendation')}`",
            "- Portfolio allocator: `defer until conditioning behavior is stable enough across costs and windows`",
            "- Deep sequence models: `defer until current conditioning evidence is strong enough to justify more complex capacity`",
            "",
            "## Deferred To Phase 3",
            "",
            "- Regime-aware forecast routing should wait for stronger stability evidence.",
            "- Stacking/meta-model orchestration should wait for a more reliable conditioning baseline.",
            "- Portfolio allocation remains deferred until single-strategy conditioning behavior is robust across costs.",
            "- Deep sequence models remain deferred because Phase 2 conditioning still needs broader validation.",
        ]
    )
    return "\n".join(lines)
