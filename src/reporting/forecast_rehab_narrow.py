"""Reporting helpers for the narrowed forecast rehabilitation cycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.median())


def _safe_std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0))


def _safe_share(mask: pd.Series) -> float:
    clean = pd.Series(mask).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.astype(bool).mean())


def _present(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def build_narrow_feature_performance_summary(
    feature_definition_summary: pd.DataFrame,
    forecast_quality_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if feature_definition_summary.empty:
        return pd.DataFrame(columns=["feature_family"])

    rows: list[dict[str, Any]] = []
    for row in feature_definition_summary.to_dict(orient="records"):
        family_name = str(row["feature_family"])
        quality = forecast_quality_summary[forecast_quality_summary["feature_family"] == family_name].copy()
        strategy = strategy_metrics[strategy_metrics["feature_family"] == family_name].copy()
        result = dict(row)
        result["median_directional_accuracy"] = _safe_median(quality.get("median_directional_accuracy", pd.Series(dtype=float)))
        result["strong_directional_accuracy_share"] = _safe_median(
            quality.get("strong_directional_accuracy_share", pd.Series(dtype=float))
        )
        result["tradable_slice_share"] = _safe_median(quality.get("tradable_slice_share", pd.Series(dtype=float)))
        result["median_rmse"] = _safe_median(quality.get("median_rmse", pd.Series(dtype=float)))
        for cost_mode, cost_group in strategy.groupby("cost_mode", sort=True):
            key = str(cost_mode).lower()
            result[f"median_sharpe_{key}"] = _safe_median(cost_group.get("sharpe", pd.Series(dtype=float)))
            result[f"median_cagr_{key}"] = _safe_median(cost_group.get("cagr", pd.Series(dtype=float)))
            result[f"median_max_drawdown_{key}"] = _safe_median(cost_group.get("max_drawdown", pd.Series(dtype=float)))
            result[f"positive_sharpe_share_{key}"] = _safe_share(
                pd.to_numeric(cost_group.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
            )
        rows.append(result)
    return pd.DataFrame(rows).sort_values("feature_family").reset_index(drop=True)


def build_narrow_model_stability_summary(
    forecast_quality_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if forecast_quality_summary.empty:
        return pd.DataFrame(columns=["model_name", "horizon", "target_name"])

    rows: list[dict[str, Any]] = []
    group_columns = _present(forecast_quality_summary, ["model_name", "horizon", "target_name"])
    for keys, quality_group in forecast_quality_summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        stability_group = model_stability_summary.merge(
            quality_group[_present(quality_group, ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name"])],
            on=_present(model_stability_summary, ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name"]),
            how="inner",
        )
        policy_group = strategy_metrics[
            (strategy_metrics["model_name"] == row["model_name"])
            & (strategy_metrics["horizon"] == row["horizon"])
            & (strategy_metrics["target_name"] == row["target_name"])
        ].copy()

        row["target_family"] = str(quality_group["target_family"].iloc[0]) if "target_family" in quality_group.columns else ""
        row["feature_family_count"] = int(quality_group["feature_family"].nunique()) if "feature_family" in quality_group.columns else int(len(quality_group))
        row["median_directional_accuracy"] = _safe_median(quality_group.get("median_directional_accuracy", pd.Series(dtype=float)))
        row["median_rmse"] = _safe_median(quality_group.get("median_rmse", pd.Series(dtype=float)))
        row["strong_directional_accuracy_share"] = _safe_median(
            quality_group.get("strong_directional_accuracy_share", pd.Series(dtype=float))
        )
        row["tradable_slice_share"] = _safe_median(quality_group.get("tradable_slice_share", pd.Series(dtype=float)))
        row["median_directional_accuracy_dispersion"] = _safe_median(
            stability_group.get("directional_accuracy_dispersion", pd.Series(dtype=float))
        )
        row["median_rmse_dispersion"] = _safe_median(stability_group.get("rmse_dispersion", pd.Series(dtype=float)))
        row["positive_slice_share"] = _safe_median(stability_group.get("positive_slice_share", pd.Series(dtype=float)))
        for cost_mode, cost_group in policy_group.groupby("cost_mode", sort=True):
            key = str(cost_mode).lower()
            row[f"median_policy_sharpe_{key}"] = _safe_median(cost_group.get("sharpe", pd.Series(dtype=float)))
            row[f"median_policy_cagr_{key}"] = _safe_median(cost_group.get("cagr", pd.Series(dtype=float)))
            row[f"positive_policy_sharpe_share_{key}"] = _safe_share(
                pd.to_numeric(cost_group.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_narrow_target_comparison_summary(
    forecast_quality_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if forecast_quality_summary.empty:
        return pd.DataFrame(columns=["target_name", "horizon"])

    rows: list[dict[str, Any]] = []
    group_columns = _present(forecast_quality_summary, ["target_name", "target_family", "horizon"])
    for keys, quality_group in forecast_quality_summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["median_directional_accuracy"] = _safe_median(quality_group.get("median_directional_accuracy", pd.Series(dtype=float)))
        row["strong_directional_accuracy_share"] = _safe_median(
            quality_group.get("strong_directional_accuracy_share", pd.Series(dtype=float))
        )
        row["tradable_slice_share"] = _safe_median(quality_group.get("tradable_slice_share", pd.Series(dtype=float)))
        row["median_rmse"] = _safe_median(quality_group.get("median_rmse", pd.Series(dtype=float)))
        policy_group = strategy_metrics[
            (strategy_metrics["target_name"] == row["target_name"])
            & (strategy_metrics["horizon"] == row["horizon"])
        ].copy()
        row["supports_policy_evaluation"] = bool(not policy_group.empty)
        for cost_mode, cost_group in policy_group.groupby("cost_mode", sort=True):
            key = str(cost_mode).lower()
            row[f"median_policy_sharpe_{key}"] = _safe_median(cost_group.get("sharpe", pd.Series(dtype=float)))
            row[f"positive_policy_sharpe_share_{key}"] = _safe_share(
                pd.to_numeric(cost_group.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_narrow_forecast_vs_policy_summary(
    forecast_quality_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if forecast_quality_summary.empty or strategy_metrics.empty:
        return pd.DataFrame(columns=["feature_family", "model_name", "target_name", "horizon", "cost_mode"])

    forecast_keys = _present(
        forecast_quality_summary,
        ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name"],
    )
    forecast_metrics = _present(
        forecast_quality_summary,
        ["median_rmse", "median_mae", "median_directional_accuracy", "strong_directional_accuracy_share", "tradable_slice_share"],
    )
    strategy_keys = _present(
        strategy_metrics,
        ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name", "cost_mode"],
    )
    strategy_metrics_columns = _present(
        strategy_metrics,
        ["sharpe", "cagr", "max_drawdown", "turnover", "trade_count", "total_return", "cost_label"],
    )
    merged = forecast_quality_summary[forecast_keys + forecast_metrics].merge(
        strategy_metrics[strategy_keys + strategy_metrics_columns],
        on=_present(strategy_metrics, forecast_keys),
        how="inner",
    )
    rank_scope = _present(merged, ["cost_mode", "horizon", "target_name", "feature_family"])
    merged["forecast_rank"] = merged.groupby(rank_scope)["median_rmse"].rank(method="dense", ascending=True)
    merged["strategy_rank"] = merged.groupby(rank_scope)["sharpe"].rank(method="dense", ascending=False)
    group_counts = merged.groupby(rank_scope)["model_name"].transform("size")
    midpoint = group_counts / 2.0
    merged["monetization_gap"] = merged["strategy_rank"] - merged["forecast_rank"]
    merged["edge_but_not_monetized"] = (merged["forecast_rank"] <= midpoint) & (merged["strategy_rank"] > midpoint)
    return merged.sort_values([*rank_scope, "strategy_rank", "model_name"]).reset_index(drop=True)


def build_cost_sensitivity_summary(strategy_metrics: pd.DataFrame) -> pd.DataFrame:
    if strategy_metrics.empty:
        return pd.DataFrame(columns=["cost_mode", "horizon", "target_name"])

    rows: list[dict[str, Any]] = []
    group_columns = _present(strategy_metrics, ["cost_mode", "target_name", "horizon"])
    for keys, group in strategy_metrics.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["median_sharpe"] = _safe_median(group.get("sharpe", pd.Series(dtype=float)))
        row["median_cagr"] = _safe_median(group.get("cagr", pd.Series(dtype=float)))
        row["median_max_drawdown"] = _safe_median(group.get("max_drawdown", pd.Series(dtype=float)))
        row["median_turnover"] = _safe_median(group.get("turnover", pd.Series(dtype=float)))
        row["median_trade_count"] = _safe_median(group.get("trade_count", pd.Series(dtype=float)))
        row["positive_sharpe_share"] = _safe_share(pd.to_numeric(group.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_f1_reference_comparison(
    *,
    reference_dir: str | Path = "artifacts/forecast_rehab",
    narrow_models: list[str] | None = None,
) -> dict[str, Any]:
    base = Path(reference_dir)
    forecast_path = base / "forecast_quality_summary.csv"
    strategy_path = base / "aggregate_strategy_metrics.csv"
    if not forecast_path.exists() or not strategy_path.exists():
        return {}

    models = set(str(model).lower() for model in (narrow_models or []))
    forecast_quality = pd.read_csv(forecast_path)
    strategy_metrics = pd.read_csv(strategy_path)
    forecast_quality["model_name"] = forecast_quality["model_name"].astype(str).str.lower()
    strategy_metrics["model_name"] = strategy_metrics["model_name"].astype(str).str.lower()
    forecast_quality = forecast_quality[
        forecast_quality["group_name"].eq("small_banks")
        & forecast_quality["horizon"].isin([5, 10])
        & forecast_quality["model_name"].isin(models)
    ].copy()
    strategy_metrics = strategy_metrics[
        strategy_metrics["group_name"].eq("small_banks")
        & strategy_metrics["horizon"].isin([5, 10])
        & strategy_metrics["target_name"].eq("forward_return")
        & strategy_metrics["model_name"].isin(models)
    ].copy()

    return {
        "reference_scope": "phase_f1_small_banks_h5_h10",
        "forecast_median_directional_accuracy": _safe_median(
            forecast_quality.get("median_directional_accuracy", pd.Series(dtype=float))
        ),
        "forecast_strong_slice_share": _safe_median(
            forecast_quality.get("strong_directional_accuracy_share", pd.Series(dtype=float))
        ),
        "forecast_tradable_slice_share": _safe_median(
            forecast_quality.get("tradable_slice_share", pd.Series(dtype=float))
        ),
        "policy_median_sharpe": _safe_median(strategy_metrics.get("sharpe", pd.Series(dtype=float))),
        "policy_positive_sharpe_share": _safe_share(
            pd.to_numeric(strategy_metrics.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
        ),
        "policy_median_cagr": _safe_median(strategy_metrics.get("cagr", pd.Series(dtype=float))),
        "policy_median_max_drawdown": _safe_median(strategy_metrics.get("max_drawdown", pd.Series(dtype=float))),
    }


def build_narrow_assessment(
    feature_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    target_comparison_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    f1_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    f1_reference = dict(f1_reference or {})
    if feature_summary.empty:
        return {
            "recommendation": "stop daily forecast rehab and reconsider the research premise",
            "phase3_blocked": True,
            "continue_one_more_cycle": False,
        }

    feature_scores = feature_summary.sort_values(
        ["positive_sharpe_share_baseline", "median_sharpe_baseline", "median_directional_accuracy", "median_rmse"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_feature = feature_scores.iloc[0]

    model_scores = model_stability_summary.sort_values(
        ["positive_policy_sharpe_share_baseline", "strong_directional_accuracy_share", "median_directional_accuracy_dispersion", "median_rmse"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    best_model = model_scores.iloc[0] if not model_scores.empty else None

    horizon_scores = target_comparison_summary[target_comparison_summary["target_name"] == "forward_return"].sort_values(
        ["positive_policy_sharpe_share_baseline", "median_policy_sharpe_baseline", "median_directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_horizon = int(horizon_scores.iloc[0]["horizon"]) if not horizon_scores.empty else None

    target_scores = target_comparison_summary.sort_values(
        [
            "supports_policy_evaluation",
            "positive_policy_sharpe_share_baseline",
            "median_policy_sharpe_baseline",
            "median_directional_accuracy",
            "strong_directional_accuracy_share",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    if len(target_scores) >= 2:
        direction_row = target_scores[target_scores["target_name"] == "direction_binary"]
        return_row = target_scores[target_scores["target_name"] == "forward_return"]
        if not direction_row.empty and not return_row.empty:
            direction_supports_policy = bool(direction_row.iloc[0].get("supports_policy_evaluation", False))
            return_supports_policy = bool(return_row.iloc[0].get("supports_policy_evaluation", False))
            direction_value = float(direction_row.iloc[0]["median_directional_accuracy"])
            return_value = float(return_row.iloc[0]["median_directional_accuracy"])
            direction_strength = float(direction_row.iloc[0]["strong_directional_accuracy_share"])
            return_strength = float(return_row.iloc[0]["strong_directional_accuracy_share"])
            if return_supports_policy and not direction_supports_policy:
                best_target = (
                    "direction_binary"
                    if direction_value > return_value + 0.05 and direction_strength > return_strength + 0.05
                    else "forward_return"
                )
            else:
                best_target = (
                    "direction_binary"
                    if direction_value > return_value + 0.03 and direction_strength >= return_strength
                    else "forward_return"
                )
        else:
            best_target = str(target_scores.iloc[0]["target_name"])
    else:
        best_target = str(target_scores.iloc[0]["target_name"]) if not target_scores.empty else None

    forward_policy = forecast_vs_policy_summary[
        forecast_vs_policy_summary["target_name"] == "forward_return"
    ].copy()
    baseline_policy = forward_policy[forward_policy["cost_mode"] == "baseline"].copy()
    elevated_policy = forward_policy[forward_policy["cost_mode"] == "elevated"].copy()
    baseline_positive_share = _safe_share(pd.to_numeric(baseline_policy.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0)
    elevated_positive_share = _safe_share(pd.to_numeric(elevated_policy.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0)
    baseline_median_sharpe = _safe_median(baseline_policy.get("sharpe", pd.Series(dtype=float)))
    elevated_median_sharpe = _safe_median(elevated_policy.get("sharpe", pd.Series(dtype=float)))

    f1_policy_median_sharpe = float(f1_reference.get("policy_median_sharpe", float("nan")))
    f1_positive_sharpe_share = float(f1_reference.get("policy_positive_sharpe_share", float("nan")))
    materially_better_than_f1 = bool(
        pd.notna(baseline_median_sharpe)
        and pd.notna(f1_policy_median_sharpe)
        and pd.notna(baseline_positive_share)
        and pd.notna(f1_positive_sharpe_share)
        and baseline_median_sharpe > f1_policy_median_sharpe + 0.15
        and baseline_positive_share > f1_positive_sharpe_share + 0.05
    )

    recommendation = "continue narrow rehab"
    if pd.notna(baseline_median_sharpe) and baseline_median_sharpe < 0.0 and elevated_positive_share < 0.15:
        recommendation = "stop daily forecast rehab and reconsider the research premise"
    elif best_model is not None and float(best_model.get("positive_policy_sharpe_share_baseline", 0.0) or 0.0) >= 0.45:
        recommendation = "freeze all but the best one or two model families"
    elif best_target == "forward_return":
        recommendation = "standardize on one target framing"
    elif not materially_better_than_f1:
        recommendation = "keep small_banks only and stop trying to generalize"

    continue_one_more_cycle = bool(
        recommendation in {"continue narrow rehab", "freeze all but the best one or two model families", "standardize on one target framing"}
        and pd.notna(baseline_positive_share)
        and baseline_positive_share >= 0.25
    )

    return {
        "recommendation": recommendation,
        "best_feature_family": str(best_feature["feature_family"]),
        "best_model_family": None if best_model is None else str(best_model["model_name"]),
        "best_horizon": best_horizon,
        "best_target_name": best_target,
        "baseline_positive_sharpe_share": baseline_positive_share,
        "baseline_median_policy_sharpe": baseline_median_sharpe,
        "elevated_positive_sharpe_share": elevated_positive_share,
        "elevated_median_policy_sharpe": elevated_median_sharpe,
        "materially_better_than_f1": materially_better_than_f1,
        "f1_reference": f1_reference,
        "phase3_blocked": True,
        "continue_one_more_cycle": continue_one_more_cycle,
    }


def render_narrow_summary_markdown(
    manifest: dict[str, Any],
    assessment: dict[str, Any],
    feature_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
) -> str:
    matrix = manifest.get("matrix", {})
    runtime = manifest.get("runtime", {})
    lines = [
        "# Narrow Forecast Rehab Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Preset: `{matrix.get('preset')}`",
        f"- Group: `{', '.join(group['group_name'] for group in matrix.get('ticker_groups', []))}`",
        f"- Horizons: `{', '.join(str(value) for value in matrix.get('horizons', []))}`",
        f"- Targets: `{', '.join(matrix.get('target_names', []))}`",
        f"- Feature families: `{', '.join(matrix.get('feature_families', []))}`",
        f"- Models: `{', '.join(matrix.get('models', []))}`",
        f"- Recommendation: `{assessment.get('recommendation')}`",
    ]

    if not feature_summary.empty:
        lines.extend(
            [
                "",
                "## Feature Families",
                "",
                "| feature_family | feature_count | median_directional_accuracy | baseline_positive_sharpe_share | elevated_positive_sharpe_share |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in feature_summary.sort_values(
            ["positive_sharpe_share_baseline", "median_sharpe_baseline", "median_directional_accuracy"],
            ascending=[False, False, False],
        ).itertuples(index=False):
            lines.append(
                f"| {row.feature_family} | {row.feature_count} | {row.median_directional_accuracy:.6f} | {getattr(row, 'positive_sharpe_share_baseline', float('nan')):.2%} | {getattr(row, 'positive_sharpe_share_elevated', float('nan')):.2%} |"
            )

    if not model_stability_summary.empty:
        lines.extend(
            [
                "",
                "## Models",
                "",
                "| model_name | median_directional_accuracy | median_directional_accuracy_dispersion | baseline_positive_sharpe_share |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        preview = model_stability_summary.groupby("model_name", sort=True)[
            ["median_directional_accuracy", "median_directional_accuracy_dispersion", "positive_policy_sharpe_share_baseline"]
        ].median().reset_index()
        preview = preview.sort_values(
            ["positive_policy_sharpe_share_baseline", "median_directional_accuracy", "median_directional_accuracy_dispersion"],
            ascending=[False, False, True],
        )
        for row in preview.itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {row.median_directional_accuracy:.6f} | {row.median_directional_accuracy_dispersion:.6f} | {row.positive_policy_sharpe_share_baseline:.2%} |"
            )
    return "\n".join(lines)


def build_narrow_report(
    manifest: dict[str, Any],
    scope_table: pd.DataFrame,
    feature_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    target_comparison_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
    cost_sensitivity_summary: pd.DataFrame,
    assessment: dict[str, Any],
) -> str:
    runtime = manifest.get("runtime", {})
    deps = manifest.get("dependency_versions", {})
    f1_reference = dict(assessment.get("f1_reference", {}))
    lines = [
        "# Narrow Forecast Rehabilitation Report",
        "",
        "## Scope",
        "",
        "- Narrowed the rehab cycle to the F1 winner cluster only: small banks, horizons 5/10, explicit compact technical feature families, tree models plus guarded statistical comparators.",
        "- Kept the Phase 2.6 execution policy fixed and added cost sensitivity instead of reopening policy optimization.",
        "",
        "## Runtime",
        "",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- statsmodels: `{deps.get('statsmodels')}`",
        f"- xgboost: `{deps.get('xgboost')}`",
        f"- lightgbm: `{deps.get('lightgbm')}`",
        "",
        "## Answers",
        "",
        f"1. Narrow scope versus broader F1: `{'better' if assessment.get('materially_better_than_f1') else 'not materially better'}` against the F1 small-bank horizon-5/10 reference. F1 reference median policy Sharpe was `{f1_reference.get('policy_median_sharpe', float('nan')):.6f}` with positive-Sharpe share `{f1_reference.get('policy_positive_sharpe_share', float('nan')):.2%}`.",
        f"2. Best balanced feature family: `{assessment.get('best_feature_family')}`.",
        f"3. Most stable model family: `{assessment.get('best_model_family')}`.",
        f"4. Horizon leadership: `{assessment.get('best_horizon')}` is the current best horizon inside the narrow scope.",
        f"5. Best next-stage target framing: `{assessment.get('best_target_name')}`.",
        f"6. Baseline positive-Sharpe share: `{assessment.get('baseline_positive_sharpe_share', float('nan')):.2%}`. Elevated-cost positive-Sharpe share: `{assessment.get('elevated_positive_sharpe_share', float('nan')):.2%}`.",
        f"7. Evidence strong enough to reopen Phase 3: `False`.",
        "",
        "## Scope Table",
        "",
    ]

    for row in scope_table.itertuples(index=False):
        lines.append(
            f"- {row.dimension}: in-scope=`{row.in_scope}` comparator-only=`{row.comparator_only}` baseline-only=`{row.baseline_only}` out/de-emphasized=`{row.out_or_deemphasized}` because {row.reason}"
        )

    if not feature_summary.empty:
        lines.extend(
            [
                "",
                "## Narrow Feature Families",
                "",
            ]
        )
        for row in feature_summary.sort_values(
            ["positive_sharpe_share_baseline", "median_sharpe_baseline", "median_directional_accuracy"],
            ascending=[False, False, False],
        ).itertuples(index=False):
            lines.append(
                f"- {row.feature_family}: count={row.feature_count}, baseline_sharpe={getattr(row, 'median_sharpe_baseline', float('nan')):.6f}, elevated_sharpe={getattr(row, 'median_sharpe_elevated', float('nan')):.6f}, rationale={row.rationale}"
            )

    if not target_comparison_summary.empty:
        lines.extend(
            [
                "",
                "## Target Framing",
                "",
            ]
        )
        for row in target_comparison_summary.sort_values(["target_name", "horizon"]).itertuples(index=False):
            baseline_value = getattr(row, "median_policy_sharpe_baseline", float("nan"))
            lines.append(
                f"- {row.target_name} h={row.horizon}: median_da={row.median_directional_accuracy:.6f}, strong_share={row.strong_directional_accuracy_share:.2%}, baseline_policy_sharpe={baseline_value:.6f}"
            )

    if not model_stability_summary.empty:
        lines.extend(
            [
                "",
                "## Model Stability",
                "",
            ]
        )
        preview = model_stability_summary.groupby("model_name", sort=True)[
            ["median_directional_accuracy", "median_directional_accuracy_dispersion", "positive_policy_sharpe_share_baseline"]
        ].median().reset_index()
        preview = preview.sort_values(
            ["positive_policy_sharpe_share_baseline", "median_directional_accuracy", "median_directional_accuracy_dispersion"],
            ascending=[False, False, True],
        )
        for row in preview.itertuples(index=False):
            lines.append(
                f"- {row.model_name}: median_da={row.median_directional_accuracy:.6f}, da_dispersion={row.median_directional_accuracy_dispersion:.6f}, baseline_positive_sharpe_share={row.positive_policy_sharpe_share_baseline:.2%}"
            )

    if not cost_sensitivity_summary.empty:
        lines.extend(
            [
                "",
                "## Cost Sensitivity",
                "",
            ]
        )
        for row in cost_sensitivity_summary.sort_values(["cost_mode", "target_name", "horizon"]).itertuples(index=False):
            lines.append(
                f"- {row.cost_mode} {row.target_name} h={row.horizon}: median_sharpe={row.median_sharpe:.6f}, median_cagr={row.median_cagr:.6f}, positive_sharpe_share={row.positive_sharpe_share:.2%}"
            )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Direct recommendation: `{assessment.get('recommendation')}`",
            f"- Continue one more rehab cycle: `{assessment.get('continue_one_more_cycle')}`",
            f"- Phase 3 blocked: `{assessment.get('phase3_blocked')}`",
        ]
    )
    return "\n".join(lines)
