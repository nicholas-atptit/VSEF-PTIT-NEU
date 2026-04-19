"""Reporting helpers for forecast-layer rehabilitation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def _present(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def build_feature_inventory_summary(feature_inventory: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the governed feature inventory into interpretable groups."""

    if feature_inventory.empty:
        return pd.DataFrame(
            columns=[
                "feature_group",
                "feature_count",
                "source",
                "suspected_usefulness",
                "suspected_risk",
            ]
        )

    rows: list[dict[str, Any]] = []
    for category, group in feature_inventory.groupby("category", sort=True):
        source_values = sorted(set(group["input_source"].astype(str)))
        selected_now = int(group["is_current_regression_baseline"].sum() + group["is_current_direction_baseline"].sum())
        if selected_now > 0:
            usefulness = "selected_in_current_baseline"
        elif int((group["status"] == "active").sum()) > 0:
            usefulness = "active_candidate_not_selected_now"
        else:
            usefulness = "mainly_experimental_or_deprecated"

        leakage_notes = " ".join(group["leakage_risk_note"].astype(str).str.lower().tolist())
        if "never forward-filled from the future" in leakage_notes or "backward-looking date alignment" in leakage_notes:
            risk = "context_alignment_requires_care"
        elif int((group["status"] == "experimental").sum()) > 0:
            risk = "experimental_surface"
        elif int((group["status"] == "deprecated").sum()) > 0:
            risk = "deprecated_aliases_present"
        else:
            risk = "low_relative"

        rows.append(
            {
                "feature_group": str(category),
                "feature_count": int(len(group)),
                "active_count": int((group["status"] == "active").sum()),
                "experimental_count": int((group["status"] == "experimental").sum()),
                "deprecated_count": int((group["status"] == "deprecated").sum()),
                "current_regression_selected_count": int(group["is_current_regression_baseline"].sum()),
                "current_direction_selected_count": int(group["is_current_direction_baseline"].sum()),
                "reduced_compact_count": int(group["is_reduced_compact"].sum()),
                "source": ",".join(source_values),
                "suspected_usefulness": usefulness,
                "suspected_risk": risk,
            }
        )
    return pd.DataFrame(rows).sort_values("feature_group").reset_index(drop=True)


def build_feature_ablation_summary(forecast_quality_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare feature families after aggregating across models."""

    if forecast_quality_summary.empty:
        return pd.DataFrame(columns=["group_name", "horizon", "target_name", "feature_family"])

    rows: list[dict[str, Any]] = []
    group_columns = _present(forecast_quality_summary, ["group_name", "horizon", "target_name", "feature_family"])
    for keys, group in forecast_quality_summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["model_count"] = int(group["model_name"].nunique()) if "model_name" in group.columns else int(len(group))
        row["mean_rmse"] = _safe_mean(group.get("mean_rmse", pd.Series(dtype=float)))
        row["median_rmse"] = _safe_median(group.get("mean_rmse", pd.Series(dtype=float)))
        row["mean_mae"] = _safe_mean(group.get("mean_mae", pd.Series(dtype=float)))
        row["mean_directional_accuracy"] = _safe_mean(group.get("mean_directional_accuracy", pd.Series(dtype=float)))
        row["median_directional_accuracy"] = _safe_median(group.get("mean_directional_accuracy", pd.Series(dtype=float)))
        row["strong_slice_share"] = _safe_mean(group.get("strong_directional_accuracy_share", pd.Series(dtype=float)))
        row["tradable_slice_share"] = _safe_mean(group.get("tradable_slice_share", pd.Series(dtype=float)))

        sortable = group.sort_values(["mean_rmse", "mean_directional_accuracy"], ascending=[True, False]).reset_index(drop=True)
        if not sortable.empty:
            row["best_rmse_model"] = str(sortable.iloc[0]["model_name"])
        directional_best = group.sort_values(["mean_directional_accuracy", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
        if not directional_best.empty:
            row["best_directional_model"] = str(directional_best.iloc[0]["model_name"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["feature_rank_by_rmse"] = summary.groupby(
            _present(summary, ["group_name", "horizon", "target_name"])
        )["mean_rmse"].rank(method="dense", ascending=True)
        summary["feature_rank_by_directional_accuracy"] = summary.groupby(
            _present(summary, ["group_name", "horizon", "target_name"])
        )["median_directional_accuracy"].rank(method="dense", ascending=False)
    return summary.sort_values(group_columns).reset_index(drop=True)


def build_forecast_quality_summary(slice_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize model quality, stability, and simple tradability proxies."""

    if slice_summary.empty:
        return pd.DataFrame(columns=["group_name", "horizon", "target_name", "feature_family", "model_name"])

    naive_key_columns = _present(slice_summary, ["group_name", "horizon", "target_name", "feature_family", "ticker", "window_id"])
    naive = slice_summary[slice_summary["model_name"] == "naive"].copy()
    naive = naive[naive_key_columns + _present(naive, ["rmse"])].rename(columns={"rmse": "naive_rmse"})
    enriched = slice_summary.merge(naive, on=naive_key_columns, how="left")
    enriched["beats_naive_rmse"] = pd.to_numeric(enriched.get("rmse"), errors="coerce") < pd.to_numeric(
        enriched.get("naive_rmse"),
        errors="coerce",
    )
    enriched["positive_directional_accuracy"] = pd.to_numeric(enriched.get("directional_accuracy"), errors="coerce") > 0.50
    enriched["strong_directional_accuracy"] = pd.to_numeric(enriched.get("directional_accuracy"), errors="coerce") >= 0.55
    enriched["tradable_slice"] = enriched["strong_directional_accuracy"] & enriched["beats_naive_rmse"].fillna(False)

    rows: list[dict[str, Any]] = []
    group_columns = _present(enriched, ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name"])
    for keys, group in enriched.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["slice_count"] = int(len(group))
        row["ticker_count"] = int(group["ticker"].nunique()) if "ticker" in group.columns else int(len(group))
        row["mean_mae"] = _safe_mean(group.get("mae", pd.Series(dtype=float)))
        row["median_mae"] = _safe_median(group.get("mae", pd.Series(dtype=float)))
        row["mean_rmse"] = _safe_mean(group.get("rmse", pd.Series(dtype=float)))
        row["median_rmse"] = _safe_median(group.get("rmse", pd.Series(dtype=float)))
        row["mean_directional_accuracy"] = _safe_mean(group.get("directional_accuracy", pd.Series(dtype=float)))
        row["median_directional_accuracy"] = _safe_median(group.get("directional_accuracy", pd.Series(dtype=float)))
        row["positive_directional_accuracy_share"] = _safe_share(group.get("positive_directional_accuracy", pd.Series(dtype=bool)))
        row["strong_directional_accuracy_share"] = _safe_share(group.get("strong_directional_accuracy", pd.Series(dtype=bool)))
        row["beats_naive_rmse_share"] = _safe_share(group.get("beats_naive_rmse", pd.Series(dtype=bool)))
        row["tradable_slice_share"] = _safe_share(group.get("tradable_slice", pd.Series(dtype=bool)))
        row["rmse_dispersion"] = _safe_std(group.get("rmse", pd.Series(dtype=float)))
        row["directional_accuracy_dispersion"] = _safe_std(group.get("directional_accuracy", pd.Series(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_model_stability_summary(slice_summary: pd.DataFrame) -> pd.DataFrame:
    """Track best-case, worst-case, and dispersion by model setup."""

    if slice_summary.empty:
        return pd.DataFrame(columns=["group_name", "horizon", "target_name", "feature_family", "model_name"])

    rows: list[dict[str, Any]] = []
    group_columns = _present(slice_summary, ["group_name", "horizon", "target_name", "target_family", "feature_family", "model_name"])
    for keys, group in slice_summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        rmse = pd.to_numeric(group.get("rmse"), errors="coerce")
        directional = pd.to_numeric(group.get("directional_accuracy"), errors="coerce")
        row["slice_count"] = int(len(group))
        row["rmse_best"] = _safe_median(rmse.nsmallest(1))
        row["rmse_worst"] = _safe_median(rmse.nlargest(1))
        row["rmse_dispersion"] = _safe_std(rmse)
        row["directional_accuracy_best"] = _safe_median(directional.nlargest(1))
        row["directional_accuracy_worst"] = _safe_median(directional.nsmallest(1))
        row["directional_accuracy_dispersion"] = _safe_std(directional)
        row["positive_slice_share"] = _safe_share(directional > 0.50)
        row["strong_slice_share"] = _safe_share(directional >= 0.55)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_forecast_vs_policy_summary(
    forecast_quality_summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forecast quality with fixed-policy economic outcomes."""

    if forecast_quality_summary.empty or strategy_metrics.empty:
        return pd.DataFrame(columns=["group_name", "horizon", "target_name", "feature_family", "model_name"])

    forecast_group_columns = _present(forecast_quality_summary, ["group_name", "horizon", "target_name", "feature_family", "model_name"])
    strategy_group_columns = _present(strategy_metrics, ["group_name", "horizon", "target_name", "feature_family", "model_name"])
    metrics = _present(strategy_metrics, ["cagr", "sharpe", "max_drawdown", "turnover", "trade_count", "total_return"])
    merged = forecast_quality_summary[forecast_group_columns + _present(forecast_quality_summary, ["mean_rmse", "mean_mae", "mean_directional_accuracy", "tradable_slice_share"])].merge(
        strategy_metrics[strategy_group_columns + metrics],
        on=_present(forecast_quality_summary, ["group_name", "horizon", "target_name", "feature_family", "model_name"]),
        how="left",
    )
    rank_scope = _present(merged, ["group_name", "horizon", "target_name", "feature_family"])
    merged["forecast_rank"] = merged.groupby(rank_scope)["mean_rmse"].rank(method="dense", ascending=True)
    merged["strategy_rank"] = merged.groupby(rank_scope)["sharpe"].rank(method="dense", ascending=False)
    group_counts = merged.groupby(rank_scope)["model_name"].transform("size")
    midpoint = group_counts / 2.0
    merged["monetization_gap"] = merged["strategy_rank"] - merged["forecast_rank"]
    merged["edge_but_not_monetized"] = (merged["forecast_rank"] <= midpoint) & (merged["strategy_rank"] > midpoint)
    return merged.sort_values([*rank_scope, "strategy_rank", "model_name"]).reset_index(drop=True)


def build_forecast_rehab_assessment(
    feature_ablation_summary: pd.DataFrame,
    forecast_quality_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a bounded Phase F1 recommendation."""

    if forecast_quality_summary.empty:
        return {
            "recommendation": "stop expansion and reconsider whether this repo has enough edge at daily frequency",
            "phase3_blocked": True,
        }

    feature_da_column = "median_directional_accuracy" if "median_directional_accuracy" in feature_ablation_summary.columns else "mean_directional_accuracy"
    feature_rmse_column = "median_rmse" if "median_rmse" in feature_ablation_summary.columns else "mean_rmse"
    quality_da_column = "median_directional_accuracy" if "median_directional_accuracy" in forecast_quality_summary.columns else "mean_directional_accuracy"
    quality_rmse_column = "median_rmse" if "median_rmse" in forecast_quality_summary.columns else "mean_rmse"

    feature_family_scores = (
        feature_ablation_summary.groupby("feature_family", sort=True)[[feature_da_column, "strong_slice_share", feature_rmse_column]]
        .median()
        .reset_index()
        .sort_values([feature_da_column, "strong_slice_share", feature_rmse_column], ascending=[False, False, True])
        if not feature_ablation_summary.empty
        else pd.DataFrame()
    )
    best_feature_family = None if feature_family_scores.empty else str(feature_family_scores.iloc[0]["feature_family"])

    horizon_scores = (
        forecast_quality_summary.groupby("horizon", sort=True)[[quality_da_column, "tradable_slice_share", quality_rmse_column]]
        .median()
        .reset_index()
        .sort_values([quality_da_column, "tradable_slice_share", quality_rmse_column], ascending=[False, False, True])
    )
    best_horizon = None if horizon_scores.empty else int(horizon_scores.iloc[0]["horizon"])

    group_scores = (
        forecast_quality_summary.groupby("group_name", sort=True)[[quality_da_column, "tradable_slice_share", quality_rmse_column]]
        .median()
        .reset_index()
        .sort_values([quality_da_column, "tradable_slice_share", quality_rmse_column], ascending=[False, False, True])
    )
    best_group = None if group_scores.empty else str(group_scores.iloc[0]["group_name"])

    target_scores = (
        forecast_quality_summary.groupby(["target_name", "target_family"], sort=True)[[quality_da_column, "tradable_slice_share", quality_rmse_column]]
        .median()
        .reset_index()
        .sort_values([quality_da_column, "tradable_slice_share", quality_rmse_column], ascending=[False, False, True])
    )
    best_target_name = None if target_scores.empty else str(target_scores.iloc[0]["target_name"])

    model_scores = (
        forecast_quality_summary.groupby("model_name", sort=True)[[quality_da_column, "strong_directional_accuracy_share", quality_rmse_column, "tradable_slice_share"]]
        .median()
        .reset_index()
        .sort_values(["strong_directional_accuracy_share", quality_da_column, quality_rmse_column], ascending=[False, False, True])
    )
    top_models = [] if model_scores.empty else model_scores.head(3)["model_name"].astype(str).tolist()

    regression_direction = forecast_quality_summary[
        forecast_quality_summary["target_family"] == "return_regression"
    ]
    direction_only = forecast_quality_summary[
        forecast_quality_summary["target_family"] == "direction_classification"
    ]
    regression_da = _safe_median(regression_direction.get(quality_da_column, pd.Series(dtype=float)))
    direction_da = _safe_median(direction_only.get(quality_da_column, pd.Series(dtype=float)))

    policy_sharpe = _safe_median(forecast_vs_policy_summary.get("sharpe", pd.Series(dtype=float)))
    positive_policy_sharpe_share = _safe_share(
        pd.to_numeric(forecast_vs_policy_summary.get("sharpe", pd.Series(dtype=float)), errors="coerce") > 0.0
    )
    edge_not_monetized_share = _safe_share(forecast_vs_policy_summary.get("edge_but_not_monetized", pd.Series(dtype=bool)))
    strong_slice_share = _safe_mean(forecast_quality_summary.get("strong_directional_accuracy_share", pd.Series(dtype=float)))
    instability = _safe_mean(model_stability_summary.get("directional_accuracy_dispersion", pd.Series(dtype=float)))

    recommendation = "continue forecast rehab with narrowed feature/model scope"
    if pd.notna(direction_da) and pd.notna(regression_da) and direction_da > regression_da + 0.03:
        recommendation = "shift from regression emphasis to direction emphasis"
    if best_group is not None and not group_scores.empty:
        if len(group_scores) >= 2:
            gap = float(group_scores.iloc[0][quality_da_column] - group_scores.iloc[-1][quality_da_column])
            if gap >= 0.03:
                recommendation = "reduce ticker universe to more favorable groups"
    if not model_scores.empty and len(model_scores) >= 4:
        leader = float(model_scores.iloc[0]["strong_directional_accuracy_share"])
        follower = float(model_scores.iloc[3]["strong_directional_accuracy_share"])
        if leader > follower + 0.10:
            recommendation = "freeze weak model families and focus on best few"
    if (
        pd.notna(policy_sharpe)
        and policy_sharpe < 0.0
        and pd.notna(strong_slice_share)
        and strong_slice_share < 0.25
        and pd.notna(edge_not_monetized_share)
        and edge_not_monetized_share < 0.35
    ):
        recommendation = "stop expansion and reconsider whether this repo has enough edge at daily frequency"

    return {
        "recommendation": recommendation,
        "best_feature_family": best_feature_family,
        "best_horizon": best_horizon,
        "best_group_name": best_group,
        "best_target_name": best_target_name,
        "top_models": top_models,
        "regression_mean_directional_accuracy": regression_da,
        "direction_mean_directional_accuracy": direction_da,
        "edge_but_not_monetized_share": edge_not_monetized_share,
        "median_policy_sharpe": policy_sharpe,
        "positive_policy_sharpe_share": positive_policy_sharpe_share,
        "mean_strong_slice_share": strong_slice_share,
        "mean_directional_accuracy_dispersion": instability,
        "phase3_blocked": True,
    }


def render_forecast_rehab_summary_markdown(
    manifest: dict[str, Any],
    assessment: dict[str, Any],
    feature_ablation_summary: pd.DataFrame,
    forecast_quality_summary: pd.DataFrame,
) -> str:
    """Render a compact markdown summary for Phase F1."""

    runtime = manifest.get("runtime", {})
    matrix = manifest.get("matrix", {})
    lines = [
        "# Forecast Rehabilitation Summary",
        "",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Commit: `{manifest.get('git', {}).get('commit_hash')}`",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Preset: `{matrix.get('preset')}`",
        f"- Groups: `{', '.join(group['group_name'] for group in matrix.get('ticker_groups', []))}`",
        f"- Horizons: `{', '.join(str(value) for value in matrix.get('horizons', []))}`",
        f"- Targets: `{', '.join(matrix.get('target_names', []))}`",
        f"- Feature families: `{', '.join(matrix.get('feature_families', []))}`",
        f"- Recommendation: `{assessment.get('recommendation')}`",
    ]

    if not feature_ablation_summary.empty:
        lines.extend(
            [
                "",
                "## Feature Families",
                "",
                "| feature_family | mean_rmse | mean_directional_accuracy | strong_slice_share |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        preview = (
            feature_ablation_summary.groupby("feature_family", sort=True)[["median_rmse", "median_directional_accuracy", "strong_slice_share"]]
            .median()
            .reset_index()
            .sort_values(["median_directional_accuracy", "strong_slice_share", "median_rmse"], ascending=[False, False, True])
        )
        for row in preview.head(10).itertuples(index=False):
            lines.append(
                f"| {row.feature_family} | {row.median_rmse:.6f} | {row.median_directional_accuracy:.6f} | {row.strong_slice_share:.6f} |"
            )

    if not forecast_quality_summary.empty:
        lines.extend(
            [
                "",
                "## Models",
                "",
                "| model_name | mean_rmse | mean_directional_accuracy | strong_directional_accuracy_share | tradable_slice_share |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        preview = (
            forecast_quality_summary.groupby("model_name", sort=True)[["median_rmse", "median_directional_accuracy", "strong_directional_accuracy_share", "tradable_slice_share"]]
            .median()
            .reset_index()
            .sort_values(["strong_directional_accuracy_share", "median_directional_accuracy", "median_rmse"], ascending=[False, False, True])
        )
        for row in preview.head(10).itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {row.median_rmse:.6f} | {row.median_directional_accuracy:.6f} | {row.strong_directional_accuracy_share:.6f} | {row.tradable_slice_share:.6f} |"
            )
    return "\n".join(lines)


def build_forecast_rehab_report(
    manifest: dict[str, Any],
    feature_inventory_summary: pd.DataFrame,
    feature_ablation_summary: pd.DataFrame,
    forecast_quality_summary: pd.DataFrame,
    model_stability_summary: pd.DataFrame,
    forecast_vs_policy_summary: pd.DataFrame,
    assessment: dict[str, Any],
) -> str:
    """Render the direct Phase F1 markdown report."""

    runtime = manifest.get("runtime", {})
    dependencies = manifest.get("dependency_versions", {})

    feature_family_scores = (
        feature_ablation_summary.groupby("feature_family", sort=True)[["median_directional_accuracy", "strong_slice_share", "median_rmse"]]
        .median()
        .reset_index()
        .sort_values(["median_directional_accuracy", "strong_slice_share", "median_rmse"], ascending=[False, False, True])
        if not feature_ablation_summary.empty
        else pd.DataFrame()
    )
    horizon_scores = (
        forecast_quality_summary.groupby("horizon", sort=True)[["median_directional_accuracy", "tradable_slice_share", "median_rmse"]]
        .median()
        .reset_index()
        .sort_values(["median_directional_accuracy", "tradable_slice_share", "median_rmse"], ascending=[False, False, True])
        if not forecast_quality_summary.empty
        else pd.DataFrame()
    )
    group_scores = (
        forecast_quality_summary.groupby("group_name", sort=True)[["median_directional_accuracy", "tradable_slice_share", "median_rmse"]]
        .median()
        .reset_index()
        .sort_values(["median_directional_accuracy", "tradable_slice_share", "median_rmse"], ascending=[False, False, True])
        if not forecast_quality_summary.empty
        else pd.DataFrame()
    )
    target_scores = (
        forecast_quality_summary.groupby(["target_name", "target_family"], sort=True)[["median_directional_accuracy", "tradable_slice_share", "median_rmse"]]
        .median()
        .reset_index()
        .sort_values(["median_directional_accuracy", "tradable_slice_share", "median_rmse"], ascending=[False, False, True])
        if not forecast_quality_summary.empty
        else pd.DataFrame()
    )
    model_scores = (
        forecast_quality_summary.groupby("model_name", sort=True)[["median_directional_accuracy", "strong_directional_accuracy_share", "median_rmse", "tradable_slice_share"]]
        .median()
        .reset_index()
        .sort_values(["strong_directional_accuracy_share", "median_directional_accuracy", "median_rmse"], ascending=[False, False, True])
        if not forecast_quality_summary.empty
        else pd.DataFrame()
    )

    best_feature = None if feature_family_scores.empty else feature_family_scores.iloc[0]
    best_horizon = None if horizon_scores.empty else horizon_scores.iloc[0]
    best_group = None if group_scores.empty else group_scores.iloc[0]
    best_target = None if target_scores.empty else target_scores.iloc[0]

    lines = [
        "# Forecast Rehabilitation Report",
        "",
        "## Scope",
        "",
        "- Reused the current walk-forward evaluator, forecast model registry, prepared `ml_5y` datasets, and the Phase 2.6 default execution baseline.",
        "- Added explicit target framing and feature-family ablations instead of further policy expansion.",
        "- Kept all work leakage-safe and benchmark-driven.",
        "",
        "## Runtime",
        "",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- statsmodels: `{dependencies.get('statsmodels')}`",
        f"- xgboost: `{dependencies.get('xgboost')}`",
        f"- lightgbm: `{dependencies.get('lightgbm')}`",
        "",
        "## Answers",
        "",
    ]

    if best_target is None:
        lines.append("1. Main problem: no target-family summary was produced.")
    else:
        lines.append(
            "1. Main problem: "
            f"best target family in this bounded sweep was `{best_target['target_name']}` "
            f"with median directional accuracy `{best_target['median_directional_accuracy']:.6f}`. "
            f"Mean directional-accuracy dispersion across setups was `{assessment.get('mean_directional_accuracy_dispersion', float('nan')):.6f}`, "
            "so instability remains part of the problem."
        )

    if best_feature is None:
        lines.append("2. Feature-set family: no feature-family ranking was produced.")
    else:
        lines.append(
            "2. Most promising feature-set family: "
            f"`{best_feature['feature_family']}` with median directional accuracy `{best_feature['median_directional_accuracy']:.6f}` "
            f"and strong-slice share `{best_feature['strong_slice_share']:.6f}`."
        )

    if best_horizon is None:
        lines.append("3. Horizon viability: no horizon ranking was produced.")
    else:
        lines.append(
            "3. Most viable horizon: "
            f"`{int(best_horizon['horizon'])}` with median directional accuracy `{best_horizon['median_directional_accuracy']:.6f}` "
            f"and tradable-slice share `{best_horizon['tradable_slice_share']:.6f}`."
        )

    if best_group is None:
        lines.append("4. Ticker-group viability: no group ranking was produced.")
    else:
        lines.append(
            "4. Most viable ticker group: "
            f"`{best_group['group_name']}` with median directional accuracy `{best_group['median_directional_accuracy']:.6f}` "
            f"and tradable-slice share `{best_group['tradable_slice_share']:.6f}`."
        )

    if model_scores.empty:
        lines.append("5. Models worth keeping: no model ranking was produced.")
    else:
        keep = ", ".join(model_scores.head(4)["model_name"].astype(str).tolist())
        lines.append(f"5. Models worth keeping: `{keep}` from the top of the strong-slice and directional-accuracy ranking.")

    regression_da = assessment.get("regression_mean_directional_accuracy", float("nan"))
    direction_da = assessment.get("direction_mean_directional_accuracy", float("nan"))
    if pd.notna(direction_da) and pd.notna(regression_da):
        framing = "direction framing" if direction_da > regression_da else "return regression"
        lines.append(
            f"6. More promising framing: `{framing}`. Mean directional accuracy was `{direction_da:.6f}` for direction framing versus `{regression_da:.6f}` for return-regression targets."
        )
    else:
        lines.append("6. Regression versus direction framing could not be ranked cleanly in this bounded run.")

    lines.append(
        "7. Continue rehab or reconsider the edge ceiling: "
        f"`{assessment.get('recommendation')}`."
    )

    if not feature_inventory_summary.empty:
        lines.extend(
            [
                "",
                "## Feature Inventory",
                "",
            ]
        )
        for row in feature_inventory_summary.itertuples(index=False):
            lines.append(
                f"- {row.feature_group}: count={row.feature_count}, selected_now={row.current_regression_selected_count + row.current_direction_selected_count}, usefulness={row.suspected_usefulness}, risk={row.suspected_risk}"
            )

    if not forecast_vs_policy_summary.empty:
        lines.extend(
            [
                "",
                "## Forecast Vs Policy",
                "",
                f"- Median policy Sharpe under the fixed Phase 2.6 execution baseline: `{assessment.get('median_policy_sharpe', float('nan')):.6f}`",
                f"- Positive-policy-Sharpe share under the fixed Phase 2.6 execution baseline: `{assessment.get('positive_policy_sharpe_share', float('nan')):.2%}`",
                f"- Edge-but-not-monetized share: `{assessment.get('edge_but_not_monetized_share', float('nan')):.2%}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Direct recommendation: `{assessment.get('recommendation')}`",
            f"- Best feature family: `{assessment.get('best_feature_family')}`",
            f"- Best horizon: `{assessment.get('best_horizon')}`",
            f"- Best group: `{assessment.get('best_group_name')}`",
            f"- Best target: `{assessment.get('best_target_name')}`",
            f"- Top models: `{', '.join(assessment.get('top_models', []))}`",
            f"- Phase 3 blocked: `{assessment.get('phase3_blocked')}`",
        ]
    )
    return "\n".join(lines)
