"""Generate Phase 6 robustness and statistical-significance artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - incomplete runtime only
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.statistics.bootstrap_eval import (  # noqa: E402
    bootstrap_hit_ratio_ci,
    bootstrap_mean_ci,
    bootstrap_metric_ci,
)
from src.ml.statistics.dm_test import diebold_mariano_test  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments"
FORECASTING_REPORT_DIR = REPO_ROOT / "reports" / "forecasting_core"
RISK_AWARE_REPORT_DIR = REPO_ROOT / "reports" / "risk_aware"
REGIME_REPORT_DIR = REPO_ROOT / "reports" / "regime_analysis"
FEATURE_REPORT_DIR = REPO_ROOT / "reports" / "feature_analysis"
DISCLAIMER = (
    "All Phase 6 outputs are robustness and statistical research artifacts only. "
    "They are not BUY / SELL / HOLD advice, capital allocation guidance, broker "
    "execution instructions, portfolio recommendations, statistical proof of "
    "future profitability, or proof of guaranteed profitable trading."
)
METRIC_DIRECTIONS = {
    "mae": "lower",
    "rmse": "lower",
    "mape": "lower",
    "directional_accuracy": "higher",
    "prediction_count": "higher",
    "coverage_count": "higher",
    "missing_prediction_rate": "lower",
}
ROBUSTNESS_NOTE_SIGNIFICANCE = (
    "The observed difference is not statistically significant under this test; "
    "it should not be treated as robust evidence of superiority."
)
WIDE_CI_NOTE = "The confidence interval is wide, so the estimate is uncertain and should be interpreted cautiously."
COST_EDGE_NOTE = (
    "The diagnostic edge weakens or disappears under cost/slippage assumptions, "
    "so the result should not be interpreted as executable strategy evidence."
)
WEAK_ROBUSTNESS_NOTE = "The result is sensitive to configuration choices and should be treated as exploratory."
STRONG_ROBUSTNESS_NOTE = (
    "The result is stable across the tested settings, but it remains diagnostic research evidence "
    "rather than investment advice."
)


WINDOW_COLUMNS = [
    "experiment_id",
    "setting_id",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "metric_name",
    "metric_value",
    "sample_size",
    "rank",
    "best_baseline_value",
    "model_vs_best_baseline_delta",
    "robustness_note",
]
UNIVERSE_COLUMNS = [
    "experiment_id",
    "universe_group",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "metric_name",
    "metric_value",
    "sample_size",
    "rank",
    "best_baseline_value",
    "model_vs_best_baseline_delta",
    "robustness_note",
]
COST_COLUMNS = [
    "experiment_id",
    "cost_scenario_id",
    "policy_id",
    "candidate_type",
    "top_n",
    "horizon",
    "transaction_cost_bps",
    "slippage_bps",
    "gross_average_realized_return",
    "net_average_realized_return",
    "net_hit_ratio",
    "net_return_volatility_proxy",
    "net_max_drawdown",
    "net_var_95",
    "net_cvar_95",
    "cost_impact",
    "diagnostic_only",
]
DM_COLUMNS = [
    "experiment_id",
    "source_experiment",
    "ticker",
    "horizon",
    "model_name",
    "baseline_name",
    "loss",
    "dm_statistic",
    "p_value",
    "mean_loss_model",
    "mean_loss_baseline",
    "mean_loss_diff",
    "effect_size",
    "sample_size",
    "significant_05",
    "significant_10",
    "warning",
]
BOOTSTRAP_COLUMNS = [
    "experiment_id",
    "source_artifact",
    "group_key",
    "policy_id",
    "candidate_type",
    "top_n",
    "horizon",
    "metric_name",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence",
    "n_bootstrap",
    "sample_size",
    "seed",
    "warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 6 robustness report artifacts.")
    parser.add_argument("--configs", nargs="+", required=True, help="Phase 6 experiment config paths")
    parser.add_argument("--output", required=True, help="Output directory, normally reports/robustness")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.configs)
    window = build_window_robustness(configs.get("EXP-RB-001", {}))
    universe = build_universe_robustness(configs.get("EXP-RB-002", {}))
    cost = build_cost_sensitivity(configs.get("EXP-RB-003", {}))
    dm_results = build_dm_results(configs.get("EXP-ST-001", {}))
    bootstrap = build_bootstrap_ci(configs.get("EXP-ST-002", {}))
    robustness_summary = build_robustness_summary(window, universe, cost)
    significance_summary = build_significance_summary(dm_results, bootstrap)
    effect_size_summary = build_effect_size_summary(dm_results)

    write_csv(output_dir / "window_robustness.csv", window)
    write_csv(output_dir / "universe_robustness.csv", universe)
    write_csv(output_dir / "cost_sensitivity.csv", cost)
    write_csv(output_dir / "dm_test_results.csv", dm_results)
    write_csv(output_dir / "bootstrap_ci.csv", bootstrap)
    write_csv(output_dir / "robustness_summary.csv", robustness_summary)
    write_csv(output_dir / "statistical_significance_summary.csv", significance_summary)
    write_csv(output_dir / "effect_size_summary.csv", effect_size_summary)

    chart_notes = generate_charts(output_dir / "charts", window, universe, cost, dm_results, bootstrap, effect_size_summary)
    write_task_reports(output_dir, configs, window, universe, cost, dm_results, bootstrap)
    main_report = render_main_report(
        configs=configs,
        window=window,
        universe=universe,
        cost=cost,
        dm_results=dm_results,
        bootstrap=bootstrap,
        robustness_summary=robustness_summary,
        significance_summary=significance_summary,
        effect_size_summary=effect_size_summary,
        chart_notes=chart_notes,
    )
    (output_dir / "ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE_REPORT.md").write_text(main_report, encoding="utf-8")
    write_optional_doc(output_dir, robustness_summary, significance_summary)
    print(f"Wrote Phase 6 robustness artifacts to {output_dir}")
    return 0


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_configs(paths: list[str]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        config = read_yaml(resolve_repo_path(raw_path))
        experiment_id = str(config.get("experiment", {}).get("id") or Path(raw_path).stem)
        configs[experiment_id] = config
    return configs


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Phase 6 configs.")
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_window_robustness(config: dict[str, Any]) -> pd.DataFrame:
    experiment_id = str(config.get("experiment", {}).get("id") or "EXP-RB-001")
    metrics = list(config.get("evaluation", {}).get("metrics") or [])
    tickers = [str(value) for value in config.get("data", {}).get("universe", [])]
    horizons = [int(value) for value in config.get("target", {}).get("horizons", [])]
    model_names = [str(value) for value in config.get("models", {}).get("include", [])]
    baseline_names = [str(value) for value in config.get("baselines", {}).get("include", [])]
    source_ids = [str(value) for value in config.get("sensitivity", {}).get("source_experiment_priority", [])]
    if not source_ids:
        source_ids = ["EXP-FC-003", "EXP-FC-001"]
    source_predictions = load_prediction_artifacts(source_ids)
    source_configs = {source_id: load_source_config(source_id) for source_id in source_ids}

    rows: list[dict[str, Any]] = []
    for window in config.get("sensitivity", {}).get("windows", []):
        setting_id = str(window.get("id") or "unknown_window")
        source_id, source_note = find_exact_window_source(window, source_configs)
        if not source_id:
            rows.append(empty_window_row(experiment_id, setting_id, source_note))
            continue
        subset = source_predictions[source_predictions["source_experiment"] == source_id].copy()
        if subset.empty:
            rows.append(empty_window_row(experiment_id, setting_id, f"missing_predictions_for_source:{source_id}"))
            continue
        subset = filter_predictions(
            subset,
            tickers=tickers,
            horizons=horizons,
            model_names=model_names,
            baseline_names=baseline_names,
            start=window.get("test_start"),
            end=window.get("test_end"),
        )
        if subset.empty:
            rows.append(empty_window_row(experiment_id, setting_id, f"no_prediction_rows_in_test_window:{source_id}"))
            continue
        computed = compute_prediction_metrics(subset, metrics)
        ranked = add_metric_ranks(computed, ["ticker", "horizon", "metric_name"])
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "setting_id": setting_id,
                    "ticker": row["ticker"],
                    "horizon": row["horizon"],
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "metric_name": row["metric_name"],
                    "metric_value": row["metric_value"],
                    "sample_size": row["sample_size"],
                    "rank": row["rank"],
                    "best_baseline_value": row["best_baseline_value"],
                    "model_vs_best_baseline_delta": row["model_vs_best_baseline_delta"],
                    "robustness_note": f"computed_from_source_predictions:{source_id};{source_note}",
                }
            )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def empty_window_row(experiment_id: str, setting_id: str, note: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "setting_id": setting_id,
        "ticker": "",
        "horizon": "",
        "model_name": "",
        "model_type": "",
        "metric_name": "",
        "metric_value": np.nan,
        "sample_size": 0,
        "rank": np.nan,
        "best_baseline_value": np.nan,
        "model_vs_best_baseline_delta": np.nan,
        "robustness_note": f"{note}; no fake rerun or substituted window metric was created",
    }


def find_exact_window_source(window: dict[str, Any], source_configs: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    exact = str(window.get("exact_source_experiment") or "")
    if exact and exact in source_configs:
        if source_matches_window(source_configs[exact], window):
            return exact, "exact_train_test_window_match"
        return None, f"declared_source_train_test_mismatch:{exact}"
    for source_id, source_config in source_configs.items():
        if source_matches_window(source_config, window):
            return source_id, "exact_train_test_window_match"
    return None, "no_exact_local_prediction_artifact_matching_train_test_window"


def source_matches_window(source_config: dict[str, Any], window: dict[str, Any]) -> bool:
    evaluation = source_config.get("evaluation", {}) or {}
    data = source_config.get("data", {}) or {}
    checks = {
        "train_start": evaluation.get("train_start"),
        "train_end": evaluation.get("train_end"),
        "test_start": evaluation.get("test_start"),
        "test_end": evaluation.get("test_end"),
    }
    if window.get("data_start") and data.get("start_date"):
        checks["data_start"] = data.get("start_date")
    if window.get("data_end") and data.get("end_date"):
        checks["data_end"] = data.get("end_date")
    return all(str(checks[key]) == str(window.get(key)) for key in checks)


def load_source_config(experiment_id: str) -> dict[str, Any]:
    for relative in ("config/resolved_config.yaml", "config/original_config.yaml"):
        path = OUTPUT_ROOT / experiment_id / relative
        if path.exists():
            return read_yaml(path)
    config_path = REPO_ROOT / "configs" / "experiments" / f"{experiment_id}.yaml"
    if config_path.exists():
        return read_yaml(config_path)
    return {}


def build_universe_robustness(config: dict[str, Any]) -> pd.DataFrame:
    experiment_id = str(config.get("experiment", {}).get("id") or "EXP-RB-002")
    metrics = list(config.get("evaluation", {}).get("metrics") or [])
    horizons = [int(value) for value in config.get("target", {}).get("horizons", [])]
    model_names = [str(value) for value in config.get("models", {}).get("include", [])]
    baseline_names = [str(value) for value in config.get("baselines", {}).get("include", [])]
    source = read_csv(FORECASTING_REPORT_DIR / "forecast_metrics.csv")
    if source.empty:
        source = load_metric_artifacts(["EXP-FC-003", "EXP-FC-001"])
    elif "experiment_id" in source.columns and (source["experiment_id"].astype(str) == "EXP-FC-003").any():
        source = source[source["experiment_id"].astype(str) == "EXP-FC-003"].copy()
    if source.empty:
        return pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "universe_group": "all",
                    "ticker": "",
                    "horizon": "",
                    "model_name": "",
                    "model_type": "",
                    "metric_name": "",
                    "metric_value": np.nan,
                    "sample_size": 0,
                    "rank": np.nan,
                    "best_baseline_value": np.nan,
                    "model_vs_best_baseline_delta": np.nan,
                    "robustness_note": "forecast_metrics_unavailable",
                }
            ],
            columns=UNIVERSE_COLUMNS,
        )

    frame = source.copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce")
    rows: list[dict[str, Any]] = []
    groups = config.get("universe_groups", {}) or {}
    for group_name, tickers in groups.items():
        tickers = [str(value) for value in tickers]
        subset = frame[
            frame["ticker"].astype(str).isin(tickers)
            & frame["horizon"].isin(horizons)
            & frame["metric_name"].astype(str).isin(metrics)
            & (
                frame["model_name"].astype(str).isin(model_names)
                | frame["model_name"].astype(str).isin(baseline_names)
            )
        ].copy()
        if subset.empty:
            rows.append(empty_universe_row(experiment_id, str(group_name), "no_rows_for_universe_group"))
            continue
        ranked = add_metric_ranks(subset, ["ticker", "horizon", "metric_name"])
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "universe_group": str(group_name),
                    "ticker": row["ticker"],
                    "horizon": row["horizon"],
                    "model_name": row["model_name"],
                    "model_type": row["model_type"],
                    "metric_name": row["metric_name"],
                    "metric_value": row["metric_value"],
                    "sample_size": row["sample_size"],
                    "rank": row["rank"],
                    "best_baseline_value": row["best_baseline_value"],
                    "model_vs_best_baseline_delta": row["model_vs_best_baseline_delta"],
                    "robustness_note": "computed_from_forecasting_core_metrics_by_ticker_group",
                }
            )
    return pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)


def empty_universe_row(experiment_id: str, universe_group: str, note: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "universe_group": universe_group,
        "ticker": "",
        "horizon": "",
        "model_name": "",
        "model_type": "",
        "metric_name": "",
        "metric_value": np.nan,
        "sample_size": 0,
        "rank": np.nan,
        "best_baseline_value": np.nan,
        "model_vs_best_baseline_delta": np.nan,
        "robustness_note": note,
    }


def build_cost_sensitivity(config: dict[str, Any]) -> pd.DataFrame:
    experiment_id = str(config.get("experiment", {}).get("id") or "EXP-RB-003")
    period_path = resolve_repo_path(
        config.get("source_artifacts", {}).get(
            "basket_period_returns",
            "outputs/experiments/EXP-RK-002/artifacts/basket_period_returns.csv",
        )
    )
    period_returns = read_csv(period_path)
    rows: list[dict[str, Any]] = []
    if period_returns.empty:
        topn_path = resolve_repo_path(
            config.get("source_artifacts", {}).get("topn_basket_metrics", "reports/risk_aware/topn_basket_metrics.csv")
        )
        return cost_from_aggregate_baskets(experiment_id, config, read_csv(topn_path))

    frame = period_returns.copy()
    frame["period_realized_return"] = pd.to_numeric(frame["period_realized_return"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["top_n"] = pd.to_numeric(frame["top_n"], errors="coerce")
    horizons = [int(value) for value in config.get("basket", {}).get("horizons", [])]
    top_ns = [int(value) for value in config.get("basket", {}).get("top_n", [])]
    if horizons:
        frame = frame[frame["horizon"].isin(horizons)]
    if top_ns:
        frame = frame[frame["top_n"].isin(top_ns)]

    for scenario in config.get("cost_scenarios", []):
        transaction = float(scenario.get("transaction_cost_bps") or 0.0)
        slippage = float(scenario.get("slippage_bps") or 0.0)
        cost_rate = (transaction + slippage) / 10000.0
        for keys, group in frame.groupby(["policy_id", "candidate_type", "top_n", "horizon"], dropna=False, sort=True):
            policy_id, candidate_type, top_n, horizon = keys
            gross = pd.to_numeric(group["period_realized_return"], errors="coerce").dropna()
            net = gross - cost_rate
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "cost_scenario_id": str(scenario.get("id") or f"cost_{transaction + slippage:g}bps"),
                    "policy_id": str(policy_id),
                    "candidate_type": str(candidate_type),
                    "top_n": int(top_n),
                    "horizon": int(horizon),
                    "transaction_cost_bps": transaction,
                    "slippage_bps": slippage,
                    "gross_average_realized_return": float(gross.mean()) if not gross.empty else np.nan,
                    "net_average_realized_return": float(net.mean()) if not net.empty else np.nan,
                    "net_hit_ratio": float((net > 0).mean()) if not net.empty else np.nan,
                    "net_return_volatility_proxy": return_volatility_proxy(net),
                    "net_max_drawdown": max_drawdown_from_returns(net),
                    "net_var_95": var_from_returns(net, 0.95),
                    "net_cvar_95": cvar_from_returns(net, 0.95),
                    "cost_impact": float(gross.mean() - net.mean()) if not gross.empty else np.nan,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows, columns=COST_COLUMNS)


def cost_from_aggregate_baskets(experiment_id: str, config: dict[str, Any], baskets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if baskets.empty:
        return pd.DataFrame(columns=COST_COLUMNS)
    for scenario in config.get("cost_scenarios", []):
        transaction = float(scenario.get("transaction_cost_bps") or 0.0)
        slippage = float(scenario.get("slippage_bps") or 0.0)
        cost_rate = (transaction + slippage) / 10000.0
        for _, row in baskets.iterrows():
            gross = to_float(row.get("average_realized_return"))
            net = gross - cost_rate if gross is not None else np.nan
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "cost_scenario_id": str(scenario.get("id") or f"cost_{transaction + slippage:g}bps"),
                    "policy_id": row.get("policy_id"),
                    "candidate_type": row.get("candidate_type"),
                    "top_n": row.get("top_n"),
                    "horizon": row.get("horizon"),
                    "transaction_cost_bps": transaction,
                    "slippage_bps": slippage,
                    "gross_average_realized_return": gross,
                    "net_average_realized_return": net,
                    "net_hit_ratio": np.nan,
                    "net_return_volatility_proxy": np.nan,
                    "net_max_drawdown": np.nan,
                    "net_var_95": np.nan,
                    "net_cvar_95": np.nan,
                    "cost_impact": cost_rate if gross is not None else np.nan,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows, columns=COST_COLUMNS)


def build_dm_results(config: dict[str, Any]) -> pd.DataFrame:
    experiment_id = str(config.get("experiment", {}).get("id") or "EXP-ST-001")
    source_cfg = config.get("source_artifacts", {}) or {}
    source_ids = [str(value) for value in source_cfg.get("forecasting_experiments", [])]
    source_ids.extend(str(value) for value in source_cfg.get("feature_experiments", []))
    comparisons = config.get("comparisons", {}) or {}
    models = [str(value) for value in comparisons.get("models", [])]
    baselines = [str(value) for value in comparisons.get("baselines", [])]
    losses = [str(value) for value in comparisons.get("losses", [])]
    horizons = [int(value) for value in comparisons.get("horizons", [])]
    predictions = load_prediction_artifacts(source_ids)
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=DM_COLUMNS)

    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    for source_id in source_ids:
        source = predictions[predictions["source_experiment"] == source_id].copy()
        if source.empty:
            continue
        tickers = sorted(source["ticker"].dropna().astype(str).unique().tolist())
        source_horizons = sorted(set(horizons) & set(source["horizon"].dropna().astype(int).unique().tolist()))
        for ticker in tickers:
            for horizon in source_horizons:
                for model_name in models:
                    for baseline_name in baselines:
                        model_rows = select_prediction_group(source, ticker, horizon, model_name, "model")
                        baseline_rows = select_prediction_group(source, ticker, horizon, baseline_name, "baseline")
                        for loss in losses:
                            row = {
                                "experiment_id": experiment_id,
                                "source_experiment": source_id,
                                "ticker": ticker,
                                "horizon": int(horizon),
                                "model_name": model_name,
                                "baseline_name": baseline_name,
                                "loss": loss,
                                "dm_statistic": np.nan,
                                "p_value": np.nan,
                                "mean_loss_model": np.nan,
                                "mean_loss_baseline": np.nan,
                                "mean_loss_diff": np.nan,
                                "effect_size": np.nan,
                                "sample_size": 0,
                                "significant_05": False,
                                "significant_10": False,
                                "warning": "",
                            }
                            if model_rows.empty or baseline_rows.empty:
                                row["warning"] = "missing_model_or_baseline_prediction_rows"
                                rows.append(row)
                                continue
                            merged = model_rows.merge(
                                baseline_rows,
                                on=["date", "ticker", "horizon"],
                                suffixes=("_model", "_baseline"),
                            )
                            if merged.empty:
                                row["warning"] = "no_aligned_model_baseline_dates"
                                rows.append(row)
                                continue
                            errors_model = (
                                pd.to_numeric(merged["y_pred_model"], errors="coerce")
                                - pd.to_numeric(merged["y_true_model"], errors="coerce")
                            )
                            errors_baseline = (
                                pd.to_numeric(merged["y_pred_baseline"], errors="coerce")
                                - pd.to_numeric(merged["y_true_baseline"], errors="coerce")
                            )
                            result = diebold_mariano_test(
                                errors_model,
                                errors_baseline,
                                loss=loss,
                                horizon=int(horizon),
                                alternative="two_sided",
                            )
                            row.update(
                                {
                                    "dm_statistic": result["dm_statistic"],
                                    "p_value": result["p_value"],
                                    "mean_loss_model": result["mean_loss_model"],
                                    "mean_loss_baseline": result["mean_loss_baseline"],
                                    "mean_loss_diff": result["mean_loss_diff"],
                                    "effect_size": result["effect_size"],
                                    "sample_size": result["sample_size"],
                                    "significant_05": result["significant_05"],
                                    "significant_10": result["significant_10"],
                                    "warning": result["warning"],
                                }
                            )
                            rows.append(row)
    return pd.DataFrame(rows, columns=DM_COLUMNS)


def build_bootstrap_ci(config: dict[str, Any]) -> pd.DataFrame:
    experiment_id = str(config.get("experiment", {}).get("id") or "EXP-ST-002")
    bootstrap_cfg = config.get("bootstrap", {}) or {}
    n_bootstrap = int(bootstrap_cfg.get("n_bootstrap") or 1000)
    confidence = float(bootstrap_cfg.get("confidence") or 0.95)
    seed = int(bootstrap_cfg.get("seed") or config.get("experiment", {}).get("seed") or 42)
    source_path = resolve_repo_path(
        config.get("source_artifacts", {}).get(
            "basket_period_returns",
            "outputs/experiments/EXP-RK-002/artifacts/basket_period_returns.csv",
        )
    )
    period_returns = read_csv(source_path)
    if period_returns.empty:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    frame = period_returns.copy()
    frame["period_realized_return"] = pd.to_numeric(frame["period_realized_return"], errors="coerce")
    rows: list[dict[str, Any]] = []
    metrics = list(bootstrap_cfg.get("metrics") or [])
    for keys, group in frame.groupby(["policy_id", "candidate_type", "top_n", "horizon"], dropna=False, sort=True):
        policy_id, candidate_type, top_n, horizon = keys
        returns = pd.to_numeric(group["period_realized_return"], errors="coerce")
        group_key = f"policy_id={policy_id}|candidate_type={candidate_type}|top_n={top_n}|horizon={horizon}"
        for metric_name in metrics:
            result = bootstrap_result_for_metric(
                metric_name,
                returns,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            )
            result["metric_name"] = metric_name
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "source_artifact": str(source_path.relative_to(REPO_ROOT)),
                    "group_key": group_key,
                    "policy_id": str(policy_id),
                    "candidate_type": str(candidate_type),
                    "top_n": int(top_n),
                    "horizon": int(horizon),
                    "metric_name": metric_name,
                    "estimate": result["estimate"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "confidence": result["confidence"],
                    "n_bootstrap": result["n_bootstrap"],
                    "sample_size": result["sample_size"],
                    "seed": result["seed"],
                    "warning": result["warning"],
                }
            )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def bootstrap_result_for_metric(
    metric_name: str,
    returns: pd.Series,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if metric_name == "average_realized_return":
        return bootstrap_mean_ci(returns, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    if metric_name == "hit_ratio":
        return bootstrap_hit_ratio_ci(returns, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    metric_functions: dict[str, Any] = {
        "return_volatility_proxy": return_volatility_proxy,
        "max_drawdown": max_drawdown_from_returns,
        "var_95": lambda sample: var_from_returns(pd.Series(sample), 0.95),
        "cvar_95": lambda sample: cvar_from_returns(pd.Series(sample), 0.95),
    }
    fn = metric_functions.get(metric_name)
    if fn is None:
        return bootstrap_mean_ci(returns, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    return bootstrap_metric_ci(
        returns,
        metric_fn=lambda sample: float(fn(pd.Series(sample))),
        metric_name=metric_name,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def load_prediction_artifacts(experiment_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id in experiment_ids:
        path = OUTPUT_ROOT / experiment_id / "predictions" / "predictions.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_experiment"] = experiment_id
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_metric_artifacts(experiment_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id in experiment_ids:
        path = OUTPUT_ROOT / experiment_id / "metrics" / "metrics.csv"
        frame = read_csv(path)
        if frame.empty:
            continue
        frame["source_experiment"] = experiment_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_predictions(
    frame: pd.DataFrame,
    tickers: list[str],
    horizons: list[int],
    model_names: list[str],
    baseline_names: list[str],
    start: Any,
    end: Any,
) -> pd.DataFrame:
    result = frame.copy()
    if tickers:
        result = result[result["ticker"].astype(str).isin(tickers)]
    if horizons:
        result = result[result["horizon"].astype(int).isin(horizons)]
    allowed = set(model_names) | set(baseline_names)
    if allowed:
        result = result[result["model_name"].astype(str).isin(allowed)]
    if start:
        result = result[result["date"] >= pd.Timestamp(start)]
    if end:
        result = result[result["date"] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)


def select_prediction_group(
    frame: pd.DataFrame,
    ticker: str,
    horizon: int,
    model_name: str,
    model_type: str,
) -> pd.DataFrame:
    return frame[
        (frame["ticker"].astype(str) == str(ticker))
        & (frame["horizon"].astype(int) == int(horizon))
        & (frame["model_name"].astype(str) == str(model_name))
        & (frame["model_type"].astype(str) == str(model_type))
    ].copy()


def compute_prediction_metrics(predictions: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = predictions.copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    for keys, group in frame.groupby(["ticker", "horizon", "model_name", "model_type"], dropna=False, sort=True):
        ticker, horizon, model_name, model_type = keys
        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        y_pred = pd.to_numeric(group["y_pred"], errors="coerce")
        valid = y_true.notna() & y_pred.notna()
        errors = y_pred[valid] - y_true[valid]
        total_count = int(len(group))

        def add(metric_name: str, value: float | None, sample_size: int) -> None:
            if metrics and metric_name not in metrics:
                return
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": int(horizon),
                    "model_name": model_name,
                    "model_type": model_type,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "sample_size": sample_size,
                }
            )

        if valid.any():
            add("mae", float(errors.abs().mean()), int(valid.sum()))
            add("rmse", float(math.sqrt(np.square(errors).mean())), int(valid.sum()))
            denominator = y_true[valid].abs()
            mape_mask = denominator > 1e-12
            add(
                "mape",
                float((errors[mape_mask].abs() / denominator[mape_mask]).mean() * 100.0) if mape_mask.any() else np.nan,
                int(mape_mask.sum()),
            )
        else:
            add("mae", np.nan, 0)
            add("rmse", np.nan, 0)
            add("mape", np.nan, 0)
        if {"actual_direction", "predicted_direction"}.issubset(group.columns):
            actual = pd.to_numeric(group.loc[valid, "actual_direction"], errors="coerce")
            predicted = pd.to_numeric(group.loc[valid, "predicted_direction"], errors="coerce")
            direction_mask = actual.notna() & predicted.notna()
            value = float((np.sign(actual[direction_mask]) == np.sign(predicted[direction_mask])).mean()) if direction_mask.any() else np.nan
            add("directional_accuracy", value, int(direction_mask.sum()))
        add("prediction_count", float(total_count), total_count)
        add("missing_prediction_rate", float(y_pred.isna().mean()) if total_count else np.nan, total_count)
    return pd.DataFrame(rows)


def add_metric_ranks(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["metric_value"] = pd.to_numeric(work["metric_value"], errors="coerce")
    work["sample_size"] = pd.to_numeric(work["sample_size"], errors="coerce")
    for _, group in work.groupby(group_keys, dropna=False, sort=True):
        metric_name = str(group["metric_name"].iloc[0])
        direction = METRIC_DIRECTIONS.get(metric_name, "lower")
        ascending = direction == "lower"
        ranked = group.dropna(subset=["metric_value"]).sort_values(
            ["metric_value", "model_name"],
            ascending=[ascending, True],
        )
        ranked["rank"] = range(1, len(ranked) + 1)
        baseline_values = ranked.loc[ranked["model_type"] == "baseline", "metric_value"]
        best_baseline = float(baseline_values.min() if ascending else baseline_values.max()) if not baseline_values.empty else np.nan
        for _, row in ranked.iterrows():
            delta = np.nan
            if pd.notna(best_baseline):
                delta = float(row["metric_value"] - best_baseline)
            rows.append(
                {
                    **row.to_dict(),
                    "rank": int(row["rank"]),
                    "best_baseline_value": best_baseline,
                    "model_vs_best_baseline_delta": delta,
                }
            )
        dropped = group[group["metric_value"].isna()]
        for _, row in dropped.iterrows():
            rows.append(
                {
                    **row.to_dict(),
                    "rank": np.nan,
                    "best_baseline_value": np.nan,
                    "model_vs_best_baseline_delta": np.nan,
                }
            )
    return pd.DataFrame(rows)


def return_volatility_proxy(returns: pd.Series | np.ndarray) -> float:
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if values.empty:
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        volatility = values.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return np.nan
    return float(values.mean() / volatility)


def max_drawdown_from_returns(returns: pd.Series | np.ndarray) -> float:
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def var_from_returns(returns: pd.Series | np.ndarray, confidence: float) -> float:
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(1.0 - confidence))


def cvar_from_returns(returns: pd.Series | np.ndarray, confidence: float) -> float:
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if values.empty:
        return np.nan
    var_value = var_from_returns(values, confidence)
    tail = values[values <= var_value]
    return float(tail.mean()) if not tail.empty else np.nan


def build_robustness_summary(window: pd.DataFrame, universe: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact_window_rows = window[window["robustness_note"].astype(str).str.contains("exact_train_test_window_match", na=False)]
    rows.append(
        {
            "section": "window",
            "item": "settings_with_exact_source_rows",
            "value": exact_window_rows["setting_id"].nunique() if not exact_window_rows.empty else 0,
            "note": "Only exact local train/test artifacts are used for window metrics.",
        }
    )
    rows.append(
        {
            "section": "window",
            "item": "settings_with_missing_exact_source",
            "value": window.loc[window["sample_size"].fillna(0).astype(float) == 0, "setting_id"].nunique(),
            "note": WEAK_ROBUSTNESS_NOTE,
        }
    )
    window_mae = window[(window["metric_name"] == "mae") & pd.to_numeric(window["metric_value"], errors="coerce").notna()].copy()
    if not window_mae.empty:
        values = pd.to_numeric(window_mae["metric_value"], errors="coerce")
        rows.extend(
            [
                {
                    "section": "window",
                    "item": "mae_mean_performance",
                    "value": float(values.mean()),
                    "note": "Mean MAE across exact-source window rows.",
                },
                {
                    "section": "window",
                    "item": "mae_worst_case_performance",
                    "value": float(values.max()),
                    "note": "Worst-case MAE across exact-source window rows; lower is better.",
                },
                {
                    "section": "window",
                    "item": "mae_metric_variance",
                    "value": float(values.var(ddof=0)),
                    "note": "Variance across available exact-source MAE rows, not across missing window reruns.",
                },
                {
                    "section": "window",
                    "item": "mae_rank_stability_std",
                    "value": float(pd.to_numeric(window_mae["rank"], errors="coerce").std(ddof=0)),
                    "note": "Rank spread across available exact-source MAE rows; full window stability needs missing reruns.",
                },
            ]
        )
    for group_name, group in universe.groupby("universe_group", dropna=False, sort=True):
        winners = group[(pd.to_numeric(group["rank"], errors="coerce") == 1) & group["metric_name"].isin(["mae", "rmse"])]
        baseline_winners = int((winners["model_type"] == "baseline").sum()) if not winners.empty else 0
        rows.append(
            {
                "section": "universe",
                "item": f"{group_name}_baseline_winners_mae_rmse",
                "value": baseline_winners,
                "note": f"{baseline_winners} of {len(winners)} MAE/RMSE first-rank rows are baselines.",
            }
        )
        group_mae = group[(group["metric_name"] == "mae") & pd.to_numeric(group["metric_value"], errors="coerce").notna()]
        if not group_mae.empty:
            values = pd.to_numeric(group_mae["metric_value"], errors="coerce")
            rows.append(
                {
                    "section": "universe",
                    "item": f"{group_name}_mae_mean_performance",
                    "value": float(values.mean()),
                    "note": "Mean MAE across this universe group.",
                }
            )
            rows.append(
                {
                    "section": "universe",
                    "item": f"{group_name}_mae_worst_case_performance",
                    "value": float(values.max()),
                    "note": "Worst-case MAE across this universe group; lower is better.",
                }
            )
            rows.append(
                {
                    "section": "universe",
                    "item": f"{group_name}_mae_metric_variance",
                    "value": float(values.var(ddof=0)),
                    "note": "Metric variance across ticker/horizon/model rows in this universe group.",
                }
            )
    if not cost.empty:
        for scenario_id, group in cost.groupby("cost_scenario_id", sort=True):
            nonpositive = int((pd.to_numeric(group["net_average_realized_return"], errors="coerce") <= 0).sum())
            rows.append(
                {
                    "section": "cost",
                    "item": f"{scenario_id}_nonpositive_net_average_rows",
                    "value": nonpositive,
                    "note": COST_EDGE_NOTE if nonpositive else STRONG_ROBUSTNESS_NOTE,
                }
            )
    return pd.DataFrame(rows)


def build_significance_summary(dm_results: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not dm_results.empty:
        rows.append(
            {
                "section": "dm_test",
                "item": "comparison_count",
                "value": int(len(dm_results)),
                "note": "Model/baseline/loss comparisons attempted.",
            }
        )
        rows.append(
            {
                "section": "dm_test",
                "item": "significant_05_count",
                "value": int(dm_results["significant_05"].fillna(False).astype(bool).sum()),
                "note": "Significant at p < 0.05; interpret with sample and multiple-comparison caution.",
            }
        )
        rows.append(
            {
                "section": "dm_test",
                "item": "significant_10_count",
                "value": int(dm_results["significant_10"].fillna(False).astype(bool).sum()),
                "note": "Significant at p < 0.10; weaker evidence than 5%.",
            }
        )
        non_sig = dm_results[pd.to_numeric(dm_results["p_value"], errors="coerce") >= 0.10]
        rows.append(
            {
                "section": "dm_test",
                "item": "non_significant_10_count",
                "value": int(len(non_sig)),
                "note": ROBUSTNESS_NOTE_SIGNIFICANCE,
            }
        )
        warnings = dm_results["warning"].fillna("").astype(str).ne("").sum()
        rows.append(
            {
                "section": "dm_test",
                "item": "warning_count",
                "value": int(warnings),
                "note": "Warnings include missing aligned rows, small samples, or invalid variance.",
            }
        )
    if not bootstrap.empty:
        return_ci = bootstrap[bootstrap["metric_name"].isin(["average_realized_return", "hit_ratio"])]
        wide = return_ci.apply(is_wide_ci, axis=1).sum()
        rows.append(
            {
                "section": "bootstrap",
                "item": "ci_rows",
                "value": int(len(bootstrap)),
                "note": "Bootstrap intervals computed from basket period returns.",
            }
        )
        rows.append(
            {
                "section": "bootstrap",
                "item": "wide_or_overlapping_key_ci_rows",
                "value": int(wide),
                "note": WIDE_CI_NOTE,
            }
        )
    return pd.DataFrame(rows)


def build_effect_size_summary(dm_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "baseline_name",
        "loss",
        "comparison_count",
        "mean_effect_size",
        "median_effect_size",
        "mean_loss_diff",
        "median_p_value",
        "significant_05_count",
        "significant_10_count",
        "warning_count",
    ]
    if dm_results.empty:
        return pd.DataFrame(columns=columns)
    frame = dm_results.copy()
    for column in ("effect_size", "mean_loss_diff", "p_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["model_name", "baseline_name", "loss"], sort=True):
        model_name, baseline_name, loss = keys
        rows.append(
            {
                "model_name": model_name,
                "baseline_name": baseline_name,
                "loss": loss,
                "comparison_count": int(len(group)),
                "mean_effect_size": numeric_mean(group, "effect_size"),
                "median_effect_size": numeric_median(group, "effect_size"),
                "mean_loss_diff": numeric_mean(group, "mean_loss_diff"),
                "median_p_value": numeric_median(group, "p_value"),
                "significant_05_count": int(group["significant_05"].fillna(False).astype(bool).sum()),
                "significant_10_count": int(group["significant_10"].fillna(False).astype(bool).sum()),
                "warning_count": int(group["warning"].fillna("").astype(str).ne("").sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def generate_charts(
    chart_dir: Path,
    window: pd.DataFrame,
    universe: pd.DataFrame,
    cost: pd.DataFrame,
    dm_results: pd.DataFrame,
    bootstrap: pd.DataFrame,
    effect_size: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"matplotlib_unavailable:{exc}"]
    chart_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not window.empty:
            counts = window.groupby("setting_id")["sample_size"].sum().reset_index()
            plot_bar(plt, counts, "setting_id", "sample_size", chart_dir / "window_robustness_sample_size.png", "Window Robustness Rows")
            notes.append("generated:window_robustness_sample_size.png")
        if not universe.empty:
            winners = universe[(pd.to_numeric(universe["rank"], errors="coerce") == 1) & universe["metric_name"].isin(["mae", "rmse"])]
            if not winners.empty:
                summary = winners.assign(is_baseline=(winners["model_type"] == "baseline").astype(float))
                summary = summary.groupby("universe_group")["is_baseline"].mean().reset_index()
                plot_bar(plt, summary, "universe_group", "is_baseline", chart_dir / "universe_robustness_baseline_win_rate.png", "Baseline First-Rank Rate")
                notes.append("generated:universe_robustness_baseline_win_rate.png")
        if not cost.empty:
            cost_plot = cost.copy()
            cost_plot["total_cost_bps"] = pd.to_numeric(cost_plot["transaction_cost_bps"], errors="coerce") + pd.to_numeric(
                cost_plot["slippage_bps"], errors="coerce"
            )
            summary = cost_plot.groupby("total_cost_bps")["net_average_realized_return"].mean().reset_index()
            plot_line(plt, summary, "total_cost_bps", "net_average_realized_return", chart_dir / "cost_sensitivity_net_average.png", "Average Net Return by Cost")
            notes.append("generated:cost_sensitivity_net_average.png")
        if not dm_results.empty:
            values = pd.to_numeric(dm_results["p_value"], errors="coerce").dropna()
            if not values.empty:
                plt.figure(figsize=(8, 4.5))
                plt.hist(values, bins=20, color="#2f6b4f", edgecolor="white")
                plt.axvline(0.05, color="#a33a3a", linestyle="--", linewidth=1)
                plt.axvline(0.10, color="#b8860b", linestyle="--", linewidth=1)
                plt.title("DM Test p-values")
                plt.xlabel("p-value")
                plt.ylabel("comparison count")
                plt.tight_layout()
                plt.savefig(chart_dir / "dm_test_pvalues_histogram.png", dpi=140)
                plt.close()
                notes.append("generated:dm_test_pvalues_histogram.png")
        if not bootstrap.empty:
            ci = bootstrap[bootstrap["metric_name"].isin(["average_realized_return", "hit_ratio"])].head(30).copy()
            if not ci.empty:
                ci["label"] = ci["candidate_type"].astype(str) + " h" + ci["horizon"].astype(str) + " top" + ci["top_n"].astype(str) + " " + ci["metric_name"].astype(str)
                plot_ci(plt, ci, chart_dir / "bootstrap_ci_key_metrics.png", "Bootstrap CI Key Metrics")
                notes.append("generated:bootstrap_ci_key_metrics.png")
        if not effect_size.empty:
            plot_frame = effect_size.copy()
            plot_frame["label"] = plot_frame["model_name"].astype(str) + " vs " + plot_frame["baseline_name"].astype(str) + " " + plot_frame["loss"].astype(str)
            plot_frame = plot_frame.sort_values("mean_effect_size").head(40)
            plot_bar(plt, plot_frame, "label", "mean_effect_size", chart_dir / "effect_size_mean_by_comparison.png", "Mean DM Effect Size")
            notes.append("generated:effect_size_mean_by_comparison.png")
    except Exception as exc:
        notes.append(f"chart_generation_failed:{exc}")
    return notes


def write_task_reports(
    output_dir: Path,
    configs: dict[str, dict[str, Any]],
    window: pd.DataFrame,
    universe: pd.DataFrame,
    cost: pd.DataFrame,
    dm_results: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    (output_dir / "EXP-RB-001_WINDOW_ROBUSTNESS.md").write_text(
        render_task_report(
            "EXP-RB-001 Window Robustness",
            "Train/test window sensitivity was evaluated only where an exact local source artifact matched the requested train/test split.",
            window,
            ["setting_id", "ticker", "horizon", "model_name", "model_type", "metric_name", "metric_value", "rank", "robustness_note"],
            configs.get("EXP-RB-001", {}),
        ),
        encoding="utf-8",
    )
    (output_dir / "EXP-RB-002_UNIVERSE_ROBUSTNESS.md").write_text(
        render_task_report(
            "EXP-RB-002 Universe Robustness",
            "Ticker universe sensitivity was computed from forecasting-core metric artifacts grouped by configured universes.",
            universe,
            ["universe_group", "ticker", "horizon", "model_name", "model_type", "metric_name", "metric_value", "rank"],
            configs.get("EXP-RB-002", {}),
        ),
        encoding="utf-8",
    )
    (output_dir / "EXP-RB-003_COST_SENSITIVITY.md").write_text(
        render_task_report(
            "EXP-RB-003 Cost Sensitivity",
            "Cost/slippage sensitivity was computed from diagnostic basket period returns when available.",
            cost,
            ["cost_scenario_id", "candidate_type", "top_n", "horizon", "gross_average_realized_return", "net_average_realized_return", "net_hit_ratio"],
            configs.get("EXP-RB-003", {}),
        ),
        encoding="utf-8",
    )
    (output_dir / "EXP-ST-001_DM_TEST.md").write_text(
        render_task_report(
            "EXP-ST-001 Diebold-Mariano Test",
            "DM tests compare aligned model and baseline forecast errors from local prediction artifacts.",
            dm_results,
            ["source_experiment", "ticker", "horizon", "model_name", "baseline_name", "loss", "dm_statistic", "p_value", "effect_size", "warning"],
            configs.get("EXP-ST-001", {}),
        ),
        encoding="utf-8",
    )
    (output_dir / "EXP-ST-002_BOOTSTRAP_CI.md").write_text(
        render_task_report(
            "EXP-ST-002 Bootstrap CI",
            "Bootstrap confidence intervals were computed from basket period return rows with a fixed seed.",
            bootstrap,
            ["candidate_type", "top_n", "horizon", "metric_name", "estimate", "ci_lower", "ci_upper", "sample_size", "warning"],
            configs.get("EXP-ST-002", {}),
        ),
        encoding="utf-8",
    )


def render_task_report(title: str, purpose: str, frame: pd.DataFrame, columns: list[str], config: dict[str, Any]) -> str:
    config_id = config.get("experiment", {}).get("id", "")
    row_count = len(frame) if frame is not None else 0
    warning_count = 0
    if frame is not None and not frame.empty:
        warning_like = [column for column in frame.columns if column in {"warning", "robustness_note"}]
        warning_count = int(sum(frame[column].fillna("").astype(str).str.contains("warning|missing|insufficient|mismatch|no_", case=False).sum() for column in warning_like))
    preview = frame[[column for column in columns if column in frame.columns]].head(20) if frame is not None and not frame.empty else pd.DataFrame()
    return "\n".join(
        [
            f"# {title}",
            "",
            f"Config: `{config_id}`",
            "",
            purpose,
            "",
            f"Rows generated: {row_count}",
            f"Warning or limitation rows: {warning_count}",
            "",
            "## Preview",
            "",
            markdown_table(preview),
            "",
            "## Interpretation Guardrail",
            "",
            "Null values and warnings indicate computations that were not supported by the available local artifacts.",
            DISCLAIMER,
            "",
        ]
    )


def render_main_report(
    configs: dict[str, dict[str, Any]],
    window: pd.DataFrame,
    universe: pd.DataFrame,
    cost: pd.DataFrame,
    dm_results: pd.DataFrame,
    bootstrap: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    significance_summary: pd.DataFrame,
    effect_size_summary: pd.DataFrame,
    chart_notes: list[str],
) -> str:
    dm_total = int(len(dm_results))
    dm_sig05 = int(dm_results["significant_05"].fillna(False).astype(bool).sum()) if not dm_results.empty else 0
    dm_sig10 = int(dm_results["significant_10"].fillna(False).astype(bool).sum()) if not dm_results.empty else 0
    non_sig = int((pd.to_numeric(dm_results.get("p_value", pd.Series(dtype=float)), errors="coerce") >= 0.10).sum()) if not dm_results.empty else 0
    window_exact = window[window["robustness_note"].astype(str).str.contains("exact_train_test_window_match", na=False)] if not window.empty else pd.DataFrame()
    missing_window = int(window.loc[pd.to_numeric(window["sample_size"], errors="coerce").fillna(0) == 0, "setting_id"].nunique()) if not window.empty else 0
    universe_summary = summarize_universe_for_report(universe)
    cost_summary = summarize_cost_for_report(cost)
    bootstrap_summary = summarize_bootstrap_for_report(bootstrap)
    acceptance = acceptance_table()
    lines = [
        "# Robustness and Statistical Significance Report",
        "",
        "## 1. Executive summary",
        "",
        (
            "Phase 6 adds robustness and statistical-significance evidence from local artifacts. "
            f"DM tests attempted {dm_total} aligned model/baseline comparisons, with {dm_sig05} significant at 5% "
            f"and {dm_sig10} significant at 10%. {ROBUSTNESS_NOTE_SIGNIFICANCE if non_sig else ''}"
        ).strip(),
        (
            f"Window sensitivity has {window_exact['setting_id'].nunique() if not window_exact.empty else 0} exact local "
            f"source setting(s) and {missing_window} requested setting(s) without exact local rerun evidence, so window "
            f"robustness remains constrained. {WEAK_ROBUSTNESS_NOTE}"
        ),
        cost_summary["headline"],
        bootstrap_summary["headline"],
        "",
        "## 2. Phase 6 objective",
        "",
        "The objective is to test whether earlier VSEF findings are stable across alternative settings and whether observed differences are statistically meaningful rather than random variation.",
        "",
        "## 3. Relation to Phase 0-5",
        "",
        "- Phase 0 froze v1 governance, provider boundaries, and diagnostic-only constraints.",
        "- Phase 1 standardized experiment execution and artifact layout.",
        "- Phase 2 found that forecasting models do not consistently beat simple baselines on MAE/RMSE.",
        "- Phase 3 found weak aggregate improvement from risk-aware ranking.",
        "- Phase 4 supported regime dependence and the no-universal-best-model thesis.",
        "- Phase 5 found strongest feature contribution evidence for rolling mean features, with mixed evidence elsewhere.",
        "",
        "## 4. Robustness design",
        "",
        "- Train/test window sensitivity uses `EXP-RB-001` and only exact local source artifacts for requested train/test splits.",
        "- Universe sensitivity uses `EXP-RB-002` ticker groups over forecasting-core metrics.",
        "- Cost/slippage sensitivity uses `EXP-RB-003` diagnostic basket period returns.",
        "- Diebold-Mariano tests use `EXP-ST-001` aligned model/baseline forecast errors.",
        "- Bootstrap confidence intervals use `EXP-ST-002` basket period returns with fixed seed reproducibility.",
        "",
        "## 5. Window robustness results",
        "",
        f"Exact source settings with computed rows: {window_exact['setting_id'].nunique() if not window_exact.empty else 0}. Missing exact settings: {missing_window}.",
        "Ranking stability and metric variance cannot be claimed across the missing windows because those train/test reruns are not present as exact local artifacts.",
        markdown_table(window[["setting_id", "ticker", "horizon", "model_name", "model_type", "metric_name", "metric_value", "rank", "robustness_note"]].head(20) if not window.empty else pd.DataFrame()),
        "",
        "## 6. Universe robustness results",
        "",
        universe_summary["text"],
        markdown_table(universe_summary["table"]),
        "",
        "## 7. Cost/slippage sensitivity",
        "",
        cost_summary["text"],
        markdown_table(cost_summary["table"]),
        "",
        "## 8. Diebold-Mariano test results",
        "",
        f"Comparisons: {dm_total}. Significant at 5%: {dm_sig05}. Significant at 10%: {dm_sig10}. Non-significant at 10%: {non_sig}.",
        ROBUSTNESS_NOTE_SIGNIFICANCE if non_sig else "Significant rows should still be interpreted with sample-size and multiple-comparison caution.",
        markdown_table(dm_results[["source_experiment", "ticker", "horizon", "model_name", "baseline_name", "loss", "dm_statistic", "p_value", "effect_size", "warning"]].head(20) if not dm_results.empty else pd.DataFrame()),
        "",
        "## 9. Bootstrap confidence interval results",
        "",
        bootstrap_summary["text"],
        markdown_table(bootstrap_summary["table"]),
        "",
        "## 10. Statistical interpretation",
        "",
        "Important differences are statistically supported only where p-values and interval evidence support them. Non-significant DM results remain exploratory.",
        "Wide or overlapping bootstrap intervals indicate uncertainty and should constrain claims.",
        "The prior baseline-competitiveness, weak aggregate risk-aware improvement, regime-dependence, and mixed feature-evidence conclusions remain best described as mixed diagnostic evidence rather than investment evidence.",
        "",
        "## 11. Limitations",
        "",
        "- Small sample sizes may affect statistical power.",
        "- Overlapping forecast horizons can induce autocorrelation; DM tests use a Newey-West style adjustment but remain approximate.",
        "- Forecast errors and returns may be non-normal.",
        "- Bootstrap intervals assume the sampled period-return rows are representative of the local diagnostic artifact.",
        "- Cost modeling uses simplified bps deductions from diagnostic period returns.",
        "- Local artifact availability constrains reproducibility; missing exact train/test reruns are disclosed instead of imputed.",
        "",
        "## 12. Acceptance criteria table",
        "",
        markdown_table(acceptance),
        "",
        "## 13. Diagnostic-only disclaimer",
        "",
        DISCLAIMER,
        "",
        "## Source configs",
        "",
        markdown_table(pd.DataFrame({"experiment_id": sorted(configs), "loaded": [True] * len(configs)})),
        "",
        "## Summary artifacts",
        "",
        markdown_table(robustness_summary.head(30)),
        "",
        "## Statistical significance summary",
        "",
        markdown_table(significance_summary.head(30)),
        "",
        "## Effect size summary",
        "",
        markdown_table(effect_size_summary.head(20)),
        "",
        "## Charts",
        "",
        "\n".join(f"- {note}" for note in chart_notes) if chart_notes else "- No chart notes.",
        "",
    ]
    return "\n".join(lines)


def write_optional_doc(output_dir: Path, robustness_summary: pd.DataFrame, significance_summary: pd.DataFrame) -> None:
    doc_path = REPO_ROOT / "docs" / "experiments" / "PHASE6_ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE.md"
    text = "\n".join(
        [
            "# Phase 6 Robustness and Statistical Significance",
            "",
            "This document indexes the Phase 6 configs and generated review artifacts.",
            "",
            "## Configs",
            "",
            "- `configs/experiments/EXP-RB-001.yaml`",
            "- `configs/experiments/EXP-RB-002.yaml`",
            "- `configs/experiments/EXP-RB-003.yaml`",
            "- `configs/experiments/EXP-ST-001.yaml`",
            "- `configs/experiments/EXP-ST-002.yaml`",
            "",
            "## Report artifacts",
            "",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE_REPORT.md'}`",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'window_robustness.csv'}`",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'universe_robustness.csv'}`",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'cost_sensitivity.csv'}`",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'dm_test_results.csv'}`",
            f"- `{output_dir.relative_to(REPO_ROOT) / 'bootstrap_ci.csv'}`",
            "",
            "## Robustness summary",
            "",
            markdown_table(robustness_summary),
            "",
            "## Statistical summary",
            "",
            markdown_table(significance_summary),
            "",
            DISCLAIMER,
            "",
        ]
    )
    doc_path.write_text(text, encoding="utf-8")


def summarize_universe_for_report(universe: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if universe.empty:
        return {"text": "No universe robustness rows were generated.", "table": pd.DataFrame()}
    for group_name, group in universe.groupby("universe_group", dropna=False, sort=True):
        winners = group[(pd.to_numeric(group["rank"], errors="coerce") == 1) & group["metric_name"].isin(["mae", "rmse"])]
        baseline_winners = int((winners["model_type"] == "baseline").sum()) if not winners.empty else 0
        model_winners = int((winners["model_type"] == "model").sum()) if not winners.empty else 0
        rows.append(
            {
                "universe_group": group_name,
                "mae_rmse_winner_rows": len(winners),
                "baseline_winners": baseline_winners,
                "model_winners": model_winners,
                "baseline_competitiveness_note": "Baseline competitiveness persists." if baseline_winners else "Models lead these ranked rows, but this remains diagnostic.",
            }
        )
    text = (
        "Universe grouping preserves the Phase 2 baseline-competitiveness caveat where baselines remain first-ranked "
        "in MAE/RMSE rows. Where models lead, the result is still a grouped diagnostic result rather than investment value."
    )
    return {"text": text, "table": pd.DataFrame(rows)}


def summarize_cost_for_report(cost: pd.DataFrame) -> dict[str, Any]:
    if cost.empty:
        return {"headline": "Cost sensitivity could not be computed from available artifacts.", "text": "No rows available.", "table": pd.DataFrame()}
    frame = cost.copy()
    frame["total_cost_bps"] = pd.to_numeric(frame["transaction_cost_bps"], errors="coerce") + pd.to_numeric(
        frame["slippage_bps"], errors="coerce"
    )
    summary = (
        frame.groupby("total_cost_bps")
        .agg(
            rows=("net_average_realized_return", "size"),
            mean_gross_average=("gross_average_realized_return", "mean"),
            mean_net_average=("net_average_realized_return", "mean"),
            nonpositive_net_rows=("net_average_realized_return", lambda value: int((pd.to_numeric(value, errors="coerce") <= 0).sum())),
            mean_net_hit_ratio=("net_hit_ratio", "mean"),
        )
        .reset_index()
    )
    weakening = summary[summary["nonpositive_net_rows"] > 0]
    if weakening.empty:
        headline = f"No tested cost level fully removed average diagnostic evidence. {STRONG_ROBUSTNESS_NOTE}"
    else:
        first = float(weakening["total_cost_bps"].min())
        headline = f"At {first:g} total bps, at least one diagnostic basket row has non-positive net average return. {COST_EDGE_NOTE}"
    text = "Gross averages decline mechanically after bps deductions. The result is a retrospective diagnostic sensitivity table, not executable strategy evidence."
    return {"headline": headline, "text": text, "table": summary}


def summarize_bootstrap_for_report(bootstrap: pd.DataFrame) -> dict[str, Any]:
    if bootstrap.empty:
        return {"headline": "Bootstrap confidence intervals could not be computed.", "text": "No rows available.", "table": pd.DataFrame()}
    frame = bootstrap.copy()
    frame["ci_width"] = pd.to_numeric(frame["ci_upper"], errors="coerce") - pd.to_numeric(frame["ci_lower"], errors="coerce")
    key = frame[frame["metric_name"].isin(["average_realized_return", "hit_ratio"])].copy()
    key["wide_or_overlap"] = key.apply(is_wide_ci, axis=1)
    wide_count = int(key["wide_or_overlap"].sum()) if not key.empty else 0
    headline = f"Bootstrap generated {len(frame)} CI rows; {wide_count} key return/hit-ratio rows are wide or overlap cautious thresholds. {WIDE_CI_NOTE if wide_count else ''}".strip()
    summary = (
        frame.groupby("metric_name")
        .agg(
            rows=("metric_name", "size"),
            mean_estimate=("estimate", "mean"),
            mean_ci_width=("ci_width", "mean"),
            min_sample_size=("sample_size", "min"),
            warning_rows=("warning", lambda value: int(pd.Series(value).fillna("").astype(str).ne("").sum())),
        )
        .reset_index()
    )
    text = "Return intervals that cross zero and hit-ratio intervals that cross 0.5 weaken superiority claims."
    return {"headline": headline, "text": text, "table": summary}


def is_wide_ci(row: pd.Series) -> bool:
    lower = to_float(row.get("ci_lower"))
    upper = to_float(row.get("ci_upper"))
    estimate = to_float(row.get("estimate"))
    metric = str(row.get("metric_name") or "")
    if lower is None or upper is None:
        return True
    if metric == "average_realized_return" and lower <= 0 <= upper:
        return True
    if metric == "hit_ratio" and lower <= 0.5 <= upper:
        return True
    width = upper - lower
    if estimate is None:
        return True
    return bool(abs(width) > max(abs(estimate) * 2.0, 1e-12))


def acceptance_table() -> pd.DataFrame:
    paths = [
        "configs/experiments/EXP-RB-001.yaml",
        "configs/experiments/EXP-RB-002.yaml",
        "configs/experiments/EXP-RB-003.yaml",
        "configs/experiments/EXP-ST-001.yaml",
        "configs/experiments/EXP-ST-002.yaml",
        "src/ml/statistics/dm_test.py",
        "src/ml/statistics/bootstrap_eval.py",
        "reports/robustness/window_robustness.csv",
        "reports/robustness/universe_robustness.csv",
        "reports/robustness/cost_sensitivity.csv",
        "reports/robustness/dm_test_results.csv",
        "reports/robustness/bootstrap_ci.csv",
        "reports/robustness/ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE_REPORT.md",
    ]
    return pd.DataFrame({"criterion": paths, "exists": [(REPO_ROOT / path).exists() for path in paths]})


def markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    clean = frame.copy().head(60)
    clean = clean.where(pd.notna(clean), "")
    headers = [str(column) for column in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in clean.iterrows():
        values = [format_markdown_value(row[column]) for column in clean.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def plot_bar(plt: Any, frame: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    data = frame.copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    plt.figure(figsize=(10, 4.8))
    plt.bar(data[x].astype(str), data[y], color="#4c6f8f")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_line(plt: Any, frame: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    data = frame.copy().sort_values(x)
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    plt.figure(figsize=(8, 4.5))
    plt.plot(data[x], data[y], marker="o", color="#2f6b4f")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_ci(plt: Any, frame: pd.DataFrame, path: Path, title: str) -> None:
    data = frame.copy().reset_index(drop=True)
    estimate = pd.to_numeric(data["estimate"], errors="coerce")
    lower = pd.to_numeric(data["ci_lower"], errors="coerce")
    upper = pd.to_numeric(data["ci_upper"], errors="coerce")
    x = np.arange(len(data))
    yerr = np.vstack([(estimate - lower).clip(lower=0), (upper - estimate).clip(lower=0)])
    plt.figure(figsize=(12, 5))
    plt.errorbar(x, estimate, yerr=yerr, fmt="o", color="#3f5f77", ecolor="#8a9bab", capsize=3)
    plt.xticks(x, data["label"].astype(str), rotation=70, ha="right", fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def numeric_median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
