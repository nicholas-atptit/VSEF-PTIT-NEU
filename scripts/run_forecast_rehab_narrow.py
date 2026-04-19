"""Run the narrowed post-F1 forecast rehabilitation benchmark."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_phase2_benchmark import build_window_garch_risk_frame, build_window_regime_frame
from src.evaluation.backtest import BacktestConfig, CostAwareBacktester
from src.evaluation.forecast_rehab_narrow import (
    BASELINE_ONLY_MODELS,
    COMPARATOR_MODELS,
    PRIMARY_MODELS,
    build_narrow_core_frame,
    build_narrow_feature_summary,
    build_narrow_matrix_config,
    build_narrow_policy_configuration,
    build_narrow_scope_table,
    create_narrow_forecast_model,
    resolve_narrow_feature_family_columns,
)
from src.evaluation.targets import build_target_spec
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator, summarize_forecasts
from src.reporting.forecast_rehab_narrow import (
    build_cost_sensitivity_summary,
    build_f1_reference_comparison,
    build_narrow_assessment,
    build_narrow_feature_performance_summary,
    build_narrow_forecast_vs_policy_summary,
    build_narrow_model_stability_summary,
    build_narrow_report,
    build_narrow_target_comparison_summary,
    render_narrow_summary_markdown,
)
from src.reporting.forecast_rehab import build_forecast_quality_summary, build_model_stability_summary
from src.reporting.manifests import (
    build_batch_manifest,
    collect_dependency_versions,
    collect_git_metadata,
    collect_runtime_metadata,
    write_run_manifest,
)
from src.reporting.summary import write_summary_markdown, write_summary_tables
from src.strategy.execution_policy import PolicyConfiguration, execute_policy_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the narrowed forecast rehabilitation benchmark.")
    parser.add_argument("--preset", choices=["smoke", "medium", "narrow_full"], default="medium")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[*PRIMARY_MODELS, *COMPARATOR_MODELS, *BASELINE_ONLY_MODELS],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/forecast_rehab_narrow")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--regime-lookback", type=int, default=20)
    parser.add_argument("--regime-bull-threshold", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold", type=float, default=-0.03)
    return parser.parse_args()


def _normalize_metadata(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    values = row if isinstance(row, dict) else row.to_dict()
    metadata = dict(values)
    tickers = metadata.get("tickers", [])
    metadata["ticker_group_members"] = ",".join(str(ticker) for ticker in tickers)
    for key, value in list(metadata.items()):
        if isinstance(value, (list, tuple)):
            metadata[key] = ",".join(str(item) for item in value)
    return metadata


def _annotate(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    annotated = frame.copy()
    for key, value in metadata.items():
        if key == "tickers":
            continue
        annotated[key] = value
    return annotated


def _build_policy_configuration(policy_baseline: dict[str, Any]) -> PolicyConfiguration:
    return PolicyConfiguration(
        policy_variant=str(policy_baseline["policy_variant"]),
        strategy_variant="forecast_plus_risk_and_regime",
        policy_label=str(policy_baseline["policy_label"]),
        threshold_policy=str(policy_baseline["threshold_policy"]),
        sizing_profile=str(policy_baseline["sizing_profile"]),
        sizing_label=str(policy_baseline["sizing_profile"]),
        use_risk_context=bool(policy_baseline["use_risk_context"]),
        use_regime_context=bool(policy_baseline["use_regime_context"]),
        use_volatility_sizing=bool(policy_baseline["use_volatility_sizing"]),
        use_drawdown_control=bool(policy_baseline["use_drawdown_control"]),
        use_regime_sizing=bool(policy_baseline["use_regime_sizing"]),
        sizing_mode=str(policy_baseline["sizing_mode"]),
        fixed_position_size=policy_baseline.get("fixed_position_size"),
        min_position_size=float(policy_baseline["min_position_size"]),
        max_position_size=float(policy_baseline["max_position_size"]),
        volatility_target_scale=float(policy_baseline["volatility_target_scale"]),
        drawdown_haircut_strength=float(policy_baseline["drawdown_haircut_strength"]),
        regime_multiplier_strength=float(policy_baseline["regime_multiplier_strength"]),
        policy_family="phase26_default_candidate",
        ablation_labels=("F1_5_NARROW_FIXED_POLICY",),
    )


def _evaluate_requested_models(
    evaluator: WalkForwardEvaluator,
    model_names: list[str],
    *,
    target_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, str]], list[str]]:
    forecast_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    evaluated_models: list[str] = []
    datasets: dict[str, Any] | None = None
    window_summary: pd.DataFrame | None = None
    target_spec = build_target_spec(target_name)

    for model_name in model_names:
        model = create_narrow_forecast_model(model_name, target_spec=target_spec)
        try:
            result = evaluator.evaluate([model])
            forecast_frames.append(result["forecasts"])
            evaluated_models.append(model_name)
            if datasets is None:
                datasets = result["datasets"]
            if window_summary is None:
                window_summary = result["window_summary"][
                    ["ticker", "window_id", "train_start", "train_end", "test_start", "test_end"]
                ].drop_duplicates()
        except Exception as exc:
            skipped.append({"model_name": model_name, "reason": str(exc)})

    if not forecast_frames or datasets is None or window_summary is None:
        raise RuntimeError("No forecast models completed successfully")

    forecast_df = pd.concat(forecast_frames, ignore_index=True).sort_values(
        ["timestamp", "ticker", "model_name"]
    ).reset_index(drop=True)
    forecast_summary = summarize_forecasts(forecast_df)
    return forecast_df, forecast_summary, window_summary, datasets, skipped, evaluated_models


@lru_cache(maxsize=32)
def _load_sample_frame(
    ticker: str,
    horizon: int,
    target_name: str,
    target_column: str,
    train_size: int,
    test_size: int,
    step_size: int,
    gap_size: int,
) -> pd.DataFrame:
    target_spec = build_target_spec(target_name, target_column=target_column)
    evaluator = WalkForwardEvaluator(
        WalkForwardConfig(
            tickers=[str(ticker)],
            horizon=int(horizon),
            train_size=int(train_size),
            test_size=int(test_size),
            step_size=int(step_size),
            gap_size=int(gap_size),
            max_windows=1,
            target_column=target_spec.target_column,
            target_type=target_spec.name,
        )
    )
    return evaluator.load_ticker_data(str(ticker)).frame.copy()


def _sample_frame_for_core_row(core_row: dict[str, Any]) -> pd.DataFrame:
    return _load_sample_frame(
        str(core_row["tickers"][0]),
        int(core_row["horizon"]),
        str(core_row["target_name"]),
        str(core_row["target_column"]),
        int(core_row["train_size"]),
        int(core_row["test_size"]),
        int(core_row["step_size"]),
        int(core_row["gap_size"]),
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    matrix_config = build_narrow_matrix_config(args.preset, include_baselines=not args.skip_baselines)
    core_frame = build_narrow_core_frame(matrix_config)
    if core_frame.empty:
        raise RuntimeError("The narrow forecast rehab matrix did not produce any core runs")

    model_names = [str(model).lower() for model in args.models]
    sample_frame = _sample_frame_for_core_row(core_frame.iloc[0].to_dict())
    scope_table = build_narrow_scope_table()
    feature_definition_summary = build_narrow_feature_summary(sample_frame)
    policy_baseline = dict(matrix_config["policy_baseline"])
    policy_config = _build_policy_configuration(policy_baseline)

    aggregate_forecasts: list[pd.DataFrame] = []
    aggregate_forecast_summary: list[pd.DataFrame] = []
    aggregate_slice_summary: list[pd.DataFrame] = []
    aggregate_strategy_metrics: list[pd.DataFrame] = []
    aggregate_signals: list[pd.DataFrame] = []
    aggregate_positions: list[pd.DataFrame] = []
    aggregate_trades: list[pd.DataFrame] = []
    skipped_model_records: list[dict[str, Any]] = []
    evaluated_models: set[str] = set()

    for core_row in core_frame.to_dict(orient="records"):
        target_spec = build_target_spec(core_row["target_name"], target_column=core_row["target_column"])
        metadata = _normalize_metadata(core_row)
        sample_target_frame = _sample_frame_for_core_row(core_row)
        feature_columns = resolve_narrow_feature_family_columns(
            sample_target_frame,
            family_name=str(core_row["feature_family"]),
        )
        metadata["resolved_feature_count"] = int(len(feature_columns))

        evaluator = WalkForwardEvaluator(
            WalkForwardConfig(
                tickers=list(core_row["tickers"]),
                horizon=int(core_row["horizon"]),
                train_size=int(core_row["train_size"]),
                test_size=int(core_row["test_size"]),
                step_size=int(core_row["step_size"]),
                gap_size=int(core_row["gap_size"]),
                max_windows=int(core_row["max_windows"]),
                feature_columns=feature_columns,
                target_column=target_spec.target_column,
                target_type=target_spec.name,
                seed=args.seed,
            )
        )

        forecast_df, forecast_summary, window_summary, datasets, skipped_models, successful_models = _evaluate_requested_models(
            evaluator,
            model_names,
            target_name=target_spec.name,
        )
        evaluated_models.update(successful_models)
        for item in skipped_models:
            skipped_model_records.append({**metadata, **item})

        annotated_forecasts = _annotate(forecast_df, metadata)
        aggregate_forecasts.append(annotated_forecasts)
        aggregate_forecast_summary.append(_annotate(forecast_summary, metadata))
        aggregate_slice_summary.append(
            summarize_forecasts(
                annotated_forecasts,
                group_columns=[
                    "core_run_id",
                    "group_name",
                    "horizon",
                    "target_name",
                    "target_family",
                    "feature_family",
                    "model_name",
                    "ticker",
                    "window_id",
                ],
            )
        )

        if not target_spec.tradable_output:
            continue

        risk_df = build_window_garch_risk_frame(datasets, window_summary, horizon=int(core_row["horizon"]))
        regime_df = build_window_regime_frame(
            datasets,
            window_summary,
            lookback=args.regime_lookback,
            bull_threshold=args.regime_bull_threshold,
            bear_threshold=args.regime_bear_threshold,
        )
        signal_df, position_df = execute_policy_configuration(
            forecast_df,
            policy_config=policy_config,
            threshold=float(policy_baseline["threshold"]),
            allow_short=args.allow_short,
            risk_df=risk_df,
            regime_df=regime_df,
            capital_config={
                "risk_budget": float(policy_baseline["risk_budget"]),
                "max_position_size": float(policy_baseline["max_position_size"]),
            },
        )
        aggregate_signals.append(_annotate(signal_df, metadata))
        aggregate_positions.append(_annotate(position_df, metadata))

        market_data = {
            ticker: dataset.frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            for ticker, dataset in datasets.items()
        }
        for cost_mode in matrix_config["cost_modes"]:
            policy_with_costs = build_narrow_policy_configuration(policy_baseline, cost_mode=cost_mode)
            backtester = CostAwareBacktester(
                BacktestConfig(
                    horizon=int(core_row["horizon"]),
                    transaction_fee_bps=float(policy_with_costs["transaction_fee_bps"]),
                    slippage_bps=float(policy_with_costs["slippage_bps"]),
                    allow_short=args.allow_short,
                )
            )
            backtest_result = backtester.run(position_df, market_data)
            cost_metadata = {
                **metadata,
                "cost_mode": str(cost_mode),
                "cost_label": f"{cost_mode}:{policy_with_costs['transaction_fee_bps']:.0f}fee_{policy_with_costs['slippage_bps']:.0f}slip",
                "transaction_fee_bps": float(policy_with_costs["transaction_fee_bps"]),
                "slippage_bps": float(policy_with_costs["slippage_bps"]),
            }
            strategy_metrics = backtest_result["strategy_metrics"].copy()
            strategy_metrics["policy_variant"] = policy_baseline["policy_variant"]
            strategy_metrics["policy_label"] = policy_baseline["policy_label"]
            strategy_metrics["sizing_profile"] = policy_baseline["sizing_profile"]
            strategy_metrics["threshold"] = float(policy_baseline["threshold"])
            aggregate_strategy_metrics.append(_annotate(strategy_metrics, cost_metadata))
            aggregate_trades.append(_annotate(backtest_result["trades"], cost_metadata))

    forecast_frame = pd.concat(aggregate_forecasts, ignore_index=True) if aggregate_forecasts else pd.DataFrame()
    forecast_summary_frame = pd.concat(aggregate_forecast_summary, ignore_index=True) if aggregate_forecast_summary else pd.DataFrame()
    slice_summary_frame = pd.concat(aggregate_slice_summary, ignore_index=True) if aggregate_slice_summary else pd.DataFrame()
    strategy_metrics_frame = pd.concat(aggregate_strategy_metrics, ignore_index=True) if aggregate_strategy_metrics else pd.DataFrame()
    signals_frame = pd.concat(aggregate_signals, ignore_index=True) if aggregate_signals else pd.DataFrame()
    positions_frame = pd.concat(aggregate_positions, ignore_index=True) if aggregate_positions else pd.DataFrame()
    trades_frame = pd.concat(aggregate_trades, ignore_index=True) if aggregate_trades else pd.DataFrame()
    forecast_quality_summary = build_forecast_quality_summary(slice_summary_frame)
    model_stability_summary = build_model_stability_summary(slice_summary_frame)

    narrow_feature_summary = build_narrow_feature_performance_summary(
        feature_definition_summary,
        forecast_quality_summary,
        strategy_metrics_frame,
    )
    narrow_model_stability_summary = build_narrow_model_stability_summary(
        forecast_quality_summary,
        model_stability_summary,
        strategy_metrics_frame,
    )
    narrow_target_comparison_summary = build_narrow_target_comparison_summary(
        forecast_quality_summary,
        strategy_metrics_frame,
    )
    narrow_forecast_vs_policy_summary = build_narrow_forecast_vs_policy_summary(
        forecast_quality_summary,
        strategy_metrics_frame,
    )
    cost_sensitivity_summary = build_cost_sensitivity_summary(strategy_metrics_frame)
    f1_reference = build_f1_reference_comparison(narrow_models=model_names)
    assessment = build_narrow_assessment(
        narrow_feature_summary,
        narrow_model_stability_summary,
        narrow_target_comparison_summary,
        narrow_forecast_vs_policy_summary,
        cost_sensitivity_summary,
        f1_reference=f1_reference,
    )

    artifact_paths = write_summary_tables(
        output_dir,
        {
            "scope_table": scope_table,
            "matrix_core_runs": core_frame,
            "aggregate_forecasts": forecast_frame,
            "aggregate_forecast_summary": forecast_summary_frame,
            "aggregate_slice_summary": slice_summary_frame,
            "forecast_quality_summary": forecast_quality_summary,
            "model_stability_summary": model_stability_summary,
            "aggregate_strategy_metrics": strategy_metrics_frame,
            "aggregate_signals": signals_frame,
            "aggregate_positions": positions_frame,
            "aggregate_trades": trades_frame,
            "narrow_feature_summary": narrow_feature_summary,
            "narrow_model_stability_summary": narrow_model_stability_summary,
            "narrow_target_comparison_summary": narrow_target_comparison_summary,
            "narrow_forecast_vs_policy_summary": narrow_forecast_vs_policy_summary,
            "cost_sensitivity_summary": cost_sensitivity_summary,
        },
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    batch_manifest = build_batch_manifest(
        git_metadata=collect_git_metadata(Path.cwd()),
        runtime=collect_runtime_metadata(),
        dependency_versions=collect_dependency_versions(
            ["pandas", "numpy", "statsmodels", "arch", "scikit-learn", "xgboost", "lightgbm"]
        ),
        command=" ".join(sys.argv),
        requested_models=model_names,
        evaluated_models=sorted(evaluated_models),
        skipped_models=skipped_model_records,
        target_type="forecast_rehab_narrow_multi_target",
        seed=args.seed,
        matrix_config=matrix_config,
        run_counts={
            "core_runs": int(len(core_frame)),
            "forecast_rows": int(len(forecast_frame)),
            "slice_rows": int(len(slice_summary_frame)),
            "policy_rows": int(len(strategy_metrics_frame)),
        },
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
        manifest_type="forecast_rehab_narrow_manifest_v1",
    )
    batch_manifest["assessment"] = assessment
    batch_manifest["f1_reference"] = f1_reference
    batch_manifest["scope_table"] = scope_table.to_dict(orient="records")

    summary_path = write_summary_markdown(
        output_dir,
        render_narrow_summary_markdown(
            batch_manifest,
            assessment,
            narrow_feature_summary,
            narrow_model_stability_summary,
        ),
    )
    report_path = write_summary_markdown(
        output_dir,
        build_narrow_report(
            batch_manifest,
            scope_table,
            narrow_feature_summary,
            narrow_model_stability_summary,
            narrow_target_comparison_summary,
            narrow_forecast_vs_policy_summary,
            cost_sensitivity_summary,
            assessment,
        ),
        filename="forecast_rehab_narrow_report.md",
    )
    batch_manifest["artifact_paths"] = {
        **artifact_paths,
        "summary": str(summary_path),
        "forecast_rehab_narrow_report": str(report_path),
    }
    manifest_path = write_run_manifest(output_dir, batch_manifest)

    print(f"Preset: {args.preset}")
    print(f"Core runs: {len(core_frame)}")
    print(f"Forecast rows: {len(forecast_frame)}")
    print(f"Policy rows: {len(strategy_metrics_frame)}")
    print(f"Recommendation: {assessment.get('recommendation')}")
    print(f"Best feature family: {assessment.get('best_feature_family')}")
    print(f"Best model family: {assessment.get('best_model_family')}")
    print(f"Best target: {assessment.get('best_target_name')}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
