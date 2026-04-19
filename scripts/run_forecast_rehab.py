"""Bounded forecast-layer rehabilitation runner."""

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
from src.evaluation.forecast_rehab import (
    build_feature_family_columns,
    build_feature_inventory_table,
    build_forecast_rehab_core_frame,
    build_forecast_rehab_matrix_config,
    create_rehab_forecast_model,
    forecast_rehab_policy_baseline,
)
from src.evaluation.targets import build_target_spec
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator, summarize_forecasts
from src.forecast.registry import supported_forecast_models
from src.reporting.forecast_rehab import (
    build_feature_ablation_summary,
    build_feature_inventory_summary,
    build_forecast_quality_summary,
    build_forecast_rehab_assessment,
    build_forecast_rehab_report,
    build_forecast_vs_policy_summary,
    build_model_stability_summary,
    render_forecast_rehab_summary_markdown,
)
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
    parser = argparse.ArgumentParser(description="Run bounded forecast-layer rehabilitation benchmarks.")
    parser.add_argument("--preset", choices=["smoke", "medium", "rehab_full"], default="medium")
    parser.add_argument("--models", nargs="+", default=list(supported_forecast_models()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/forecast_rehab")
    parser.add_argument("--allow-short", action="store_true")
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
        ablation_labels=("F1_FIXED_POLICY",),
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
        model = create_rehab_forecast_model(model_name, target_spec=target_spec)
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


@lru_cache(maxsize=64)
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


def _sample_frame_for_feature_inventory(core_row: dict[str, Any]) -> pd.DataFrame:
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

    matrix_config = build_forecast_rehab_matrix_config(args.preset)
    core_frame = build_forecast_rehab_core_frame(matrix_config)
    policy_baseline = forecast_rehab_policy_baseline()
    policy_config = _build_policy_configuration(policy_baseline)

    if core_frame.empty:
        raise RuntimeError("The forecast rehab matrix did not produce any core runs")

    sample_frame = _sample_frame_for_feature_inventory(core_frame.iloc[0].to_dict())
    feature_inventory = build_feature_inventory_table(sample_frame)

    evaluated_model_names: set[str] = set()
    skipped_model_records: list[dict[str, Any]] = []
    aggregate_forecast_frames: list[pd.DataFrame] = []
    aggregate_forecast_summary_frames: list[pd.DataFrame] = []
    aggregate_slice_summary_frames: list[pd.DataFrame] = []
    aggregate_strategy_metric_frames: list[pd.DataFrame] = []
    aggregate_position_frames: list[pd.DataFrame] = []
    aggregate_trade_frames: list[pd.DataFrame] = []
    aggregate_signal_frames: list[pd.DataFrame] = []

    for core_row in core_frame.to_dict(orient="records"):
        target_spec = build_target_spec(core_row["target_name"], target_column=core_row["target_column"])
        metadata = _normalize_metadata(core_row)
        sample_target_frame = _sample_frame_for_feature_inventory(core_row)
        feature_columns = build_feature_family_columns(
            sample_target_frame,
            family_name=str(core_row["feature_family"]),
            target_name=target_spec.name,
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

        forecast_df, forecast_summary, window_summary, datasets, skipped_models, evaluated_models = _evaluate_requested_models(
            evaluator,
            [model.lower() for model in args.models],
            target_name=target_spec.name,
        )
        evaluated_model_names.update(evaluated_models)
        for item in skipped_models:
            skipped_model_records.append({**metadata, **item})

        annotated_forecasts = _annotate(forecast_df, metadata)
        aggregate_forecast_frames.append(annotated_forecasts)
        aggregate_forecast_summary_frames.append(_annotate(forecast_summary, metadata))
        aggregate_slice_summary_frames.append(
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

        if target_spec.tradable_output:
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
            backtester = CostAwareBacktester(
                BacktestConfig(
                    horizon=int(core_row["horizon"]),
                    transaction_fee_bps=float(policy_baseline["transaction_fee_bps"]),
                    slippage_bps=float(policy_baseline["slippage_bps"]),
                    allow_short=args.allow_short,
                )
            )
            market_data = {
                ticker: dataset.frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                for ticker, dataset in datasets.items()
            }
            backtest_result = backtester.run(position_df, market_data)
            aggregate_signal_frames.append(_annotate(signal_df, metadata))
            aggregate_position_frames.append(_annotate(position_df, metadata))
            aggregate_trade_frames.append(_annotate(backtest_result["trades"], metadata))
            strategy_metrics = backtest_result["strategy_metrics"].copy()
            strategy_metrics["policy_variant"] = policy_baseline["policy_variant"]
            strategy_metrics["policy_label"] = policy_baseline["policy_label"]
            strategy_metrics["sizing_profile"] = policy_baseline["sizing_profile"]
            strategy_metrics["threshold"] = float(policy_baseline["threshold"])
            aggregate_strategy_metric_frames.append(_annotate(strategy_metrics, metadata))

    aggregate_forecasts = pd.concat(aggregate_forecast_frames, ignore_index=True) if aggregate_forecast_frames else pd.DataFrame()
    aggregate_forecast_summary = (
        pd.concat(aggregate_forecast_summary_frames, ignore_index=True) if aggregate_forecast_summary_frames else pd.DataFrame()
    )
    aggregate_slice_summary = (
        pd.concat(aggregate_slice_summary_frames, ignore_index=True) if aggregate_slice_summary_frames else pd.DataFrame()
    )
    aggregate_strategy_metrics = (
        pd.concat(aggregate_strategy_metric_frames, ignore_index=True) if aggregate_strategy_metric_frames else pd.DataFrame()
    )
    aggregate_positions = pd.concat(aggregate_position_frames, ignore_index=True) if aggregate_position_frames else pd.DataFrame()
    aggregate_trades = pd.concat(aggregate_trade_frames, ignore_index=True) if aggregate_trade_frames else pd.DataFrame()
    aggregate_signals = pd.concat(aggregate_signal_frames, ignore_index=True) if aggregate_signal_frames else pd.DataFrame()

    feature_inventory_summary = build_feature_inventory_summary(feature_inventory)
    forecast_quality_summary = build_forecast_quality_summary(aggregate_slice_summary)
    feature_ablation_summary = build_feature_ablation_summary(forecast_quality_summary)
    model_stability_summary = build_model_stability_summary(aggregate_slice_summary)
    forecast_vs_policy_summary = build_forecast_vs_policy_summary(forecast_quality_summary, aggregate_strategy_metrics)
    assessment = build_forecast_rehab_assessment(
        feature_ablation_summary,
        forecast_quality_summary,
        model_stability_summary,
        forecast_vs_policy_summary,
    )

    artifact_paths = write_summary_tables(
        output_dir,
        {
            "matrix_core_runs": core_frame,
            "aggregate_forecasts": aggregate_forecasts,
            "aggregate_forecast_summary": aggregate_forecast_summary,
            "aggregate_slice_summary": aggregate_slice_summary,
            "aggregate_strategy_metrics": aggregate_strategy_metrics,
            "aggregate_positions": aggregate_positions,
            "aggregate_trades": aggregate_trades,
            "aggregate_signals": aggregate_signals,
            "feature_inventory_summary": feature_inventory_summary,
            "feature_ablation_summary": feature_ablation_summary,
            "forecast_quality_summary": forecast_quality_summary,
            "forecast_vs_policy_summary": forecast_vs_policy_summary,
            "model_stability_summary": model_stability_summary,
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
        requested_models=[model.lower() for model in args.models],
        evaluated_models=sorted(evaluated_model_names),
        skipped_models=skipped_model_records,
        target_type="forecast_rehab_multi_target",
        seed=args.seed,
        matrix_config=matrix_config,
        run_counts={
            "core_runs": int(len(core_frame)),
            "forecast_rows": int(len(aggregate_forecasts)),
            "slice_rows": int(len(aggregate_slice_summary)),
            "policy_rows": int(len(aggregate_strategy_metrics)),
        },
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
        manifest_type="forecast_rehab_manifest_v1",
    )
    batch_manifest["assessment"] = assessment
    batch_manifest["policy_baseline"] = dict(policy_baseline)

    summary_path = write_summary_markdown(
        output_dir,
        render_forecast_rehab_summary_markdown(
            batch_manifest,
            assessment,
            feature_ablation_summary,
            forecast_quality_summary,
        ),
    )
    report_path = write_summary_markdown(
        output_dir,
        build_forecast_rehab_report(
            batch_manifest,
            feature_inventory_summary,
            feature_ablation_summary,
            forecast_quality_summary,
            model_stability_summary,
            forecast_vs_policy_summary,
            assessment,
        ),
        filename="forecast_rehab_report.md",
    )
    batch_manifest["artifact_paths"] = {
        **artifact_paths,
        "summary": str(summary_path),
        "forecast_rehab_report": str(report_path),
    }
    manifest_path = write_run_manifest(output_dir, batch_manifest)

    print(f"Preset: {args.preset}")
    print(f"Core runs: {len(core_frame)}")
    print(f"Forecast rows: {len(aggregate_forecasts)}")
    print(f"Policy rows: {len(aggregate_strategy_metrics)}")
    print(f"Recommendation: {assessment.get('recommendation')}")
    print(f"Best feature family: {assessment.get('best_feature_family')}")
    print(f"Best target: {assessment.get('best_target_name')}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
