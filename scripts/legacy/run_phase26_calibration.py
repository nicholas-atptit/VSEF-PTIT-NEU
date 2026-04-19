"""Bounded Phase 2.6 calibration runner for policy ablations."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_phase2_benchmark import Phase2BenchmarkSpec, run_phase2_core
from src.evaluation.backtest import BacktestConfig, CostAwareBacktester
from src.evaluation.calibration import (
    build_phase26_core_frame,
    build_phase26_matrix_config,
    build_phase26_sweep_frame,
)
from src.forecast.registry import supported_forecast_models
from src.reporting.calibration import (
    build_forecast_vs_policy_summary,
    build_phase26_assessment,
    build_phase26_report,
    build_policy_ablation_summary,
    build_policy_cost_sensitivity_summary,
    build_policy_run_summary,
    build_regime_value_summary,
    build_sizing_calibration_summary,
    build_threshold_calibration_summary,
    render_phase26_summary_markdown,
)
from src.reporting.hardening import build_regime_stability_summary, build_risk_stability_summary
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
    parser = argparse.ArgumentParser(description="Run the bounded Phase 2.6 policy calibration matrix.")
    parser.add_argument("--preset", choices=["smoke", "medium", "calibration_full"], default="medium")
    parser.add_argument("--models", nargs="+", default=list(supported_forecast_models()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-budget", type=float, default=0.02)
    parser.add_argument("--max-position-size", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--regime-lookback", type=int, default=20)
    parser.add_argument("--regime-bull-threshold", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold", type=float, default=-0.03)
    parser.add_argument("--output-dir", default="artifacts/phase26_calibration")
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


def _build_policy_configuration(sweep_row: dict[str, Any]) -> PolicyConfiguration:
    return PolicyConfiguration(
        policy_variant=str(sweep_row["policy_variant"]),
        strategy_variant=str(sweep_row["strategy_variant"]),
        policy_label=str(sweep_row.get("policy_label", sweep_row["policy_variant"])),
        threshold_policy=str(sweep_row.get("threshold_policy", "fixed")),
        sizing_profile=str(sweep_row.get("sizing_profile", "fixed_fraction_full")),
        sizing_label=str(sweep_row.get("sizing_label", sweep_row.get("sizing_profile", "fixed_fraction_full"))),
        use_risk_context=bool(sweep_row.get("use_risk_context", False)),
        use_regime_context=bool(sweep_row.get("use_regime_context", False)),
        use_volatility_sizing=bool(sweep_row.get("use_volatility_sizing", False)),
        use_drawdown_control=bool(sweep_row.get("use_drawdown_control", False)),
        use_regime_sizing=bool(sweep_row.get("use_regime_sizing", False)),
        sizing_mode=str(sweep_row.get("sizing_mode", "fixed_fraction")),
        fixed_position_size=(
            None if pd.isna(sweep_row.get("fixed_position_size")) else float(sweep_row.get("fixed_position_size"))
        ),
        min_position_size=float(sweep_row.get("min_position_size", 0.0)),
        max_position_size=float(sweep_row.get("max_position_size", 1.0)),
        volatility_target_scale=float(sweep_row.get("volatility_target_scale", 1.0)),
        drawdown_haircut_strength=float(sweep_row.get("drawdown_haircut_strength", 1.0)),
        regime_multiplier_strength=float(sweep_row.get("regime_multiplier_strength", 1.0)),
        policy_family=str(sweep_row.get("policy_family", sweep_row["policy_variant"])),
        ablation_labels=tuple(str(item) for item in sweep_row.get("ablation_labels", [])),
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    matrix_config = build_phase26_matrix_config(args.preset)
    core_frame = build_phase26_core_frame(matrix_config)
    sweep_frame = build_phase26_sweep_frame(matrix_config)

    evaluated_model_names: set[str] = set()
    skipped_model_records: list[dict[str, Any]] = []
    aggregate_results_frames: list[pd.DataFrame] = []
    aggregate_model_metric_frames: list[pd.DataFrame] = []
    aggregate_position_frames: list[pd.DataFrame] = []
    aggregate_trade_frames: list[pd.DataFrame] = []
    aggregate_signal_frames: list[pd.DataFrame] = []
    aggregate_forecast_summary_frames: list[pd.DataFrame] = []
    aggregate_regime_frames: list[pd.DataFrame] = []
    aggregate_risk_frames: list[pd.DataFrame] = []

    core_outputs_dir = output_dir / "core_runs"
    core_outputs_dir.mkdir(parents=True, exist_ok=True)

    for core_row in core_frame.to_dict(orient="records"):
        core_metadata = _normalize_metadata(core_row)
        core_spec = Phase2BenchmarkSpec(
            tickers=list(core_row["tickers"]),
            models=[model.lower() for model in args.models],
            horizon=int(core_row["horizon"]),
            train_size=int(core_row["train_size"]),
            test_size=int(core_row["test_size"]),
            step_size=int(core_row["step_size"]),
            gap_size=int(core_row["gap_size"]),
            max_windows=int(core_row["max_windows"]),
            threshold=float(matrix_config["thresholds"][0]),
            risk_budget=args.risk_budget,
            max_position_size=args.max_position_size,
            transaction_fee_bps=float(matrix_config["cost_modes"][0]["transaction_fee_bps"]),
            slippage_bps=float(matrix_config["cost_modes"][0]["slippage_bps"]),
            seed=args.seed,
            allow_short=args.allow_short,
            regime_lookback=args.regime_lookback,
            regime_bull_threshold=args.regime_bull_threshold,
            regime_bear_threshold=args.regime_bear_threshold,
            group_name=str(core_row["group_name"]),
            run_label=str(core_row["core_run_id"]),
        )
        core_result = run_phase2_core(core_spec)
        evaluated_model_names.update(core_result["evaluated_models"])
        window_count = int(core_result["window_summary"]["window_id"].nunique()) if not core_result["window_summary"].empty else 0
        core_metadata["window_count"] = window_count

        for item in core_result["skipped_models"]:
            skipped_model_records.append(
                {
                    **core_metadata,
                    "model_name": item["model_name"],
                    "reason": item["reason"],
                }
            )

        aggregate_forecast_summary_frames.append(_annotate(core_result["forecast_summary"], core_metadata))
        aggregate_regime_frames.append(_annotate(core_result["regime_summary"], core_metadata))
        aggregate_risk_frames.append(_annotate(core_result["risk_summary"], core_metadata))

        write_summary_tables(
            core_outputs_dir / str(core_row["core_run_id"]),
            {
                "forecast_summary": core_result["forecast_summary"],
                "forecast_summary_by_horizon": core_result["forecast_summary_by_horizon"],
                "window_summary": core_result["window_summary"],
                "risk_summary": core_result["risk_summary"],
                "regime_summary": core_result["regime_summary"],
            },
        )

        core_sweep = sweep_frame[sweep_frame["core_run_id"] == core_row["core_run_id"]].copy()
        for sweep_row in core_sweep.to_dict(orient="records"):
            run_metadata = _normalize_metadata(sweep_row)
            run_metadata["window_count"] = window_count
            policy_config = _build_policy_configuration(sweep_row)

            signal_df, position_df = execute_policy_configuration(
                core_result["forecasts"],
                policy_config=policy_config,
                threshold=float(sweep_row["threshold"]),
                allow_short=args.allow_short,
                risk_df=core_result["risk_summary"],
                regime_df=core_result["regime_summary"],
                capital_config={
                    "risk_budget": args.risk_budget,
                    "max_position_size": args.max_position_size,
                },
            )
            backtester = CostAwareBacktester(
                BacktestConfig(
                    horizon=int(sweep_row["horizon"]),
                    transaction_fee_bps=float(sweep_row["transaction_fee_bps"]),
                    slippage_bps=float(sweep_row["slippage_bps"]),
                    allow_short=args.allow_short,
                )
            )
            backtest_result = backtester.run(position_df, core_result["market_data"])

            signals = _annotate(signal_df, run_metadata)
            positions = _annotate(position_df, run_metadata)
            trades = _annotate(backtest_result["trades"], run_metadata)
            strategy_metrics = _annotate(backtest_result["strategy_metrics"], run_metadata)

            aggregate_signal_frames.append(signals)
            aggregate_position_frames.append(positions)
            aggregate_trade_frames.append(trades)
            aggregate_model_metric_frames.append(strategy_metrics)
            aggregate_results_frames.append(_annotate(build_policy_run_summary(strategy_metrics), run_metadata))

    aggregate_results = pd.concat(aggregate_results_frames, ignore_index=True) if aggregate_results_frames else pd.DataFrame()
    aggregate_model_metrics = (
        pd.concat(aggregate_model_metric_frames, ignore_index=True) if aggregate_model_metric_frames else pd.DataFrame()
    )
    aggregate_positions = pd.concat(aggregate_position_frames, ignore_index=True) if aggregate_position_frames else pd.DataFrame()
    aggregate_trades = pd.concat(aggregate_trade_frames, ignore_index=True) if aggregate_trade_frames else pd.DataFrame()
    aggregate_signals = pd.concat(aggregate_signal_frames, ignore_index=True) if aggregate_signal_frames else pd.DataFrame()
    aggregate_forecast_summary = (
        pd.concat(aggregate_forecast_summary_frames, ignore_index=True) if aggregate_forecast_summary_frames else pd.DataFrame()
    )
    aggregate_regime = pd.concat(aggregate_regime_frames, ignore_index=True) if aggregate_regime_frames else pd.DataFrame()
    aggregate_risk = pd.concat(aggregate_risk_frames, ignore_index=True) if aggregate_risk_frames else pd.DataFrame()

    policy_ablation_summary = build_policy_ablation_summary(aggregate_results)
    threshold_calibration_summary = build_threshold_calibration_summary(aggregate_results)
    sizing_calibration_summary = build_sizing_calibration_summary(aggregate_results, aggregate_positions)
    regime_value_summary = build_regime_value_summary(aggregate_results)
    forecast_vs_policy_summary = build_forecast_vs_policy_summary(aggregate_forecast_summary, aggregate_model_metrics)
    regime_stability_summary = build_regime_stability_summary(aggregate_regime)
    risk_stability_summary = build_risk_stability_summary(aggregate_positions)
    cost_sensitivity_summary = build_policy_cost_sensitivity_summary(aggregate_results)
    assessment = build_phase26_assessment(
        policy_ablation_summary,
        regime_value_summary,
        forecast_vs_policy_summary,
        cost_sensitivity_summary,
    )

    artifact_paths = write_summary_tables(
        output_dir,
        {
            "matrix_core_runs": core_frame,
            "matrix_policy_runs": sweep_frame,
            "aggregate_results": aggregate_results,
            "aggregate_model_strategy_metrics": aggregate_model_metrics,
            "aggregate_positions": aggregate_positions,
            "aggregate_trades": aggregate_trades,
            "aggregate_signals": aggregate_signals,
            "aggregate_forecast_summary": aggregate_forecast_summary,
            "policy_ablation_summary": policy_ablation_summary,
            "threshold_calibration_summary": threshold_calibration_summary,
            "sizing_calibration_summary": sizing_calibration_summary,
            "regime_value_summary": regime_value_summary,
            "forecast_vs_policy_summary": forecast_vs_policy_summary,
            "regime_stability_summary": regime_stability_summary,
            "risk_stability_summary": risk_stability_summary,
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
        requested_models=[model.lower() for model in args.models],
        evaluated_models=sorted(evaluated_model_names),
        skipped_models=skipped_model_records,
        target_type="forward_return",
        seed=args.seed,
        matrix_config=matrix_config,
        run_counts={
            "core_runs": int(len(core_frame)),
            "policy_runs": int(len(sweep_frame)),
            "policy_result_rows": int(len(aggregate_results)),
            "model_result_rows": int(len(aggregate_model_metrics)),
        },
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
        manifest_type="phase26_calibration_manifest_v1",
    )

    summary_path = write_summary_markdown(
        output_dir,
        render_phase26_summary_markdown(
            batch_manifest,
            policy_ablation_summary,
            threshold_calibration_summary,
            sizing_calibration_summary,
            regime_value_summary,
            assessment,
        ),
    )
    report_path = write_summary_markdown(
        output_dir,
        build_phase26_report(
            batch_manifest,
            policy_ablation_summary,
            threshold_calibration_summary,
            sizing_calibration_summary,
            regime_value_summary,
            forecast_vs_policy_summary,
            cost_sensitivity_summary,
            assessment,
        ),
        filename="phase26_report.md",
    )
    batch_manifest["artifact_paths"] = {
        **artifact_paths,
        "summary": str(summary_path),
        "phase26_report": str(report_path),
    }
    batch_manifest["assessment"] = assessment
    manifest_path = write_run_manifest(output_dir, batch_manifest)

    print(f"Preset: {args.preset}")
    print(f"Core runs: {len(core_frame)}")
    print(f"Policy runs: {len(sweep_frame)}")
    print(f"Policy result rows: {len(aggregate_results)}")
    print(f"Model result rows: {len(aggregate_model_metrics)}")
    print(f"Recommendation: {assessment['recommendation']}")
    print(f"Default policy candidate: {assessment.get('default_policy_candidate')}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
