"""Legacy module.
Retained for historical compatibility or migration reference.
Not part of canonical governed runtime.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_phase2_benchmark import Phase2BenchmarkSpec, run_phase2_core, run_phase2_strategy_suite
from src.evaluation.hardening import (
    build_phase25_core_frame,
    build_phase25_matrix_config,
    build_phase25_sweep_frame,
)
from src.forecast.registry import supported_forecast_models
from src.reporting.hardening import (
    build_cost_sensitivity_summary,
    build_grouped_metric_summary,
    build_phase25_report,
    build_phase25_stability_summary,
    build_phase3_readiness_assessment,
    build_regime_stability_summary,
    build_risk_stability_summary,
    render_phase25_summary_markdown,
)
from src.reporting.manifests import (
    build_batch_manifest,
    collect_dependency_versions,
    collect_git_metadata,
    collect_runtime_metadata,
    write_run_manifest,
)
from src.reporting.summary import write_summary_markdown, write_summary_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2.5 hardening benchmark matrix.")
    parser.add_argument("--preset", choices=["smoke", "medium", "full"], default="medium")
    parser.add_argument("--models", nargs="+", default=list(supported_forecast_models()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-budget", type=float, default=0.02)
    parser.add_argument("--max-position-size", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--regime-lookback", type=int, default=20)
    parser.add_argument("--regime-bull-threshold", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold", type=float, default=-0.03)
    parser.add_argument("--output-dir", default="artifacts/phase25_hardening")
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


def _build_ticker_trade_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["ticker", "strategy_variant", "trade_count"])
    return (
        trades_df.groupby(
            ["ticker", "strategy_variant", "horizon", "threshold", "cost_mode", "sizing_mode"],
            sort=True,
        )
        .agg(
            trade_count=("net_trade_return", "size"),
            mean_net_trade_return=("net_trade_return", "mean"),
            median_net_trade_return=("net_trade_return", "median"),
            mean_position_size=("position_size", "mean"),
        )
        .reset_index()
        .sort_values(["ticker", "strategy_variant", "horizon", "threshold", "cost_mode", "sizing_mode"])
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    matrix_config = build_phase25_matrix_config(args.preset)
    core_frame = build_phase25_core_frame(matrix_config)
    sweep_frame = build_phase25_sweep_frame(matrix_config)

    evaluated_model_names: set[str] = set()
    skipped_model_records: list[dict[str, Any]] = []
    aggregate_results_frames: list[pd.DataFrame] = []
    aggregate_comparison_frames: list[pd.DataFrame] = []
    aggregate_strategy_metric_frames: list[pd.DataFrame] = []
    aggregate_position_frames: list[pd.DataFrame] = []
    aggregate_trade_frames: list[pd.DataFrame] = []
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
            policy_spec = Phase2BenchmarkSpec(
                tickers=list(sweep_row["tickers"]),
                models=[model.lower() for model in args.models],
                horizon=int(sweep_row["horizon"]),
                train_size=int(sweep_row["train_size"]),
                test_size=int(sweep_row["test_size"]),
                step_size=int(sweep_row["step_size"]),
                gap_size=int(sweep_row["gap_size"]),
                max_windows=int(sweep_row["max_windows"]),
                threshold=float(sweep_row["threshold"]),
                risk_budget=args.risk_budget,
                max_position_size=args.max_position_size,
                transaction_fee_bps=float(sweep_row["transaction_fee_bps"]),
                slippage_bps=float(sweep_row["slippage_bps"]),
                seed=args.seed,
                allow_short=args.allow_short,
                regime_lookback=args.regime_lookback,
                regime_bull_threshold=args.regime_bull_threshold,
                regime_bear_threshold=args.regime_bear_threshold,
                strategy_modes=tuple(matrix_config["strategy_variants"]),
                sizing_mode=str(sweep_row["sizing_mode"]),
                fixed_position_size=sweep_row.get("fixed_position_size"),
                group_name=str(sweep_row["group_name"]),
                cost_mode=str(sweep_row["cost_mode"]),
                run_label=str(sweep_row["run_id"]),
            )
            strategy_result = run_phase2_strategy_suite(core_result, policy_spec)

            positions = _annotate(strategy_result["positions"], run_metadata)
            positions["configured_max_position_size"] = float(args.max_position_size)
            positions["configured_fixed_position_size"] = (
                float(sweep_row["fixed_position_size"])
                if pd.notna(sweep_row.get("fixed_position_size"))
                else np.nan
            )

            aggregate_results_frames.append(_annotate(strategy_result["conditioning_mode_summary"], run_metadata))
            aggregate_comparison_frames.append(_annotate(strategy_result["phase2_comparison_summary"], run_metadata))
            aggregate_strategy_metric_frames.append(_annotate(strategy_result["strategy_metrics"], run_metadata))
            aggregate_position_frames.append(positions)
            aggregate_trade_frames.append(_annotate(strategy_result["trades"], run_metadata))

    aggregate_results = pd.concat(aggregate_results_frames, ignore_index=True) if aggregate_results_frames else pd.DataFrame()
    aggregate_model_comparison = (
        pd.concat(aggregate_comparison_frames, ignore_index=True) if aggregate_comparison_frames else pd.DataFrame()
    )
    aggregate_strategy_metrics = (
        pd.concat(aggregate_strategy_metric_frames, ignore_index=True) if aggregate_strategy_metric_frames else pd.DataFrame()
    )
    aggregate_positions = pd.concat(aggregate_position_frames, ignore_index=True) if aggregate_position_frames else pd.DataFrame()
    aggregate_trades = pd.concat(aggregate_trade_frames, ignore_index=True) if aggregate_trade_frames else pd.DataFrame()
    aggregate_forecast_summary = (
        pd.concat(aggregate_forecast_summary_frames, ignore_index=True) if aggregate_forecast_summary_frames else pd.DataFrame()
    )
    aggregate_regime = pd.concat(aggregate_regime_frames, ignore_index=True) if aggregate_regime_frames else pd.DataFrame()
    aggregate_risk = pd.concat(aggregate_risk_frames, ignore_index=True) if aggregate_risk_frames else pd.DataFrame()

    stability_summary = build_phase25_stability_summary(aggregate_results)
    regime_stability_summary = build_regime_stability_summary(aggregate_regime)
    risk_stability_summary = build_risk_stability_summary(aggregate_positions)
    cost_sensitivity_summary = build_cost_sensitivity_summary(aggregate_results)
    group_comparison_summary = build_grouped_metric_summary(
        aggregate_results,
        group_columns=["group_name", "strategy_variant"],
    )
    horizon_comparison_summary = build_grouped_metric_summary(
        aggregate_results,
        group_columns=["horizon", "strategy_variant"],
    )
    threshold_comparison_summary = build_grouped_metric_summary(
        aggregate_results,
        group_columns=["threshold", "strategy_variant"],
    )
    ticker_forecast_summary = (
        aggregate_forecast_summary.groupby(["ticker", "horizon", "model_name"], sort=True)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["ticker", "horizon", "model_name"])
        .reset_index(drop=True)
        if not aggregate_forecast_summary.empty
        else pd.DataFrame(columns=["ticker", "horizon", "model_name"])
    )
    ticker_trade_summary = _build_ticker_trade_summary(aggregate_trades)

    artifact_paths = write_summary_tables(
        output_dir,
        {
            "matrix_core_runs": core_frame,
            "matrix_sweep_runs": sweep_frame,
            "aggregate_results": aggregate_results,
            "aggregate_model_comparison": aggregate_model_comparison,
            "aggregate_strategy_metrics": aggregate_strategy_metrics,
            "aggregate_forecast_summary": aggregate_forecast_summary,
            "stability_summary": stability_summary,
            "regime_stability_summary": regime_stability_summary,
            "risk_stability_summary": risk_stability_summary,
            "cost_sensitivity_summary": cost_sensitivity_summary,
            "group_comparison_summary": group_comparison_summary,
            "horizon_comparison_summary": horizon_comparison_summary,
            "threshold_comparison_summary": threshold_comparison_summary,
            "ticker_forecast_summary": ticker_forecast_summary,
            "ticker_trade_summary": ticker_trade_summary,
        },
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    assessment = build_phase3_readiness_assessment(
        aggregate_results,
        regime_stability_summary,
        cost_sensitivity_summary,
    )
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
            "conditioning_rows": int(len(aggregate_results)),
        },
        artifact_paths=artifact_paths,
        started_at=started_at,
        completed_at=completed_at,
    )

    summary_path = write_summary_markdown(
        output_dir,
        render_phase25_summary_markdown(
            batch_manifest,
            aggregate_results,
            stability_summary,
            cost_sensitivity_summary,
            assessment,
        ),
    )
    report_path = write_summary_markdown(
        output_dir,
        build_phase25_report(
            batch_manifest,
            aggregate_results,
            stability_summary,
            regime_stability_summary,
            risk_stability_summary,
            cost_sensitivity_summary,
            assessment,
        ),
        filename="phase25_report.md",
    )
    batch_manifest["artifact_paths"] = {
        **artifact_paths,
        "summary": str(summary_path),
        "phase25_report": str(report_path),
    }
    batch_manifest["assessment"] = assessment
    manifest_path = write_run_manifest(output_dir, batch_manifest)

    print(f"Preset: {args.preset}")
    print(f"Core runs: {len(core_frame)}")
    print(f"Policy runs: {len(sweep_frame)}")
    print(f"Aggregate conditioning rows: {len(aggregate_results)}")
    print("Stability summary:")
    print(stability_summary.to_string(index=False))
    print("Cost sensitivity summary:")
    print(cost_sensitivity_summary.to_string(index=False))
    print(f"Recommendation: {assessment['recommendation']}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
