"""Top-level quant-core governance and orchestration runner."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.model_governance import RUN_MODES
from src.evaluation.consensus import build_model_consensus_summary
from src.evaluation.quant_core import (
    PRESET_CONFIGS,
    build_quant_core_core_frame,
    build_quant_core_matrix_config,
    run_quant_core_scenario,
)
from src.evaluation.targets import supported_target_specs
from src.forecast.registry import forecast_model_governance_table, supported_forecast_models
from src.reporting.analysis_packets import (
    build_analysis_packets,
    build_decision_lane_candidates,
    write_analysis_packets_jsonl,
)
from src.reporting.manifests import (
    collect_dependency_versions,
    collect_git_metadata,
    collect_runtime_metadata,
    write_run_manifest,
)
from src.reporting.model_health import build_model_health_summary
from src.reporting.quant_core import build_quant_core_manifest, render_quant_core_summary_markdown
from src.reporting.summary import write_summary_markdown, write_summary_tables
from src.risk_governance import run_risk_governance, write_risk_governance_outputs
from src.scenario import ScenarioEngineConfig, run_scenario_evaluation, write_scenario_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the governed quant-core orchestration path.")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), default="medium")
    parser.add_argument("--run-mode", choices=list(RUN_MODES), default="full_forecast")
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit custom ticker list")
    parser.add_argument("--ticker-groups", nargs="+", default=None, help="Named ticker groups to run")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--target-types", nargs="+", default=None, choices=supported_target_specs())
    parser.add_argument("--models", nargs="+", default=None, help="Optional explicit model-name subset")
    parser.add_argument("--model-roles", nargs="+", default=None, help="Optional role filter on top of run mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Accepted for repeated-seed orchestration compatibility; single-run Quant Core is currently serial.",
    )
    parser.add_argument("--risk-budget", type=float, default=0.02)
    parser.add_argument("--max-position-size", type=float, default=1.0)
    parser.add_argument("--transaction-fee-bps", type=float, default=15.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--regime-lookback", type=int, default=20)
    parser.add_argument("--regime-bull-threshold", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold", type=float, default=-0.03)
    parser.add_argument("--output-dir", default="artifacts/quant_core")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--enable-scenario-engine", action="store_true")
    parser.add_argument("--enable-risk-governance", action="store_true")
    parser.add_argument("--scenario-calibration-lookback", type=int, default=252)
    parser.add_argument("--scenario-probability-method", choices=["deterministic_v1"], default="deterministic_v1")
    return parser.parse_args()


def _build_custom_ticker_groups(args: argparse.Namespace) -> tuple[list[str] | None, dict[str, list[str]] | None]:
    if args.tickers:
        return ["cli_custom"], {"cli_custom": [str(ticker).upper() for ticker in args.tickers]}
    if args.ticker_groups:
        return [str(name) for name in args.ticker_groups], None
    return None, None


def _annotate(frame: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    annotated = frame.copy()
    for key, value in metadata.items():
        annotated[key] = value
    return annotated


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    group_names, custom_groups = _build_custom_ticker_groups(args)
    matrix_config = build_quant_core_matrix_config(
        args.preset,
        group_names=group_names,
        ticker_groups=custom_groups,
        horizons=args.horizons,
        target_names=args.target_types,
    )
    core_frame = build_quant_core_core_frame(matrix_config)
    if core_frame.empty:
        raise RuntimeError("The quant-core scenario matrix did not produce any runs")

    governance_frame = forecast_model_governance_table(
        run_mode=args.run_mode,
        roles=args.model_roles,
        model_names=args.models,
        include_parked=False,
    )
    requested_models = [str(model).lower() for model in args.models] if args.models else list(
        supported_forecast_models(
            run_mode=args.run_mode,
            roles=args.model_roles,
            include_parked=False,
        )
    )

    aggregate_forecasts: list[pd.DataFrame] = []
    aggregate_forecast_summary: list[pd.DataFrame] = []
    aggregate_forecast_summary_by_horizon: list[pd.DataFrame] = []
    aggregate_window_summary: list[pd.DataFrame] = []
    aggregate_risk_summary: list[pd.DataFrame] = []
    aggregate_regime_summary: list[pd.DataFrame] = []
    aggregate_signals: list[pd.DataFrame] = []
    aggregate_positions: list[pd.DataFrame] = []
    aggregate_trades: list[pd.DataFrame] = []
    aggregate_strategy_metrics: list[pd.DataFrame] = []
    aggregate_equity_curve: list[pd.DataFrame] = []
    aggregate_policy_summary: list[pd.DataFrame] = []
    aggregate_execution_log: list[pd.DataFrame] = []
    skipped_models: list[dict[str, object]] = []
    evaluated_models: set[str] = set()

    for core_row in core_frame.to_dict(orient="records"):
        scenario_result = run_quant_core_scenario(
            core_row,
            run_mode=args.run_mode,
            requested_model_names=requested_models,
            requested_roles=args.model_roles,
            include_ensemble=not args.no_ensemble,
            seed=args.seed,
            allow_short=args.allow_short,
            risk_budget=args.risk_budget,
            max_position_size=args.max_position_size,
            transaction_fee_bps=args.transaction_fee_bps,
            slippage_bps=args.slippage_bps,
            regime_lookback=args.regime_lookback,
            regime_bull_threshold=args.regime_bull_threshold,
            regime_bear_threshold=args.regime_bear_threshold,
        )
        evaluated_models.update(scenario_result["evaluated_models"])
        skipped_models.extend(scenario_result["skipped_models"])
        aggregate_forecasts.append(scenario_result["forecasts"])
        aggregate_forecast_summary.append(scenario_result["forecast_summary"])
        aggregate_forecast_summary_by_horizon.append(scenario_result["forecast_summary_by_horizon"])
        aggregate_window_summary.append(scenario_result["window_summary"])
        aggregate_risk_summary.append(scenario_result["risk_summary"])
        aggregate_regime_summary.append(scenario_result["regime_summary"])
        if not scenario_result["signals"].empty:
            aggregate_signals.append(scenario_result["signals"])
        if not scenario_result["positions"].empty:
            aggregate_positions.append(scenario_result["positions"])
        if not scenario_result["trades"].empty:
            aggregate_trades.append(scenario_result["trades"])
        if not scenario_result["strategy_metrics"].empty:
            aggregate_strategy_metrics.append(scenario_result["strategy_metrics"])
        if not scenario_result["equity_curve"].empty:
            aggregate_equity_curve.append(scenario_result["equity_curve"])
        if not scenario_result["policy_summary"].empty:
            aggregate_policy_summary.append(scenario_result["policy_summary"])
        aggregate_execution_log.append(scenario_result["model_execution_log"])

    forecasts = pd.concat(aggregate_forecasts, ignore_index=True) if aggregate_forecasts else pd.DataFrame()
    forecast_summary = pd.concat(aggregate_forecast_summary, ignore_index=True) if aggregate_forecast_summary else pd.DataFrame()
    forecast_summary_by_horizon = (
        pd.concat(aggregate_forecast_summary_by_horizon, ignore_index=True)
        if aggregate_forecast_summary_by_horizon
        else pd.DataFrame()
    )
    window_summary = pd.concat(aggregate_window_summary, ignore_index=True) if aggregate_window_summary else pd.DataFrame()
    risk_summary = pd.concat(aggregate_risk_summary, ignore_index=True) if aggregate_risk_summary else pd.DataFrame()
    regime_summary = pd.concat(aggregate_regime_summary, ignore_index=True) if aggregate_regime_summary else pd.DataFrame()
    signals = pd.concat(aggregate_signals, ignore_index=True) if aggregate_signals else pd.DataFrame()
    positions = pd.concat(aggregate_positions, ignore_index=True) if aggregate_positions else pd.DataFrame()
    trades = pd.concat(aggregate_trades, ignore_index=True) if aggregate_trades else pd.DataFrame()
    strategy_metrics = pd.concat(aggregate_strategy_metrics, ignore_index=True) if aggregate_strategy_metrics else pd.DataFrame()
    equity_curve = pd.concat(aggregate_equity_curve, ignore_index=True) if aggregate_equity_curve else pd.DataFrame()
    policy_summary = pd.concat(aggregate_policy_summary, ignore_index=True) if aggregate_policy_summary else pd.DataFrame()
    model_execution_log = pd.concat(aggregate_execution_log, ignore_index=True) if aggregate_execution_log else pd.DataFrame()
    model_consensus_summary = build_model_consensus_summary(
        forecasts,
        signals_df=signals,
    )
    analysis_packets = build_analysis_packets(
        forecasts,
        model_consensus_summary,
        risk_df=risk_summary,
        regime_df=regime_summary,
        signals_df=signals,
        positions_df=positions,
        strategy_metrics_df=strategy_metrics,
    )
    model_health_summary = build_model_health_summary(
        model_execution_log,
        forecasts,
        strategy_metrics,
    )
    scenario_result = None
    if args.enable_scenario_engine:
        scenario_result = run_scenario_evaluation(
            forecasts_df=forecasts,
            consensus_df=model_consensus_summary,
            risk_df=risk_summary,
            regime_df=regime_summary,
            strategy_metrics_df=strategy_metrics,
            analysis_packets_df=analysis_packets,
            model_health_df=model_health_summary,
            config=ScenarioEngineConfig(
                probability_method=args.scenario_probability_method,
                calibration_lookback=args.scenario_calibration_lookback,
            ),
        )
        analysis_packets = scenario_result.analysis_packets
    decision_lane_candidates = build_decision_lane_candidates(analysis_packets)
    risk_governance_result = None
    if args.enable_risk_governance:
        risk_governance_result = run_risk_governance(
            candidates_df=decision_lane_candidates,
            packets_df=analysis_packets,
            risk_df=risk_summary,
            consensus_df=model_consensus_summary,
            model_health_df=model_health_summary,
            scenario_dominance_df=scenario_result.scenario_dominance_summary if scenario_result is not None else None,
            scenario_uncertainty_df=(
                scenario_result.scenario_uncertainty_summary if scenario_result is not None else None
            ),
            scenario_probability_df=scenario_result.scenario_probability if scenario_result is not None else None,
        )

    table_paths = write_summary_tables(
        output_dir,
        {
            "scenario_matrix": core_frame,
            "model_governance": governance_frame,
            "full_model_predictions": forecasts,
            "forecast_summary": forecast_summary,
            "forecast_summary_by_horizon": forecast_summary_by_horizon,
            "window_summary": window_summary,
            "risk_summary": risk_summary,
            "regime_summary": regime_summary,
            "policy_summary": policy_summary,
            "signals": signals,
            "positions": positions,
            "trades": trades,
            "strategy_metrics": strategy_metrics,
            "equity_curve": equity_curve,
            "model_execution_log": model_execution_log,
            "model_consensus_summary": model_consensus_summary,
            "model_health_summary": model_health_summary,
            "decision_lane_candidates": decision_lane_candidates,
        },
    )
    scenario_artifact_paths = write_scenario_outputs(output_dir, scenario_result) if scenario_result is not None else {}
    risk_governance_artifact_paths = (
        write_risk_governance_outputs(output_dir, risk_governance_result)
        if risk_governance_result is not None
        else {}
    )
    analysis_packets_path = write_analysis_packets_jsonl(output_dir, analysis_packets)

    completed_at = datetime.now(timezone.utc).isoformat()
    run_counts = {
        "scenario_count": int(len(core_frame)),
        "forecast_rows": int(len(forecasts)),
        "risk_rows": int(len(risk_summary)),
        "regime_rows": int(len(regime_summary)),
        "strategy_rows": int(len(strategy_metrics)),
        "analysis_packet_rows": int(len(analysis_packets)),
    }
    if scenario_result is not None:
        run_counts.update(
            {
                "scenario_probability_rows": int(len(scenario_result.scenario_probability)),
                "scenario_ranking_rows": int(len(scenario_result.scenario_rankings)),
                "scenario_dominance_rows": int(len(scenario_result.scenario_dominance_summary)),
            }
        )
    if risk_governance_result is not None:
        run_counts.update(
            {
                "risk_governance_rows": int(len(risk_governance_result.risk_governance_summary)),
                "risk_adjusted_candidate_rows": int(len(risk_governance_result.risk_adjusted_candidates)),
                "risk_override_rows": int(len(risk_governance_result.risk_override_log)),
            }
        )
    manifest = build_quant_core_manifest(
        git_metadata=collect_git_metadata(Path.cwd()),
        runtime=collect_runtime_metadata(),
        dependency_versions=collect_dependency_versions(
            ["pandas", "numpy", "statsmodels", "arch", "scikit-learn", "xgboost", "lightgbm"]
        ),
        command=" ".join(sys.argv),
        requested_models=requested_models,
        evaluated_models=sorted(evaluated_models),
        skipped_models=skipped_models,
        seed=args.seed,
        matrix_config=matrix_config,
        run_counts=run_counts,
        artifact_paths={
            **dict(table_paths),
            **scenario_artifact_paths,
            **risk_governance_artifact_paths,
            "analysis_packets": str(analysis_packets_path),
        },
        started_at=started_at,
        completed_at=completed_at,
        run_mode=args.run_mode,
        requested_model_roles=list(args.model_roles or []),
        governance_frame=governance_frame,
    )

    manifest_path = write_run_manifest(output_dir, manifest)
    summary_path = write_summary_markdown(
        output_dir,
        render_quant_core_summary_markdown(
            manifest,
            core_frame,
            governance_frame,
            forecast_summary,
            strategy_metrics,
        ),
    )

    print(f"Run mode: {args.run_mode}")
    print(f"Preset: {args.preset}")
    print(f"Scenarios: {len(core_frame)}")
    print(f"Requested models: {', '.join(requested_models)}")
    print(f"Evaluated models: {', '.join(sorted(evaluated_models))}")
    print(f"Forecast rows: {len(forecasts)}")
    print(f"Strategy rows: {len(strategy_metrics)}")
    if scenario_result is not None:
        print(f"Scenario engine rows: {len(scenario_result.scenario_probability)}")
    if risk_governance_result is not None:
        print(f"Risk governance rows: {len(risk_governance_result.risk_governance_summary)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
