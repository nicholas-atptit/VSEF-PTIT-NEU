"""Validation benchmark for the Phase 2 forecasting, risk, and regime stack."""

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

from src.ensemble.weighted import WeightedEnsembleModel
from src.evaluation.backtest import BacktestConfig, CostAwareBacktester
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator, summarize_forecasts
from src.forecast.registry import create_forecast_model, supported_forecast_models
from src.regime.markov_switching import MarkovSwitchingRegimeModel
from src.reporting.manifests import (
    build_run_manifest,
    collect_dependency_versions,
    collect_git_metadata,
    collect_runtime_metadata,
    write_run_manifest,
)
from src.reporting.summary import (
    build_conditioning_mode_summary,
    build_phase2_conditioning_summary,
    render_phase2_summary_markdown,
    write_summary_markdown,
    write_summary_tables,
)
from src.risk.garch import GARCHRiskModel
from src.strategy.execution_policy import BasicExecutionPolicy, RegimeAwareExecutionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Phase 2 forecasting benchmark.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker subset to benchmark")
    parser.add_argument("--models", nargs="+", default=list(supported_forecast_models()), help="Forecast models to evaluate")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--step-size", type=int, default=21)
    parser.add_argument("--gap-size", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--risk-budget", type=float, default=0.02)
    parser.add_argument("--max-position-size", type=float, default=1.0)
    parser.add_argument("--transaction-fee-bps", type=float, default=15.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/phase2_benchmark")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--regime-lookback", type=int, default=20)
    parser.add_argument("--regime-bull-threshold", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold", type=float, default=-0.03)
    return parser.parse_args()


def evaluate_requested_models(
    evaluator: WalkForwardEvaluator,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, str]], list[str]]:
    forecast_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    evaluated_models: list[str] = []
    datasets: dict[str, Any] | None = None
    window_summary: pd.DataFrame | None = None

    for model_name in model_names:
        model = create_forecast_model(model_name)
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


def build_window_garch_risk_frame(
    datasets: dict[str, Any],
    window_summary: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    risk_rows: list[dict[str, Any]] = []
    for row in window_summary.itertuples(index=False):
        dataset = datasets[str(row.ticker)]
        frame = dataset.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        train_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.train_end))
        ].copy()
        test_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.test_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        returns = pd.to_numeric(train_df["daily_return"], errors="coerce").dropna()
        if returns.empty or test_df.empty:
            continue

        metrics = GARCHRiskModel().fit(returns).forecast_risk(horizon=horizon)
        for timestamp in pd.to_datetime(test_df["timestamp"], errors="coerce").dropna():
            risk_rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": str(row.ticker),
                    "window_id": str(row.window_id),
                    "source_model": metrics["risk_model"],
                    **metrics,
                }
            )
    return pd.DataFrame(risk_rows).sort_values(["timestamp", "ticker", "window_id"]).reset_index(drop=True)


def build_window_regime_frame(
    datasets: dict[str, Any],
    window_summary: pd.DataFrame,
    *,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
) -> pd.DataFrame:
    regime_frames: list[pd.DataFrame] = []
    for row in window_summary.itertuples(index=False):
        dataset = datasets[str(row.ticker)]
        frame = dataset.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        train_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.train_end))
        ].copy()
        history_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        test_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.test_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        if train_df.empty or history_df.empty or test_df.empty:
            continue
        train_df["window_id"] = str(row.window_id)
        history_df["window_id"] = str(row.window_id)
        model = MarkovSwitchingRegimeModel(
            return_column="daily_return",
            lookback=lookback,
            bull_threshold=bull_threshold,
            bear_threshold=bear_threshold,
            min_train_observations=max(80, min(len(train_df), 120)),
        ).fit(train_df, config={"window_id": str(row.window_id)})
        regime_history = model.predict(history_df)
        regime_frames.append(
            regime_history[regime_history["timestamp"].isin(pd.to_datetime(test_df["timestamp"], errors="coerce"))].copy()
        )
    if not regime_frames:
        return pd.DataFrame(columns=["timestamp", "ticker", "regime_label", "regime_prob_bull", "regime_prob_bear", "regime_prob_sideway", "source_model", "window_id"])
    return pd.concat(regime_frames, ignore_index=True).sort_values(["timestamp", "ticker", "window_id"]).reset_index(drop=True)


def build_market_data(datasets: dict[str, Any]) -> dict[str, pd.DataFrame]:
    market_data: dict[str, pd.DataFrame] = {}
    for ticker, dataset in datasets.items():
        market_data[ticker] = dataset.frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    return market_data


def execute_strategy_variant(
    *,
    strategy_variant: str,
    forecasts: pd.DataFrame,
    backtester: CostAwareBacktester,
    market_data: dict[str, pd.DataFrame],
    threshold: float,
    allow_short: bool,
    risk_df: pd.DataFrame | None,
    regime_df: pd.DataFrame | None,
    risk_budget: float,
    max_position_size: float,
) -> dict[str, pd.DataFrame]:
    capital_config = {
        "risk_budget": risk_budget,
        "max_position_size": max_position_size,
    }
    if strategy_variant == "forecast_only":
        policy = BasicExecutionPolicy(
            threshold=threshold,
            allow_short=allow_short,
            capital_config=capital_config,
        )
        signal_df = policy.generate_signal(forecasts)
        position_df = policy.size_positions(signal_df, capital_config=capital_config)
    elif strategy_variant == "forecast_plus_risk":
        policy = BasicExecutionPolicy(
            threshold=threshold,
            allow_short=allow_short,
            capital_config=capital_config,
        )
        signal_df = policy.generate_signal(forecasts, risk_df=risk_df)
        position_df = policy.size_positions(signal_df, risk_df=risk_df, capital_config=capital_config)
    elif strategy_variant == "forecast_plus_risk_and_regime":
        policy = RegimeAwareExecutionPolicy(
            threshold=threshold,
            allow_short=allow_short,
            capital_config=capital_config,
        )
        signal_df = policy.generate_signal(forecasts, risk_df=risk_df, regime_df=regime_df)
        position_df = policy.size_positions(signal_df, risk_df=risk_df, capital_config=capital_config)
    else:
        raise ValueError(f"Unsupported strategy variant '{strategy_variant}'")

    signal_df = signal_df.copy()
    signal_df["strategy_variant"] = strategy_variant
    position_df = position_df.copy()
    position_df["strategy_variant"] = strategy_variant
    backtest_result = backtester.run(position_df, market_data)
    return {
        "signals": signal_df,
        "positions": position_df,
        "trades": backtest_result["trades"],
        "strategy_metrics": backtest_result["strategy_metrics"],
        "equity_curve": backtest_result["equity_curve"],
    }


def build_phase2_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2 Implementation Report",
            "",
            "## Added",
            "",
            "- Regime layer v1 with `MarkovSwitchingRegimeModel` and deterministic fallback",
            "- Risk layer v2 with `GARCHRiskModel` volatility and tail-risk forecasts",
            "- Regime-aware thresholding plus volatility/drawdown-aware sizing",
            "- Conditioning-mode evaluation for `forecast_only`, `forecast_plus_risk`, and `forecast_plus_risk_and_regime`",
            "",
            "## Reused From Phase 1",
            "",
            "- Shared forecast contracts and model registry",
            "- Leakage-safe `WalkForwardEvaluator`",
            "- Cost-aware `CostAwareBacktester`",
            "- Weighted ensemble forecast combiner",
            "",
            "## Validated",
            "",
            f"- Runtime: `{manifest.get('runtime', {}).get('python_executable')}`",
            f"- Dependencies: `statsmodels={manifest.get('dependency_versions', {}).get('statsmodels')}`, `arch={manifest.get('dependency_versions', {}).get('arch')}`",
            f"- Benchmark modes: `{', '.join(manifest.get('benchmark_modes', []))}`",
            "",
            "## Deferred To Phase 3",
            "",
            "- Regime-aware forecast routing",
            "- Stacking/meta-model orchestration",
            "- Portfolio allocator",
            "- Broad hyperparameter search",
        ]
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    evaluator = WalkForwardEvaluator(
        WalkForwardConfig(
            tickers=[ticker.upper().strip() for ticker in args.tickers],
            horizon=args.horizon,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            gap_size=args.gap_size,
            max_windows=args.max_windows,
            seed=args.seed,
        )
    )

    forecast_df, forecast_summary, window_summary, datasets, skipped_models, evaluated_models = evaluate_requested_models(
        evaluator,
        [model.lower() for model in args.models],
    )

    per_model_frames = [group.copy() for _, group in forecast_df.groupby("model_name", sort=True)]
    ensemble = WeightedEnsembleModel()
    ensemble_forecast = ensemble.combine(
        per_model_frames,
        context={"forecast_summary": forecast_summary},
    )
    all_forecasts = pd.concat([forecast_df, ensemble_forecast], ignore_index=True).sort_values(
        ["timestamp", "ticker", "model_name"]
    ).reset_index(drop=True)
    all_forecast_summary = summarize_forecasts(all_forecasts)
    forecast_summary_by_horizon = summarize_forecasts(all_forecasts, group_columns=["model_name", "horizon"])

    risk_df = build_window_garch_risk_frame(datasets, window_summary, horizon=args.horizon)
    regime_df = build_window_regime_frame(
        datasets,
        window_summary,
        lookback=args.regime_lookback,
        bull_threshold=args.regime_bull_threshold,
        bear_threshold=args.regime_bear_threshold,
    )

    backtester = CostAwareBacktester(
        BacktestConfig(
            horizon=args.horizon,
            transaction_fee_bps=args.transaction_fee_bps,
            slippage_bps=args.slippage_bps,
            allow_short=args.allow_short,
        )
    )
    market_data = build_market_data(datasets)

    strategy_modes = [
        "forecast_only",
        "forecast_plus_risk",
        "forecast_plus_risk_and_regime",
    ]
    signal_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    strategy_metric_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []

    for strategy_variant in strategy_modes:
        result = execute_strategy_variant(
            strategy_variant=strategy_variant,
            forecasts=all_forecasts,
            backtester=backtester,
            market_data=market_data,
            threshold=args.threshold,
            allow_short=args.allow_short,
            risk_df=risk_df if strategy_variant != "forecast_only" else None,
            regime_df=regime_df if strategy_variant == "forecast_plus_risk_and_regime" else None,
            risk_budget=args.risk_budget,
            max_position_size=args.max_position_size,
        )
        signal_frames.append(result["signals"])
        position_frames.append(result["positions"])
        trade_frames.append(result["trades"])
        strategy_metric_frames.append(result["strategy_metrics"])
        equity_frames.append(result["equity_curve"])

    all_signals = pd.concat(signal_frames, ignore_index=True).sort_values(
        ["strategy_variant", "timestamp", "ticker", "model_name"]
    ).reset_index(drop=True)
    all_positions = pd.concat(position_frames, ignore_index=True).sort_values(
        ["strategy_variant", "timestamp", "ticker", "model_name"]
    ).reset_index(drop=True)
    all_trades = pd.concat(trade_frames, ignore_index=True).sort_values(
        ["strategy_variant", "model_name", "ticker", "entry_timestamp"]
    ).reset_index(drop=True)
    all_strategy_metrics = pd.concat(strategy_metric_frames, ignore_index=True).sort_values(
        ["strategy_variant", "model_name"]
    ).reset_index(drop=True)
    all_equity = pd.concat(equity_frames, ignore_index=True).sort_values(
        ["strategy_variant", "model_name", "timestamp"]
    ).reset_index(drop=True)

    mode_summary = build_conditioning_mode_summary(all_strategy_metrics)
    phase2_comparison_summary = build_phase2_conditioning_summary(all_forecast_summary, all_strategy_metrics)

    table_paths = write_summary_tables(
        output_dir,
        {
            "forecasts": all_forecasts,
            "forecast_summary": all_forecast_summary,
            "forecast_summary_by_horizon": forecast_summary_by_horizon,
            "window_summary": window_summary,
            "risk_summary": risk_df,
            "regime_summary": regime_df,
            "signals": all_signals,
            "positions": all_positions,
            "trades": all_trades,
            "strategy_metrics": all_strategy_metrics,
            "equity_curve": all_equity,
            "conditioning_mode_summary": mode_summary,
            "phase2_comparison_summary": phase2_comparison_summary,
        },
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = build_run_manifest(
        manifest_type="phase2_run_manifest_v1",
        git_metadata=collect_git_metadata(Path.cwd()),
        runtime=collect_runtime_metadata(),
        dependency_versions=collect_dependency_versions(
            ["pandas", "numpy", "statsmodels", "arch", "scikit-learn", "xgboost", "lightgbm"]
        ),
        command=" ".join(sys.argv),
        tickers=[ticker.upper().strip() for ticker in args.tickers],
        requested_models=[model.lower() for model in args.models],
        evaluated_models=evaluated_models + [ensemble.model_name],
        skipped_models=skipped_models,
        benchmark_modes=strategy_modes,
        target_type="forward_return",
        horizon=args.horizon,
        seed=args.seed,
        costs={
            "transaction_fee_bps": args.transaction_fee_bps,
            "slippage_bps": args.slippage_bps,
        },
        evaluation_config={
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step_size": args.step_size,
            "gap_size": args.gap_size,
            "max_windows": args.max_windows,
        },
        regime_context={
            "model_name": "markov_switching",
            "lookback": args.regime_lookback,
            "bull_threshold": args.regime_bull_threshold,
            "bear_threshold": args.regime_bear_threshold,
        },
        risk_context={
            "model_name": "garch",
            "distribution": "t",
            "confidence_levels": [0.95, 0.99],
        },
        strategy_context={
            "threshold_policy": "regime_aware_thresholding_v1",
            "sizing_policy": "volatility_drawdown_aware_v1",
            "base_threshold": args.threshold,
            "risk_budget": args.risk_budget,
            "max_position_size": args.max_position_size,
            "allow_short": args.allow_short,
        },
        artifact_paths=table_paths,
        started_at=started_at,
        completed_at=completed_at,
    )
    manifest_path = write_run_manifest(output_dir, manifest)
    summary_path = write_summary_markdown(
        output_dir,
        render_phase2_summary_markdown(manifest, mode_summary, phase2_comparison_summary),
    )
    report_path = write_summary_markdown(output_dir, build_phase2_report(manifest), filename="phase2_report.md")

    print(f"Forecast rows: {len(all_forecasts)}")
    print(f"Evaluated models: {', '.join(evaluated_models + [ensemble.model_name])}")
    print(f"Benchmark modes: {', '.join(strategy_modes)}")
    if skipped_models:
        print("Skipped models:")
        for item in skipped_models:
            print(f"  - {item['model_name']}: {item['reason']}")
    print("Conditioning mode summary:")
    print(mode_summary.to_string(index=False))
    print("Top phase 2 comparison rows:")
    print(phase2_comparison_summary.head(8).to_string(index=False))
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Phase 2 report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
