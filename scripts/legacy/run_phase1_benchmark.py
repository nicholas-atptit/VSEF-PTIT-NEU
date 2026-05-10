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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ensemble.weighted import WeightedEnsembleModel
from src.evaluation.backtest import BacktestConfig, CostAwareBacktester
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator, summarize_forecasts
from src.forecast.registry import create_forecast_model, supported_forecast_models
from src.reporting.manifests import build_run_manifest, collect_git_metadata, write_run_manifest
from src.reporting.summary import (
    build_model_comparison_summary,
    render_summary_markdown,
    write_summary_markdown,
    write_summary_tables,
)
from src.risk.drawdown import DrawdownRiskModel
from src.risk.monte_carlo import MonteCarloRiskModel
from src.risk.var_cvar import VaRCVaRRiskModel
from src.strategy.execution_policy import BasicExecutionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Phase 1 forecasting benchmark.")
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
    parser.add_argument("--output-dir", default="artifacts/phase1_benchmark")
    parser.add_argument("--allow-short", action="store_true")
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


def build_window_risk_frame(
    datasets: dict[str, Any],
    window_summary: pd.DataFrame,
    *,
    horizon: int,
    seed: int,
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

        monte_carlo = MonteCarloRiskModel(simulations=1000, random_seed=seed).fit(returns).forecast_risk(horizon=horizon)
        var_cvar = VaRCVaRRiskModel().fit(returns).forecast_risk(horizon=horizon)
        drawdown = DrawdownRiskModel().fit(returns).forecast_risk(horizon=horizon)
        merged = {**monte_carlo, **var_cvar, **drawdown}
        for timestamp in pd.to_datetime(test_df["timestamp"], errors="coerce").dropna():
            risk_rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": str(row.ticker),
                    "window_id": str(row.window_id),
                    **merged,
                }
            )
    return pd.DataFrame(risk_rows).sort_values(["timestamp", "ticker", "window_id"]).reset_index(drop=True)


def build_market_data(datasets: dict[str, Any]) -> dict[str, pd.DataFrame]:
    market_data: dict[str, pd.DataFrame] = {}
    for ticker, dataset in datasets.items():
        market_data[ticker] = dataset.frame[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].copy()
    return market_data


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

    risk_df = build_window_risk_frame(datasets, window_summary, horizon=args.horizon, seed=args.seed)
    execution_policy = BasicExecutionPolicy(
        threshold=args.threshold,
        allow_short=args.allow_short,
        capital_config={
            "risk_budget": args.risk_budget,
            "max_position_size": args.max_position_size,
        },
    )
    signal_df = execution_policy.generate_signal(all_forecasts)
    position_df = execution_policy.size_positions(signal_df, risk_df=risk_df)

    backtester = CostAwareBacktester(
        BacktestConfig(
            horizon=args.horizon,
            transaction_fee_bps=args.transaction_fee_bps,
            slippage_bps=args.slippage_bps,
            allow_short=args.allow_short,
        )
    )
    backtest_result = backtester.run(position_df, build_market_data(datasets))
    comparison_summary = build_model_comparison_summary(all_forecast_summary, backtest_result["strategy_metrics"])

    table_paths = write_summary_tables(
        output_dir,
        {
            "forecasts": all_forecasts,
            "forecast_summary": all_forecast_summary,
            "forecast_summary_by_horizon": forecast_summary_by_horizon,
            "window_summary": window_summary,
            "risk_summary": risk_df,
            "signals": signal_df,
            "positions": position_df,
            "trades": backtest_result["trades"],
            "strategy_metrics": backtest_result["strategy_metrics"],
            "comparison_summary": comparison_summary,
        },
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = build_run_manifest(
        git_metadata=collect_git_metadata(Path.cwd()),
        command=" ".join(sys.argv),
        tickers=[ticker.upper().strip() for ticker in args.tickers],
        requested_models=[model.lower() for model in args.models],
        evaluated_models=evaluated_models + [ensemble.model_name],
        skipped_models=skipped_models,
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
            "threshold": args.threshold,
            "risk_budget": args.risk_budget,
            "max_position_size": args.max_position_size,
            "allow_short": args.allow_short,
        },
        artifact_paths=table_paths,
        started_at=started_at,
        completed_at=completed_at,
    )
    manifest_path = write_run_manifest(output_dir, manifest)
    summary_markdown = render_summary_markdown(manifest, comparison_summary)
    summary_path = write_summary_markdown(output_dir, summary_markdown)

    print(f"Forecast rows: {len(all_forecasts)}")
    print(f"Evaluated models: {', '.join(evaluated_models + [ensemble.model_name])}")
    if skipped_models:
        print("Skipped models:")
        for item in skipped_models:
            print(f"  - {item['model_name']}: {item['reason']}")
    print("Top comparison rows:")
    print(comparison_summary.head(5).to_string(index=False))
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
