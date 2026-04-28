"""Run the full walk-forward all-model stacking experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.walk_forward_all_models_stacking import (
    FINAL_STACKING_MODEL_NAME,
    WalkForwardAllModelsStackingConfig,
    WalkForwardAllModelsStackingRunner,
)

REQUIRED_STEP_CSVS = {
    "predictions_detailed.csv": "base_df",
    "stacking_predictions_detailed.csv": "stack_df",
    "actual_comparison_summary.csv": "actual_comparison_summary",
    "summary_by_horizon.csv": "summary_by_horizon",
    "summary_by_ticker.csv": "summary_by_ticker",
    "summary_by_model.csv": "summary_by_model",
    "stacking_vs_all_models.csv": "stacking_vs_all_models",
    "backtest_summary.csv": "backtest_summary",
    "buy_and_hold_comparison.csv": "buy_and_hold_comparison",
    "forecast_coverage_summary.csv": "forecast_coverage_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward all-model forecasting with time-series-safe stacking")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["MSN", "MWG", "DGC", "SSI", "FPT", "ACB"],
        help="Ticker symbols to evaluate",
    )
    parser.add_argument("--history-start", type=str, default="2018-01-01")
    parser.add_argument("--history-end", type=str, default="2026-03-31")
    parser.add_argument("--initial-train-start", type=str, default="2018-01-01")
    parser.add_argument("--initial-train-end", type=str, default="2024-12-31")
    parser.add_argument("--forecast-start", type=str, default="2025-01-01")
    parser.add_argument("--forecast-end", type=str, default="2026-03-31")
    parser.add_argument(
        "--horizons",
        type=str,
        default="short_5d,short_10d,short_20d,short_30d,long_3m,long_6m",
        help="Comma-separated horizon names",
    )
    parser.add_argument(
        "--step-sizes",
        type=str,
        default="1,2",
        help="Comma-separated walk-forward step sizes in trading days",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Optional comma-separated model list. Default is all supported non-stacking models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/walkforward_all_models",
        help="Root output directory. Step folders are created under <output-dir>/<ticker-slug>/",
    )
    parser.add_argument(
        "--foreign-flow-path",
        type=str,
        default=None,
        help=(
            "Optional foreign-flow CSV artifact path for context joins. "
            "When omitted, the default data/foreign_flow.csv loader behavior is preserved."
        ),
    )
    parser.add_argument(
        "--foreign-flow-mode",
        choices=["auto", "path", "disabled"],
        default="auto",
        help=(
            "Foreign-flow context loading mode. auto preserves existing behavior; "
            "path requires --foreign-flow-path; disabled intentionally skips all "
            "foreign-flow artifact loading."
        ),
    )
    parser.add_argument(
        "--ohlcv-data-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit directory containing per-ticker OHLCV CSV files. "
            "When supplied, ticker files are loaded from this directory instead of "
            "the live provider/default tracked CSV fallback."
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--criterion", type=str, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-min-rows", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=60)
    parser.add_argument("--meta-model-alpha", type=float, default=1.0)
    parser.add_argument("--meta-min-samples", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count across tickers")
    args = parser.parse_args()
    if args.foreign_flow_mode == "path" and not args.foreign_flow_path:
        parser.error("--foreign-flow-mode path requires --foreign-flow-path")
    if args.foreign_flow_mode == "disabled" and args.foreign_flow_path:
        parser.error("--foreign-flow-mode disabled cannot be combined with --foreign-flow-path")
    return args


def _ticker_slug(tickers: list[str]) -> str:
    return "_".join(str(ticker).strip().lower() for ticker in tickers if str(ticker).strip())


def _ensure_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in columns:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
    return prepared


def _coverage_chart(coverage_df: pd.DataFrame, output_path: Path) -> None:
    if coverage_df.empty:
        return
    chart_frame = (
        coverage_df.groupby(["horizon"], as_index=False)["coverage_ratio"]
        .mean()
        .sort_values("horizon")
        .reset_index(drop=True)
    )
    plt.figure(figsize=(9, 4.5))
    plt.bar(chart_frame["horizon"], chart_frame["coverage_ratio"], color="#2563eb")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Coverage Ratio")
    plt.title("Forecast Coverage / Evaluation Eligibility")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _write_step_report(
    *,
    step_size: int,
    step_dir: Path,
    tickers: list[str],
    fetch_summary: pd.DataFrame,
    algorithms: list[str],
    skipped_algorithms: list[dict[str, str]],
    config: WalkForwardAllModelsStackingConfig,
    summary_by_horizon: pd.DataFrame,
    summary_by_model: pd.DataFrame,
    stacking_vs_all_models: pd.DataFrame,
    stack_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> Path:
    step_fetch = fetch_summary.sort_values("ticker").reset_index(drop=True)
    step_horizon = summary_by_horizon[(summary_by_horizon["step_size"] == step_size) & (summary_by_horizon["ticker"] == "OVERALL")].copy()
    step_model = summary_by_model[summary_by_model["step_size"] == step_size].copy()
    step_stacking_vs = stacking_vs_all_models[stacking_vs_all_models["step_size"] == step_size].copy()
    step_stack = stack_df[stack_df["step_size"] == step_size].copy()
    eligible_stack = step_stack[step_stack["evaluation_eligible"] == True].copy()
    worst_divergence = eligible_stack.sort_values("absolute_error", ascending=False).head(10)
    best_horizon_rows = (
        step_horizon.sort_values(["horizon", "rmse", "mae", "model_name"])
        .groupby("horizon", as_index=False)
        .first()
        if not step_horizon.empty
        else pd.DataFrame()
    )
    overall_stacking = step_model[step_model["model_name"] == FINAL_STACKING_MODEL_NAME].copy()
    coverage_by_horizon = (
        coverage_df[coverage_df["step_size"] == step_size]
        .groupby("horizon", as_index=False)[["total_predictions", "evaluation_eligible_predictions", "evaluation_ineligible_predictions", "coverage_ratio"]]
        .sum()
        .sort_values("horizon")
        .reset_index(drop=True)
    )
    coverage_note = (
        f"Per-ticker rows below record the actual tradable date ranges after filtering the requested "
        f"{config.history_start} through {config.history_end} history window."
    )

    lines = [
        "# Walk-Forward Forecasting Report",
        "",
        "## Experiment Setup",
        f"- Ticker universe: {', '.join(tickers)}",
        f"- Actual data source used: {', '.join(f'{row.ticker}={row.source}' for row in step_fetch.itertuples(index=False))}",
        f"- Requested history coverage status: {coverage_note}",
        f"- Historical input window: {config.history_start} through {config.history_end}",
        f"- OHLCV data directory override: `{config.ohlcv_data_dir}`" if config.ohlcv_data_dir else "- OHLCV data directory override: not supplied",
        f"- Training window: {config.initial_train_start} through {config.initial_train_end}",
        f"- Forecast window: {config.forecast_start} through {config.forecast_end}",
        f"- Horizons: {', '.join(config.horizons)}",
        f"- Step size: {step_size}",
        f"- All models actually run: {', '.join(algorithms)}",
        "- Stacking method used: prequential Ridge regression over prior out-of-sample base-model predictions within the same horizon and step size, with mean fallback before enough realized rows exist.",
        "- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.",
        "- Evaluation-eligible rows: rows where the realized `target_date` close exists inside fetched history; rows without enough future market data remain in the detailed prediction tables with `evaluation_eligible=False`.",
        "",
        "## Per-Ticker Coverage",
    ]
    for row in step_fetch.itertuples(index=False):
        lines.append(
            f"- {row.ticker}: source={row.source}, rows={row.rows}, available_range={row.fetched_min_date} through {row.fetched_max_date}"
        )
    if skipped_algorithms:
        lines.extend(["", "## Skipped Algorithms"])
        for item in skipped_algorithms:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    lines.extend(["", "## Best-Performing Models and Horizons"])
    if not overall_stacking.empty:
        row = overall_stacking.sort_values(["rmse", "mae"]).iloc[0]
        lines.append(
            f"- Overall stacking for step_size={step_size}: RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, directional_accuracy={row.directional_accuracy:.4f}"
        )
    for row in best_horizon_rows.itertuples(index=False):
        lines.append(
            f"- Horizon {row.horizon}: best model by RMSE was `{row.model_name}` with RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, directional_accuracy={row.directional_accuracy:.4f}"
        )
    lines.extend(["", "## Stacking vs Individual Models"])
    if step_stacking_vs.empty:
        lines.append("- No pairwise stacking comparison rows were available.")
    else:
        summary = (
            step_stacking_vs.groupby(["scope", "horizon"], as_index=False)[
                ["stacking_better_mae", "stacking_better_rmse", "stacking_better_directional_accuracy"]
            ]
            .mean()
            .sort_values(["scope", "horizon"])
            .reset_index(drop=True)
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"- scope={row.scope}, horizon={row.horizon}: stacking beat the field on MAE in {row.stacking_better_mae:.2%} of pairwise comparisons, on RMSE in {row.stacking_better_rmse:.2%}, and on directional accuracy in {row.stacking_better_directional_accuracy:.2%}."
            )
    lines.extend(["", "## Where Predictions Diverged Most"])
    if worst_divergence.empty:
        lines.append("- No evaluation-eligible stacking rows were available for divergence analysis.")
    else:
        for row in worst_divergence.itertuples(index=False):
            lines.append(
                f"- {row.ticker} {row.horizon} prediction_date={pd.Timestamp(row.prediction_date).date()}: predicted={row.final_predicted_return:.6f}, actual={row.actual_realized_forward_return:.6f}, absolute_error={row.absolute_error:.6f}"
            )
    lines.extend(["", "## Evaluation Coverage"])
    for row in coverage_by_horizon.itertuples(index=False):
        lines.append(
            f"- {row.horizon}: eligible={int(row.evaluation_eligible_predictions)}, ineligible={int(row.evaluation_ineligible_predictions)}, coverage_ratio={row.coverage_ratio:.4f}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "- The repo-local daily CSV cache was insufficient for the requested start date, so the experiment depends on the live vnstock KBS path for historical backfill.",
            "- Long horizons near the end of the requested forecast window are intentionally kept in the outputs but excluded from scored metrics when realized target closes are not yet available.",
            "- Strategy metrics are computed on overlapping forecast windows, so they are technical usefulness diagnostics rather than execution-ready portfolio PnL.",
            "- The final stack is a regression meta-learner, so no final calibrated probability is emitted.",
            "",
            "## Output Paths",
            f"- csv/: `{step_dir / 'csv'}`",
            f"- charts/: `{step_dir / 'charts'}`",
            f"- report.md: `{step_dir / 'report.md'}`",
        ]
    )
    report_path = step_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _export_step_outputs(
    *,
    base_output_dir: Path,
    result: dict[str, object],
    config: WalkForwardAllModelsStackingConfig,
) -> dict[int, dict[str, str]]:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    combined_chart_dir = Path(str(result["charts_dir"]))
    fetch_summary = result["fetch_summary"]
    exported: dict[int, dict[str, str]] = {}

    for step_size in config.step_sizes:
        step_value = int(step_size)
        step_dir = base_output_dir / f"step_{step_value}"
        csv_dir = step_dir / "csv"
        charts_dir = step_dir / "charts"
        csv_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)

        for filename, result_key in REQUIRED_STEP_CSVS.items():
            frame = result[result_key]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{result_key} is not a DataFrame")
            step_frame = frame[frame["step_size"] == step_value].copy() if "step_size" in frame.columns else frame.copy()
            step_frame.to_csv(csv_dir / filename, index=False)

        # Recursively copy the entire charts directory to preserve the horizon/type hierarchy
        if combined_chart_dir.exists():
            for chart_item in combined_chart_dir.rglob("*"):
                if chart_item.is_file():
                    # Check if the file belongs to the current step size
                    # Our filenames start with step{step_value}_ or are overall summaries
                    if chart_item.name.startswith(f"step{step_value}_") or "summary" in chart_item.name:
                        # Determine relative path from combined_chart_dir and strip the step prefix for the final destination
                        rel_path = chart_item.relative_to(combined_chart_dir)
                        final_filename = chart_item.name
                        prefix_to_strip = f"step{step_value}_"
                        if final_filename.startswith(prefix_to_strip):
                            final_filename = final_filename[len(prefix_to_strip):]
                        
                        target_path = charts_dir / rel_path.parent / final_filename
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(chart_item, target_path)
        coverage_df = result["forecast_coverage_summary"]
        if isinstance(coverage_df, pd.DataFrame):
            _coverage_chart(coverage_df[coverage_df["step_size"] == step_value].copy(), charts_dir / "forecast_coverage_summary.png")

        report_path = _write_step_report(
            step_size=step_value,
            step_dir=step_dir,
            tickers=[ticker.upper() for ticker in config.tickers],
            fetch_summary=fetch_summary if isinstance(fetch_summary, pd.DataFrame) else pd.DataFrame(fetch_summary),
            algorithms=list(result["available_algorithms"]),
            skipped_algorithms=list(result["skipped_algorithms"]),
            config=config,
            summary_by_horizon=result["summary_by_horizon"],
            summary_by_model=result["summary_by_model"],
            stacking_vs_all_models=result["stacking_vs_all_models"],
            stack_df=result["stack_df"],
            coverage_df=coverage_df,
        )
        exported[step_value] = {
            "csv_dir": str(csv_dir),
            "charts_dir": str(charts_dir),
            "report_path": str(report_path),
        }
    return exported


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper().strip() for ticker in args.tickers if ticker.strip()]
    step_sizes = [int(value.strip()) for value in args.step_sizes.split(",") if value.strip()]
    ticker_slug = _ticker_slug(tickers)
    output_root = Path(args.output_dir) / ticker_slug
    combined_output_dir = output_root / "_combined_internal"

    config = WalkForwardAllModelsStackingConfig(
        tickers=tickers,
        history_start=args.history_start,
        history_end=args.history_end,
        initial_train_start=args.initial_train_start,
        initial_train_end=args.initial_train_end,
        forecast_start=args.forecast_start,
        forecast_end=args.forecast_end,
        output_dir=str(combined_output_dir),
        horizons=[value.strip().lower() for value in args.horizons.split(",") if value.strip()],
        step_sizes=step_sizes,
        algorithms=(
            [value.strip().lower() for value in args.algorithms.split(",") if value.strip()]
            if args.algorithms
            else None
        ),
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        validation_fraction=args.validation_fraction,
        validation_min_rows=args.validation_min_rows,
        min_train_rows=args.min_train_rows,
        meta_model_alpha=args.meta_model_alpha,
        meta_min_samples=args.meta_min_samples,
        max_workers=args.max_workers,
        foreign_flow_path=args.foreign_flow_path,
        foreign_flow_mode=args.foreign_flow_mode,
        ohlcv_data_dir=args.ohlcv_data_dir,
    )

    result = WalkForwardAllModelsStackingRunner(config).run()
    exported = _export_step_outputs(base_output_dir=output_root, result=result, config=config)

    print("Walk-forward all-model stacking experiment completed.")
    print("\nModels run:")
    print(", ".join(result["available_algorithms"]))
    if result["skipped_algorithms"]:
        print("\nSkipped algorithms:")
        for item in result["skipped_algorithms"]:
            print(f"{item['algorithm']}: {item['reason']}")
    print("\nStep output locations:")
    for step_size in step_sizes:
        step_info = exported[int(step_size)]
        print(f"step_{int(step_size)}/csv: {step_info['csv_dir']}")
        print(f"step_{int(step_size)}/charts: {step_info['charts_dir']}")
        print(f"step_{int(step_size)}/report.md: {step_info['report_path']}")
    print("\nOverall summary by model:")
    print(result["summary_by_model"].to_string(index=False))


if __name__ == "__main__":
    main()
