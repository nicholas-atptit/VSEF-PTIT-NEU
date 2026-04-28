"""Run signal-effectiveness diagnostics on saved forecast prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.signal_effectiveness import (
    DEFAULT_COST_PER_TRADE_VALUES,
    DEFAULT_MINIMUM_SIGNAL_COUNTS,
    DEFAULT_PREDICTED_RETURN_THRESHOLDS,
    DEFAULT_SLIPPAGE_VALUES,
    POLICY_DIRECTION_AND_RETURN_THRESHOLD,
    POLICY_RETURN_THRESHOLD,
    POLICY_STRICT_BUY_PRECISION_PROBE,
    SUCCESS_COST_ADJUSTED_POSITIVE,
    SUCCESS_RAW_POSITIVE,
    SUCCESS_TARGET_RETURN,
    SignalEffectivenessConfig,
    SignalEffectivenessRunner,
)


def _split_items(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    normalized = raw_value.replace(",", " ")
    values = [item.strip() for item in normalized.split() if item.strip()]
    return values or None


def _split_floats(raw_value: str | None, default: list[float]) -> list[float]:
    values = _split_items(raw_value)
    if values is None:
        return list(default)
    return [float(value) for value in values]


def _split_ints(raw_value: str | None, default: list[int]) -> list[int]:
    values = _split_items(raw_value)
    if values is None:
        return list(default)
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BUY/HOLD/AVOID signal effectiveness from saved forecast outputs."
    )
    parser.add_argument("--predictions-path", required=True, help="Saved prediction CSV to evaluate")
    parser.add_argument("--output-dir", required=True, help="Directory for signal-effectiveness reports")
    parser.add_argument("--models", default=None, help="Optional comma- or space-separated model filter")
    parser.add_argument("--horizons", default=None, help="Optional comma- or space-separated horizon filter")
    parser.add_argument("--tickers", default=None, help="Optional comma- or space-separated ticker filter")
    parser.add_argument(
        "--policy",
        default=POLICY_STRICT_BUY_PRECISION_PROBE,
        choices=[
            POLICY_RETURN_THRESHOLD,
            POLICY_DIRECTION_AND_RETURN_THRESHOLD,
            POLICY_STRICT_BUY_PRECISION_PROBE,
        ],
        help="Rule policy used to generate BUY/HOLD/AVOID labels",
    )
    parser.add_argument(
        "--threshold-grid",
        default=",".join(str(value) for value in DEFAULT_PREDICTED_RETURN_THRESHOLDS),
        help="Comma-separated predicted return thresholds",
    )
    parser.add_argument(
        "--cost-per-trade",
        default=",".join(str(value) for value in DEFAULT_COST_PER_TRADE_VALUES),
        help="Comma-separated per-side cost rates, e.g. 0.001",
    )
    parser.add_argument(
        "--slippage",
        default=",".join(str(value) for value in DEFAULT_SLIPPAGE_VALUES),
        help="Comma-separated per-side slippage rates, e.g. 0.0005",
    )
    parser.add_argument(
        "--success-definition",
        default=SUCCESS_COST_ADJUSTED_POSITIVE,
        choices=[SUCCESS_RAW_POSITIVE, SUCCESS_COST_ADJUSTED_POSITIVE, SUCCESS_TARGET_RETURN],
        help="BUY correctness definition",
    )
    parser.add_argument(
        "--target-return-threshold",
        type=float,
        default=0.01,
        help="Return threshold used when --success-definition target_return is selected",
    )
    parser.add_argument(
        "--minimum-signal-count",
        default=",".join(str(value) for value in DEFAULT_MINIMUM_SIGNAL_COUNTS),
        help="Comma-separated minimum BUY signal counts for filtered precision tables",
    )
    parser.add_argument(
        "--probability-up-threshold-grid",
        default="0.55,0.60,0.65,0.70",
        help="Optional probability-up thresholds used only when a usable probability column exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SignalEffectivenessConfig(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        models=_split_items(args.models),
        horizons=_split_items(args.horizons),
        tickers=_split_items(args.tickers),
        policy=args.policy,
        predicted_return_thresholds=_split_floats(args.threshold_grid, DEFAULT_PREDICTED_RETURN_THRESHOLDS),
        cost_per_trade_values=_split_floats(args.cost_per_trade, DEFAULT_COST_PER_TRADE_VALUES),
        slippage_values=_split_floats(args.slippage, DEFAULT_SLIPPAGE_VALUES),
        probability_up_thresholds=_split_floats(args.probability_up_threshold_grid, [0.55, 0.60, 0.65, 0.70]),
        success_definition=args.success_definition,
        target_return_threshold=args.target_return_threshold,
        minimum_signal_counts=_split_ints(args.minimum_signal_count, DEFAULT_MINIMUM_SIGNAL_COUNTS),
    )
    result = SignalEffectivenessRunner(config).run()

    print("Signal-effectiveness backtest completed.")
    print(f"Input predictions: {args.predictions_path}")
    print("\nOutput files:")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    summary = result["signal_effectiveness_summary"]
    if not summary.empty:
        preview = summary[
            [
                "model_name",
                "horizon",
                "predicted_return_threshold",
                "minimum_signal_count",
                "buy_signal_count",
                "buy_precision",
                "net_average_return_after_buy",
            ]
        ].head(12)
        print("\nSummary preview:")
        print(preview.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
