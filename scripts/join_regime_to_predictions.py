"""Join precomputed regime labels to saved prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.backtest.signal_regime_join import (
    DEFAULT_PREDICTION_DATE_COLUMN,
    DEFAULT_REGIME_DATE_COLUMN,
    DEFAULT_TICKER_COLUMN,
    JOIN_MODE_DATE,
    JOIN_MODE_TICKER_DATE,
    join_regime_csvs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join safe precomputed regime labels onto a copy of saved prediction outputs."
    )
    parser.add_argument("--predictions-path", required=True, help="Saved prediction CSV to enrich")
    parser.add_argument("--regime-path", required=True, help="Precomputed regime label CSV")
    parser.add_argument("--output-path", required=True, help="Path for enriched prediction CSV")
    parser.add_argument("--summary-path", required=True, help="Path for JSON or CSV join coverage summary")
    parser.add_argument(
        "--join-mode",
        default=JOIN_MODE_DATE,
        choices=[JOIN_MODE_DATE, JOIN_MODE_TICKER_DATE],
        help="Join by prediction date only or by ticker/date",
    )
    parser.add_argument(
        "--prediction-date-column",
        default=DEFAULT_PREDICTION_DATE_COLUMN,
        help="Prediction date column in the prediction CSV",
    )
    parser.add_argument(
        "--regime-date-column",
        default=DEFAULT_REGIME_DATE_COLUMN,
        help="Date column in the regime CSV",
    )
    parser.add_argument(
        "--ticker-column",
        default=DEFAULT_TICKER_COLUMN,
        help="Ticker column used when --join-mode ticker_date is selected",
    )
    parser.add_argument(
        "--regime-column",
        default=None,
        help="Optional explicit regime label column in the regime CSV",
    )
    parser.add_argument(
        "--overwrite-regime",
        action="store_true",
        help="Overwrite an existing prediction regime column with joined labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _enriched, summary = join_regime_csvs(
        predictions_path=args.predictions_path,
        regime_path=args.regime_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        join_mode=args.join_mode,
        prediction_date_column=args.prediction_date_column,
        regime_date_column=args.regime_date_column,
        ticker_column=args.ticker_column,
        regime_column=args.regime_column,
        overwrite_regime=args.overwrite_regime,
    )

    print("Signal regime join completed.")
    print(f"Input predictions: {args.predictions_path}")
    print(f"Input regimes: {args.regime_path}")
    print(f"Output predictions: {args.output_path}")
    print(f"Summary: {args.summary_path}")
    print(
        "Coverage: "
        f"{summary['matched_prediction_rows']}/{summary['prediction_rows']} "
        f"matched ({summary['matched_rate']:.6f})"
    )
    print(f"Join governance: {summary['join_governance']}")
    if summary["duplicate_regime_keys_exist"]:
        print(f"Warning: duplicate regime keys detected: {summary['duplicate_regime_key_count']}", file=sys.stderr)
    if summary["suspicious_columns_present"]:
        print(
            "Warning: suspicious regime source columns detected: "
            + ", ".join(summary["suspicious_columns"]),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
