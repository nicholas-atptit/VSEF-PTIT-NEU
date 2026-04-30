"""Run deterministic Portfolio Allocator v1 on saved Quant Core outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.allocation.portfolio_allocator import PortfolioAllocatorConfig, run_portfolio_allocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create allocation-candidate artifacts from Quant Core diagnostic outputs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Quant Core output CSVs")
    parser.add_argument("--output-dir", default=None, help="Directory for allocator outputs; defaults to input-dir")
    parser.add_argument("--max-ticker-weight", type=float, default=0.10)
    parser.add_argument("--max-total-exposure", type=float, default=0.60)
    parser.add_argument("--cash-buffer", type=float, default=0.40)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--min-model-agreement", type=float, default=0.5)
    parser.add_argument("--max-risk-score", type=float, default=1.0)
    parser.add_argument("--risk-penalty-strength", type=float, default=0.5)
    parser.add_argument("--agreement-penalty-strength", type=float, default=0.5)
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PortfolioAllocatorConfig(
        max_ticker_weight=args.max_ticker_weight,
        max_total_exposure=args.max_total_exposure,
        cash_buffer=args.cash_buffer,
        min_candidate_score=args.min_candidate_score,
        min_model_agreement=args.min_model_agreement,
        max_risk_score=args.max_risk_score,
        risk_penalty_strength=args.risk_penalty_strength,
        agreement_penalty_strength=args.agreement_penalty_strength,
        allow_short=bool(args.allow_short),
    )
    result = run_portfolio_allocator(
        args.input_dir,
        args.output_dir,
        config=config,
    )
    summary_row = result.summary.iloc[0].to_dict() if not result.summary.empty else {}
    print(f"Portfolio label: {summary_row.get('portfolio_label', 'unknown')}")
    print(f"Allocation count: {summary_row.get('allocation_count', 0)}")
    print(f"Invested exposure: {summary_row.get('invested_exposure', 0.0)}")
    print(f"Cash weight: {summary_row.get('cash_weight', 1.0)}")
    print("Outputs:")
    for name, path in sorted(result.output_paths.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
