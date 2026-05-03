"""Portfolio Allocator v1 public API."""

from src.portfolio_allocator.allocation import allocate_portfolio, run_portfolio_allocator
from src.portfolio_allocator.gating import evaluate_allocation_gate
from src.portfolio_allocator.reporting import run_portfolio_allocator_from_files, write_portfolio_allocator_outputs
from src.portfolio_allocator.schema import (
    PORTFOLIO_ALLOCATOR_VERSION,
    PortfolioAllocatorConfig,
    PortfolioAllocatorResult,
)
from src.portfolio_allocator.sizing import calculate_raw_weight

__all__ = [
    "PORTFOLIO_ALLOCATOR_VERSION",
    "PortfolioAllocatorConfig",
    "PortfolioAllocatorResult",
    "allocate_portfolio",
    "calculate_raw_weight",
    "evaluate_allocation_gate",
    "run_portfolio_allocator",
    "run_portfolio_allocator_from_files",
    "write_portfolio_allocator_outputs",
]
