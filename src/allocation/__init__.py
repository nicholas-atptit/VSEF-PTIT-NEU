"""Legacy module.
Retained for historical compatibility or migration reference.
Not part of canonical governed runtime.
"""

from .portfolio_allocator import (
    ALLOCATION_OUTPUT_FILES,
    ALLOCATION_LABELS,
    PortfolioAllocatorConfig,
    PortfolioAllocatorResult,
    run_portfolio_allocator,
)

__all__ = [
    "ALLOCATION_OUTPUT_FILES",
    "ALLOCATION_LABELS",
    "PortfolioAllocatorConfig",
    "PortfolioAllocatorResult",
    "run_portfolio_allocator",
]
