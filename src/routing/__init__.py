"""Legacy deterministic routing utilities for governed Quant Core outputs."""

from src.routing.phase3_router import (
    LEGACY_PHASE3_ROUTE_LABELS,
    LEGACY_PHASE3_ROUTER_OUTPUT_FILES,
    PHASE3_ROUTE_LABELS,
    PHASE3_ROUTER_OUTPUT_FILES,
    Phase3RouterConfig,
    Phase3RouterResult,
    run_phase3_router,
)

__all__ = [
    "LEGACY_PHASE3_ROUTE_LABELS",
    "LEGACY_PHASE3_ROUTER_OUTPUT_FILES",
    "PHASE3_ROUTE_LABELS",
    "PHASE3_ROUTER_OUTPUT_FILES",
    "Phase3RouterConfig",
    "Phase3RouterResult",
    "run_phase3_router",
]
