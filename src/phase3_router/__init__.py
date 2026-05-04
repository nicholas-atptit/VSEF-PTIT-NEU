"""Phase 3 Router v1 public API."""

from src.phase3_router.guards import RouteGuardDecision, evaluate_route_guard
from src.phase3_router.reporting import run_phase3_router_from_files, write_phase3_router_outputs
from src.phase3_router.routing import route_portfolio_allocations, run_phase3_router
from src.phase3_router.schema import (
    LEGACY_ROUTER_ARTIFACT_FILENAMES,
    PHASE3_ROUTER_VERSION,
    ROUTE_DECISIONS,
    ROUTER_ARTIFACT_FILENAMES,
    ROUTER_DECISION_COLUMNS,
    Phase3RouterConfig,
    Phase3RouterResult,
    PortfolioContext,
)

__all__ = [
    "LEGACY_ROUTER_ARTIFACT_FILENAMES",
    "PHASE3_ROUTER_VERSION",
    "ROUTE_DECISIONS",
    "ROUTER_ARTIFACT_FILENAMES",
    "ROUTER_DECISION_COLUMNS",
    "Phase3RouterConfig",
    "Phase3RouterResult",
    "PortfolioContext",
    "RouteGuardDecision",
    "evaluate_route_guard",
    "route_portfolio_allocations",
    "run_phase3_router",
    "run_phase3_router_from_files",
    "write_phase3_router_outputs",
]
