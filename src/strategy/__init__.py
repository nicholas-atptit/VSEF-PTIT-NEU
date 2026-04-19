"""Strategy contracts plus Phase 1 and Phase 2 execution policies."""

from .execution_policy import BasicExecutionPolicy, RegimeAwareExecutionPolicy
from .regime_thresholding import generate_regime_aware_signals
from .sizing import size_positions
from .signal_rules import StrategyModel
from .thresholding import generate_threshold_signals

__all__ = [
    "StrategyModel",
    "BasicExecutionPolicy",
    "RegimeAwareExecutionPolicy",
    "generate_threshold_signals",
    "generate_regime_aware_signals",
    "size_positions",
]
