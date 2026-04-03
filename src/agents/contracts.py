from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["BUY", "SELL", "HOLD"]
SignalRegime = Literal["trend", "range", "unclear"]


@dataclass(slots=True)
class MarketSignal:
    ticker: str
    current_price: float
    pred_return: float
    confidence: float
    volatility: float
    sentiment_score: float = 0.0
    trend_up_prob: float = 0.0
    trend_down_prob: float = 0.0
    trend_sideways_prob: float = 1.0
    rsi_14: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    atr_14: float | None = None
    regime: SignalRegime = "unclear"
    source: str = "ml_pipeline"


@dataclass(slots=True)
class AnalystDecision:
    ticker: str
    action: Action
    confidence: float
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskDecision:
    ticker: str
    approved: bool
    action: Action
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_holding_days: int
    veto_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PositionProposal:
    ticker: str
    action: Action
    weight: float
    confidence: float
    rationale: str


@dataclass(slots=True)
class PortfolioProposal:
    positions: list[PositionProposal]
    gross_exposure: float
    cash_buffer: float
    notes: list[str] = field(default_factory=list)
