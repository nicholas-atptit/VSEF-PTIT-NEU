"""Event-Driven Backtester.

Prevents absolute Look-ahead bias by injecting mandatory timestamp constraints.
Executes Walk-Forward Optimization routines.
Injects Slippage and Fees into execution modeling.
"""

from __future__ import annotations

import datetime as dt

from src.data.context.rag_service import ZonedRAGService
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_safe_rag_context(
    ticker: str,
    allowed_zones: list[str],
    current_time: dt.datetime,
) -> str:
    """Fetch RAG context with absolute time constraints.

    This prevents Look-ahead bias by strictly ensuring no Vector DB document
    is retrieved if its metadata timestamp > current_time.
    """
    normalized_time = current_time if current_time.tzinfo else current_time.replace(tzinfo=dt.UTC)
    header = f"[safe_rag] ticker={ticker.upper()} as_of={normalized_time.date().isoformat()}"

    try:
        rag_service = ZonedRAGService()
        rag_context = rag_service.query(
            ticker=ticker.upper(),
            allowed_zones=allowed_zones,
            as_of=normalized_time,
        )
        return f"{header}\n{rag_context}"
    except Exception as exc:
        logger.warning(
            "safe_rag_fallback",
            ticker=ticker.upper(),
            as_of=normalized_time.isoformat(),
            error=str(exc),
        )
        return f"{header}\n[rag] Khong truy van duoc ChromaDB: {str(exc)[:120]}"


def simulate_execution_cost(
    entry_price: float,
    volume: int,
    action: str = "BUY",
) -> tuple[float, float, float]:
    """Simulates Slippage and Fees heavily punishing algorithms.

    Requirements:
        - Trading Fee: 0.15%
        - Tax (Sell only): 0.10%
        - Slippage: 0.20%

    Args:
        entry_price: Theoretical execution price.
        volume: Number of shares.
        action: "BUY" or "SELL".

    Returns:
        tuple (adjusted_price, total_fees, total_slippage_cost)
    """
    nominal_value = entry_price * volume

    # 1. Slippage applies to price directly making it worse
    slippage_rate = 0.0020  # 0.20%
    if action == "BUY":
        adjusted_price = entry_price * (1.0 + slippage_rate)
    else:  # SELL
        adjusted_price = entry_price * (1.0 - slippage_rate)

    slippage_cost = abs(adjusted_price - entry_price) * volume

    # 2. Fees applied to nominal value
    trading_fee = nominal_value * 0.0015  # 0.15%
    tax = nominal_value * 0.0010 if action == "SELL" else 0.0  # 0.10% on Sale

    total_fees = trading_fee + tax

    return adjusted_price, total_fees, slippage_cost


def walk_forward_chunks(
    start_year: int,
    end_year: int,
    train_years: int = 5,
    test_years: int = 1,
) -> list[dict[str, int]]:
    """Yield Sliding Windows for Walk-Forward Optimization.

    Prevents training over 10 years at once to simulate live degradation.

    Returns:
        List of dicts defining train/test year bounds.
    """
    chunks = []
    current_start = start_year

    while current_start + train_years < end_year:
        train_end = current_start + train_years
        test_start = train_end
        test_end = test_start + test_years

        if test_end > end_year:
            test_end = end_year

        chunks.append({
            "train_start": current_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        current_start += 1  # Slide forward 1 year

    return chunks
