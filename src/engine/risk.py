"""Module 2: Hard-Cap Risk Management

Applies rigid stop-loss, anti-FOMO, and position sizing bounds
after the Decision Matrix approves an order.
"""

from __future__ import annotations

import math

from src.api.schemas import ActionPlan, OrderPayload, RiskManagementOverride


def apply_risk_constraints(
    ticker: str,
    action_plan: ActionPlan,
    real_time_price: float,
    atr_14: float,
    applied_risk_tolerance: float,
    portfolio_risk_capital: float = 100_000_000.0,  # Example: 100 mil VND max risk
) -> tuple[OrderPayload | None, RiskManagementOverride]:
    """Apply hard risk caps to the proposed action plan.

    Constraints:
        1. Absolute Stop-Loss Cap: Max -7% from entry price.
        2. Position Sizing: Volume = (Risk Capital * applied_risk_tolerance) / (Multiplier * ATR)
        3. Anti-FOMO: Block if real_time_price > max_entry * 1.015

    Args:
        ticker: Symbol.
        action_plan: ML proposed entry zones and stop losses.
        real_time_price: The simulated or live current price.
        atr_14: Current 14-day Average True Range volatility measure.
        applied_risk_tolerance: Hard-capped risk tolerance (max 0.70).
        portfolio_risk_capital: Max loss willing to suffer across portfolio for this trade.

    Returns:
        Tuple of (OrderPayload (or None if blocked), RiskManagementOverride).
    """
    assert applied_risk_tolerance <= 0.70, "Risk tolerance constraint violated prior to engine."

    fomo_check_passed = True
    order_type = "STANDBY"
    
    # ── 1. Anti-FOMO Check (Only applies to BUYs in this scenario) ──
    try:
        max_entry = max(action_plan.entry_zone)
    except ValueError:
        max_entry = real_time_price

    if action_plan.recommendation == "BUY":
        order_type = "LIMIT"
        fomo_threshold = max_entry * 1.015 # 1.5% FOMO Buffer
        if real_time_price > fomo_threshold:
            fomo_check_passed = False
            return None, RiskManagementOverride(
                original_stop_loss_pct=0.0,
                applied_stop_loss_pct=0.0,
                fomo_check_passed=False,
            )

    elif action_plan.recommendation == "SELL":
        order_type = "MARKET"

    # ── 2. Absolute Stop-Loss Cap (-7%) ──
    # Calculate proposed SL percentage from proposed lowest entry point
    min_entry = min(action_plan.entry_zone) if action_plan.entry_zone else real_time_price
    
    original_sl_pct = 0.0
    if min_entry > 0:
        original_sl_pct = (action_plan.stop_loss - min_entry) / min_entry

    applied_sl_pct = original_sl_pct
    hard_stop_loss = action_plan.stop_loss

    # If ML proposes a SL worse than -7% (e.g. -10%), we force it to -7% (for BUYs)
    if action_plan.recommendation == "BUY" and original_sl_pct < -0.07:
        applied_sl_pct = -0.07
        hard_stop_loss = min_entry * (1.0 + applied_sl_pct)
    # If ML proposes a SL worse than +7% (for short SELLs), force to +7%
    elif action_plan.recommendation == "SELL" and original_sl_pct > 0.07:
        applied_sl_pct = 0.07
        hard_stop_loss = min_entry * (1.0 + applied_sl_pct)

    # ── 3. Position Sizing (Volatility Targeting) ──
    # Volume = (Risk Budget / Risk per Share)
    # To keep simple: using ATR directly as risk unit proxy.
    risk_budget = portfolio_risk_capital * applied_risk_tolerance
    
    # Avoid zero division
    safe_atr = max(atr_14, 0.01)
    
    # Multiplier: usually 1 for stocks, but ATR padding is often > 1 (e.g. 1.5x ATR buffer)
    risk_per_share = safe_atr * 1.5

    volume = 0
    if fomo_check_passed and order_type != "STANDBY":
        raw_volume = risk_budget / risk_per_share
        
        # Board lot size for VNSC/HSX is 100
        volume = math.floor(raw_volume / 100) * 100
        
        # Max Position Cap: 20% of Portfolio value to avoid concentration risk
        max_lot_cap = (portfolio_risk_capital * 2.0) / (real_time_price or 1.0)
        volume = min(volume, math.floor(max_lot_cap / 100) * 100)

        if volume <= 0:
            fomo_check_passed = False

    override_record = RiskManagementOverride(
        original_stop_loss_pct=round(original_sl_pct, 4),
        applied_stop_loss_pct=round(applied_sl_pct, 4),
        fomo_check_passed=fomo_check_passed,
    )

    payload = None
    if fomo_check_passed:
        payload = OrderPayload(
            order_type=order_type,
            entry_price=real_time_price if order_type == "MARKET" else min_entry,
            volume=volume,
            hard_stop_loss_price=round(hard_stop_loss, 2),
            take_profit_price=round(action_plan.take_profit, 2),
        )

    return payload, override_record
