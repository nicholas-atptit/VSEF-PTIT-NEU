"""Canonical claim-boundary wording for research artifacts."""

from __future__ import annotations

CLAIM_BOUNDARY = {
    "offline_diagnostic_only": "offline diagnostic only",
    "no_trading": "no trading",
    "no_buy_sell": "no BUY/SELL",
    "no_live_deployment": "no live deployment",
    "no_production": "no production",
    "no_daily_t1": "no daily T+1",
    "final_scoring_only": "final rows scoring-only",
    "future_blind_required": "future-blind required",
}


def claim_label(*, final_rows: bool = False) -> str:
    return "final_scoring_only" if final_rows else "offline_diagnostic_only"


def claim_statement(*, include_final_rows: bool = True) -> str:
    keys = [
        "offline_diagnostic_only",
        "no_trading",
        "no_buy_sell",
        "no_live_deployment",
        "no_production",
        "no_daily_t1",
        "future_blind_required",
    ]
    if include_final_rows:
        keys.insert(1, "final_scoring_only")
    return "; ".join(CLAIM_BOUNDARY[key] for key in keys) + "."
