"""VN100 Universe Loader.

Provides utilities to retrieve and manage the VN100 constituent list
for data synchronization, model training, and inference.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Literal

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════ LEGACY STATIC LIST ═══════════════════════════
# This list is used as a fallback if the live API retrieval fails.
# It matches the list currently hardcoded in scripts/sync_all_data.py.
VN100_BACKUP_TICKERS = [
    "AAA", "ACB", "ANV", "ASM", "BAF", "BCG", "BCM", "BID", "BMI", "BMP",
    "BVH", "BWE", "CII", "CMG", "CTD", "CTG", "CTR", "DBC", "DCM", "DGC",
    "DGW", "DIG", "DPM", "DPR", "DXG", "EIB", "EVF", "FCN", "FPT", "FRT",
    "FTS", "GAS", "GEX", "GIL", "GMD", "GVR", "HAG", "HAH", "HCM", "HDB",
    "HDC", "HDG", "HHV", "HPG", "HSG", "HT1", "IJC", "KBC", "KDC", "KDH",
    "LCG", "LPB", "MBB", "MSB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL",
    "OCB", "PAN", "PC1", "PDR", "PET", "PHR", "PLX", "PNJ", "POW", "PTB",
    "PVD", "PVT", "REE", "SAB", "SBT", "SHB", "SSB", "SSI", "STB", "SZC",
    "TCB", "TCH", "TNH", "TPB", "VCB", "VCG", "VCI", "VGC", "VGI", "VHC",
    "VHM", "VIB", "VIC", "VIX", "VJC", "VND", "VNM", "VOS", "VPB", "VPI",
    "VRE", "VSH", "VTK", "VTP",
]

VIETTEL_TICKERS = ["VTP", "VGI", "CTR", "FOX"]


def get_vn100_universe(
    as_of_date: Optional[dt.date] = None, 
    mode: Literal["current", "current_vn100", "current_plus_viettel"] = "current"
) -> List[str]:
    """Retrieve the VN100 constituent list.

    Args:
        as_of_date: The date for which to retrieve the constituents.
            Currently only supports 'None' (today's constituents).
        mode: The retrieval mode:
            - 'current' or 'current_vn100': Returns the current VN100 list.
            - 'current_plus_viettel': Returns VN100 plus 4 Viettel tickers.

    Returns:
        Sorted list of unique ticker symbols.

    Note:
        Historical constituent data is not yet supported and remains future work.
    """
    if as_of_date is not None:
        logger.warning(
            "historical_universe_not_supported", 
            requested_date=as_of_date.isoformat(),
            fallback="returning_current_universe"
        )

    tickers: List[str] = []

    # 1. Attempt to fetch current VN100 from Adapter (vnstock 3.0)
    try:
        adapter = VnstockAdapter()
        tickers = adapter.get_vn100_tickers()
        
        if not tickers:
            logger.warning("vnstock_returned_empty_vn100", action="falling_back_to_static")
            tickers = VN100_BACKUP_TICKERS.copy()
        else:
            logger.info("vn100_resolved_via_adapter", count=len(tickers))
            
    except Exception as e:
        logger.error("vn100_resolution_error", error=str(e), action="falling_back_to_static")
        tickers = VN100_BACKUP_TICKERS.copy()

    # 2. Handle Universe Extensions
    if mode == "current_plus_viettel":
        # Union of VN100 and Viettel special list
        tickers_set = set(tickers) | set(VIETTEL_TICKERS)
        tickers = sorted(list(tickers_set))
        logger.debug("extended_universe_resolved", mode=mode, total_count=len(tickers))
    else:
        tickers = sorted(list(set(tickers)))

    return tickers
