import sys
from pathlib import Path
import asyncio
import datetime as dt
import json
import os
import pandas as pd

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.data.historical.backdate import BackdateIngestor
from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.utils.logging import get_logger

logger = get_logger("sync_all")

# ═══════════════════════════ VN100 + VIETTEL TICKER UNIVERSE ═══════════════════
# Legacy Static List (maintained for backward compatibility)
VN100_TICKERS = [
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
VN100_PLUS_VIETTEL = sorted(set(VN100_TICKERS + VIETTEL_TICKERS))

async def sync_benchmark(ingestor: BackdateIngestor, start_date: dt.date, end_date: dt.date) -> int:
    """Sync VNINDEX benchmark data."""
    logger.info("sync_benchmark_starting", symbol="VNINDEX")
    try:
        rows = await ingestor.run(tickers=["VNINDEX"], start_date=start_date, end_date=end_date, force_refresh=True)
        logger.info("sync_benchmark_done", symbol="VNINDEX", rows=rows)
        return rows
    except Exception as e:
        logger.error("sync_benchmark_error", symbol="VNINDEX", error=str(e))
        return 0

async def main(
    train_vn100: bool = False,
    universe_mode: str = "all",
    ticker: str | None = None,
    start_str: str | None = None,
    end_str: str | None = None,
    force_refresh: bool = False,
    save_raw_copy: bool = False,
    benchmark: bool = False
):
    tickers = []
    
    # 0. Handle single ticker override
    if ticker:
        tickers = [ticker.upper()]
        logger.info("sync_mode_single_ticker", ticker=ticker)
    # 1. Resolve Universe Mode
    elif universe_mode == "current_vn100":
        adapter = VnstockAdapter()
        tickers = adapter.get_vn100_tickers()
        logger.info("sync_universe_dynamic_vn100", count=len(tickers))
    elif train_vn100:  # Backward compatibility for --vn100 flag
        tickers = VN100_PLUS_VIETTEL
        logger.info("sync_universe_legacy_vn100", count=len(tickers))
    else:
        vip_path = os.path.join(str(root_dir), 'data/listing/danh_sach_VIP_LLM_ready.jsonl')
        if os.path.exists(vip_path):
            with open(vip_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    tickers.append(data['symbol'])
            logger.info("sync_universe_vip", count=len(tickers))
        else:
            logger.warning("vip_list_not_found", path=vip_path, fallback="using_hardcoded_vn100")
            tickers = VN100_PLUS_VIETTEL

    # 2. Resolve Dates
    end_date = dt.date.today()
    if end_str:
        end_date = dt.datetime.strptime(end_str, "%Y-%m-%d").date()
        
    start_date = end_date - dt.timedelta(days=180)
    if start_str:
        start_date = dt.datetime.strptime(start_str, "%Y-%m-%d").date()

    logger.info("sync_starting", ticker_count=len(tickers), start=start_date.isoformat(), end=end_date.isoformat())
    
    ingestor = BackdateIngestor()
    
    # 3. Synchronize Tickers
    total_rows = await ingestor.run(
        tickers=tickers, 
        start_date=start_date, 
        end_date=end_date,
        force_refresh=force_refresh,
        save_raw_copy=save_raw_copy
    )

    # 4. Synchronize Benchmark if requested
    benchmark_rows = 0
    if benchmark:
        benchmark_rows = await sync_benchmark(ingestor, start_date, end_date)

    # 5. Final Summary
    logger.info(
        "sync_summary",
        status="complete",
        total_tickers=len(tickers),
        total_rows_ingested=total_rows,
        benchmark_ingested=benchmark_rows
    )
    
    print("\n" + "="*50)
    print("SYNC SUMMARY")
    print(f"Tickers Processed: {len(tickers)}")
    print(f"Total Rows Ingested: {total_rows}")
    print(f"Benchmark Rows: {benchmark_rows}")
    print(f"Date Range: {start_date} to {end_date}")
    if save_raw_copy:
        print(f"Raw copies saved to: data/raw/")
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync Market Data")
    
    # Legacy flag
    parser.add_argument("--vn100", action="store_true", help="Sync using legacy hardcoded VN100 list")
    
    # New flags
    parser.add_argument("--universe_mode", choices=["all", "current_vn100"], default="all", 
                        help="Select ticker universe (all or current_vn100)")
    parser.add_argument("--ticker", help="Sync only a single ticker (e.g., HPG)")
    parser.add_argument("--start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--force_refresh", action="store_true", help="Ignore existing database progress")
    parser.add_argument("--save_raw_copy", action="store_true", help="Save local CSV copies of fetched data")
    parser.add_argument("--benchmark", action="store_true", help="Sync VNINDEX benchmark data")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        train_vn100=args.vn100,
        universe_mode=args.universe_mode,
        ticker=args.ticker,
        start_str=args.start_date,
        end_str=args.end_date,
        force_refresh=args.force_refresh,
        save_raw_copy=args.save_raw_copy,
        benchmark=args.benchmark
    ))
