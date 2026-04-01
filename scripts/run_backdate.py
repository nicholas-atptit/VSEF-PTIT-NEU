"""Entry point: Run the historical backdate ingestion.

Usage:
    python scripts/run_backdate.py --tickers HPG VIC VNM --start 2014-01-01 --end 2025-12-31
    python scripts/run_backdate.py --all --start 2020-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.data.historical.backdate import BackdateIngestor
from src.utils.logging import setup_logging, get_logger

import datetime as dt


# Default tickers for backdate (top VN30 stocks)
DEFAULT_TICKERS = [
    "VNM", "VIC", "VHM", "HPG", "MSN", "VCB", "BID", "CTG",
    "TCB", "MBB", "ACB", "FPT", "MWG", "PNJ", "REE", "SSI",
    "VND", "HCM", "GAS", "PLX", "POW", "PVD", "DPM", "DCM",
    "VRE", "NVL", "KDH", "DXG", "PDR", "VJC",
]


async def main() -> None:
    """Run backdate ingestion."""
    parser = argparse.ArgumentParser(description="Historical data backdate ingestion")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific ticker symbols to backfill",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Backfill all default VN30 tickers",
    )
    parser.add_argument(
        "--start", type=str, default="2014-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )

    args = parser.parse_args()

    setup_logging()
    logger = get_logger("run_backdate")

    # Determine tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all:
        import json
        logger.info("fetching_all_market_symbols")
        df_listing = None
        try:
            # Lấy dữ liệu siêu tốc bằng quyền Insider (nguồn vnd)
            from vnstock_data import Listing
            logger.info("using_vnstock_data_insider_source")
            df_listing = Listing(source='vnd').all_symbols()
        except ImportError:
            logger.warning("vnstock_data_not_found_falling_back_to_vnstock_vci")
            try:
                from vnstock import Vnstock
                df_listing = Vnstock().stock(symbol="VND", source="VCI").listing.all_symbols()
            except Exception as backup_e:
                logger.error("failed_to_fetch_symbols_backup", error=str(backup_e))
                sys.exit(1)
        except Exception as e:
            logger.error("failed_to_fetch_symbols", error=str(e))
            sys.exit(1)

        if df_listing is not None and not df_listing.empty:
            # Determine ticker column
            ticker_col = 'ticker' if 'ticker' in df_listing.columns else df_listing.columns[0]
            tickers = df_listing[ticker_col].tolist()
            
            # Export CSV (Requested by user)
            csv_path = 'danh_sach_VIP_14_cot.csv'
            df_listing.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # Export LLM-Optimized Format (.jsonl)
            jsonl_path = 'danh_sach_VIP_LLM_ready.jsonl'
            try:
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for _, row in df_listing.iterrows():
                        # Create a textual representation suitable for instruction tuning / RAG
                        record = row.to_dict()
                        t_sym = record.get(ticker_col, "")
                        
                        # Create a rich natural language summary of the row
                        desc_parts = [f"This record describes the stock symbol {t_sym}."]
                        for key, val in record.items():
                            if key != ticker_col and pd.notna(val):
                                desc_parts.append(f"Its {key} is {val}.")
                        
                        llm_text = " ".join(desc_parts)
                        
                        llm_obj = {
                            "text": llm_text,
                            "metadata": {"ticker": t_sym, "type": "company_listing"}
                        }
                        f.write(json.dumps(llm_obj, ensure_ascii=False) + "\n")
                
                logger.info(
                    "fetched_all_symbols_and_exported", 
                    count=len(tickers), 
                    saved_csv=csv_path,
                    saved_jsonl=jsonl_path
                )
                print(f"Lụm lúa! Đã xuất {csv_path} và {jsonl_path} (tối ưu cho LLM) \nMở thư mục lên check hàng thôi bác ơi!")
                
                # Bulk insert Company Profiles to DB to avoid 1500 API calls later
                logger.info("bulk_inserting_company_profiles")
                from src.data.database.connection import get_session
                from src.ml.models.company import CompanyProfile
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                
                async def _insert_profiles():
                    company_values = []
                    for _, row in df_listing.iterrows():
                        rec = row.to_dict()
                        t = rec.get(ticker_col, "")
                        if not t:
                            continue
                        # Map known columns from vnstock_data VIP listing if available
                        company_values.append({
                            "ticker": t,
                            "exchange": str(rec.get("exchangeName", rec.get("exchange", "HOSE"))),
                            "industry": str(rec.get("industryName", rec.get("industry", ""))),
                            "company_name": str(rec.get("organName", rec.get("companyName", ""))),
                            "short_name": str(rec.get("organShortName", rec.get("shortName", t))),
                            "issue_share": float(rec.get("issueShare", rec.get("outstandingShare", 0))),
                            "charter_capital": float(rec.get("charterCapital", 0)),
                            "foreign_percent": float(rec.get("foreignPercent", 0)),
                        })
                    
                    if company_values:
                        async with get_session() as session:
                            stmt = pg_insert(CompanyProfile).values(company_values)
                            stmt = stmt.on_conflict_do_update(
                                index_elements=["ticker"],
                                set_={
                                    "exchange": stmt.excluded.exchange,
                                    "industry": stmt.excluded.industry,
                                    "company_name": stmt.excluded.company_name,
                                    "short_name": stmt.excluded.short_name,
                                }
                            )
                            await session.execute(stmt)
                            
                # Run the bulk insert
                try:
                    await _insert_profiles()
                    logger.info("bulk_insert_company_profiles_done")
                except Exception as db_e:
                    logger.error("bulk_insert_company_profiles_failed", error=str(db_e))
                    
            except Exception as e:
                logger.warning("failed_to_export_files", error=str(e))
        else:
            logger.error("listing_dataframe_is_empty")
            sys.exit(1)
    else:
        print("Please specify --tickers or --all")
        print("Example: python scripts/run_backdate.py --tickers HPG VIC --start 2020-01-01")
        sys.exit(1)

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    logger.info(
        "backdate_starting",
        tickers=tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    ingestor = BackdateIngestor()
    await ingestor.run(tickers, start_date, end_date)

    logger.info("backdate_complete")


if __name__ == "__main__":
    asyncio.run(main())
