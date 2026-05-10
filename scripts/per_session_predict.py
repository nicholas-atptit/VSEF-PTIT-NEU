import asyncio
import json
import os
import sys
import time
import argparse
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import select, desc, text

# Add current dir to path for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.ml.trainer import DualModelTrainer
from src.ml.signal_generator import SignalGenerator
from config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.time_utils import now_vn, is_trading_day
from src.data.universe import get_vn100_universe
from src.ml.data_loader import VN100DataLoader

logger = get_logger(__name__)

# Fetch dynamic universe if needed
def load_target_universe(mode="current_plus_viettel"):
    try:
        return get_vn100_universe(mode=mode)
    except Exception as e:
        logger.error("universe_load_failed", error=str(e))
        # Fallback to a minimal safe set or empty
        return []

async def predict_ticker(trainer, sg, ticker, semaphore):
    async with semaphore:
        try:
            # Use VN100DataLoader for robust multi-source loading
            # Join market, fundamentals, and sentiment to match training contract
            loader = VN100DataLoader(prefer_source="csv")
            df = await asyncio.to_thread(
                loader.build_inference_dataset, 
                tickers=[ticker], 
                lookback_days=300,
                join_market=True,
                join_fundamentals=True,
                join_sentiment=True,
                join_sectors=True
            )
            
            if df.empty or len(df) < 30:
                logger.warning("insufficient_data", ticker=ticker, count=len(df))
                return ticker, None

            # df from builder has 'ticker' column, we only need the data for this ticker
            df = df[df["ticker"] == ticker].copy()
            if df.empty: return ticker, None
            
            # Ensure columns are what trainer expects (lowercase)
            col_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=col_map)
            # Run Multi-Horizon Feature Computation
            features_df = trainer.compute_features_for_ticker(ticker, df)
            
            if features_df is None or features_df.empty:
                logger.warning("features_empty", ticker=ticker)
                return ticker, None
            
            current_close = float(df.iloc[-1]["close"])
            
            horizons = ["1w", "1m", "6m"]
            multi_signals = {}
            
            for h in horizons:
                try:
                    pred = trainer.predict(ticker, features_df, horizon=h)
                    if pred:
                        signal = await sg.generate(ticker, current_close, pred, volatility_score=pred.get("volatility"))
                        multi_signals[h] = signal
                    else:
                        logger.debug("no_prediction_for_horizon", ticker=ticker, horizon=h)
                except Exception as e:
                    logger.warning(f"horizon_failed:{h}", ticker=ticker, error=str(e))
            
            if not multi_signals:
                logger.warning("no_multi_signals_generated", ticker=ticker)
                return ticker, None
            
            # --- Unified Agent Payload (Phase 3 Evolution) ---
            # Use the short horizon as the primary serialized payload view.
            final_payload = multi_signals.get("1w", {}).copy()
            final_payload["multi_horizon"] = multi_signals # Preserve for TUI 'Forecast Radar'
            
            return ticker, final_payload

        except Exception as e:
            logger.error("prediction_failed", ticker=ticker, error=str(e))
            return ticker, None

def get_wait_time_until_session_end():
    """Calculate seconds until next VN session end (11:35 or 15:15)."""
    now = now_vn()
    h, m = now.hour, now.minute
    
    # Target checkpoints (VN Time)
    checkpoints = [
        (11, 35), # Morning session end + 5 mins buffer
        (15, 10), # Afternoon session end + 10 mins buffer
        (23, 59)  # End of day (to reset for next day)
    ]
    
    for ch_h, ch_m in checkpoints:
        target = now.replace(hour=ch_h, minute=ch_m, second=0, microsecond=0)
        if target > now:
            diff = (target - now).total_seconds()
            return diff, f"{ch_h:02d}:{ch_m:02d}"
            
    # If all checkpoints passed, wait until tomorrow 11:35
    tomorrow_target = (now + dt.timedelta(days=1)).replace(hour=11, minute=35, second=0, microsecond=0)
    return (tomorrow_target - now).total_seconds(), "Tomorrow 11:35"

async def main_loop(interval_mins, session_mode=False, tickers_list=None, train_vn100=False, batch_mode=False, limit=None):
    settings = get_settings()
    trainer = DualModelTrainer()
    sg = SignalGenerator()
    
    cache_path = root_dir / "data" / "latest_predictions.json"
    current_results = {}
    
    # Load existing cache if any
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                current_results = json.load(f)
        except Exception: pass

    while True:
        if session_mode:
            wait_sec, target_label = get_wait_time_until_session_end()
            print(f"[SESSION MODE] Next run scheduled at {target_label} VN Time.")
            print(f"Sleeping for {wait_sec/3600:.2f} hours...")
            await asyncio.sleep(wait_sec)
        
        # Check if today is a trading day
        if not is_trading_day():
            logger.info("not_a_trading_day_skipping")
            if session_mode:
                # Wait longer if in session mode
                await asyncio.sleep(3600)
                continue
            elif interval_mins > 0:
                await asyncio.sleep(interval_mins * 60)
                continue
            else:
                break

        start_time = time.time()
        print(f"Starting ML prediction cycle...")
        
        # Determine all tickers with models
        model_dir = root_dir / "models"
        if tickers_list:
            all_tickers = [t.upper() for t in tickers_list if (model_dir / t.upper()).exists()]
        elif train_vn100 or batch_mode:
            target_set = set(load_target_universe(mode="current_plus_viettel"))
            all_tickers = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.upper() in target_set]
            print(f"[UNIVERSE MODE] Target set: {len(target_set)} tickers. Models found: {len(all_tickers)}")
        else:
            all_tickers = [d.name for d in model_dir.iterdir() if d.is_dir()]
        
        # Apply limit if specified
        if limit:
            all_tickers = all_tickers[:limit]
            print(f"Limited to first {limit} tickers.")

        # Prioritize ticker from TUI filter if active
        tui_ticker_path = root_dir / "data" / ".tui_ticker"
        if tui_ticker_path.exists() and not batch_mode:
            try:
                prio = tui_ticker_path.read_text().strip().upper()
                if prio in all_tickers:
                    all_tickers.remove(prio)
                    all_tickers.insert(0, prio)
            except Exception: pass

        semaphore = asyncio.Semaphore(15) 
        tasks = [predict_ticker(trainer, sg, ticker, semaphore) for ticker in all_tickers]
        
        completed = 0
        total = len(all_tickers)
        batch_results = {}
        
        for future in asyncio.as_completed(tasks):
            ticker, res = await future
            if res:
                current_results[ticker] = res
                batch_results[ticker] = res
                logger.info("prediction_success", ticker=ticker)
            else:
                logger.warning("prediction_failed_or_skipped", ticker=ticker)
            
            completed += 1
            if completed % 25 == 0 or completed == total:
                try:
                    tmp_path = cache_path.with_suffix(".tmp")
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(current_results, f, indent=4)
                    if cache_path.exists(): cache_path.unlink()
                    tmp_path.rename(cache_path)
                    print(f"DEBUG: Progress {completed}/{total} - Cache Updated.", end="\r")
                except Exception: pass
        
        print() 
        elapsed = time.time() - start_time
        print(f"Cycle complete. Processed {total} tickers in {elapsed:.1f}s.")

        # Batch reporting
        if batch_mode:
            report_dir = root_dir / "data" / "processed"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"batch_inference_{now_vn().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "timestamp": now_vn().isoformat(),
                        "elapsed_sec": round(elapsed, 2),
                        "total_tickers": total,
                        "success_count": len(batch_results),
                        "predictions": batch_results
                    }, f, indent=4)
                print(f"Batch Report saved to: {report_path}")
            except Exception as e:
                logger.error("batch_report_failed", error=str(e))
        
        if not session_mode:
            if interval_mins <= 0: break
            wait_time = max(10, (interval_mins * 60) - elapsed)
            print(f"Waiting {wait_time:.0f}s until next cycle...")
            await asyncio.sleep(wait_time)
        else:
            await asyncio.sleep(120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Session ML Prediction Engine")
    parser.add_argument("--loop", type=int, default=0, help="Interval in minutes between updates (0 = run once)")
    parser.add_argument("--session", action="store_true", help="Wait and run only at session ends (11:35 and 15:15 VN)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated list of tickers to predict")
    parser.add_argument("--vn100", action="store_true", dest="train_vn100", help="Predict only VN100 + 4 Viettel tickers")
    parser.add_argument("--batch", action="store_true", help="Batch inference mode with structured reporting")
    parser.add_argument("--limit", type=int, help="Limit number of tickers to process")
    args = parser.parse_args()
    
    t_list = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else []
    
    try:
        asyncio.run(main_loop(args.loop, args.session, t_list, args.train_vn100, args.batch, args.limit))
    except KeyboardInterrupt:
        print("\n👋 Prediction Engine stopped.")
