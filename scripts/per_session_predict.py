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

from src.database.connection import get_session
from src.models.price import RawPrice
from src.ml.trainer import DualModelTrainer
from src.ml.signal_generator import SignalGenerator
from config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.time_utils import now_vn, is_trading_day

logger = get_logger(__name__)

async def predict_ticker(trainer, sg, ticker, semaphore):
    async with semaphore:
        try:
            async with get_session() as session:
                # Fetch last 200 days for feature engineering
                stmt = select(RawPrice).filter(RawPrice.ticker == ticker).order_by(desc(RawPrice.timestamp)).limit(200)
                res = await session.execute(stmt)
                prices = res.scalars().all()
                
                if not prices or len(prices) < 30:
                    return ticker, None

                # Convert to DF
                df = pd.DataFrame([
                    {
                        "time": p.timestamp,
                        "open": float(p.open),
                        "high": float(p.high),
                        "low": float(p.low),
                        "close": float(p.close),
                        "volume": int(p.volume)
                    } for p in reversed(prices)
                ])

                # Run Multi-Horizon Prediction
                features_df = trainer.compute_features_for_ticker(ticker, df)
                if features_df is None or features_df.empty: return ticker, None
                
                last_row = features_df.iloc[-1]
                current_close = float(df.iloc[-1]["close"])
                
                horizons = ["1w", "1m", "6m"]
                multi_signals = {}
                
                for h in horizons:
                    try:
                        pred = trainer.predict(ticker, last_row, horizon=h)
                        if pred:
                            signal = sg.generate(ticker, current_close, pred)
                            multi_signals[h] = signal
                    except Exception as e:
                        logger.debug(f"horizon_failed:{h}", ticker=ticker, error=str(e))
                
                if not multi_signals: return ticker, None
                
                # Combine for TUI (use 1w at root as legacy/default)
                final_payload = multi_signals.get("1w", {}).copy()
                final_payload["multi_horizon"] = multi_signals
                
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

async def main_loop(interval_mins, session_mode=False, tickers_list=None):
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
            print(f"⏳ [SESSION MODE] Next run scheduled at {target_label} VN Time.")
            print(f"💤 Sleeping for {wait_sec/3600:.2f} hours...")
            await asyncio.sleep(wait_sec)
        
        start_time = time.time()
        print(f"🚀 [CRON] Starting ML prediction cycle for all tickers...")
        
        # Determine all tickers with models
        model_dir = root_dir / "models"
        if tickers_list:
            all_tickers = [t.upper() for t in tickers_list if (model_dir / t.upper()).exists()]
        else:
            all_tickers = [d.name for d in model_dir.iterdir() if d.is_dir()]
        
        # Prioritize ticker from TUI filter if active
        tui_ticker_path = root_dir / "data" / ".tui_ticker"
        if tui_ticker_path.exists():
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
        
        for future in asyncio.as_completed(tasks):
            ticker, res = await future
            if res:
                current_results[ticker] = res
            
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
        print(f"✅ Cycle complete. Processed {total} tickers in {elapsed:.1f}s.")
        
        if not session_mode:
            if interval_mins <= 0: break
            wait_time = max(10, (interval_mins * 60) - elapsed)
            print(f"💤 Waiting {wait_time:.0f}s until next cycle...")
            await asyncio.sleep(wait_time)
        else:
            # In session mode, wait at least a minute before checking next checkpoint 
            # to avoid instant re-trigger
            await asyncio.sleep(120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Session ML Prediction Engine")
    parser.add_argument("--loop", type=int, default=0, help="Interval in minutes between updates (0 = run once)")
    parser.add_argument("--session", action="store_true", help="Wait and run only at session ends (11:35 and 15:15 VN)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated list of tickers to predict")
    args = parser.parse_args()
    
    t_list = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else []
    
    try:
        asyncio.run(main_loop(args.loop, args.session, t_list))
    except KeyboardInterrupt:
        print("\n👋 Prediction Engine stopped.")
