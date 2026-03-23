import asyncio
import json
import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import select, desc, text

# Add current dir to path for imports
sys.path.append(os.getcwd())

from src.database.connection import get_session
from src.models.price import RawPrice
from src.ml.trainer import DualModelTrainer
from src.ml.signal_generator import SignalGenerator
from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def predict_ticker(trainer, sg, ticker, semaphore):
    async with semaphore:
        try:
            async with get_session() as session:
                # Fetch last 200 days for feature engineering
                stmt = select(RawPrice).filter(RawPrice.ticker == ticker).order_by(desc(RawPrice.timestamp)).limit(200)
                res = await session.execute(stmt)
                rows = res.scalars().all()
                if not rows: 
                    return ticker, None
                
                # Convert to DF
                df = pd.DataFrame([{
                    "time": r.timestamp,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume)
                } for r in rows]).sort_values("time")
                
                # Compute features
                feat_df = trainer.compute_features_for_ticker(ticker, df)
                if feat_df is None or feat_df.empty: return ticker, None
                
                # Get latest row for prediction
                latest_features = feat_df.iloc[-1]
                latest_price = float(df.iloc[-1]["close"])
                
                # Predict trend & range
                pred = trainer.predict(ticker, latest_features)
                
                # Generate signal & action plan
                p = sg.generate(ticker, latest_price, pred)
                
                # Format for TUI
                result = {
                    "ml_prediction": {
                        "trend_probabilities": p["quantitative_signals"]["trend_probabilities"],
                        "expected_range": p["quantitative_signals"]["expected_range"],
                        "max_upside_pct": p["quantitative_signals"].get("max_upside_pct", 0.0),
                        "max_downside_pct": p["quantitative_signals"].get("max_downside_pct", 0.0),
                        "action_plan": p["quantitative_signals"]["action_plan"]
                    },
                    "llm_analysis": {
                        "overall_outlook": "POSITIVE" if p["quantitative_signals"]["trend_probabilities"]["up"] > 0.5 else "NEGATIVE" if p["quantitative_signals"]["trend_probabilities"]["down"] > 0.5 else "NEUTRAL",
                        "reasoning": f"ML Consensus: Trend score {p['quantitative_signals']['trend_probabilities']['up']:.1%}. Entry Zone: {p['quantitative_signals']['action_plan']['entry_zone']}",
                        "news_headlines": "",
                        "rl_recommendation": {"suggested_allocation_pct": 0.05 if p["quantitative_signals"]["trend_probabilities"]["up"] > 0.5 else 0.01},
                        "deep_learning_context": {
                            "tft_forecast": f"Proj: {p['quantitative_signals']['expected_range']['median_50th']:,.0f}",
                            "cnn_microstructure": "Stable"
                        }
                    }
                }
                return ticker, result
        except Exception as e:
            logger.error("prediction_failed", ticker=ticker, error=str(e))
            return ticker, None

async def run_batch(trainer, sg, tickers):
    semaphore = asyncio.Semaphore(10)  # Limit concurrency to 10 tickers
    tasks = [predict_ticker(trainer, sg, ticker, semaphore) for ticker in tickers]
    
    results = {}
    completed = 0
    total = len(tickers)
    
    for future in asyncio.as_completed(tasks):
        ticker, res = await future
        if res:
            results[ticker] = res
        completed += 1
        if completed % 20 == 0 or completed == total:
            print(f"DEBUG: Processed {completed}/{total} tickers...", end="\r")
    
    print() # New line
    return results

def get_all_tickers_with_models():
    model_dir = Path("models")
    if not model_dir.exists(): return []
    return [d.name for d in model_dir.iterdir() if d.is_dir() and (d / "trend_classifier.joblib").exists()]

async def main_loop(interval_mins):
    trainer = DualModelTrainer()
    sg = SignalGenerator()
    cache_path = Path("data/prediction_cache/latest_predictions.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    while True:
        start_time = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔮 Starting new prediction cycle...")
        
        all_tickers = get_all_tickers_with_models()
        
        # Priority check
        prio_ticker = None
        tui_ticker_path = Path("data/.tui_ticker")
        if tui_ticker_path.exists():
            try:
                prio_ticker = tui_ticker_path.read_text().strip().upper()
                if prio_ticker in all_tickers:
                    all_tickers.remove(prio_ticker)
                    all_tickers.insert(0, prio_ticker)
                    print(f"DEBUG: Prioritizing active TUI ticker: {prio_ticker}")
            except Exception: pass
            
        print(f"DEBUG: Found {len(all_tickers)} tickers with models.")
        
        # Incremental processing
        current_results = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    current_results = json.load(f)
            except Exception: pass

        semaphore = asyncio.Semaphore(15) # Slightly higher concurrency
        tasks = [predict_ticker(trainer, sg, ticker, semaphore) for ticker in all_tickers]
        
        completed = 0
        total = len(all_tickers)
        
        for future in asyncio.as_completed(tasks):
            ticker, res = await future
            if res:
                current_results[ticker] = res
            
            completed += 1
            if completed % 25 == 0 or completed == total:
                # Incremental Save
                try:
                    tmp_path = cache_path.with_suffix(".tmp")
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(current_results, f, indent=4)
                    if cache_path.exists(): cache_path.unlink()
                    tmp_path.rename(cache_path)
                    print(f"DEBUG: Progress {completed}/{total} - Persistent Cache Updated.", end="\r")
                except Exception: pass
        
        print() # New line
        elapsed = time.time() - start_time
        print(f"✅ Cycle complete. Processed {total} tickers in {elapsed:.1f}s.")
        
        if interval_mins <= 0: break
        wait_time = max(10, (interval_mins * 60) - elapsed)
        print(f"💤 Waiting {wait_time:.0f}s until next cycle...")
        await asyncio.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Session ML Prediction Engine")
    parser.add_argument("--loop", type=int, default=0, help="Interval in minutes between updates (0 = run once)")
    args = parser.parse_args()
    
    try:
        asyncio.run(main_loop(args.loop))
    except KeyboardInterrupt:
        print("\n👋 Prediction Engine stopped.")
