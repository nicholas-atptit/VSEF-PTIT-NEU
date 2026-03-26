import sys
import os
from pathlib import Path
import subprocess
import datetime as dt
import argparse

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Try to import VN time if available
try:
    from src.utils.time_utils import now_vn
    today_str = now_vn().date().isoformat()
except:
    today_str = dt.date.today().isoformat()

def run_step(name, cmd_list):
    print(f"\n--- 🛠️ STEP: {name} ---")
    print(f"Executing: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified Market Update Orchestrator")
    parser.add_argument("--all", action="store_true", help="Sync all 1500+ tickers (otherwise uses top-10)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to look back (default: 1)")
    parser.add_argument("--start", type=str, help="Explicit start date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Calculate start date
    # Base it on now_vn() if available
    current_time = dt.datetime.now()
    try:
        from src.utils.time_utils import now_vn
        current_time = now_vn()
    except: pass

    if args.start:
        start_date = args.start
    else:
        start_date = (current_time - dt.timedelta(days=args.days-1)).date().isoformat()
    
    # Refresh today_str for the end boundary
    current_today_str = current_time.date().isoformat()
    
    print(f"🚀 [MARKET REFRESH] Manual Update Triggered: {start_date} to {current_today_str}")
    
    # We use the same python interpreter
    python_exe = sys.executable
    
    # 1. Sync Latest Prices
    price_cmd = [python_exe, "scripts/run_backdate.py", "--start", start_date, "--end", current_today_str]
    if args.all: 
        price_cmd.append("--all")
    else: 
        price_cmd.extend(["--tickers", "FPT", "VGI", "VHM", "VIC", "SSI", "TCB", "VNM", "HPG", "MWG", "DGC"])
    
    run_step("Syncing Prices", price_cmd)
    
    # 2. Update ML Decisions (Forecasts)
    run_step("Updating ML Predictions", [python_exe, "scripts/per_session_predict.py"])

    # 3. Harvest Latest News
    news_cmd = [python_exe, "scripts/update_news.py"]
    if args.all: 
        news_cmd.append("--all")
    run_step("Harvesting Latest News", news_cmd)
    
    print(f"\n✅ [DONE] Full System Refresh Complete! Price, ML, and News are now up to date.")

if __name__ == "__main__":
    main()
