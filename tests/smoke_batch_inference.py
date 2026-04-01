"""Smoke test for VN100 Batch Inference.

Runs a small batch inference on 3 tickers and verifies:
1. Cache data/latest_predictions.json exists.
2. Batch report data/processed/batch_inference_*.json exists.
3. Volatility field is present (even if Null).
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Force UTF-8 for stdout to prevent UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_batch_inference():
    print("--- Starting Batch Inference Smoke Test ---")
    
    # Clean old results if any
    report_dir = PROJECT_ROOT / "data" / "processed"
    if report_dir.exists():
        for f in report_dir.glob("batch_inference_*.json"):
            f.unlink()
    
    # Run a limited batch inference (3 known tickers)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "per_session_predict.py"),
        "--batch",
        "--tickers", "ABS,ACB,ACC"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    # Use encoding='utf-8' to handle emojis in the script's output
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    # 1. Check exit code
    if result.returncode != 0:
        print(f"[ERROR] Script failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

    # 2. Check if cache exists
    cache_path = PROJECT_ROOT / "data" / "latest_predictions.json"
    if not cache_path.exists():
        print("[ERROR] latest_predictions.json not created.")
        return False
    print("[OK] latest_predictions.json exists.")

    # 3. Check if batch report exists
    reports = list(report_dir.glob("batch_inference_*.json"))
    if not reports:
        print("[ERROR] No batch report found in data/processed/")
        print(f"STDOUT: {result.stdout}")
        return False
    
    report_path = reports[0]
    print(f"[OK] Batch report found: {report_path.name}")

    # 4. Validate report schema
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "predictions" not in data:
            print("[ERROR] 'predictions' key missing from report.")
            print(f"STDOUT: {result.stdout}")
            return False
            
        preds = data["predictions"]
        if not preds:
            print("[ERROR] No predictions entries in report.")
            print(f"STDOUT: {result.stdout}")
            # print(f"FILE CONTENT: {json.dumps(data, indent=2)}")
            return False
            
        # Check first entry for volatility
        first_ticker = list(preds.keys())[0]
        first_entry = preds[first_ticker]
        
        if "volatility" not in first_entry:
            print(f"[ERROR] 'volatility' field missing for {first_ticker}.")
            return False
        
        print(f"[OK] Schema validation passed for {first_ticker}.")
        
    except Exception as e:
        print(f"[ERROR] During JSON validation: {str(e)}")
        return False

    print("\n--- Smoke Test Passed! ---")
    return True

if __name__ == "__main__":
    success = test_batch_inference()
    sys.exit(0 if success else 1)
