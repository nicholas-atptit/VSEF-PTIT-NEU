
import json
from pathlib import Path

import pandas as pd

def main():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "listing" / "danh_sach_VIP_14_cot.csv"
    output_path = repo_root / "data" / "listing" / "danh_sach_VIP_LLM_ready.jsonl"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Filter for listed stocks
    listed = df[(df['status'] == 'listed') & (df['type'] == 'STOCK')]
    
    # Include ALL listed stocks
    symbols = listed['symbol'].tolist()
    
    with output_path.open('w', encoding='utf-8') as f:
        for s in symbols:
            f.write(json.dumps({'symbol': s}) + '\n')
            
    print(f"Successfully wrote {len(symbols)} symbols to {output_path}")

if __name__ == "__main__":
    main()
