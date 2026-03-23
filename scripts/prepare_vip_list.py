
import pandas as pd
import json
import os

def main():
    csv_path = 'H:/AI-ML-LLM in Stock_march26_PTIT_NEU/data/listing/danh_sach_VIP_14_cot.csv'
    output_path = 'H:/AI-ML-LLM in Stock_march26_PTIT_NEU/data/listing/danh_sach_VIP_LLM_ready.jsonl'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Filter for listed stocks
    listed = df[(df['status'] == 'listed') & (df['type'] == 'STOCK')]
    
    # Include ALL listed stocks
    symbols = listed['symbol'].tolist()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in symbols:
            f.write(json.dumps({'symbol': s}) + '\n')
            
    print(f"Successfully wrote {len(symbols)} symbols to {output_path}")

if __name__ == "__main__":
    main()
