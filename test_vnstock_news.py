from vnstock import Vnstock
import pandas as pd

def test_news():
    try:
        vn = Vnstock()
        ticker = 'VGI'
        s = vn.stock(symbol=ticker, source='VCI') 
        
        print("\n--- Vnstock (vn) attributes ---")
        for m in sorted(dir(vn)):
            if not m.startswith('_'):
                print(f"vn.{m}")
                
        print(f"\n--- vn.stock('{ticker}') (s) attributes ---")
        for m in sorted(dir(s)):
            if not m.startswith('_'):
                print(f"s.{m}")
                
        if hasattr(s, 'quote'):
            print(f"\n--- s.quote attributes ---")
            for m in sorted(dir(s.quote)):
                if not m.startswith('_'):
                    print(f"s.quote.{m}")
                    
        if hasattr(s, 'listing'):
            print(f"\n--- s.listing attributes ---")
            for m in sorted(dir(s.listing)):
                if not m.startswith('_'):
                    print(f"s.listing.{m}")
                    
        if hasattr(s, 'company'):
            print(f"\n--- Testing s.company.news() ---")
            df = s.company.news()
            print(f"Type: {type(df)}")
            if df is not None and not df.empty:
                print("Public Dates:")
                for date in df['public_date'].tolist()[:5]:
                    print(f"  - {date}")
                print(df.head())
            else:
                print("No news found or empty DataFrame.")

    except Exception as e:
        print(f"Error testing vnstock: {e}")

if __name__ == "__main__":
    test_news()
