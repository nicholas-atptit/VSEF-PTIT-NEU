from vnstock import Vnstock
import pandas as pd

def test():
    v = Vnstock()
    print("🚀 Testing Vnstock Financials for HPG...")
    try:
        # Get ratios
        df_ratio = v.stock(symbol='HPG', source='VCI').finance.ratio(period='quarterly', lang='vi')
        if df_ratio is not None and not df_ratio.empty:
            print(f"✅ Ratios found: {df_ratio.columns.tolist()[:10]}...")
            print(df_ratio.head(2))
        
        # Get sector
        print("\n🚀 Testing Listing Info...")
        df_info = v.stock(symbol='HPG', source='VCI').listing.info()
        print(f"✅ Info: {df_info}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test()
