from vnstock import Vnstock
import pandas as pd

def test_listing_all():
    v = Vnstock()
    s = v.stock(symbol='HPG')
    print("🚀 Trying s.listing.all_symbols()...")
    try:
        df = s.listing.all_symbols()
        print(f"✅ Success (all_symbols): {df.columns.tolist()}")
        print(df.head(2))
    except Exception as e: print(f"❌ Fail (all_symbols): {e}")

    print("\n🚀 Trying s.listing.symbols_by_industries()...")
    try:
        # Some versions need arguments or return a dict
        res = s.listing.symbols_by_industries()
        print(f"✅ Success (industries) type: {type(res)}")
        if isinstance(res, pd.DataFrame):
             print(res.head(2))
        else:
             print(str(res)[:200])
    except Exception as e: print(f"❌ Fail (industries): {e}")

if __name__ == "__main__":
    test_listing_all()
