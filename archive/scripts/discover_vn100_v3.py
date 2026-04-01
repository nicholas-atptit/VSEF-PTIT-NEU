from vnstock import Vnstock
import pandas as pd

try:
    vn = Vnstock()
    print("Vnstock initialized")
    
    # Try symbols_by_group
    try:
        # Some versions have it as a top-level or under market
        if hasattr(vn, 'market') and hasattr(vn.market, 'symbols_by_group'):
            df = vn.market.symbols_by_group(group='VN100')
            print("vn.market.symbols_by_group found")
            print(df.head())
        elif hasattr(vn, 'symbols_by_group'):
            df = vn.symbols_by_group(group='VN100')
            print("vn.symbols_by_group found")
            print(df.head())
    except Exception as e:
        print(f"symbols_by_group failed: {e}")

    # Try components
    try:
        s = vn.stock(symbol="VN100", source="TCBS")
        if hasattr(s, 'listing') and hasattr(s.listing, 'components'):
            df = s.listing.components()
            print("s.listing.components found")
            print(df.head())
    except Exception as e:
        print(f"s.listing.components failed: {e}")

except Exception as e:
    print(f"Vnstock init failed: {e}")
