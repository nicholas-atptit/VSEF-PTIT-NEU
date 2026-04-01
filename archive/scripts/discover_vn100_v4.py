from vnstock import Vnstock
import pandas as pd

try:
    vn = Vnstock()
    print("Vnstock initialized")
    
    # Try components with supported source
    try:
        s = vn.stock(symbol="VN100", source="VCI")
        if hasattr(s, 'listing') and hasattr(s.listing, 'components'):
            df = s.listing.components()
            print("s.listing.components found")
            print(df.head())
    except Exception as e:
        print(f"s.listing.components failed: {e}")

except Exception as e:
    print(f"Vnstock init failed: {e}")
