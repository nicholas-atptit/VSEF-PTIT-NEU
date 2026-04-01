from vnstock import Vnstock
import pandas as pd

try:
    vn = Vnstock()
    print("Vnstock initialized")
    
    # List all attributes to find market/listing related ones
    attrs = [a for a in dir(vn) if not a.startswith('_')]
    print(f"Attributes: {attrs}")
    
    # Try stock-based listing
    try:
        s = vn.stock(symbol="VN100", source="VCI")
        print("Stock VN100 initialized")
        s_attrs = [a for a in dir(s) if not a.startswith('_')]
        print(f"Stock Attributes: {s_attrs}")
        
        if 'listing' in s_attrs:
            print("Listing attribute found in Stock")
            l_attrs = [a for a in dir(s.listing) if not a.startswith('_')]
            print(f"Listing Attributes: {l_attrs}")
    except Exception as e:
        print(f"Stock VN100 failed: {e}")

except Exception as e:
    print(f"Vnstock init failed: {e}")
