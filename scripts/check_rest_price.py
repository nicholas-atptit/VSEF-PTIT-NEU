
import sys
try:
    from vnstock3 import Vnstock
    stock = Vnstock().stock(symbol="FPT", source="VCI")
    df = stock.quote.history(interval="1m", count=1)
    if not df.empty:
        print(f"LATEST_PRICE_REST: {df.iloc[-1]['close']}")
    else:
        print("REST_EMPTY")
except Exception as e:
    print(f"REST_ERROR: {str(e)}")
