"""Best-effort contract probe for the canonical vnstock_data provider.

Canonical provider: vnstock_data (NOT vnstock).
This file checks the provider contract when vnstock_data is importable in the active interpreter.
It does not assert runtime availability by itself; missing imports are reported as a soft skip.
"""

import datetime as dt
import pandas as pd


def test_ohlcv_schema_from_vnstock_data():
    """Verify vnstock_data.Quote returns expected OHLCV columns and types when available."""
    try:
        from vnstock_data import Quote

        ticker = "VCB"
        end = dt.date.today()
        start = end - dt.timedelta(days=30)

        df = Quote(source="VCI", symbol=ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1D",
            get_all=True,
        )

        assert df is not None, "Quote returned None"
        assert not df.empty, f"Quote returned empty DataFrame for {ticker}"

        # Canonical column check
        date_col = "time" if "time" in df.columns else "date"
        required_cols = [date_col, "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"Missing OHLCV columns: {missing}. Available: {list(df.columns)}"

        # Type assertions
        assert df["close"].dtype in [float, "float64", "float32"], f"close dtype: {df['close'].dtype}"
        assert df["volume"].dtype in [int, "int64", "int32", float, "float64"], f"volume dtype: {df['volume'].dtype}"

        print(f"\n[PASS] vnstock_data OHLCV contract: {ticker}, {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(df.tail(3))

    except ImportError:
        print("[SKIP] vnstock_data not installed. Run: pip install vnstock_data")


def test_listing_from_vnstock_data():
    """Verify vnstock_data.Listing returns a valid symbols table when available."""
    try:
        from vnstock_data import Listing

        df = Listing(source="vnd").all_symbols()
        assert df is not None, "Listing returned None"
        assert not df.empty, "Listing returned empty DataFrame"

        ticker_col = "ticker" if "ticker" in df.columns else df.columns[0]
        assert len(df[ticker_col]) > 100, f"Expected >100 symbols, got {len(df[ticker_col])}"

        print(f"\n[PASS] vnstock_data Listing contract: {len(df)} symbols")
        print(f"Columns: {list(df.columns)}")
        print(df.head(3))

    except ImportError:
        print("[SKIP] vnstock_data not installed. Run: pip install vnstock_data")


if __name__ == "__main__":
    test_ohlcv_schema_from_vnstock_data()
    test_listing_from_vnstock_data()
