# Vnstock Adapter layer

This module provides a thin wrapper around `vnstock>=3.0` to centralize data acquisition for market data, fundamental data, and news.

## Features

- **Automatic API key injection**: Uses `config/settings.py` to configure `vnstock` environment variables.
- **Standarized OHLCV fetching**: Methods for both ticker symbols and market indices.
- **Financial Ratios**: Fetches latest yearly financial metrics.
- **News Aggregation**: Retrieves ticker-specific news from multiple vnstock sources.
- **VN100 constituent list**: Provides easy access to the list of VN100 companies.

## Usage

```python
from src.adapters.vnstock_adapter import VnstockAdapter

# Initialize adapter (API keys are injected automatically)
adapter = VnstockAdapter()

# Fetch daily OHLCV for SSI
ohlc_df = adapter.get_ohlc("SSI", "2024-01-01", "2024-03-31")

# Fetch VN-Index data
vnindex_df = adapter.get_index_ohlcv("VNINDEX", "2024-01-01", "2024-03-31")

# Fetch latest financial ratios for HPG
fund_df = adapter.get_financial_ratios("HPG")

# Fetch latest 10 news items for VCB
news_df = adapter.get_news("VCB", count=10)

# Get the full VN100 ticker list
vn100_tickers = adapter.get_vn100_tickers()
```

## API Methods

### `__init__(symbol_list: Optional[List[str]] = None)`
Initializes the `Vnstock` client and injects API keys from `Settings`.

### `get_ohlc(symbol, start_date, end_date, interval="1D")`
Fetches historical OHLCV data for a ticker symbol. Standardizes column names from `vnstock` (renaming `time` to `date`).

### `get_index_ohlcv(symbol, start_date, end_date, interval="1D")`
Fetches historical OHLCV data for a market index (e.g., `VNINDEX`). This is a thin wrapper over `get_ohlc`.

### `get_financial_ratios(symbol)`
Fetches latest yearly financial ratios (P/E, P/B, ROE, etc.) for a symbol.

### `get_news(ticker, count=10)`
Fetches recent news items for a ticker with automatic fallback between `stock.news()` and `stock.company.news()`.

### `get_vn100_tickers()`
Returns the list of ticker symbols that constitute the VN100 index.
