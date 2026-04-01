# VN100 Universe Loader

The `VN100 Universe Loader` provides a centralized interface for retrieving the list of constituent symbols in the VN100 index. This module ensures that all scripts, ML training pipelines, and inference tasks operate on the same set of tickers.

## Architecture

The loader is located at `src/data/universe.py`. It integrates with `src/adapters/vnstock_adapter.py` to retrieve live constituent lists from `vnstock` and provides a robust fallback to a hardcoded "snapshot" if the API is unavailable.

## Usage

### Basic Retrieval

To get the current list of VN100 tickers:

```python
from src.data.universe import get_vn100_universe

# Default: returns current VN100 tickers
tickers = get_vn100_universe()
print(f"Loaded {len(tickers)} VN100 tickers.")
```

### Modes

The loader supports different modes to cater to various use cases:

| Mode | Description |
| :--- | :--- |
| `current` | (Default) Current VN100 constituents. |
| `current_vn100` | Alias for `current`. |
| `current_plus_viettel` | VN100 plus 4 specific Viettel tickers used for enhanced coverage (`VTP`, `VGI`, `CTR`, `FOX`). |

```python
# Extended universe for training/inference
full_universe = get_vn100_universe(mode="current_plus_viettel")
```

### Batch Ingestion Integration

Replace local static lists in `scripts/sync_all_data.py`:

```python
from src.data.universe import get_vn100_universe

async def main(universe_mode="all", ...):
    if universe_mode == "current_vn100":
        tickers = get_vn100_universe(mode="current")
    # ...
```

## Future Work

- **Historical Constituents**: Support for `as_of_date` to retrieve historical VN100 members. Currently, any date provided will log a warning and return the current list.
- **Dynamic Source Switching**: Better selection between VCI, TCBS, or other sources for constituent data.
