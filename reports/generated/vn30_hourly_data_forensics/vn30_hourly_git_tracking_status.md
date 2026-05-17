# VN30 Hourly Data Git Tracking Status

## Git Ignore Status

| Directory | Ignored By | Tracked by Git |
|-----------|-----------|----------------|
| `data/market_cache/` | `.gitignore:38` | NO |
| `data/raw/` | `.gitignore:39` | NO |
| `outputs/` | `.gitignore:59` | NO |
| `archive/generated_data_snapshots/` | `.gitignore:60` | NO |

## Implications

1. **NO data files are tracked by Git**. All data directories are explicitly ignored.
2. **Git cannot restore missing data**. If data is deleted from the filesystem, it cannot be recovered from Git history.
3. **Data exists only in the local filesystem**. The data is not backed up in the Git repository.
4. **Archive snapshots are also ignored**. Even the archive snapshots are not tracked by Git.

## Can Git Restore Missing Data?

**NO.** All data directories are in `.gitignore`. Git has never tracked any CSV, parquet, or JSON data files in these directories.

## Where Data Actually Exists

- **Active cache**: `data/market_cache/vnstock_data/` (local filesystem only)
- **Raw fetch**: `data/raw/vnstock_fetch/` (local filesystem only)
- **Outputs**: `outputs/` (local filesystem only)
- **Archive**: `archive/generated_data_snapshots/` (local filesystem only)

## Conclusion

The data is not deleted - it exists in the local filesystem in multiple locations. However, it is NOT tracked by Git and cannot be restored from version control. If the local filesystem is lost, the data would need to be re-fetched from the vendor.
