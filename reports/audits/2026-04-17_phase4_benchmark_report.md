**Phase 4 Benchmark**

Date: `2026-04-17`
Environment: `Python 3.12`
Scope: feature build speed, fragmentation warnings, artifact cache behavior, incremental rebuild behavior.

**Feature Build Modes**

| Mode | Tickers | Total Seconds | Seconds / Ticker | Mean Memory MB | Mean Columns | Performance Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_research_mode` | 5 | 0.575304 | 0.115061 | 2.996186 | 468 | 0 |
| `fast_core_mode` | 5 | 0.259671 | 0.051934 | 1.337559 | 214 | 0 |
| `regime_risk_mode` | 5 | 0.306030 | 0.061206 | 1.561860 | 233 | 0 |

Key observations:
- Main-path pandas fragmentation warnings were eliminated in the benchmarked feature builds.
- `fast_core_mode` reduced per-ticker build time by about `54.9%` vs `full_research_mode`.
- `regime_risk_mode` reduced per-ticker build time by about `46.8%` vs `full_research_mode`.
- Default behavior remains `full_research_mode`; the faster modes are explicit opt-in only.

**Artifact Cache**

| Artifact | Rows | Cold Seconds | Warm Seconds | Speedup x | Warm Cache Status | Provenance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `market_breadth.csv` | 2897 | 0.010042 | 0.000323 | 31.118 | `hit` | `derived_from_vnstock_data` |
| `sector_proxies.csv` | 8015 | 0.011729 | 0.000243 | 48.208 | `hit` | `derived_from_vnstock_data` |
| `macro_context.csv` | unavailable | n/a | n/a | n/a | n/a | n/a |

Notes:
- `macro_context.csv` was not present in the workspace, so only cache mechanics were validated in tests, not on a live artifact.
- `foreign_flow.csv` was also unavailable in the workspace at benchmark time, so it is covered by tests rather than a live artifact timing row.

**Incremental Rebuild Sample**

Sample universe: `4` local vnstock-grounded ticker CSVs copied into a temp workspace.

| Builder | Full Seconds | Incremental Seconds | Speedup x | Latest Date |
| --- | ---: | ---: | ---: | --- |
| `build_market_breadth_from_csv` | 0.059467 | 0.056357 | 1.055 | `2026-03-23` |
| `build_sector_proxies_from_csv` | 0.017126 | 0.021524 | 0.796 | `2026-03-23` |

Interpretation:
- Breadth incremental rebuild was modestly faster on the small sample.
- Sector incremental rebuild was slightly slower on the small sample because the universe was tiny and cache/setup overhead dominated.
- The incremental path is still operationally valuable for larger real universes because it avoids deterministic full-history rebuilds when only recent dates changed.

**Conclusion**

- Financial semantics were preserved.
- Main-path feature generation is materially lighter and warning-free.
- Artifact cache behavior is now explicit, test-covered, and measurable.
- Build modes are documented and manifest-visible.
