# Refactor Risk Register

| Risk | Severity | Mitigation | Status |
|---|---:|---|---|
| Protected evidence is changed or deleted | Critical | Add artifact policy registry; do not move/delete evidence; inspect staged diff | Mitigated; final staged inspection pending |
| Existing dirty work is overwritten | Critical | Snapshot on `safety/pre-refactor-dirty-work-snapshot-v1` before refactor | Mitigated locally; remote push pending |
| Large QML/model-universe runners change results | Critical | Keep runner algorithms and output code unchanged; compile-check only | Mitigated by scope |
| Runner-to-runner imports remain fragile | High | Add reusable APIs now; document incremental migration debt | Accepted debt |
| Missing forecast-engine/range-lab runners are invented | High | Record missing status; do not invent outputs or smoke results | Mitigated |
| Final-period data influences selection | Critical | Centralize strict split and validation-only selection policy; add tests | Mitigated by reusable policy and tests |
| Point-in-time feature leakage | Critical | Require grouped trailing/lagged transforms and add focused tests | Mitigated for new common builders |
| Metric definitions diverge | High | Add canonical metrics package and tests without rewriting evidence | Mitigated for new callers |
| GitHub push stalls because repository history is large | High | Keep local commits; retry explicit branch pushes; report exact status | Mitigated; safety and refactor branches pushed |
| Broad structure move breaks imports/references | High | No moves in this pass; migration map records decision | Mitigated |
| Live data/provider calls occur during validation | Critical | Run only policy checks, unit tests, compile checks, and documented offline smoke if runner exists | Mitigated; no fetch performed |
| Existing `config/` and `configs/` split remains confusing | Medium | Document architecture; defer risky migration | Accepted debt |
