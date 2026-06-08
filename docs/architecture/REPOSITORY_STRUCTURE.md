# Repository Structure

## Reusable code

| Path | Responsibility |
|---|---|
| `src/data/` | Provider contracts, adapters, local loaders, and data validation. |
| `src/features/builders/` | Point-in-time-safe common feature builders. |
| `src/forecasting/panels/` | Canonical offline forecast-panel contract. |
| `src/forecasting/selectors/` | Validation-only candidate selection. |
| `src/evaluation/metrics/` | Direction, return/price, interval, and ranking metrics. |
| `src/evaluation/baselines/` | Direction, return/price, and interval baselines. |
| `src/governance/` | Split, claim-boundary, and artifact policies. |
| `src/utils/` | Paths, timestamps, serialization, logging, and manifests. |

Existing `src/ml/`, `src/forecast/`, and `src/engine/` modules remain supported.
They were not broadly moved because many tests, scripts, and evidence files
depend on their current paths.

## Runners and configuration

- `scripts/research/`: one-off research, audit, and benchmark orchestration.
- `scripts/validation/`: validation entry points where present.
- `scripts/maintenance/`: maintenance entry points where present.
- `configs/`: active experiment, feature, policy, and universe configuration.
- `config/`: legacy/current runtime configuration retained to avoid import risk.

Large QML and Model Universe runners remain in place. Their stable split and I/O
helpers now come from reusable `src/` modules.

## Evidence and artifacts

- `reports/_index/ACTIVE_EVIDENCE_INDEX.md`: primary evidence navigation.
- `reports/results/`, `reports/claims/`, `reports/protocols/`: active evidence.
- `reports/generated/`: generated evidence; preserve in place.
- `reports/paper/`: paper materials and paper evidence.
- `data/`, `outputs/`, `archive/`, `paper_evidence_export/`,
  `paper_evidence_raw_full_export/`: protected data/evidence roots.

See `docs/governance/DATA_AND_ARTIFACT_POLICY.md` before moving or deleting any
artifact.
