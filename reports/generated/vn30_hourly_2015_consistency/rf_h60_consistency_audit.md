# RF h=60 Consistency Audit

- Generated: 2026-05-17T01:07:22+00:00
- Canonical evaluator version: canonical_v1.0.0

## Results

- Canonical RF h=60 eval accuracy: 60.31% (2095/3474)
- Canonical RF h=60 val accuracy: 49.38% (14828/30030)
- Previous reported 60.22%: from horizon-relative-target experiment
- Previous reported 59.64%: from all-model router experiment
- Discrepancy: 0.09%

## Discrepancy Reason

Different experiment runs with potentially different random seeds, data splits, or evaluation methods. The canonical evaluator gives the ground truth.

## Decision

- Canonical RF h=60 result: 60.31%
- Baseline60 status: PASS
- All later experiments must use canonical evaluator.
