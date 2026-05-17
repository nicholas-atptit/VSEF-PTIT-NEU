# Figure 2: Walk-Forward Validation Design

```text
Raw daily request:  2006-01-01 -> 2015-12-31
Raw hourly request: 2016-01-01 -> 2025-12-31

Training labels allowed through: 2024-12-31
Cutoff rule: target_timestamp <= train_cutoff

Official evaluation window: 2025-01-01 -> 2025-12-31
Effective daily evaluation: 2025-01-02 -> 2025-12-31
Effective hourly evaluation: 2025-01-02 -> 2025-12-31
```

Source artifact: `run_config.json` and `manifest.json`.

Claim supported: 2025 outcomes are evaluated out of sample after a 2024-12-31 training-label cutoff.

Limitation: the current evidence covers one official 2025 window.

Status: ready.
