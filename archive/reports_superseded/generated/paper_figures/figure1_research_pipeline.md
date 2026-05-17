# Figure 1: Research Pipeline

```text
VN100 official cache artifacts
  -> cache usability and date/schema verification
  -> train-label cutoff: target_timestamp <= 2024-12-31
  -> walk-forward out-of-sample prediction on 2025 labels
  -> model and baseline accuracy summaries
  -> confidence-filter and regime diagnostics
  -> paper tables, figures, and claim register
```

Source artifact: `run_config.json`, `manifest.json`, and official daily/hourly summary CSV files.

Claim supported: the paper pipeline separates cache validation, train-cutoff enforcement, held-out evaluation, and diagnostic reporting.

Limitation: this figure is a methodology schematic, not a new empirical result.

Status: ready.
