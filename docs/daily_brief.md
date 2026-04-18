# Daily Brief Reporting Guide

The Daily Brief module provides a high-level summary of the latest VN100 batch inference results, optimized for quick decision-making and risk monitoring.

## Purpose
- Consolidate predictions across the VN100 universe.
- Identify the most bullish and highest return prospects.
- Highlight high-risk/high-volatility names.
- Report on system data quality and processing statistics.

## Module Structure
- **Module**: `src/reporting/daily_brief.py`
- **Output Directory**: `reports/`
- **Formats**: Markdown (.md), CSV (.csv), HTML (.html)

## How to Run

### Automatic (Latest Batch)
To generate a report for the most recent batch inference run:
```bash
python -m src.reporting.daily_brief --latest
```

### Specific File
To generate a report for a specific JSON result:
```bash
python -m src.reporting.daily_brief --file data/processed/batch_inference_20260402_030000.json
```

## Report Sections

### 1. Executive Summary
- **Universe Size**: Total tickers targeted.
- **Success Rate**: Percentage of tickers successfully predicted.
- **Model Mode**: Shows the trainer and thresholds used.

### 2. Top Picks
- **Bullish**: Ranked by `prob_up` (Probability of upward trend).
- **Expected Return**: Calculated as `(median_target / current_price) - 1`.
- **Action**: The recommended fusion action (BUY, HOLD, SELL, WAIT).

### 3. High Risk / Volatility
- **Volatility Score**: Predicted annualized volatility.
- **Heuristic Scenario Risk**: Residual-based scenario VaR/CVaR summary around the point forecast. This is not calibrated confidence.

### 4. Data Quality Warnings
- Alerts for missing tickers or predictions with unusually weak scenario-risk or data-quality context.
