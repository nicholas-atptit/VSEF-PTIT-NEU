# Prompt Run: daily-brief

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/reporting/daily_brief.py`
  - `docs/daily_brief.md`

## Objective
- Implement an automated daily reporting module for the VN100 stock prediction system.
- Produce Markdown and CSV reports with "Executive Summary", "Top Picks", and "High Risk" sections.

## Files Created
- `src/reporting/daily_brief.py`
- `docs/daily_brief.md`

## Files Modified
- None (New module)

## Key Changes
- **Targeted Insights**: Generates "Top Bullish", "High Expected Return", and "High Volatility" lists.
- **Action Recommendations**: Maps predictions to specific actions (BUY, HOLD, SELL, WAIT).
- **Data Quality Alerts**: Identifies low-confidence predictions or missing tickers.
- **Reporting Formats**: Standardized Markdown and CSV outputs in `reports/`.

## Implementation Details
- **`daily_brief.py`**:
  - `--latest` flag: Automatically finds the most recent batch inference JSON.
  - Consolidated view: Merges classification and regression outputs into a single board.
- **Metrics**: Displays `prob_up`, `median_return`, and `volatility_score`.

## Algorithms / Methods / Rules Applied
- Ranking by `prob_up` and expected return.
- Threshold-based action mapping (e.g., BUY if `prob_up` > 0.75).

## Data Flow Impact
- Input: `data/processed/batch_inference_*.json`.
- Output: `reports/daily_brief_<date>.md` and `.csv`.

## Backward Compatibility
- ✅ Independent module; does not affect existing pipelines.

## Risks / Limitations
- Relies on the completion of the batch inference run.
- Action recommendations are rule-based and not backtested yet.

## Verification
- `python -m src.reporting.daily_brief --latest`
- Reviewing `docs/daily_brief.md`.

## Open TODOs
- PDF export support.
- Email/Telegram notification integration.
