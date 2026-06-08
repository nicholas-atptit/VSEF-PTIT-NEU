# Research Workflow

1. Write a protocol with universe, local data scope, frequency, horizon,
   timestamp split, metrics, baselines, and claim boundary.
2. Validate provider policy and local cache coverage without fetching data.
3. Build point-in-time-safe features. Use trailing or lagged inputs only.
4. Assign strict splits using both feature and target timestamps.
5. Fit on train rows and select candidates using validation rows only.
6. Score locked candidates on final rows without selection or relocking.
7. Compare against task-appropriate baselines.
8. Write artifacts, result summary, claim boundary, and evidence-index updates.

Do not translate diagnostic metrics into trading, profitability, BUY/SELL,
recommendation, live-deployment, production, or daily T+1 claims.

Large historical runners remain one-off orchestrators. New reusable logic belongs
under `src/`, with focused tests, before a runner consumes it.
