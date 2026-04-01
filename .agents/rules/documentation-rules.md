---
trigger: always_on
---

After every implementation task, automatically generate a markdown report under docs/prompt_runs/.

Required filename format:
- <sequence_number>_<short_task_name>.md

Required sections:
- Prompt Intent
- Actual Outcome
- Files Created
- Files Modified
- Key Code Changes
- Functions / Classes / Scripts Added or Updated
- Algorithms / Methods / Logic Introduced
- Config / CLI / Environment Changes
- Storage / Persistence Impact
- Compatibility Notes
- Tests / Validation
- Remaining Gaps

Strict requirements:
1. Do not invent features, algorithms, or changes.
2. Only document what is actually present in code, config, tests, docs, or git diff.
3. If the mapping between prompt and change is uncertain, explicitly state that.
4. If the task contains only infrastructure or wiring, write:
   - "No new predictive algorithm introduced in this task."
5. Update docs/prompt_runs/INDEX.md after every report.