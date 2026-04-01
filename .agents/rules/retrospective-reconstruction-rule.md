---
trigger: always_on
---

If multiple implementation tasks have already been completed before documentation was generated, create best-effort reconstruction reports under docs/prompt_runs/ based on:
- current codebase
- git history/diff if available
- commit messages if available
- tests/docs/scripts/config

In retrospective mode:
1. Never claim exact prompt-to-change mapping unless it can be verified.
2. Explicitly label uncertain reports as:
   - Best-effort reconstruction
3. Also maintain:
   - docs/prompt_runs/INDEX.md
   - docs/prompt_runs/CHANGELOG_SUMMARY.md
