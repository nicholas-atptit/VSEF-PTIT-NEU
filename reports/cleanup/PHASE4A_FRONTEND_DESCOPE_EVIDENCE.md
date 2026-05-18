# Phase 4A Frontend De-Scope Evidence

Date: 2026-05-10
Scope: remove repository web UI surfaces from the governed runtime.

## Summary

All repository web UI surfaces were removed from the governed runtime:

- standalone Vite frontend under `src/api/ui/web/`
- root-level served dashboard under `web/`
- FastAPI static `/web` mount
- FastAPI `/dashboard` redirect to deleted files

Backend API business logic, ML/model/training logic, allocator logic, VN100 logic,
feature catalogue logic, repository hygiene policy, and API governance semantics
were not redesigned in this phase.

## Required Pre-Edit Search

Command:

```powershell
rg -n "src/api/ui/web|api/ui/web|/web|/dashboard|StaticFiles|web-dashboard|frontend|dashboard|vite|npm run|package.json|package-lock|index.html" .
```

Result summary:

- Found FastAPI static serving in `src/api/main.py`.
- Found tracked Vite files under `src/api/ui/web/`.
- Found active root-level served dashboard files under `web/`.
- Found comments/docstrings referring to dashboard/frontend consumers in
  `src/api/routes.py`, `src/api/routes_v2.py`, and `scripts/per_session_predict.py`.
- Found historical/report references in `reports/` and archived docs.

## Removed Frontend Paths

Tracked paths removed:

```text
src/api/ui/web/index.html
src/api/ui/web/package-lock.json
src/api/ui/web/package.json
src/api/ui/web/src/App.jsx
src/api/ui/web/src/index.css
src/api/ui/web/src/main.jsx
src/api/ui/web/vite.config.js
web/app.js
web/index.html
web/style.css
```

Untracked local frontend build/dependency paths removed if present:

```text
src/api/ui/web/node_modules
src/api/ui/web/dist
web/dist
```

Post-removal path checks:

```text
Test-Path src/api/ui/web              -> False
Test-Path web                         -> False
Test-Path src/api/ui/web/node_modules -> False
Test-Path src/api/ui/web/dist         -> False
Test-Path web/dist                    -> False
```

## Backend Cleanup

Updated `src/api/main.py`:

- removed `StaticFiles` import
- removed `RedirectResponse` import
- removed `Path` import
- removed `_web_dir` resolution
- removed `app.mount("/web", ..., name="web-dashboard")`
- changed `/dashboard` from a redirect to `/web/index.html` into a JSON
  response reporting that the web dashboard has been removed

Current `/dashboard` behavior:

```json
{
  "status": "removed",
  "detail": "The web dashboard is no longer part of the governed runtime.",
  "docs": "/docs"
}
```

Confirmation:

- `/web` is no longer mounted.
- `/dashboard` no longer redirects to deleted files.
- root metadata does not advertise `/web` or `/dashboard`.

## Comments and Docs

Updated active code comments/docstrings:

- `src/api/routes.py`: changed stock-history charting docstring from frontend
  wording to API-client wording.
- `src/api/routes_v2.py`: changed price endpoint comment/docstring from web
  dashboard/frontend polling to API-client polling.
- `scripts/per_session_predict.py`: changed serialized payload comment so it no
  longer refers to a dashboard root.

Deferred historical references:

- Archived docs and prior audit reports still contain historical dashboard/UI
  references. They were not rewritten because they are evidence snapshots, not
  active runtime instructions.
- Generic governance mentions of dashboards remain where they do not claim an
  active repository web UI.

## Tests Added

Updated `tests/test_api.py`:

- root endpoint test now asserts `/web` and `/dashboard` are not advertised
- added coverage that `/web/index.html` returns 404
- added coverage that `/dashboard` returns the removal JSON and does not point
  at `/web/index.html`

## Verification

Source-level static-serving check:

```powershell
rg -n 'StaticFiles|web-dashboard|app\.mount\("/web|RedirectResponse|_web_dir' src/api/main.py
```

Result: exit code 1; no matches. `src/api/main.py` no longer imports or uses the
FastAPI static web serving hooks.

| Command | Exit | Result |
| --- | --- | --- |
| `python -m pytest tests/test_api.py -q` | 0 | `19 passed, 3 warnings in 12.34s` |
| `python -m pytest tests -q` | 0 | `808 passed, 5 skipped, 33 warnings in 265.54s` |
| `python scripts/check_repo_hygiene.py` | 0 | `Repository hygiene check passed.` |

Dedicated commit message:

```text
chore: remove frontend web ui surfaces
```

Final `git status --short` after the dedicated removal commit:

```text
?? reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

## Unresolved Impact

- Direct external bookmarks to `/web/index.html` now receive 404.
- Direct external callers of `/dashboard` now receive JSON removal metadata
  instead of a browser UI redirect.
- Historical reports still mention removed frontend paths for audit
  traceability.

## Protected File Check

`reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` remains untracked and was not
modified. It appeared only as an untracked path in status and broad text-search
output.
