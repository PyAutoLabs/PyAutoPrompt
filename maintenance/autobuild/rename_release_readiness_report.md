Rename PyAutoBuild's "Release Readiness Report" to remove readiness ambiguity.

## Context

The Health Agent boundary audit (see PyAutoHeart `health_agent/
pyautobuild_boundary_audit.md`) confirmed PyAutoBuild contains **no** drifted
health/readiness gating logic — every gate lives in PyAutoHeart, and
`pyauto-heart readiness` is the single authoritative verdict.

The one residual is **naming**. `autobuild/aggregate_results.py` builds a report
titled "Release Readiness Report" with a top-level `ready` boolean
(`ready = not has_failures`), and `create_analysis_issue.py` posts it as a
"Release Readiness Report — <date>" issue. This is really a **workspace
script-run aggregation** (an executor primitive that Heart's `workspace-
validation.yml` reuses and whose `report.json` Heart's `test_run` check
consumes), NOT a release gate. The shared "readiness" vocabulary risks a future
reader mistaking Build's report for Heart's authoritative verdict.

## Task (low risk, optional, naming-only)

In PyAutoBuild, rename the report's user-facing strings to something like
"Workspace Validation Report" and the field from `ready` to e.g.
`scripts_passed` / `validation_passed`:

- `autobuild/aggregate_results.py` — `generate_markdown` title + status string,
  and the report `ready` key (keep a back-compat alias if any consumer reads it).
- `autobuild/create_analysis_issue.py` — issue title + description.

Then update Heart's `test_run.py` if it keys off the field name, and the
`report.json` contract notes. This is a documentation/naming change only — do not
move or add any gating logic into Build. Confirm `pytest` passes in both repos.
