# NSS optional dependency workspace release blocker

## Original user request

continue

## Context

During release hygiene for the PyAutoBuild release run, `autofit_workspace/scripts/searches/nest.py` failed because it constructs `af.NSS()` when the optional `nss` dependency stack is not installed.

The related PyAutoFit unit-test blocker was already fixed by skipping NSS-only tests when the optional dependency is absent. The workspace script should be made release-safe in the same spirit: runnable in the standard release environment while still preserving the NSS example for users who install the optional extra.

## Failure

Full PyAutoBuild run:

- Run: `2026-06-08T16-10-15Z`
- Report: `PyAutoBuild/test_results/runs/2026-06-08T16-10-15Z/report.md`
- Failure: `autofit_workspace/scripts/searches/nest.py`
- Error tail: optional `nss` dependency stack is unavailable; PyAutoFit points users to the `pyproject.toml` `[project.optional-dependencies] nss` pins.

## Goal

Update the `autofit_workspace` NSS example so the workspace release checks pass without requiring the optional `nss` extra, while retaining useful documentation and runnable behavior for environments where the extra is installed.
