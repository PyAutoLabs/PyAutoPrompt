# Add a TestPyPI-only "rehearsal" mode to release.yml

Type: feature
Target: PyAutoBuild
Repos:
- PyAutoBuild
Status: planned
Milestone: M1 — prerequisite for `feature/pyautoheart/release_validation.md` (M2)

## Why

The Heart-owned release-validation pipeline (see
`feature/pyautoheart/release_validation.md`) needs to test the organism against
**built wheels**, not source checkouts — that is the only thing that catches
packaging-layer bugs (a bad `MANIFEST`, a missing data file, a too-loose or
direct-URL dependency). Two real incidents in `complete.md` motivate this: the
PyAutoFit `[nss]` `git+` direct URL silently broke every TestPyPI upload for
weeks, and the nufftax/JAX dependency-floor mismatch produced broken installs —
neither was caught by the source-based validation.

`release.yml` already knows how to build and publish to TestPyPI (its
`release_test_pypi` job), but that capability is **coupled to the full
release** (TestPyPI → PyPI → tag → notebook commits in one flow). We need to be
able to build + publish the current source to TestPyPI **and stop**, so Heart
can install and validate those wheels before any PyPI promotion.

## Task

Add a `rehearsal` (TestPyPI-only) execution mode to `release.yml`:

- A new `workflow_dispatch` input (e.g. `rehearsal: true`, or reuse/extend the
  existing skip flags) that:
  - builds every package from current source and **publishes to TestPyPI**,
  - then **STOPS** — no PyPI upload, no git tag, no `tag_and_merge`, no notebook
    generation/commit to workspaces, no Colab-URL bumps.
- Emit the resolved TestPyPI version string as a workflow output / artifact so
  the caller (Heart / Brain health agent) can install exactly those wheels.
- Keep the existing full-release path untouched and default.

This is intentionally small and isolated — it is the highest-value, lowest-risk
piece and unblocks the rest of the redesign. It does not change the full release
flow; it just exposes "build + TestPyPI, then halt" as a first-class mode.

## Notes / footguns

- Respect the "pure executor" boundary: `release.yml` runs no readiness checks;
  it just builds/publishes. The gate lives in Heart.
- Verify the TestPyPI upload step tolerates re-runs of the same version
  (TestPyPI rejects duplicate filenames) — the rehearsal will be dispatched
  repeatedly. Use a dev/local version suffix or `--skip-existing` semantics as
  appropriate.
- Confirm no `git+` direct URLs leak into any uploaded wheel's metadata
  (the original `[nss]` failure) — a rehearsal that can't upload is the bug
  this whole effort exists to surface early.

## Validation

- Dispatch the rehearsal mode manually; confirm wheels appear on TestPyPI, the
  version string is emitted, and the workflow halts before PyPI/tag/notebook
  steps.
- `pytest` in PyAutoBuild stays green.

## PR

"PyAutoBuild: TestPyPI-only rehearsal mode in release.yml".
