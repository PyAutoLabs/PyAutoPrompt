Follow-up to the autolens_assistant API-drift task (issue: stale `al.Kernel2D` /
version-pin baseline + drift-check). That task adds an `api_audit_baseline.json` to
autolens_assistant recording the PyAuto* versions + a hash of the public API surface, and
a startup drift-check that warns when the installed autolens differs from the pinned
baseline.

Right now that baseline is regenerated manually (by running `work/audit_skill_apis.py`).
We want it refreshed automatically as part of the release process, so the assistant's
version pin always tracks the latest published autolens with no human in the loop.

Build this into **@PyAutoBuild**'s release pipeline:

- @PyAutoBuild already verifies workspace versions during a build (see
  `@PyAutoBuild/verify_workspace_versions.sh` and `@PyAutoBuild/pre_build.sh` /
  `@PyAutoBuild/autobuild/`). Add a step that, on a new autolens release, regenerates the
  assistant's API baseline (PyAuto* versions + public-API-surface hash) and commits/opens
  a PR against `@autolens_assistant` updating `wiki/core/api_audit_baseline.json`.
- Decide the mechanism: either PyAutoBuild invokes the assistant's
  `work/audit_skill_apis.py --write-baseline` against the freshly released stack, or
  PyAutoBuild computes the same hash itself (shared helper) so the two never diverge.
- The drift-check on the assistant side then becomes "are you on the released version the
  assistant was last built for" — and a release automatically advances that pin.

Open questions to resolve when picking this up:
- Where the baseline regeneration runs (the release CI job vs a post-release dispatch to
  autolens_assistant).
- Whether the API-surface hash helper should live in PyAutoBuild and be imported by the
  assistant's audit script, to guarantee both sides hash identically.
- How this interacts with the existing `verify_workspace_versions.sh` workspace-version
  check (same release hook, related concern).

Do not start until the parent autolens_assistant baseline/drift-check task has shipped —
this depends on the `api_audit_baseline.json` format it defines.
