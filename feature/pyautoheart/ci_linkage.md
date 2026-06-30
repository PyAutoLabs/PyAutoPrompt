# Heart ↔ CI linkage: read release-grade CI from the Actions server, gate the right repos

Type: feature
Target: PyAutoHeart
Repos:
- PyAutoHeart
Status: planned
Milestone: M0 — foundational; the release-validation gate
(`feature/pyautoheart/release_validation.md`) builds on a trustworthy CI signal.

## Why

The final design review of the Brain → Health → Heart chain found the CI signal
Heart consumes is too coarse and too narrow to gate a release on, and its repo
registry is stale. These are linkage bugs: Heart can read "CI", but not reliably
the *right* CI. Fix the link before layering the deep release-validation gate on
top of it.

## Findings to fix (all verified in current code)

1. **CI granularity is wrong for a gate.** `heart/checks/ci_status.sh` runs
   `gh run list --repo <repo> --limit 1` — the single most recent run, *any
   workflow, any branch*. But each workspace has THREE gating workflows
   (`smoke_tests.yml`, `navigator_check.yml`, `url_check.yml`) on **two Pythons
   (3.12, 3.13)**, and libraries gate on `pytest`. So "latest run" can report a
   green `url_check` while `smoke_tests` was red, or a run on a feature branch.
   For release readiness Heart must read the **conclusion of each *required*
   workflow on the `main` HEAD commit**, not "the newest run."

2. **readiness coverage is library-only.** `heart/readiness.py`'s hard gate loops
   `DEFAULT_LIBRARIES` (the 5 libs) for CI/branch/dirty/behind. Workspace CI
   conclusions are *observed* (`ci_status` writes per-repo sidecars) but **never
   folded into the verdict** — only the aggregate `test_run` + `version_skew`
   represent workspaces. Decision to make and implement: for release readiness,
   **gate the workspaces' `smoke_tests` + `navigator_check` conclusions on main**
   (RED on failure), or document explicitly why they stay advisory. (Recommended:
   gate them — a red workspace smoke on main is a real release blocker.)

3. **Read via the CI server, not via local reports.** The canonical continuous
   "did it pass" signal should be the **GitHub Actions run conclusion** queried
   from the Actions API — which is reachable from mobile via Brain's MCP GitHub
   tools — with `report.json` kept as *detail enrichment only*, never a hard
   dependency. This directly fixes the mobile fragility where a missing local
   `report.json` makes `test_run` report "unknown → YELLOW" even though the cloud
   `workspace-validation` run is green and queryable. (`test_run.py` already has a
   `_cloud_verdict()` that reads the `workspace-validation.yml` conclusion via
   `gh`; generalise that pattern: prefer the server conclusion, enrich with the
   report when present, and make the server query work through the agent's MCP
   path when `gh` is absent.)

4. **`config/repos.yaml` is stale — broken linkage.** It still lists
   `PyAutoPrompt` (renamed → PyAutoMind) under `build_workflow`, excludes
   `PyAutoPaper` (renamed → PyAutoMemory), and does not poll the organism repos
   `PyAutoBrain` / `PyAutoHeart` / `PyAutoMemory`. Update the registry to the
   current names and add the organism repos so Heart watches what actually
   exists. (Heart watching itself is fine and useful.)

## Scope

- Rework `ci_status` to record per-required-workflow conclusions on the `main`
  HEAD per repo (keep the cheap one-line summary; the readiness consumer reads
  the structured per-workflow detail).
- Extend `readiness.py` to gate workspace CI per the decision in finding 2.
- Make the run-conclusion the primary `test_run` signal (server-first, report as
  enrichment), with a path that works without `gh` (agent-supplied via MCP).
- Fix `config/repos.yaml` names + add organism repos.
- Keep the `<30s` tick budget: the tick still reads conclusions cheaply (one
  `gh`/API call per repo); the heavier per-workflow detail is fine because it is
  still just metadata reads, no execution.

## Validation

- `pytest tests/` green; add cases for: per-workflow gating (a red `smoke_tests`
  with a green `url_check` → RED, not green), workspace-CI gating, and the
  server-first `test_run` resolution (report absent but server green → not
  "unknown").
- Run `pyauto-heart tick` + `readiness` and confirm the verdict reflects the
  per-workflow, per-repo reality.

## PR

"PyAutoHeart: release-grade CI linkage (per-workflow gating, workspace coverage,
server-first signal, registry refresh)".
