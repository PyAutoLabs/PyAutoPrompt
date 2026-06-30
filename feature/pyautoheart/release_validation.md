# Heart-owned deep release-validation ("full test mode") + tracked report + readiness gate

Type: feature
Target: PyAutoHeart (+ PyAutoBrain health agent, PyAutoBuild dispatch target)
Repos:
- PyAutoHeart
- PyAutoBrain
- PyAutoBuild
- autolens_workspace_test / autogalaxy_workspace_test / autofit_workspace_test (env profiles)
- autolens_workspace / autogalaxy_workspace / autofit_workspace (env profiles)
Status: planned
Milestone: M2 (this prompt) — depends on M1 (`feature/pyautobuild/release_yml_testpypi_rehearsal_mode.md`)

## Why

Today `pyauto-heart readiness` goes GREEN on *observed* signals: each library's
CI conclusion, the workspace-validation workflow conclusion, version skew, and a
(deep, on-demand) `verify_install`. It never forces a fresh, end-to-end
**rehearsal of the exact source about to ship**, built and installed as wheels,
exercised by the full integration surface, before green-lighting a release.

That is the gap this fills. The organism should be able to answer "is it safe to
release?" with: *"the current source was built, published to TestPyPI, installed
from the wheel, and passed unit tests + the full workspace and workspace-test
integration surface at release fidelity — here is the report."* And that whole
thing should be drivable from a phone via the Brain health agent.

This is a new **third Heart tier**: not the cheap `<30s` monitoring `tick`, and
not the existing on-demand deep checks (`verify_install`, `url_sweep`) — a
**release-grade validation** that *composes* the existing executors (Build's
`release.yml`, the Heart-owned `workspace-validation.yml`, each library's CI) and
**owns the resulting report and the gate**. Heart owns the spec, the report, and
the verdict; Build still executes the build/publish (Heart dispatches it). This
is consistent with the intended `Brain → Heart → Build` call chain — the
"observer never triggers Build" rule applies to the passive monitoring daemon,
not to a deliberate `pyauto-heart validate` invocation.

## Assessment findings this must address (grounded in current code)

Read before designing — these are real, verified gaps:

1. **The current validation tests SOURCE, not BUILDS.**
   `PyAutoHeart/.github/workflows/workspace-validation.yml` installs
   `autolens[optional]` only for transitive deps, then **shadows the PyAuto
   packages with source checkouts via `PYTHONPATH`** (~line 175). So the gating
   run never touches a wheel. This is exactly the blind spot that let the
   PyAutoFit `[nss]` git-URL silently break every TestPyPI upload for weeks, and
   the nufftax/JAX dependency-floor bug ship (both in `complete.md`). The
   rehearsal MUST `pip install` the TestPyPI wheels and NOT put source on
   `PYTHONPATH`.

2. **The current validation runs in SMOKE mode, not at release fidelity.**
   Both tiers' `config/build/env_vars.yaml` default to `PYAUTO_TEST_MODE=2`
   (skip sampler) + `PYAUTO_SMALL_DATASETS=1` (15×15 toy grids) +
   `PYAUTO_DISABLE_JAX=1` + `PYAUTO_FAST_PLOTS=1`. Correct for a per-PR
   structural smoke; wrong for a release gate. The intended release
   infrastructure is already documented in `PyAutoBuild/.github/workflows/
   release.yml:693-698`:

   | Tier | PYAUTO_TEST_MODE | PYAUTO_SMALL_DATASETS | PYAUTO_FAST_PLOTS |
   |------|------------------|------------------------|-------------------|
   | user workspaces | `1` (reduced) | `1` (capped) | `1` |
   | `*_workspace_test` | `0` (real searches, `n_like_max` caps) | unset (full-res) | unset |

   Formalise this as a named **`release` profile** distinct from the **`smoke`
   profile**. The `env_vars.yaml` header already *claims* to serve "pre-release
   checks" but only encodes smoke values — fix that conflation. Per-script
   `overrides:` (unset `SMALL_DATASETS` for full-res FITS scripts, keep JAX on
   for `jax_likelihood_functions/` and `jax_substructure/`, etc.) still layer on
   top of the selected profile.

   Both gaps are implemented in **M3** (wheel-based integration at release
   fidelity); capture them here as acceptance criteria so they are not lost.

## Design

### 1. `pyauto-heart validate` — the deep validation pipeline

A new on-demand subcommand, NEVER in the `tick`, with four stages:

- **Stage 0 — preflight.** Reuse `repo_state` + `version_skew`: all 5 libraries
  clean, on `main`, not behind; workspace version pins consistent. Abort early
  (RED) if not — no point building a dirty tree.
- **Stage 1 — unit.** Each library's `pytest` green. Observe the latest library
  CI conclusion; optionally dispatch + await a fresh run.
- **Stage 2 — rehearse (build on TestPyPI).** Dispatch Build's `release.yml` in
  the new **TestPyPI-only / rehearsal mode** (M1): build current source →
  publish to TestPyPI → STOP (no PyPI, no tag, no notebook commits). Capture the
  resolved TestPyPI version.
- **Stage 3 — integrate on wheels at release fidelity.** Dispatch the evolved
  `workspace-validation.yml`: `pip install` the Stage-2 TestPyPI wheels (NO
  source on PYTHONPATH — Gap A), run all `workspace` + `workspace_test`
  integration scripts under the **`release` profile** (Gap B) + `verify_install`
  A–E against the same wheels.

### 2. `validation_report.json` — the tracked artifact (the new bit)

Heart ingests the per-workflow `report.json` artifacts into a single
`validation_report.json` stored in Heart state and **committed** so health
history is tracked in Heart. Schema (at least):

- `release_ready` (top-level boolean)
- `testpypi_version` (the rehearsed version)
- `commit_shas` — per repo, so readiness can confirm the report is for *this*
  source (a report for an older SHA is stale, not green)
- `profile` — which env profile each tier ran under (must read `release`)
- per-stage status + per-project pass/fail/skip/timeout counts
- failure logs / GitHub run URLs
- timestamp

### 3. readiness gains a hard gate

GREEN-for-release now requires a fresh, passing `validation_report` whose
`commit_shas` match the current `main` HEADs and whose `profile == release`.
Absent/stale/source-not-matching → YELLOW ("no release rehearsal for current
source"); failing → RED. The existing soft signals are unchanged.

### 4. Brain health agent drives it (mobile, no `gh`)

Cloud/mobile sessions have NO `gh` CLI. So the GitHub dispatch + poll is done by
**Brain's health agent via its MCP GitHub tools**; Heart stays credential-free
(defines the spec, ingests the downloaded report artifacts, computes the
verdict). `agents/health/health.sh` (or a sibling `validate` entry point) gains
a mode that: dispatches the workflows, polls for completion, downloads the
report artifacts, hands them to `pyauto-heart validate --ingest`, and prints the
RED/YELLOW/GREEN verdict with per-stage reasons. This makes "run the full
release validation from my phone via Brain" the native path.

## Scope of THIS prompt (M2 foundation)

- `validation_report.json` schema + `pyauto-heart validate --ingest <artifacts>`.
- readiness hard-gate consuming it (stale-by-SHA detection).
- The Brain health-agent dispatch/poll/ingest driver (MCP-based).
- Define (don't yet fully wire) the `release` env profile and the wheel-install
  requirement as acceptance criteria for M3.

Out of scope here (later milestones): M1 = the TestPyPI-only mode in `release.yml`
(separate prompt, prerequisite); M3 = actually re-pointing `workspace-validation.yml`
at TestPyPI wheels + release profile; M4 = the full `validate` orchestrator
sequencing Stages 0–3; M5 = polish of the mobile UX.

## Validation

- `pytest tests/` in PyAutoHeart stays green (add tests for ingest + the new
  readiness gate, incl. the stale-by-SHA path → YELLOW, fresh-pass → GREEN,
  fail → RED).
- Run the Brain health agent end-to-end against a hand-placed sample
  `validation_report.json` and confirm the verdict flips correctly.

## PR

One PR per milestone. This one: "PyAutoHeart: release-validation report + readiness gate".
