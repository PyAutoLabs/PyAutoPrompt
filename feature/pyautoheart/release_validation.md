# Heart-owned deep release-validation ("full test mode") + tracked report + readiness gate

Type: feature
Target: PyAutoHeart (+ PyAutoBrain Release Agent orchestrates / Health Agent judges, PyAutoBuild dispatch target)
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
thing should be drivable from a phone via Brain (the Release Agent orchestrates
the run; the Health Agent reports the verdict).

This is a new **third Heart tier**: not the cheap `<30s` monitoring `tick`, and
not the existing on-demand deep checks (`verify_install`, `url_sweep`) — a
**release-grade validation** whose spec, ingested report, and verdict are
**owned by Heart**, while the actual building (Build's `release.yml`) and the
heavy integration run (`workspace-validation.yml`) execute on CI and are
**dispatched by the Brain Release Agent**, not by Heart.

## Boundary: Heart never mutates a repo and never triggers a build

This is non-negotiable and matches Heart's existing rule ("the daemon must be a
pure observer; mutations belong in `pyauto-heart fix`, which only EMITS context
for a fresh session"). Applied to this feature:

- **Heart MAY** run/read tests (it already executes throwaway `verify_install`
  venvs) and READ CI conclusions (via `gh`/API). It writes ONLY to
  `~/.pyauto-heart/`.
- **Heart MUST NOT** edit/commit/push any repo, and **MUST NOT** dispatch
  `release.yml`, `workspace-validation.yml`, or any other build/CI workflow.
- **All dispatching** (rehearsal build, validation run, await, artifact
  collection) is the **Brain Release Agent's** job.
- **All mutation** (code/config fixes, the eventual PyPI promotion) is a **dev
  agent** or **Hands/Build**, never Heart.
- Heart and Build therefore **never call each other** — the Release Agent sits
  between them. `pyauto-heart validate` is **ingest-and-judge only**: it consumes
  already-collected artifacts/conclusions and emits the verdict.

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

3. **The `release` env profile is broader than the 4 smoke vars — but it is an
   env-var profile, NOT a Heart-owned config-mutation.** The full library
   env-var surface is 13 `PYAUTO_*` vars (canonical entry:
   `PyAutoConf/autoconf/test_mode.py`), not the 4 in the smoke defaults:
   `PYAUTO_TEST_MODE, PYAUTO_SMALL_DATASETS, PYAUTO_FAST_PLOTS, PYAUTO_OUTPUT_MODE,
   PYAUTO_DISABLE_JAX, PYAUTO_SKIP_FIT_OUTPUT, PYAUTO_SKIP_VISUALIZATION,
   PYAUTO_SKIP_CHECKS, PYAUTO_SKIP_LATENTS, PYAUTO_SKIP_WORKSPACE_VERSION_CHECK,
   PYAUTO_LATENT_NAN_INJECT, PYAUTO_DISABLE_IPYTHON_DISPLAY, PYAUTO_LIVE_VIEWER_LOG`.
   Plus a few per-script switches outside any yaml default (`PYAUTO_MASS_MODE` /
   `PYAUTO_MASS_FAST`, `JAX_PILOT` / `JAX_PLATFORM_NAME` / `JAX_PLATFORMS`).

   **Explicitly NOT in scope for Heart:** `config/general.yaml`'s `test:` block
   (`check_likelihood_function`, `lh_timeout_seconds`,
   `disable_positions_lh_inversion_check`) and `version:` toggles
   (`python_version_check`, …) are **workspace-run/user settings, not Heart
   checks** — the release validation runs the scripts *as the workspace ships
   them* and does not mutate these. Heart's only version signal is the existing
   `version_skew` check. Do not have Heart set or assert the `test:`/`version:`
   config toggles.

4. **Wheel-based config-resolution footgun (this one IS validation-relevant).**
   autoconf resolves the *workspace's* `config/` only when scripts run from
   inside the workspace checkout; a bare wheel falls back to the library's
   *packaged* defaults (`autolens/config/`). So the rehearsal must `pip install`
   the PyAuto wheels (Gap A) but still execute scripts **from within the
   workspace checkout** (for `config/` + `dataset/`), with NO source on
   `PYTHONPATH`. Otherwise the workspace's own settings silently revert to
   library defaults and the validation exercises something other than the shipped
   configuration.

## Design

### 1. The validation pipeline — four stages (Release Agent orchestrates, Heart judges)

The pipeline has four stages. **The Brain Release Agent dispatches and awaits
each stage; Heart defines the stage spec and ingests each result.** Heart's
`pyauto-heart validate --ingest` consumes the collected artifacts/conclusions and
emits the verdict — it never dispatches (see Boundary above).

- **Stage 0 — preflight.** Heart's existing `repo_state` + `version_skew` signals:
  all 5 libraries clean, on `main`, not behind; workspace version pins
  consistent. The Release Agent reads these and aborts early (RED) if not — no
  point building a dirty tree.
- **Stage 1 — unit.** Each library's `pytest` green. Read the library CI
  conclusion (Stage-0 signal); the Release Agent may dispatch + await a fresh run
  if stale.
- **Stage 2 — rehearse (build on TestPyPI).** The Release Agent dispatches Build's
  `release.yml` in the new **TestPyPI-only / rehearsal mode** (M1): build current
  source → publish to TestPyPI → STOP (no PyPI, no tag, no notebook commits). The
  resolved TestPyPI version is captured and passed to Stage 3.
- **Stage 3 — integrate on wheels at release fidelity.** The Release Agent
  dispatches the evolved `workspace-validation.yml`: `pip install` the Stage-2
  TestPyPI wheels (NO source on PYTHONPATH — Gap A), run all `workspace` +
  `workspace_test` integration scripts under the **`release` profile** (Gap B) +
  `verify_install` A–E against the same wheels.

After the stages complete, the Release Agent hands every artifact/conclusion to
`pyauto-heart validate --ingest`, which writes the report and computes the gate.

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

### 4. The Brain agents drive it (mobile, no `gh`) — orchestration vs. judgement

Cloud/mobile sessions have NO `gh` CLI, so GitHub dispatch + poll runs through
Brain's MCP GitHub tools. **But mind the agent boundary** (verified against
`PyAutoBrain/agents/health/AGENTS.md`): the Health Agent is a strict
*read-and-reason* role — *"Never write into any repo, run a build, or trigger a
release."* So dispatching workflows is **NOT** the Health Agent's job.

Split it across the two existing Brain agents:

- **Release Agent** (`agents/release/`) **orchestrates**: dispatches the Build
  TestPyPI rehearsal (M1) and the validation workflow, polls for completion,
  downloads the report artifacts, hands them to `pyauto-heart validate --ingest`,
  then **consults the Health Agent** for the gate via the existing
  `consult_health_agent_verdict` pattern in `agents/_common.sh`.
- **Health Agent** stays read-only: reasons over Heart's verdict, returns
  GREEN/YELLOW/RED. Unchanged.
- **Hands/Build** promotes to PyPI only on GREEN (an explicit human/Release-Agent
  action — never auto-promoted).

So the chain is: `Mind → Release Agent (orchestrate) → Heart (measure) ↘ Health
Agent (judge) → Hands/Build (promote on GREEN)`. From a phone you run the Release
Agent to kick the rehearsal+validation; it consults the Health Agent for the
verdict. Heart stays credential-free (spec + ingest + verdict only).

### 5. Register the new capability in Heart's manifest

The Brain agents are manifest-driven — they read Heart's
`health_agent/capabilities.yaml` (NOT hardcoded check names). So the new
`validate` capability and the `validation_report` signal MUST be added to that
manifest, or the agents won't surface them. Heart's
`HEART_CAPABILITIES.md` cross-reference and the Health Agent's reasoning then
pick them up with no Brain edits.

## Scope of THIS prompt (M2 foundation)

- `validation_report.json` schema + `pyauto-heart validate --ingest <artifacts>`.
- readiness hard-gate consuming it (stale-by-SHA detection).
- The Brain **Release Agent** dispatch/poll/ingest driver (MCP-based); the
  Health Agent (read-only) reports the resulting verdict.
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
