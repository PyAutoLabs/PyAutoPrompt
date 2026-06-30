# Release env profile + wheel-based integration for the validation run

Type: feature
Target: PyAutoHeart (validation workflow) + the six workspaces (env profile)
Repos:
- PyAutoHeart
- autolens_workspace_test / autogalaxy_workspace_test / autofit_workspace_test
- autolens_workspace / autogalaxy_workspace / autofit_workspace
- PyAutoBuild (run_python.py profile selection)
Status: planned
Milestone: M3 — closes Gaps A & B from `feature/pyautoheart/release_validation.md`;
depends on M1 (TestPyPI rehearsal) and M2 (report + gate).

## Why

The release-validation gate (M2) is only as trustworthy as the run it ingests.
Two verified gaps in the current `workspace-validation.yml` mean today's run is a
*structural smoke against source*, not a *release validation against builds*:

- **Gap A — tests source, not builds.** `workspace-validation.yml` installs
  `autolens[optional]` only for transitive deps, then shadows the PyAuto packages
  with source checkouts via `PYTHONPATH` (~line 175). It never exercises a wheel.
- **Gap B — runs the smoke profile, not release fidelity.** Both tiers'
  `config/build/env_vars.yaml` default to `PYAUTO_TEST_MODE=2` (skip sampler) +
  `PYAUTO_SMALL_DATASETS=1` (15×15 toy grids) + `PYAUTO_DISABLE_JAX=1` +
  `PYAUTO_FAST_PLOTS=1`.

This milestone makes the validation run install the **TestPyPI wheels** and run
at **release fidelity**.

## Task

### A. Wheel-based execution

Add a wheel mode to the validation run (a parallel job, or a parameter on
`workspace-validation.yml`): `pip install` the PyAuto packages from the
Stage-2 TestPyPI version, and **do NOT put source on `PYTHONPATH`**. Scripts must
still execute **from inside the workspace checkout** so autoconf resolves the
workspace `config/` + `dataset/` (the wheel-config-resolution footgun — without
this the run silently uses the library's packaged config defaults).

### B. A named `release` env profile

Formalise a `release` profile distinct from the existing `smoke` defaults, driven
by a selector (e.g. `PYAUTO_PROFILE=release`, or a `profiles:` block in
`config/build/env_vars.yaml`). The `release` profile mirrors the tier split
already documented in `PyAutoBuild/.github/workflows/release.yml:693-698`:

| Tier | PYAUTO_TEST_MODE | PYAUTO_SMALL_DATASETS | PYAUTO_FAST_PLOTS |
|------|------------------|------------------------|-------------------|
| user workspaces | `1` (reduced) | `1` (capped) | `1` |
| `*_workspace_test` | `0` (real searches, `n_like_max` caps) | unset (full-res) | unset |

The per-script `overrides:` (unset `SMALL_DATASETS` for full-res FITS scripts,
keep JAX on for `jax_likelihood_functions/` + `jax_substructure/`, etc.) still
layer on top of whichever profile is selected. `smoke` stays the default for the
per-PR gate; `release` is selected only by the validation run.

NB this is an **env-var profile only**. Do NOT touch `config/general.yaml`'s
`test:` / `version:` toggles (`check_likelihood_function`, `python_version_check`,
…) — those are workspace-run/user settings, not part of this profile and not
Heart's concern. The scripts run as the workspace ships them.

## Scope notes

- `run_python.py` (PyAutoBuild) needs to understand profile selection so the
  same runner serves both the smoke gate and the release validation.
- Keep `smoke` behaviour byte-for-byte unchanged — this adds a profile, it does
  not change the per-PR gate.
- The full library env-var surface is 13 `PYAUTO_*` vars (canonical:
  `PyAutoConf/autoconf/test_mode.py`); the profile only needs the fidelity subset
  above, but document the full set so a future profile can extend it.

## Validation

- Dispatch the wheel-based validation against a TestPyPI rehearsal version;
  confirm scripts import from the wheel (not source) and run under the `release`
  profile (sampler actually runs in `*_test`; full-res datasets).
- Confirm the `smoke` per-PR gate is unchanged (same pass/skip counts).
- The resulting `report.json` records `profile: release` so M2's gate can verify
  the run was release-grade.

## PR

"Release env profile + wheel-based validation run" (workspace env_vars + the
PyAutoHeart validation workflow + PyAutoBuild run_python profile selection).
