## heart-ci-linkage
- prompt: PyAutoMind/feature/pyautoheart/ci_linkage.md
- status: planned
- filed: 2026-06-30
- classification: organism (PyAutoHeart CI signal + registry)
- suggested-branch: feature/heart-ci-linkage
- milestone: M0 (foundational — release-validation gate builds on a trustworthy CI signal)
- summary: |
    Final-review finding: Heart's CI signal is too coarse/narrow to gate a
    release. ci_status reads `gh run list --limit 1` (newest run, any workflow,
    any branch) but workspaces gate on 3 workflows × 2 Pythons; readiness gates
    only the 5 libraries' CI (workspace CI observed but never gated); and the
    signal should come from the Actions server (mobile-reachable via MCP) with
    report.json as enrichment, not a hard dependency. Plus repos.yaml is stale
    (PyAutoPrompt→Mind, PyAutoPaper→Memory; organism repos unpolled). Rework
    ci_status to per-required-workflow-on-main, gate workspace CI, make the run
    conclusion the primary test_run signal, refresh the registry.
- affected-repos:
  - PyAutoHeart

## heart-release-validation
- prompt: PyAutoMind/feature/pyautoheart/release_validation.md
- status: planned
- filed: 2026-06-30
- classification: organism (PyAutoHeart deep validation + report + readiness gate)
- suggested-branch: feature/heart-release-validation
- milestone: M2 (depends on M1 = build-testpypi-rehearsal-mode)
- boundary: |
    Heart never mutates a repo and never triggers a build. The Brain Release
    Agent dispatches the rehearsal + validation workflows and awaits them; Heart's
    `validate` is ingest-and-judge only; the Health Agent (read-only) reports the
    verdict. Heart and Build never call each other.

## heart-release-profile-wheel-integration
- prompt: PyAutoMind/feature/pyautoheart/release_profile_and_wheel_integration.md
- status: planned
- filed: 2026-06-30
- classification: organism (validation fidelity — wheels + release env profile)
- suggested-branch: feature/heart-release-profile-wheel-integration
- milestone: M3 (depends on M1 + M2; closes Gaps A & B)
- summary: |
    Make the validation run install the TestPyPI wheels (no source on PYTHONPATH,
    scripts run from inside the workspace checkout so autoconf resolves workspace
    config/) and run at release fidelity via a named `release` env profile
    (user workspaces TEST_MODE=1+small+fast; *_test TEST_MODE=0, full-res),
    mirroring release.yml's tier split. Env-var profile only — does not touch
    config/general.yaml test:/version: toggles.
- affected-repos:
  - PyAutoHeart
  - PyAutoBuild
  - autolens_workspace_test / autogalaxy_workspace_test / autofit_workspace_test
  - autolens_workspace / autogalaxy_workspace / autofit_workspace
- summary: |
    New third Heart tier: a release-grade `pyauto-heart validate` that composes
    a TestPyPI build rehearsal + unit tests + the full workspace/workspace_test
    integration surface, ingests the run reports into a tracked
    `validation_report.json`, and hard-gates `readiness` GREEN on a fresh pass
    for the current source SHAs. Driven from mobile via the Brain health agent
    (GitHub dispatch/poll via MCP; Heart stays credential-free). Bakes in two
    verified gaps the current `workspace-validation.yml` has: it tests source
    not wheels (PYTHONPATH-shadow), and it runs the smoke profile
    (PYAUTO_TEST_MODE=2 + PYAUTO_SMALL_DATASETS=1) not a release-fidelity profile.
- affected-repos:
  - PyAutoHeart
  - PyAutoBrain
  - PyAutoBuild

## build-testpypi-rehearsal-mode
- prompt: PyAutoMind/feature/pyautobuild/release_yml_testpypi_rehearsal_mode.md
- status: planned
- filed: 2026-06-30
- classification: organism (PyAutoBuild executor capability)
- suggested-branch: feature/build-testpypi-rehearsal-mode
- milestone: M1 (prerequisite for M2 = heart-release-validation)
- summary: |
    Add a TestPyPI-only "rehearsal" dispatch mode to release.yml: build current
    source, publish to TestPyPI, emit the version string, and STOP before
    PyPI/tag/notebook steps — so Heart can install and validate the actual wheels
    before any release. Small, isolated, highest-value first piece.
- affected-repos:
  - PyAutoBuild

## jax-point-source-point-smoke-sentinel
- prompt: PyAutoMind/issued/jax_point_source_point_smoke_sentinel.md
- status: planned
- filed: 2026-05-21
- classification: library (triage; routing TBD by bisect)
- suggested-branch: feature/jax-point-source-point-smoke-sentinel
- summary: |
    Pre-existing regression surfaced during fast-viz-zero-contour-perf smoke.
    `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/point.py`
    fails its hardcoded `-83.38049778` literal — `fitness._vmap` returns the
    `-1e99` non-finite-likelihood sentinel from `FitPositionsImagePairAll` on
    canonical main of all three libraries. Last known good: 2026-05-08
    (autolens_workspace_test@362cfa8 rebaseline). Sibling JAX point-source
    profiling drift already tracked as PyAutoLens#514; this is a more severe
    symptom on a different file — held as two hypotheses (same root cause /
    independent regression) for triage.

    Affected repos (when resumed):
      - PyAutoLens (likely primary — PointSolver / FitPositionsImagePairAll)
      - PyAutoGalaxy or PyAutoArray (possible — bisect will say)
      - autolens_workspace_test (literal rebaseline OR no change, depending on outcome)

    Sibling smoke scripts to check while triaging: image_plane.py,
    source_plane.py in the same dir — they share the seed dataset.

## nfw-truncated-potential-accuracy
- prompt: PyAutoMind/bug/autogalaxy/nfw_truncated_potential_accuracy.md
- status: planned
- filed: 2026-06-05
- classification: library (accuracy bug)
- suggested-branch: feature/nfw-truncated-potential-accuracy
- summary: |
    Pre-existing accuracy bug surfaced while shipping dark-matter-potentials.
    NFWTruncatedSph.potential_2d_from (MGE) fails grad(psi)=alpha self-
    consistency in autolens_workspace_test/scripts/mass/dark.py (med 7.1e-2 vs
    ~8e-4 for every other NFW/gNFW/cNFW variant). Deflections pass, only the
    potential is off — likely the MGE sigma range (radii_max = truncation_radius
    * 5) is too narrow. Reproduce on clean main first.
- affected-repos:
  - PyAutoGalaxy

## piemass-potential
- prompt: PyAutoMind/feature/autogalaxy/piemass_potential.md
- status: planned
- filed: 2026-06-05
- classification: library (missing feature)
- suggested-branch: feature/piemass-potential
- summary: |
    PIEMass (Lenstool-ported PIE) has no potential_2d_from, so it now raises a
    clean NotImplementedError (post dark-matter-potentials) and crashes tracer
    visualization (potential FITS extension) — same class as the original NFW
    bug, different profile. No MGE/CSE decomposition hook exists; needs an
    analytic port (Kassiola & Kovner 1993, or the dPIEMass r_s->inf limit) or a
    new convergence-MGE hook. Validate via grad(psi)=alpha self-consistency.
- affected-repos:
  - PyAutoGalaxy
