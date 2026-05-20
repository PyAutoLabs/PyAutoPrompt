## smoke-test-optimization
- issue: https://github.com/rhayes777/PyAutoFit/issues/1183
- session: claude --resume "profile-smoke-test-runtime"
- status: profiling-and-optimization
- location: cli-in-progress
- branch: main
- repos: all on main (previous PRs merged)
- summary: |
    Done: profiled imaging scripts, achieved 80-96% runtime reductions
    Next: investigate cosmology distance calc, profile interferometer scripts

### Completed optimizations
- `PYAUTO_WORKSPACE_SMALL_DATASETS=1` — caps grids/masks to 15x15, forces over_sample_size=2, skips radial bins (PyAutoArray)
- `PYAUTO_DISABLE_JAX=1` — forces use_jax=False in Analysis.__init__ (PyAutoFit)
- `PYAUTO_FAST_PLOTS=1` — skips tight_layout + savefig + critical curve/caustic overlays (PyAutoArray/Galaxy/Lens)
- Skip print_vram_use(), model.info, result_info, pre/post-fit I/O in test_mode >= 2 (PyAutoFit)
- Moved test_mode to autoconf (fixes PyAutoArray CI — no autofit dependency)

### Profiled scripts
- `imaging/simulator.py`: ~100s → 3.6s (96% reduction)
- `interferometer/simulator.py`: ~100s → 4.4s
- `imaging/modeling.py`: ~100s → 19.5s (80% reduction)

### Remaining work — next session
- Investigate cosmology distance calc (176 calls for 2-plane lens) in subplot_fit_imaging
- Investigate repeated ray-tracing in subplot panels
- Profile interferometer/modeling.py and other scripts
- Consider caching cosmology distances per redshift pair

## psf-oversampling
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/299
- session: claude --resume "psf-oversampling"
- status: parked
- parked: 2026-05-06
- classification: library (then workspace follow-up)
- suggested-branch: feature/psf-oversampling
- summary: |
    Parked — no resources claimed. Task worktree was created during /start_library
    but removed without edits; local feature/psf-oversampling branches deleted
    from PyAutoArray and PyAutoGalaxy. Both repos are free for other tasks.

    Affected repos (when resumed):
      - PyAutoArray (library, primary)
      - PyAutoGalaxy (library)
      - autolens_workspace_test (workspace follow-up)
      - autolens_workspace (workspace follow-up)

    To resume: run /start_library — it will recreate the worktree and the
    feature/psf-oversampling branches off origin/main. Then start with
    Phase 1 (over_sample_util helpers) per the agreed phasing below.

    Phasing (smaller tasks, agreed mid-session):
      1. over_sample_util: Mask2D upscale-by-N + fine->native sum-reduce helpers + tests
      2. Convolver: add convolve_over_sample_size kwarg (default 1, no behaviour change) + test
      3. Convolver: bin-down branch in all four conv paths, gated > 1 + brute-force test
      4. Imaging dataset: kwargs + 2 construction-time guards (adaptive over-sample, sparse)
      5. GridsDataset: expose oversampled grids when > 1
      6. OperateImage + FitImaging caller threading (PyAutoGalaxy)
      7. Inversion mapping audit + assertion (mapping.py / abstract.py)
      8. End-to-end library integration test
      (workspace) extend convolution.py + new convolution_oversampled.py + simulator.py

## grid-respect-small-datasets
- session: claude (cluster-h triage verification)
- status: library-shipped, workspace-pending
- location: cli-in-progress
- worktree: ~/Code/PyAutoLabs-wt/grid-respect-small-datasets
- branch: feature/grid-respect-small-datasets
- library-pr:
    PyAutoArray: https://github.com/PyAutoLabs/PyAutoArray/pull/327
    PyAutoGalaxy: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/431
- repos:
    PyAutoArray: feature/grid-respect-small-datasets
    PyAutoGalaxy: feature/grid-respect-small-datasets
    autolens_workspace_test: feature/grid-respect-small-datasets
- summary: |
    Cluster H triage said cluster/visualization.py was failing due to a simulator
    regression or LensCalc multi-plane plumbing bug. Verification on clean main
    showed both diagnoses were wrong: simulator.py passes, mass.csv exists, host
    halo is the expected 10^15.3 M_sun NFW, and LensCalc returns the right curve
    when given a grid of adequate extent.

    Real root cause: PYAUTO_SMALL_DATASETS=1 shrinks any Grid2D.uniform >15x15
    to (15, 15) @ 0.6" (≈8" extent). The visualization.py viz_grid AND the
    internal evaluation grid built by PyAutoGalaxy's @evaluation_grid decorator
    both got shrunk well inside the cluster's tangential critical curve, so
    LensCalc.tangential_critical_curve_list_from silently returned [].

    Fix:
      - PyAutoArray: new `respect_small_datasets: bool = True` kwarg on
        Grid2D.uniform (default preserves existing behaviour).
      - PyAutoGalaxy: evaluation_grid decorator passes
        respect_small_datasets=False on its internal Grid2D.uniform call.
      - autolens_workspace_test: viz_grid in cluster/visualization.py passes
        respect_small_datasets=False (one-line comment explains why).

    Merge order: PyAutoArray PR must merge before PyAutoGalaxy PR (PyAutoGalaxy
    test + code call the new kwarg). Workspace ships via /ship_workspace after
    both library PRs merge.

## truncated-gaussian-fast-path
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1284
- session: claude --resume "truncated-gaussian-fast-path"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/truncated-gaussian-fast-path
- repos:
- summary: |
    Replace scipy.stats.norm.cdf/ppf inside
    TruncatedGaussianPrior.value_for with direct erf/erfinv calls from
    scipy.special / jax.scipy.special. cProfile (PR #17 baseline)
    showed this scipy.stats wrapper accounts for 33% of graphical
    wall time and 16% of EP at N=10. Target reductions: ~30%
    graphical, ~17% EP. Numerics must match pre-fix baseline to
    1e-6 relative tolerance — that's the merge gate, ahead of the
    speed gate.

    Follow-up after the library PR merges: refresh
    autofit_workspace_developer/{graphical,ep}/profiles/baseline.json
    in a separate workspace PR.

    First sub-task spawned by graphical-ep-scale-up scoping
    (PyAutoPrompt/graphical_ep/{graphical,ep}_scoping.md).

