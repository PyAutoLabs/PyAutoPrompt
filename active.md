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

## rst-to-myst-md-pass3
- issue: none — direct followup to PyAutoFit#1245 (issue closed) and pass2
- session: claude --resume "rst-to-myst-md-pass3"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/rst-to-myst-md-pass3
- repos:
  - autofit_workspace: feature/rst-to-myst-md-pass3
  - autofit_workspace_developer: feature/rst-to-myst-md-pass3
  - autofit_workspace_test: feature/rst-to-myst-md-pass3
  - autogalaxy_workspace: feature/rst-to-myst-md-pass3
  - autogalaxy_workspace_test: feature/rst-to-myst-md-pass3
  - autolens_workspace: feature/rst-to-myst-md-pass3
  - autolens_workspace_test: feature/rst-to-myst-md-pass3
  - autolens_base_project: feature/rst-to-myst-md-pass3
  - euclid_strong_lens_modeling_pipeline: feature/rst-to-myst-md-pass3
  - PyAutoFit: feature/rst-to-myst-md-pass3 (lib prose-ref tail)
  - PyAutoGalaxy: feature/rst-to-myst-md-pass3 (lib prose-ref tail)
  - PyAutoLens: feature/rst-to-myst-md-pass3 (lib prose-ref tail)
- summary: |
    Convert prose .rst → MyST .md across the workspace ecosystem (285 files, 9 repos).
    Plus tail: update docs/general/{configs,workspace}.md prose refs in
    PyAutoFit/Galaxy/Lens (deferred from pass 2 because workspaces still had .rst).
    12 PRs total. Workspace order smallest first; then 3 lib prose tweaks.
