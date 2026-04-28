## weak-shear-simulator
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/476
- session: claude --resume "weak-shear-simulator"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/weak-shear-simulator
- repos:

## positions-test-mode-fallback
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/477
- session: claude --resume "positions-test-mode-fallback"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/positions-test-mode-fallback
- repos:

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

## merge-results-start-here
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/95
- session: claude --resume "merge-results-start-here"
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/merge-results-start-here
- repos:

## dashboard-followup-commands
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/9
- session: claude --resume "dashboard-followup-commands"
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/dashboard-followup-commands
- repos: PyAutoPrompt
