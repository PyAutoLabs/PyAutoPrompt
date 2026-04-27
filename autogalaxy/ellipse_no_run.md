- All ellipse example scripts under `autogalaxy_workspace/scripts/ellipse/` are currently in
  `autogalaxy_workspace/config/build/no_run.yaml` with a `NEEDS_FIX 2026-04-24` marker.

  The five entries are:

  - `ellipse/simulator`
  - `ellipse/fit`
  - `ellipse/modeling`
  - `ellipse/multipoles`
  - `ellipse/database`

  They were parked because the ellipse model needs a refactor and JAX support (tracked separately in
  `PyAutoPrompt/autogalaxy/ellipse_fitting_jax.md`). In particular, `ellipse/modeling` and
  `ellipse/multipoles` time out under `PYAUTO_TEST_MODE=1` in the mega-run, and
  `ellipse/modeling` additionally raises a `KeyError` on `ellipses.0.centre_0` kwargs after API drift.

  When the JAX refactor lands:

  1. Try running each ellipse script with `PYAUTO_TEST_MODE=2` first — some may just need the
     stronger sampler bypass.
  2. Remove the five `ellipse/*` lines from `autogalaxy_workspace/config/build/no_run.yaml`.
  3. If the refactor also unlocks aggregator-style usage, the `ellipse/database` entry in
     `PyAutoBuild/autobuild/config/no_run.yaml` (the fallback list) can be removed too.
  4. Re-run the mega-run (`run_all_script_fix_failures` skill in `autogalaxy_workspace`) to confirm
     every ellipse script passes.
