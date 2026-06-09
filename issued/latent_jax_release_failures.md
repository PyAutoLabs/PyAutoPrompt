# Fix latent/JAX release failures after latent refactor

Original user request:

> ok fix the first group

Context:

The adjusted PyAutoBuild report at
`PyAutoBuild/test_results/runs/2026-06-08T20-19-35Z/report_adjusted_after_latent_refactor.md`
groups the active release failures after rerunning the latent-refactor-suspect
scripts. Group A still fails after the latent class refactor and should be fixed
first.

Target failures:

- `autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py`
  - `Analysis has no attribute _jitted_fit_from`
- `autogalaxy_workspace/scripts/guides/results/start_here.py`
  - invalid/zero noise-map values
- `autogalaxy_workspace_test/scripts/ellipse/modeling_visualization_jit.py`
  - expected `jax.Array`, got `numpy.float64`
- `autogalaxy_workspace_test/scripts/ellipse/visualization_jax.py`
  - `fit_ellipse.png` not produced
- `autogalaxy_workspace_test/scripts/interferometer/modeling_visualization_jit.py`
  - `compute_latent_samples` empty stack

Start with the smallest PyAutoFit-level failure,
`autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py`, because it
asserts the JAX visualization dispatch state directly and may clarify the
downstream JAX visualization failures.

Likely affected repositories:

- `@PyAutoFit`
- `@autofit_workspace_test`
- `@autogalaxy_workspace`
- `@autogalaxy_workspace_test`
- possibly `@PyAutoGalaxy`
