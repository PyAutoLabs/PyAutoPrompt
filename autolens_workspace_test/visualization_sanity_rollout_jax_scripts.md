# Phase D.2.a — `__Visualization Sanity__` rollout across `visualization_jax*.py`

Companion to Phase D.1 (PR `autolens_workspace_test#113`, PR
`autogalaxy_workspace_test#54`). D.1 covered the `modeling_visualization_jit*.py`
scripts (JIT-cached path with Nautilus search). This phase covers the
sibling `visualization_jax*.py` scripts (single-shot JAX-backed
visualization, no Nautilus).

Scope deliberately limited to **existing** `visualization_jax*.py`
scripts — six files across both workspace_test repos. Authoring the
missing dataset coverage (ellipse, weak lensing, quantity viz_jit) is
Phase D.2.b, to author once this lands.

## What `visualization_jax*.py` scripts do

Each script builds an Analysis with
`use_jax=True, use_jax_for_visualization=True`, takes the
prior-median instance, and calls
`Visualizer.visualize(analysis=..., paths=..., instance=..., during_analysis=False)`
to drive the JAX-backed visualization end-to-end. The assertion at the
end is that `fit.png` lands on disk. No Nautilus search, no Part-1/
Part-2 split — just a single-shot validation that the JAX path renders
plots.

The same regression class the D.1 blocks catch can also slip through
this code path silently (silent zero, cache-busting, deflection
collapse), so the same Sanity-block template applies — just inserted
at the **end** of the script (after the existing `fit.png` assertion),
since these scripts have no Nautilus to insert before.

## Files to update

Six scripts gain a Sanity block. Insertion point on every file: after
the existing `print("PILOT SUCCEEDED — ...")` line (or equivalent
end-of-script status print).

### 1. `autolens_workspace_test/scripts/imaging/visualization_jax.py`

Imaging-template SIE Sanity (same as D.1 imaging pilot from PR #111):
non-empty tangential CC + finite positive Einstein radius + warm-call
< 100 ms on a deterministic SIE tracer.

### 2. `autolens_workspace_test/scripts/interferometer/visualization_jax.py`

Imaging-template SIE Sanity + interferometer-specific `fit.model_data`
(complex Visibilities) finite + non-zero. Run the model fit via
`analysis.fit_from(instance=instance)` since the script doesn't keep a
cached `fit_2` reference.

### 3. `autolens_workspace_test/scripts/point_source/visualization_jax.py`

SIE Sanity only (no point-source-specific FoM assertion — same
reasoning as the D.1 point_source block: the prior-median position can
legitimately give chi² = -inf).

### 4. `autogalaxy_workspace_test/scripts/imaging/visualization_jax.py`

Non-lensing template (no Tracer / no lensing latents): build the fit
via `analysis.fit_from(instance=instance)`, assert `fit.model_data`
finite + non-zero, `fit.figure_of_merit` finite.

### 5. `autogalaxy_workspace_test/scripts/interferometer/visualization_jax.py`

Same as #4 but `fit.model_data` is the `aa.Visibilities` complex
array. `np.abs().sum()` handles both shapes so the assertion is
dataset-type-agnostic.

### 6. `autogalaxy_workspace_test/scripts/quantity/visualization_jax.py`

Non-lensing template applied to `FitQuantity`. The model field /
residual map are the analogues of the imaging model image — assert
`fit.model_data` (or the `FitQuantity` equivalent — verify the
attribute name at implementation time) is finite + non-zero, and
`fit.figure_of_merit` is finite.

## Common helpers

Each file's Sanity block opens with the same imports:

```python
"""
__Visualization Sanity__

<dataset-specific prose explaining the failure mode this block catches>
"""
import time as _sanity_time
import numpy as np  # add only if not already imported at module top
from autogalaxy.operate.lens_calc import LensCalc as _SanityLensCalc

# Build sanity tracer / fit + assertions ...
```

For non-lensing autogalaxy scripts, omit the `LensCalc` import (the
block has no SIE-tracer section).

## Out of scope

- **New dataset coverage.** `autogalaxy_workspace_test/scripts/ellipse/`
  has `visualization_jax.py` but no `modeling_visualization_jit.py`;
  `autogalaxy_workspace_test/scripts/quantity/` is the same;
  `autolens_workspace_test/scripts/weak/` doesn't exist at all. These
  gaps are Phase D.2.b.
- **Library code changes.** All work is in workspace_test scripts.
- **Modifying the existing JAX-validation assertion** (the `fit.png`
  exists check). The new Sanity block runs after it.

## Verification

Each modified script runs cleanly end-to-end under the build-server
`run_all_scripts.sh` path (no `PYAUTO_TEST_MODE`). Local validation:
each script can be run directly with `python scripts/<path>` and the
Sanity block prints its three PASS lines. Failure modes (e.g. legacy
fit.model_data attribute name differing on FitQuantity) should be
flagged at implementation time, not patched silently.

Workspace smoke test (`/smoke_test autolens_workspace_test` and
`/smoke_test autogalaxy_workspace_test`) should pass — modulo the
pre-existing `point.py` JAX-vmap CI failure on autolens_workspace_test
(documented in the previous three tasks' notes).

## References

- `z_features/fast_visualization.md` — parent tracker, Phase D section.
- PR `autolens_workspace_test#113` + PR `autogalaxy_workspace_test#54`
  — Phase D.1 (modeling_visualization_jit*.py rollout). Template for
  the Sanity-block prose and assertions.
- PR `autolens_workspace_test#111` — original imaging Sanity pilot.
- `feedback_no_silent_guards` (memory) — codebase rule against silent
  catch-and-degrade; what these Sanity blocks enforce at workspace-test
  layer.
- `feedback_jax_closure_cache_busts` (memory) — the perf regression
  the warm-call assertion catches.
