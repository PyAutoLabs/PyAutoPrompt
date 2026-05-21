# Phase D.1 — `__Visualization Sanity__` rollout across `modeling_visualization_jit*.py`

Mechanical sweep of the Phase D pilot pattern (from PR
`autolens_workspace_test#111`) across the remaining JIT-cached
visualization scripts. Catches the silent-zero / collapsed-source /
unconstrained-latent regression class on every dataset type that
exercises the JIT'd `fit_for_visualization` path.

Scope deliberately limited to the `modeling_visualization_jit*.py`
scripts only — the JIT-cached path where regression risk is highest.
The `visualization_jax*.py` scripts and the missing-dataset coverage
(ellipse / weak lensing / quantity viz_jit) are Phase D.2, to author
once this lands.

## What already exists

The pilot block in `autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py`
(merged in PR #111) is the template. It:

- Builds a deterministic SIE tracer (`einstein_radius=1.2`,
  `ell_comps=(0.1, 0.0)`) rather than using the script's model prior
  medians, so the assertions are independent of the model under test.
- Wraps the tracer in `LensCalc.from_mass_obj(tracer)` to expose the
  zero_contour methods (Tracer doesn't proxy them).
- Asserts (a) non-empty tangential critical curve, (b) finite positive
  Einstein radius, (c) warm-call latency < 100 ms (guards against
  closure cache-busting regression).

## Files to update

Six scripts gain a Sanity block following the pilot pattern, with the
per-dataset assertion shape laid out in `z_features/fast_visualization.md`
under "Per-dataset assertion shapes". Insertion point: after the
existing Part-1 caching probe block (matching where the pilot landed
in `imaging/modeling_visualization_jit.py`), before the live Nautilus
search section.

### 1. `autolens_workspace_test/scripts/imaging/modeling_visualization_jit_delaunay.py`

Same SIE-tracer Sanity assertions as the imaging pilot. Pixelization
source model means the script's prior median is even less likely to
produce strong-enough lensing than the MGE+NFW imaging case; the
SIE sanity tracer keeps the assertion deterministic.

### 2. `autolens_workspace_test/scripts/imaging/modeling_visualization_jit_rectangular.py`

Identical to the delaunay variant — same SIE Sanity tracer, same
three assertions. Rectangular vs delaunay is a regularization /
mesh-class difference that doesn't affect lensing-side latents.

### 3. `autolens_workspace_test/scripts/interferometer/modeling_visualization_jit.py`

Lensing assertions identical to imaging (SIE tracer → non-empty CC +
finite Einstein radius + < 100 ms warm). Plus one
interferometer-specific assertion on model visibilities, using the
script's actual `analysis_mge` to build a `fit`:

```python
# Interferometer-specific: model visibilities must not collapse to zero / nan.
instance_mge = model_mge.instance_from_prior_medians()
fit = analysis_mge.fit_from(instance=instance_mge)
mv = np.asarray(fit.model_visibilities)
assert np.isfinite(mv).all(), "model visibilities have nan/inf"
assert float(np.abs(mv).sum()) > 0.0, "model visibilities all-zero"
```

This is the failure-mode characteristic for interferometer pipelines —
the NUFFT-via-JAX path returning zeros / NaNs from the linear
inversion silently when something deeper has gone wrong.

### 4. `autolens_workspace_test/scripts/point_source/modeling_visualization_jit.py`

No source-plane reconstructed *image*, so the imaging assertions don't
transfer. Use the per-dataset shape from the tracker (positions-based):

```python
# Use the script's analysis_mge + a prior-median instance because the
# point-source path's "fit" depends on the dataset's positions, not on
# a separately-constructed SIE.
instance_mge = model_mge.instance_from_prior_medians()
fit = analysis_mge.fit_from(instance=instance_mge)
sp = np.asarray(fit.positions_source_plane)
data_n = len(fit.positions)
assert len(sp) == data_n, "lost source-plane positions (deflection regression)"
assert np.isfinite(sp).all(), "non-finite source-plane positions"
assert float(np.max(np.linalg.norm(sp - sp.mean(axis=0), axis=1))) > 0.0, (
    "source-plane positions all coincident — deflections collapsed"
)
```

Plus the lensing-side latent assertion using a separately-constructed
SIE sanity tracer (same as imaging) — the same `Tracer` machinery
applies to point-source models, so the Einstein-radius assertion
still catches the cache-busting / silent-zero failure mode.

Verify the exact attribute name (`fit.positions_source_plane` vs
`fit.source_plane_positions`) by reading the relevant
`FitPointDataset` class at prompt-implementation time. If the name
differs, follow the actual API.

### 5. `autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py`

Single-galaxy (no Tracer / no source plane via lensing) so the
lensing-side assertions from the imaging pilot don't apply. Use the
per-dataset shape from the tracker's *Quantity* / *non-lensing* family
for autogalaxy — assert the fit's model image is non-zero and finite:

```python
instance_mge = model_mge.instance_from_prior_medians()
fit = analysis_mge.fit_from(instance=instance_mge)
model_image = np.asarray(fit.model_data)
assert np.isfinite(model_image).all(), "model image has nan/inf"
assert float(np.abs(model_image).sum()) > 0.0, "model image all-zero"
assert np.isfinite(float(fit.figure_of_merit)), "FoM nan/inf — fit collapsed"
```

Verify the model-image attribute (`fit.model_data`, `fit.model_image`,
or similar) against the live `FitImaging` API at implementation time.

### 6. `autogalaxy_workspace_test/scripts/interferometer/modeling_visualization_jit.py`

Same shape as autogalaxy imaging (no lensing → no Einstein-radius
assertion) plus the interferometer visibility assertion:

```python
instance_mge = model_mge.instance_from_prior_medians()
fit = analysis_mge.fit_from(instance=instance_mge)
mv = np.asarray(fit.model_visibilities)
assert np.isfinite(mv).all(), "model visibilities have nan/inf"
assert float(np.abs(mv).sum()) > 0.0, "model visibilities all-zero"
assert np.isfinite(float(fit.figure_of_merit)), "FoM nan/inf"
```

## Common helpers

Each file's Sanity block opens with the same imports + helper
construction so the body stays focused on the assertions:

```python
"""
__Visualization Sanity__

<dataset-specific prose explaining the failure mode this block catches>
"""
import time
import numpy as np
from autogalaxy.operate.lens_calc import LensCalc

# Build the relevant fit / lens_calc handles ...
# Assertions ...
```

Match the prose style of the existing `__Likelihood Sanity__` /
`__Visualization Sanity__` blocks — explanatory paragraph naming the
failure mode this guards against (cite PyAutoGalaxy `abd7b717` /
PyAutoFit `#1280` for the imaging/lensing scripts, the relevant
analysis path for the non-lensing ones).

## Out of scope

- **`visualization_jax*.py` scripts.** Different code path (JAX-only
  validation, no Nautilus search). Defer to Phase D.2.
- **Missing-dataset coverage.** ellipse / weak lensing / quantity
  `modeling_visualization_jit.py` don't exist yet — author in Phase D.2.
- **Code changes outside the new Sanity blocks.** No edits to library
  code, no edits to existing assertions.

## Verification

1. **Script-level execution under `PYAUTO_TEST_MODE=2`:** each of the
   six modified scripts runs cleanly through Part 1 + new Sanity block
   (the Nautilus search itself is skipped by TEST_MODE=2). The Sanity
   block's assertions all pass.
2. **Workspace smoke list:** `/smoke_test autolens_workspace_test` and
   `/smoke_test autogalaxy_workspace_test` pass (modulo pre-existing
   CI gaps from earlier tasks: NSS extras on autofit_workspace,
   point.py JAX vmap on autolens_workspace_test).
3. **Manual regression check:** in a scratch branch, temporarily
   re-broaden the broad `except` in
   `PyAutoLens/autolens/imaging/plot/fit_imaging_plots.py:52` back to
   bare `except Exception:` and confirm the imaging Sanity block now
   fails (proves the safety net catches the failure mode). Don't
   commit the rollback. Skip if PyAutoLens is no longer in the
   worktree by then.

## References

- `z_features/fast_visualization.md` — parent tracker, Phase D section
  with the Coverage audit and per-dataset assertion shapes.
- `complete.md::fast-viz-zero-contour-perf` — the imaging pilot
  shipped in PR `autolens_workspace_test#111` with the
  closure-cache-busting bug fix. Template for the rollout.
- `feedback_no_silent_guards` (memory) — codebase rule against
  silent catch-and-degrade; what the Sanity blocks enforce at
  workspace-test layer.
- `feedback_jax_closure_cache_busts` (memory) — the perf regression
  the warm-call assertion catches.
