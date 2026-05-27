# Audit uncached @property on Fit classes — cache immutable computed results

## Problem

`FitImaging.model_data` is a plain `@property` that recomputes the
entire inversion pipeline on every access. Any code that touches
`fit.model_data` more than once (residuals, chi-squared, visualization,
aggregator scripts) pays the full cost again. For a Delaunay source
with 1500 mesh pixels, each recomputation is ~5-20s.

This was discovered during quick-update rendering profiling: accessing
`fit.model_data` once and `fit.subtracted_images_of_planes_list` once
(which internally accesses `model_data` again) cost 13.7s just from
redundant recomputation.

`FitImaging` is constructed once and never mutated — the tracer,
dataset, and settings are fixed at construction time. All computed
properties should be safe to cache.

## Scope

### Primary: FitImaging / FitInterferometer / FitDataset

Sweep these classes for `@property` methods that:
1. Compute from other properties (creating a cascade of recomputation)
2. Are accessed more than once in normal usage patterns
3. Have no side effects and depend only on immutable constructor args

Known candidates from profiling:
- `FitImaging.model_data` — recomputes blurred image / inversion
- `FitImaging.subtracted_images_of_planes_list` — accesses model_data
- `FitImaging.model_images_of_planes_list` — may also recompute
- `FitDataset.residual_map` — accesses model_data
- `FitDataset.normalized_residual_map` — accesses residual_map
- `FitDataset.chi_squared_map` — accesses residual_map + noise_map
- `FitDataset.log_likelihood` — accesses chi_squared_map

### Secondary: Other Fit classes

- `FitInterferometer` — same pattern, different dataset type
- `FitEllipse` / `FitQuantity` — check if same issue exists
- `FitPointDataset` — likely simpler but worth checking

### Tertiary: Non-fit classes with expensive @property

While auditing, note any other classes in PyAutoArray / PyAutoGalaxy /
PyAutoLens where `@property` methods do expensive computation that
should be cached. Common patterns:
- Grid transformations computed from immutable mask geometry
- Convolver matrices derived from fixed PSF + mask
- Tracer plane images derived from fixed galaxy list

## Implementation

For each candidate:
1. Verify the class is immutable after construction (no setattr on
   the attributes the property depends on)
2. Change `@property` to `@cached_property` (from `functools` or
   `autoconf`)
3. If the class uses `__getstate__` / `__setstate__` (for pickling),
   ensure cached values are included or excluded appropriately
4. If the class is pytree-registered (for JAX), check that cached
   values appear in `__dict__` and are handled by the flatten/unflatten
   functions — cached properties that produce non-JAX-compatible types
   (e.g. Python lists) may need to be excluded from the pytree

## Testing

- `pytest test_autoarray/` — must pass
- `pytest test_autogalaxy/` — must pass  
- `pytest test_autolens/` — must pass
- Profiling: re-run `autolens_profiling/quick_update/imaging.py` and
  `imaging_delaunay.py` to confirm model_data is no longer recomputed
  during rendering
- Smoke tests across workspaces

## What this unblocks

- Quick-update rendering becomes even faster (model_data computed once
  per fit, not once per property access)
- Aggregator scripts that iterate over many fits and access multiple
  properties get a proportional speedup
- Any user code that accesses fit properties in a loop benefits
