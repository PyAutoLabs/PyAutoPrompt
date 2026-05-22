# `geometry_profiles.transformed_to_reference_frame_grid_from` — make `self.centre` JAX-traceable

Small library fix in PyAutoGalaxy that unblocks `AnalysisQuantity +
use_jax=True` under `jax.vmap` (Nautilus's fitness path). Currently
fails with `TracerArrayConversionError` when the mass profile's
`centre` is a tuple of traced scalars under vmap.

## What's broken

`autogalaxy/profiles/geometry_profiles.py:168`:

```python
return xp.subtract(grid.array, xp.array(self.centre))
```

Under JAX vmap, `self.centre` is a tuple `(traced_scalar, traced_scalar)`
because the model's `centre.centre_0` / `centre.centre_1` priors are
batched. `jnp.array((tracer, tracer))` calls `__array__()` on each
element, which raises `TracerArrayConversionError`.

## What works as a fix

```python
return xp.subtract(grid.array, xp.stack([self.centre[0], self.centre[1]]))
```

`xp.stack` of a list of tracers builds a length-2 array without
materializing the elements via `__array__`. Numpy semantics are
identical so the non-JAX path keeps working.

A more robust alternative: use `xp.array(list(self.centre))` after
checking whether the list-of-tracers form survives. Or check whether
the elements are tracers via `hasattr(x, "aval")` and switch between
forms. Pick whichever is cleanest at implementation time.

## Where to verify the fix

After the library edit, re-add **Part 2** to
`autogalaxy_workspace_test/scripts/quantity/modeling_visualization_jit.py`
(currently truncated to Part 1 + Sanity only — see that file's
docstring). Mirror the imaging variant's Part-2 Nautilus quick-update
block. The script should then run end-to-end on a real (short) Nautilus
search.

## Out of scope

- Other `xp.array(self.X)` sites in the profile hierarchy. If you find
  them while patching this one, batch them together — same fix shape.
- Unrelated fixes to the AnalysisQuantity dispatch / fit_quantity
  module. Stay focused on the geometry_profiles site.

## Verification

1. Unit test in PyAutoGalaxy: wrap a representative
   `transformed_to_reference_frame_grid_from(...)` call in `jax.vmap`
   on a batched centre and confirm it returns a finite array. Add to
   `test_autogalaxy/profiles/test_geometry_profiles.py`.
2. End-to-end: restore Part 2 of
   `autogalaxy_workspace_test/scripts/quantity/modeling_visualization_jit.py`
   and run it. The fit.png assertion should succeed.

## References

- `autogalaxy_workspace_test/scripts/quantity/modeling_visualization_jit.py`
  docstring — documents the precise traceback that surfaced this bug
  (Phase D.2.b.i of the fast-visualization roadmap).
- `feedback_jax_validation_vmap_not_jit` (memory) — sibling vmap-vs-jit
  validation rule. Same bug class as the convert.axis_ratio_and_angle_from
  fixes during ellipse_fitting_jax.
