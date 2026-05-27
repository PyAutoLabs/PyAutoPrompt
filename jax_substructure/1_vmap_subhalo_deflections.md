Context: PyAutoLens issue #542 asks for a JIT/vmap-able multi-plane substructure
forward simulator. This is prompt 1 of 4 — building the vectorized per-plane
deflection computation that everything else stacks on top of.

## Background

Today, when a Tracer has N subhalos on a single plane, their deflections are
summed via a Python generator loop:

```python
# tracer_util.py line 262
deflections_yx_2d = sum(
    (g.deflections_yx_2d_from(grid=scaled_grid, xp=xp) for g in galaxies)
)
```

Under `jax.jit`, JAX unrolls this into N separate traced operations. For 5
galaxies that's fine. For 1000 halos it produces a massive XLA graph (slow
compilation) and recompiles whenever N changes between realizations.

The fix is a **vmapped deflection function** that takes stacked parameter arrays
and computes all N deflections in a single GPU launch, then sums them.

All four dark matter profiles already accept `xp=jnp` and produce correct
JAX-traced outputs — the individual deflection math is ready. What's missing
is the batching orchestration.

## What to build

A pure-function module (suggest `autolens/lens/substructure_util.py` or similar)
containing functions like:

```python
def deflections_nfw_truncated_sph_from(
    grid,        # (M, 2) image-plane grid
    params,      # (N, 4) — mass_at_200, concentration, centre_y, centre_x
    mask,        # (N,) boolean — which slots are active halos
    cosmology,   # for MCR variants that need kappa_s / scale_radius
    redshift,    # halo redshift (scalar, shared across the batch)
    xp=jnp,
):
    """Compute summed deflections from N NFWTruncatedSph halos via vmap."""
    ...
```

The inner single-halo function should call the existing deflection math from
the profile classes. Look at how `NFWTruncatedSph.deflections_yx_2d_from`
works in `autogalaxy/profiles/mass/dark/nfw_truncated.py` — it calls through
the `@aa.decorators.transform` and `@aa.decorators.to_vector_yx` decorator
chain. For the vmapped path you'll want to call the underlying math directly
(pre-transform the grid by subtracting `centre`, call the radial deflection
functions, post-transform back) to avoid the decorator overhead that wraps
results in autoarray objects.

The key profiles to cover:

- `NFWTruncatedSph` — `autogalaxy/profiles/mass/dark/nfw_truncated.py`
- `cNFWSph` — `autogalaxy/profiles/mass/dark/cnfw.py`
- Their MCR Ludlow variants (`nfw_truncated_mcr.py`, `cnfw_mcr.py`) which
  derive `kappa_s` and `scale_radius` from `mass_at_200` via
  `autogalaxy/profiles/mass/dark/mcr_util.py`

For the MCR variants, the Ludlow concentration-mass relation
(`mcr_util.kappa_s_and_scale_radius_for_ludlow` and
`mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow`) is already
JAX-native — it auto-detects JAX arrays and uses `jnp` internally. So you
can vmap through the full MCR → deflection chain.

The `mask` parameter handles the padding: pad `params` to `max_N` rows, set
`mask=False` for unused slots, and zero out their deflection contribution
before summing. This way the array shape is fixed regardless of the actual
number of halos, so `jax.jit` compiles once.

## Integration test

This is the key validation: build a Tracer the normal way with ~10 subhalos
(using the existing Galaxy/profile API), compute deflections via the
Python-loop path, then compute the same deflections via the new vmapped
path, and assert they match to numerical tolerance.

Put this in `autolens_workspace_test/scripts/jax_substructure/` (new directory).
Something like:

```python
# 1. Build 10 NFWTruncatedSph halos as Galaxy objects
halos = [ag.Galaxy(redshift=0.5, mass=ag.mp.NFWTruncatedSph(...)) for _ in range(10)]
tracer = al.Tracer(galaxies=[macro_galaxy, *halos, source_galaxy])

# 2. Get deflections via existing path
deflections_old = tracer_util.traced_grid_2d_list_from(..., xp=jnp)

# 3. Stack same parameters into arrays
params = jnp.array([[mass_i, conc_i, cy_i, cx_i] for ...])
mask = jnp.ones(10, dtype=bool)

# 4. Get deflections via new vmapped path
deflections_new = deflections_nfw_truncated_sph_from(grid, params, mask, ...)

# 5. Assert match
assert jnp.allclose(deflections_old, deflections_new, atol=1e-8)
```

Do this for all four profile types. Also test that masked-out slots contribute
zero deflection.

## Scope boundaries

- This prompt covers **single-plane** vectorized deflections only. Multi-plane
  scan is prompt 2.
- Don't modify the existing Tracer or Galaxy classes. This is a parallel path.
- The macro lens (PowerLaw + ExternalShear) doesn't need vmapping here — there's
  only one macro lens per realization. It will be called directly in prompt 2.
- Light profiles (source image) are also not in scope here — just mass deflections.
