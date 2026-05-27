Context: PyAutoLens issue #542, prompt 2 of 4. Prompt 1 built vmapped per-plane
deflection functions. This prompt wires them into a `jax.lax.scan` over redshift
planes to replace the Python loops in multi-plane ray-tracing.

## Background

The current multi-plane ray-tracing lives in
`autolens/lens/tracer_util.py : traced_grid_2d_list_from` (lines 174-268).
It has three nested Python loops:

1. **Outer loop** (line 232): `for plane_index, galaxies in enumerate(planes):`
2. **Scaling loop** (line 238): `for previous_plane_index in range(plane_index):`
   — applies cosmological scaling factors from all previous planes
3. **Galaxy sum** (line 262): `sum(g.deflections_yx_2d_from(...) for g in galaxies)`
   — sums deflections from all galaxies on the current plane

For the substructure use case (~8 planes, ~1000 total halos), these Python loops
unroll into a huge XLA graph and recompile whenever the galaxy count changes.

## What to build

A standalone pure-function that does the same multi-plane ray-tracing but using
`jax.lax.scan` over planes and the vmapped deflection functions from prompt 1.
Suggest placing this in the same module as prompt 1
(`autolens/lens/substructure_util.py`).

### Input representation

The key design decision is how to represent the per-plane halo populations as
fixed-shape arrays. The natural structure is:

```python
# Per-plane halo parameters, padded to max_halos_per_plane
halo_params: jnp.array     # shape (n_planes, max_halos_per_plane, n_halo_params)
halo_mask: jnp.array       # shape (n_planes, max_halos_per_plane) — bool
plane_redshifts: jnp.array # shape (n_planes,)
```

The macro lens (PowerLaw + ExternalShear) should be handled separately from the
halo stacks — it's a single galaxy evaluated directly, not vmapped. The source
light profile is also separate (evaluated on the final traced grid).

### Precomputed scaling-factor matrix

The cosmological scaling factors between all plane pairs can be precomputed
**outside jit** as a `(n_planes, n_planes)` matrix:

```python
# scaling_matrix[i, j] = scaling_factor from plane j to plane i (0 if j >= i)
scaling_matrix = precompute_scaling_matrix(plane_redshifts, cosmology)
```

The cosmology module at `autogalaxy/cosmology/model.py` already has
`scaling_factor_between_redshifts_from(redshift_0, redshift_1, redshift_final, xp)`
which is xp-threaded. Call it for each `(j, i)` pair where `j < i`.

This matrix is a static input to the jitted function — it only depends on
redshifts, which are fixed for a given realization.

### The scan function

```python
def traced_grids_via_scan(
    grid,              # (M, 2) image-plane grid
    macro_params,      # dict or array of PowerLaw + ExternalShear params
    halo_params,       # (n_planes, max_N, n_halo_params)
    halo_mask,         # (n_planes, max_N)
    scaling_matrix,    # (n_planes, n_planes)
    source_params,     # Sersic params for the source
    ...
):
    def scan_step(carry, plane_inputs):
        # carry: (current_grid, all_prev_deflections as (n_planes, M, 2) buffer)
        # plane_inputs: (this_plane_halo_params, this_plane_mask, scaling_row)

        grid, deflection_buffer, plane_idx = carry
        plane_halo_params, plane_mask, scaling_row = plane_inputs

        # 1. Apply scaled deflections from all previous planes
        #    scaling_row is (n_planes,) — entries for j >= plane_idx are 0
        scaled_deflections = jnp.einsum('p,pmd->md', scaling_row, deflection_buffer)
        current_grid = grid - scaled_deflections

        # 2. Compute macro deflections (if this is the lens plane)
        #    ... call PowerLaw + ExternalShear deflection directly ...

        # 3. Compute halo deflections via vmapped function from prompt 1
        halo_deflections = deflections_nfw_truncated_sph_from(
            current_grid, plane_halo_params, plane_mask, ...
        )

        # 4. Store total plane deflections in buffer
        total_deflections = macro_deflections + halo_deflections
        deflection_buffer = deflection_buffer.at[plane_idx].set(total_deflections)

        return (grid, deflection_buffer, plane_idx + 1), current_grid

    init_carry = (grid, jnp.zeros((n_planes, M, 2)), 0)
    _, traced_grids = jax.lax.scan(scan_step, init_carry, plane_stack)
    return traced_grids
```

The exact API will need refinement — the sketch above shows the idea. The macro
lens only contributes on one plane (the main lens plane), so use `jax.lax.cond`
or `jnp.where` to conditionally add its deflections based on `plane_idx`.

### Where the macro lens fits

The macro galaxy (PowerLaw + ExternalShear) is evaluated directly — not vmapped,
since there's only one. Its deflection function is already JAX-traceable
(`autogalaxy/profiles/mass/total/power_law.py` uses a `jax.lax.scan` series
expansion). Call it on the lens-plane grid and add it to that plane's deflection
buffer alongside the halo contribution.

### Where the source fits

After the scan produces `traced_grids` for all planes, evaluate the source light
profile (e.g. `SersicCore`) on the final plane's traced grid to produce the
lensed image. `SersicCore.image_2d_via_radii_from` already accepts `xp` — call
it directly on the source-plane grid.

## Integration test

Extend the test from prompt 1. Build a Tracer with:
- 1 PowerLaw + ExternalShear macro at z=0.5
- 10 NFWTruncatedSph subhalos at z=0.5 (lens plane)
- 5 NFWTruncatedSph LOS halos at z=0.25 (foreground plane)
- 5 NFWTruncatedSph LOS halos at z=0.75 (background plane)
- 1 Sersic source at z=1.0

Compute the final source-plane grid via both paths:
1. `tracer_util.traced_grid_2d_list_from(planes, grid, cosmology, xp=jnp)`
2. `traced_grids_via_scan(grid, macro_params, halo_params, ...)`

Assert the source-plane grids match to numerical tolerance. This validates that
the scan + vmap path reproduces the existing Python-loop path.

Also test that the scan path compiles once and reuses the compiled code when
only parameter values change (same shapes, different halo masses/positions).

Put tests in `autolens_workspace_test/scripts/jax_substructure/`.

## Scope boundaries

- This covers multi-plane ray-tracing and source-plane grid computation.
- PSF convolution and noise are prompt 3.
- The LOSSampler output stays as-is — it runs outside jit and produces the
  parameter arrays that feed into this function. The conversion from
  `LOSSampler.galaxies_from()` output to `(halo_params, halo_mask)` arrays
  is a small helper, not a refactor of LOSSampler itself.
- Don't modify the existing Tracer class or tracer_util. This is a parallel path.

## Existing patterns to follow

- `jax.lax.scan` is already used in `autogalaxy/profiles/mass/total/jax_utils.py`
  (omega series expansion) and `autoarray/operators/transformer.py` (chunked
  NUFFT). Look at those for the carry/accumulator pattern.
- `jax.lax.fori_loop` is used in `autoarray/inversion/mesh/interpolator/knn.py`.
- Pytree registration: `autoarray/abstract_ndarray.py` has `register_instance_pytree`.
  The new function takes raw arrays, so pytree registration isn't needed for
  the function itself — just ensure inputs are plain `jnp.arrays`.
