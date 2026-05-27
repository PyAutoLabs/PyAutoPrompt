Context: PyAutoLens issue #542, prompt 3 of 4. Prompts 1-2 built the vectorized
deflection and scan-based ray-tracing. This prompt wires them through PSF
convolution and Poisson noise to produce the end-to-end `jax.jit(simulate)`
function.

## Background

The existing simulator call chain is:

```
SimulatorImaging.via_tracer_from(tracer, grid)
  -> tracer.padded_image_2d_from(grid, psf_shape_2d)
       -> image_2d_from (sum light profiles on traced grids)
  -> SimulatorImaging.via_image_from(image)
       -> PSF convolution (FFT or real-space, both JAX-ready)
       -> add background sky
       -> Poisson noise via jax.random.poisson (when xp=jnp)
       -> return Imaging dataset
```

The downstream half (PSF convolution onward) is already JAX-friendly. The
upstream half (image from traced grids) is now handled by the scan path from
prompt 2. This prompt connects them and fixes the remaining gaps.

## Gap 1: PRNGKey support for Poisson noise

`autoarray/dataset/preprocess.py : poisson_noise_via_data_eps_from` (line 455)
currently takes an integer `seed` parameter. On the JAX path (line 488) it
converts this to a PRNGKey:

```python
effective_seed = seed if seed != -1 else int(time.time() * 1e6) & 0xFFFFFFFF
key = jax.random.PRNGKey(effective_seed)
```

This works for single calls but blocks `vmap` over noise seeds — you can't
vmap a function that calls `int(time.time())` inside.

Add an optional `prng_key` parameter:

```python
def poisson_noise_via_data_eps_from(
    data_eps, exposure_time_map, seed=-1, prng_key=None, xp=np
):
    ...
    if prng_key is not None:
        key = prng_key
    elif xp is not np:
        effective_seed = seed if seed != -1 else int(time.time() * 1e6) & 0xFFFFFFFF
        key = jax.random.PRNGKey(effective_seed)
    ...
```

Thread this parameter through `data_eps_with_poisson_noise_added` (line 500)
and up through `SimulatorImaging.via_image_from` in
`autoarray/dataset/imaging/simulator.py`.

## Gap 2: Over-sampler xp threading

`Grid2D.padded_grid_from` in `autoarray/structures/grids/uniform_2d.py`
(line 1140) uses `np.pad` which is not xp-aware. Similarly the OverSampler
binning path uses numpy operations.

For the substructure fast path, the simplest approach is to **skip the
autoarray grid/over-sampler machinery entirely** and handle padding and
sub-gridding with plain jnp operations in the standalone simulate function.
The grid is uniform and the over-sample factor is fixed, so this is
straightforward:

```python
# Pad grid for PSF
padded_shape = image_shape + psf_shape - 1
padded_grid = make_uniform_grid(padded_shape, pixel_scale)  # pure jnp

# Evaluate source on sub-grid if over_sample > 1
sub_grid = make_sub_grid(padded_grid, over_sample_size)  # pure jnp
sub_images = source_image_fn(sub_grid, source_params)
image = sub_images.reshape(...).mean(axis=-1)  # bin down
```

This avoids modifying the autoarray grid classes while giving us a fully
jnp-native path.

## The end-to-end simulate function

Combine everything into a single jittable function:

```python
@jax.jit
def simulate_substructure(
    macro_params,        # PowerLaw + ExternalShear parameters
    halo_params,         # (n_planes, max_N, n_halo_params)
    halo_mask,           # (n_planes, max_N)
    source_params,       # Sersic parameters
    # --- static / precomputed (passed via jax.jit static_argnums or closure) ---
    grid,                # (M, 2) image-plane grid (padded for PSF)
    psf_kernel,          # (K, K) PSF array
    scaling_matrix,      # (n_planes, n_planes)
    exposure_time,       # scalar
    background_sky,      # scalar
    prng_key,            # jax.random.PRNGKey for Poisson noise
):
    # 1. Multi-plane ray-trace (from prompt 2)
    traced_grids = traced_grids_via_scan(
        grid, macro_params, halo_params, halo_mask, scaling_matrix
    )

    # 2. Evaluate source light on final traced grid
    source_grid = traced_grids[-1]
    image = sersic_image_from(source_grid, source_params)

    # 3. PSF convolution (FFT)
    image = jax.scipy.signal.fftconvolve(image, psf_kernel, mode='same')

    # 4. Add background sky
    image = image + background_sky

    # 5. Poisson noise
    image_counts = image * exposure_time
    noisy_counts = jax.random.poisson(prng_key, image_counts)
    noisy_image = noisy_counts / exposure_time

    # 6. Subtract sky
    noisy_image = noisy_image - background_sky

    return noisy_image
```

The PSF convolution can use `jax.scipy.signal.fftconvolve` directly — the
existing Convolver FFT path in `autoarray/operators/convolver.py` already
does essentially this with `jnp.fft.rfft2 / irfft2`, so either approach works.
For the standalone function, the scipy one-liner is simpler.

## Integration test / smoke test

Build a representative substructure configuration and verify the end-to-end
simulate function against the existing OO path:

```python
# Build via existing API
tracer = al.Tracer(galaxies=[macro, *subhalos_10, source])
simulator = al.SimulatorImaging(
    exposure_time=300.0, background_sky_level=1.0,
    psf=al.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.1, ...),
    noise_seed=42,
)
imaging_old = simulator.via_tracer_from(tracer=tracer, grid=grid)

# Build via new pure-function path (same parameters, same seed)
key = jax.random.PRNGKey(42)
image_new = simulate_substructure(
    macro_params, halo_params, halo_mask, source_params,
    grid, psf_kernel, scaling_matrix, 300.0, 1.0, key,
)

# Compare (tolerance for Poisson noise RNG differences — compare
# the deterministic part first, then the noisy part with the same seed)
assert jnp.allclose(image_new, imaging_old.data, atol=1e-6)
```

Also verify that `jax.jit(simulate_substructure)` compiles successfully
and that calling it a second time with different parameter values (same
shapes) reuses the compiled code (no recompilation).

Put tests in `autolens_workspace_test/scripts/jax_substructure/`.

## Scope boundaries

- This prompt produces a working `jit(simulate)` for a single realization.
- `vmap` over a batch of parameter vectors is prompt 4.
- The LOSSampler conversion helper (Galaxy list -> padded arrays) should be
  a small utility, not a refactor. If it's simple enough, include it here;
  otherwise defer to prompt 4.
- Don't modify the existing SimulatorImaging class beyond adding the
  `prng_key` parameter to the noise functions in preprocess.py.
