Context: PyAutoLens issue #542, prompt 4 of 4 (stretch goal). Prompts 1-3 built
`jax.jit(simulate_substructure)` for a single realization. This prompt extends
it to `vmap(jit(simulate))(thetas, keys)` for batched evaluation — ~1024 lensed
images per GPU launch.

## Background

The issue author's use case evaluates `theta -> noisy image` of order 10^6
times. After prompt 3, each call is a single jitted GPU kernel. The next
speedup is batching: evaluate many theta vectors in one launch, saturating
GPU parallelism.

## What to build

### Batched simulate function

```python
batched_simulate = jax.vmap(simulate_substructure, in_axes=(
    0,     # macro_params: (batch, n_macro_params) — varies per realization
    0,     # halo_params: (batch, n_planes, max_N, n_halo_params) — varies
    0,     # halo_mask: (batch, n_planes, max_N) — varies (different N per draw)
    0,     # source_params: (batch, n_source_params) — varies
    None,  # grid: shared across batch
    None,  # psf_kernel: shared
    None,  # scaling_matrix: shared (same redshift structure)
    None,  # exposure_time: shared
    None,  # background_sky: shared
    0,     # prng_key: (batch,) — different key per realization
))
```

Call with:

```python
keys = jax.random.split(master_key, batch_size)
images = jax.jit(batched_simulate)(
    macro_params_batch,   # (1024, n_macro)
    halo_params_batch,    # (1024, n_planes, max_N, n_halo)
    halo_mask_batch,      # (1024, n_planes, max_N)
    source_params_batch,  # (1024, n_source)
    grid, psf_kernel, scaling_matrix, exposure_time, background_sky,
    keys,                 # (1024,)
)
# images shape: (1024, H, W)
```

### LOSSampler → padded array conversion

The LOSSampler at `autolens/lens/los.py` produces a `List[ag.Galaxy]` per
realization. For the batched path, we need a helper that converts many
realizations into padded arrays:

```python
def los_realizations_to_arrays(
    realizations: List[List[ag.Galaxy]],
    max_halos_per_plane: int,
    n_planes: int,
    plane_redshifts: np.ndarray,
):
    """Convert a batch of LOSSampler outputs to padded arrays.

    Returns:
        halo_params: (batch, n_planes, max_halos_per_plane, n_params)
        halo_mask: (batch, n_planes, max_halos_per_plane)
    """
    ...
```

This runs in numpy (outside jit) and produces the fixed-shape arrays that
feed into the vmapped function. The LOSSampler itself doesn't need to change.

### Memory considerations

1024 images of size 100x100 at float32 = 1024 * 100 * 100 * 4 bytes = ~40 MB.
Fine for any GPU. But the intermediate arrays (per-halo deflections across all
batch elements) can be larger: 1024 * max_N * M * 2 * 4 bytes. For max_N=200
and M=10000 grid points, that's ~16 GB — may exceed GPU memory.

Mitigation strategies:
- Process in sub-batches (e.g. 128 at a time) and concatenate results
- Reduce max_N by using separate halo types per plane (most planes have
  few halos; only the lens plane has many subhalos)
- Use `jax.checkpoint` to trade compute for memory on the scan steps

Include a utility that estimates peak memory for a given configuration and
suggests a batch size.

### What varies vs what's shared across the batch

For the issue author's use case (fixed lens macro, varying substructure):

| Input | Varies? | Notes |
|-------|---------|-------|
| macro_params | Maybe | Could be fixed or sampled |
| halo_params | Yes | Different SHMF draw per realization |
| halo_mask | Yes | Different N per draw |
| source_params | Maybe | Could be fixed or sampled |
| grid | No | Same image grid |
| psf_kernel | No | Same instrument |
| scaling_matrix | No | Same redshift planes (if plane structure is fixed) |
| prng_key | Yes | Different noise per realization |

If the plane redshift structure also varies between realizations (different
LOS plane redshifts per draw), then `scaling_matrix` would need to be batched
too. But the issue author mentions 8 fixed planes, so it's likely shared.

## Integration test

Verify batch consistency:

```python
# Single-image results
images_single = [simulate_substructure(p, h, m, s, ..., k)
                 for p, h, m, s, k in zip(params...)]

# Batched results
images_batch = batched_simulate(params_stacked..., keys)

# Must match
for i in range(batch_size):
    assert jnp.allclose(images_single[i], images_batch[i], atol=1e-6)
```

Also benchmark: measure wall-clock time for 1024 sequential calls vs one
batched call. The batched version should be significantly faster (the whole
point).

Put tests in `autolens_workspace_test/scripts/jax_substructure/`.

## Scope boundaries

- This is the final prompt in the series. After this, the user has a complete
  `vmap(jit(simulate))(thetas, keys)` path.
- If memory is a hard constraint, the sub-batching utility is sufficient —
  don't try to implement gradient checkpointing in this prompt.
- The LOSSampler conversion helper is simple numpy reshaping, not a refactor
  of the sampler itself.
