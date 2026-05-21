The likelihood-side `apply_sparse_operator` path for interferometer datasets at ALMA-realistic visibility counts (1M+) OOM-kills on A100 host RAM even at 384 GB cgroup limit. This is a different OOM from the simulator-side one in `@PyAutoPrompt/autoarray/nufft_simulator_chunking.md` — the simulator fails inside `nufftax.spread.interp_2d_impl` during a one-shot forward NUFFT, whereas the likelihood path fails downstream in the W-Tilde / precision-operator precompute chain.

Concrete repro: `@autolens_profiling/likelihood_runtime/interferometer/delaunay.py --instrument alma --config-name hpc_a100_fp64` on RAL HPC with `--mem=384gb`, JAX_ENABLE_X64=True, XLA_PYTHON_CLIENT_PREALLOCATE=false, the alma preset (1M vis, 800×800 grid, mask_radius=3.5). Two retries (314018 + 314019) both `OUT_OF_MEMORY` after ~30 min wall; the precompute chain runs (we see "Finished W-Tilde (JAX) in 1.4s" + "Computing NUFFT Precision Operator…" log lines repeatedly) then the cgroup OOM-killer fires before any per-call timing lands.

The likelihood path was supposed to scale here: `apply_sparse_operator(use_jax=True)` precomputes a W-Tilde matrix of shape `(2 N_y, 2 N_x)` where `N_y × N_x` is the real-space grid — for alma that's `1600 × 1600 × 8 ≈ 20 MB`. The on-disk inputs are small too: 16 MB uv_wavelengths.fits, 16 MB data.fits, 16 MB noise_map.fits. Where the 384 GB goes is unclear — likely culprits:

- **`@PyAutoArray/autoarray/inversion/inversion/interferometer/inversion_interferometer_util.py`** — block-accumulator code (`acc = jnp.zeros(gy_block.shape, dtype=jnp.float64)` at lines ~435, the `out = jnp.zeros((2 * y_shape, 2 * x_shape), dtype=jnp.float64)` at ~457). The block sizes may scale with N_visibilities and not stream/chunk.
- **`@PyAutoArray/autoarray/dataset/interferometer/dataset.py:psf_precision_operator_from`** — uses `chunk_k=2048` for the visibility chunking but the gathered intermediates per chunk may still be large at N_unmasked × chunk_k × dtype.
- **JAX XLA program retained tensors / compilation cache** — JIT can keep large intermediate arrays alive longer than naive accounting suggests.

The first-pass investigation is "instrument the precompute path with explicit memory snapshots and find where 384 GB goes." `jax.live_arrays()` + `psutil.Process().memory_info().rss` between major steps will localise the leak / accumulation. From there the fix is most likely chunking in `inversion_interferometer_util` — analogous to but distinct from the simulator-side nufftax chunking.

Datacube cell (`@autolens_profiling/likelihood_runtime/datacube/delaunay.py`) hits a related but slightly different failure mode: the per-channel precompute completes (one channel ~12 min on A100, mostly NUFFT precision-matrix construction) but 34 channels in series exceed the 2-hour SLURM walltime. Once per-channel memory is sorted, two follow-ups:

1. Bump the datacube SLURM wall time to 8h+ to accommodate 34 × per-channel precompute, OR
2. Look at sharing the precision-matrix across channels — for the canonical "all channels share lens model + uv coverage" case the precision matrix is channel-invariant; current code seems to recompute it 34×. That would be a substantial speedup.

Plumbing concerns:
- The fix here is in PyAutoArray inversion utilities, not autoarray's transformer (unlike the simulator-side prompt). Different code path, different reviewers.
- The fix must preserve the W-Tilde mathematical equivalence — this isn't a chunking-of-the-final-output situation, it's chunking the intermediate accumulation.
- Verify on the `alma` preset first (1M vis) before chasing `alma_high` (5M or 10M); if the alma OOM yields to a simple chunking fix the alma_high case is informative for whether `chunk_k` needs to be `O(N_vis)`-aware or stay at a fixed 2048.

Verification: rerun the four blocked SLURM submits — `@z_projects/profiling/hpc/batch_gpu/submit_interferometer_delaunay_a100_alma_{fp64,mp}` and `@z_projects/profiling/hpc/batch_gpu/submit_datacube_delaunay_a100_alma_{fp64,mp}` — and confirm they land cleanly with `JSON` files at `output/runtime/{interferometer,datacube}/delaunay/alma/hpc_a100_{fp64,mp}.json`.

**This task feeds back into the open profiling work**: the A100 sweep of `interferometer/delaunay + datacube/delaunay × {sma, alma, alma_high} × {fp64, mp}` was started today (PR-in-progress on `autolens_profiling` shipping the 4 SMA-only cells) and explicitly punted alma + alma_high on this blocker. Once this prompt's fix lands, re-run the sweep on the 4 alma cells + (separately, when the sibling `nufft_simulator_chunking` prompt also lands) the 4 alma_high cells. The aggregator at `@autolens_profiling/likelihood_runtime/aggregate.py` and the corresponding section in `@autolens_profiling/likelihood_runtime/OPTIMIZATION_NOTES.md` already have placeholders pointing at this work.
