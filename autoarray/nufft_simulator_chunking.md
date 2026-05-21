The `al.SimulatorInterferometer` path that uses `al.TransformerNUFFT` (nufftax-backed) can't scale to ALMA-realistic visibility counts. At ~5M visibilities on an 800×800 real-space grid it OOMs on an A100 (80 GB) with a single ~15.7 GB allocation; at 10M it's ~31 GB. The likelihood path scales fine to the same regime because `apply_sparse_operator` precomputes a small W-Tilde matrix bounded by `N_source_pixels` (~thousands), not by `N_visibilities`. The simulator has no equivalent escape valve — every forward call does one dense nufftax spread.

The blocker is upstream in nufftax. `nufftax.transforms.nufft2.nufft2d2` calls `_interp_2d_dispatch` → `interp_2d_impl` which, at line `fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, kernel_params.nspread, kernel_params.nspread)`, materialises the full gather buffer in one shot. With `M = 5_000_000` and the default `eps=1e-6` (nspread=14), that's `2 × 5e6 × 14² × 8 ≈ 15.7 GB` for a single intermediate, and JAX's other intermediates push us past A100 headroom even with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The likelihood path proves the scaling is achievable. We need an equivalent batching escape valve for the simulator side. Two reasonable places to put it:

1. **`@PyAutoArray/autoarray/operators/transformer.py:TransformerNUFFT._forward_native`** — wrap the `nufftax.nufft2d2(self._x, self._y, image_flipped, eps, -1)` call in a chunked loop over `M`. Split `(self._x, self._y)` into batches of e.g. 200k visibilities, run `nufft2d2` per chunk, concatenate the resulting per-batch visibilities. The forward NUFFT is linear in visibility batch, so the result is bit-identical to the one-shot call.
2. **Upstream `@nufftax/transforms/nufft2.py:nufft2d2`** — add a `chunk_size` arg that does the same internal chunking. Cleaner and benefits any nufftax caller, not just autoarray.

Option 1 is the right scope for this task — keeps the change inside our codebase, lands without an upstream PR. Option 2 can be a follow-up to `nufftax` once the autoarray-side batching proves the math.

Plumbing concerns to settle while implementing:
- The constructor of `TransformerNUFFT` (currently in `@PyAutoArray/autoarray/operators/transformer.py`) needs a knob — probably `chunk_size: int | None = None` defaulting to "no chunking" so existing small-N callers (`sma` with 190 visibilities) don't pay the chunk-loop overhead.
- Equivalent batching for `TransformerNUFFT.image_from` (the adjoint via `nufft2d1`) should land in the same PR — the adjoint has the same gather pattern and same memory ceiling on big problems. Out-of-scope today, but flag it.
- Chunking interacts with JIT: a Python-level `for` loop unrolls in JAX. Use `jax.lax.scan` or `jax.lax.map` so the compiled HLO graph stays bounded regardless of `M / chunk_size`. Otherwise the forward call is fine eagerly but JIT compile time blows up.
- Picking a default `chunk_size`: needs profiling. Memory budget = `2 × chunk_size × nspread² × dtype_size`. For nspread=14 + complex64 + a 40 GB A100 working budget, `chunk_size ≈ 1_000_000` is the natural ceiling.

Verification: re-run `autolens_profiling/simulators/interferometer.py --instrument alma_high` on an A100 (currently OOMs in the simulate jobs under `@z_projects/profiling/hpc/batch_gpu/submit_simulate_interferometer_alma_high`). With the batching in place, it should land cleanly and produce the same data the un-chunked call would have on a hypothetical 200 GB GPU. Then the downstream `@autolens_profiling/likelihood_runtime/interferometer/delaunay.py` and `@autolens_profiling/likelihood_runtime/datacube/delaunay.py` A100 sweeps that depend on alma_high stop being blocked.

Once the simulator chunks cleanly, the runtime/`apply_sparse_operator` path needs no change — it already lives inside the W-Tilde envelope and doesn't trip the nufftax gather.
