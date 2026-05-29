# `af.NSS`: add chunked-vmap support so inversion-heavy likelihoods don't OOM the GPU

## Context

Surfaced by the first-class A100 NSS profiling sweep in
`autolens_profiling/searches/`. The Nautilus baseline cell
(`searches/nautilus/imaging/mge × hst × fp64`, A100 job 322560)
clocked **12.1 ms/eval** in 14 min. The NSS comparison
(`searches/nss/imaging/mge × hst × fp64`, A100 job 322590) clocked
**1.61 ms/eval** in 11 min — NSS 7.5× faster per eval, same
posterior mode.

The Delaunay extension of the comparison
(`searches/nautilus/imaging/delaunay × hst × fp64`, A100 job 322601)
completed cleanly with `n_batch=16` (from the `vram/config.py` per-cell
probe) at 84.8 ms/eval, 45 min. The NSS counterpart
(`searches/nss/imaging/delaunay × hst × fp64`, A100 jobs 322592 / 322596
/ 322600 / 322602) **cannot run at A100 scale**:

```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED:
  Out of memory while trying to allocate 27668233200 bytes.
  in PyAutoArray:autoarray/inversion/mappers/mapper_util.py:315
    mat = mat.at[flat_parent, flat_pixidx].add(flat_contrib_out)
```

The scatter operation feeds the source-mesh-to-image-pixel mapping
matrix. For HST Delaunay: 15,361 masked image pixels × 1,500 source-
mesh pixels × 8 bytes (fp64) = 184 MB per replica. With NSS's slice-
MCMC inner steps, scatter temp buffers, and `num_delete=16` particles
in flight, the working set exceeds A100's 80 GB.

`autolens_profiling/searches/_samplers.py` already reads
`vram.config.VMAP_BATCH` and caps `num_delete` at the per-cell
probe value (16 for HST Delaunay). That fix prevents some OOMs but
isn't enough on its own: NSS keeps additional state per particle
that scales worse than Nautilus's straight likelihood batches.

## Update — control test confirms vmap fan-out is the root cause

After filing the initial version, a closer audit of the Delaunay path
turned up `jax.pure_callback` usage at
`PyAutoArray:autoarray/inversion/mesh/interpolator/delaunay.py:80,249`
(wrapping `scipy.spatial.Delaunay`), so the obvious suspicion was that
the callback's HLO retention under vmap was the cause. The decisive
control was to submit the same cell with `RectangularAdaptImage`
(pure-JAX mesh, no callback) at essentially identical memory budget
per the probe (931 vs 922 MB / replica on HST).

Result (A100 jobs 322603 Nautilus + 322604 NSS, run 2026-05-28):
**NSS pixelization OOMs at 28,055,330,400 bytes, identical site
(`mapper_util.py:315` → `scatter_op`), to ~1.4% of NSS Delaunay's
27,668,233,200 bytes.** The 387 MB delta matches the 9 MB/replica
budget difference, scaled by num_delete=16 and the scatter's ~3×
working-set overhead.

The `pure_callback` is NOT the root cause for the NSS OOM. The
chunked-vmap fix proposed below is the right primary intervention.

A separate prompt
(`PyAutoPrompt/autoarray/delaunay_interpolator_pure_callback_vmap_memory.md`)
still tracks the `pure_callback` as a minor efficiency follow-up — it's
not free under vmap, just not the dominant cost — but that prompt no
longer claims to fix the OOM.

## The bug

The vmap in
`blackjax:blackjax/ns/from_mcmc.py:85-86` (handley-lab fork at SHA
`ef45acd2`):

```python
sample_keys = jax.random.split(sample_key, num_delete)
return jax.vmap(mcmc_kernel)(sample_keys, start_state)
```

is a **full vmap with no chunking**. Peak memory scales linearly with
`num_delete` (and with everything `mcmc_kernel` allocates per particle:
MCMC state, slice-walk temp buffers, scatter outputs from the
likelihood).

For low-allocation likelihoods (HST MGE, point-source) `num_delete=50`
fits comfortably. For inversion-heavy likelihoods (Delaunay,
RectangularAdaptImage on JWST/AO, datacube) we can't reduce
`num_delete` enough to fit without destroying convergence:

- `num_delete=1` makes the outer loop trivially slow (every dead
  particle is a separate JIT-compiled step).
- `num_delete=16` is the practical floor on A100 for HST Delaunay,
  and even that OOMs because the MCMC inner-step state per particle
  is larger than the bare likelihood eval.

## Desired fix

Add a `chunk_size` parameter to NSS's MCMC step builder that replaces
the bare `jax.vmap` with `jax.lax.map(..., batch_size=chunk_size)`.
`jax.lax.map(batch_size=k)` (stable since JAX 0.4.30; HPC venv runs
0.4.38) maps in chunks of `k`, vmapping within each chunk and looping
across chunks. Peak memory becomes `chunk_size × per_particle_state`
instead of `num_delete × per_particle_state`.

Sketch (in
`blackjax:blackjax/ns/from_mcmc.py:build_kernel`):

```python
def build_kernel(
    *,
    init_state_fn,
    logdensity_fn,
    mcmc_init_fn,
    mcmc_step_fn,
    num_inner_steps,
    update_inner_kernel_params_fn,
    num_delete,
    chunk_size: Optional[int] = None,    # NEW
):
    ...
    if chunk_size is None or chunk_size >= num_delete:
        batched_kernel = jax.vmap(mcmc_kernel)
    else:
        # lax.map vmaps within each chunk_size-wide block, loops across blocks
        batched_kernel = lambda sample_keys, start_state: jax.lax.map(
            lambda x: mcmc_kernel(x[0], x[1]),
            (sample_keys, start_state),
            batch_size=chunk_size,
        )
    return batched_kernel(sample_keys, start_state)
```

Then expose `chunk_size` up the API: `nss.from_mcmc.build_kernel` →
`blackjax.nss(...)` constructor → `nss.ns.run_nested_sampling(...)` →
`af.NSS(..., chunk_size=N)`. The
`autolens_profiling/searches/_samplers.py:build_nss` factory would set
`chunk_size = vmap_batch_for_cell(dataset_class, model_type, instrument)`
the same way it currently sets `num_delete`.

For Nautilus parity, the natural API is
`af.NSS(..., chunk_size=...)`; the corresponding Nautilus knob is
`af.Nautilus(..., n_batch=...)`. Both express the same idea: limit
the vmap fan-out to fit GPU memory.

## Implementation paths

1. **Upstream `handley-lab/blackjax`.** The cleanest place; the change
   lives where the vmap lives. Touches one function in
   `blackjax/ns/from_mcmc.py:build_kernel`, optionally also
   `blackjax/ns/nss.py` to thread the new arg through, plus a unit
   test that verifies `chunk_size=k` produces bit-identical results
   to the un-chunked path on a small problem.

2. **Wrap at the `yallup/nss` level.** `nss.ns.run_nested_sampling`
   constructs `algo = blackjax.nss(...)` — if blackjax exposes
   `chunk_size`, `nss.ns` just forwards it. If not, NSS would need
   to re-implement the step. Strictly worse than path 1.

3. **Wrap at the `af.NSS` level.** Pre-chunk the user's
   `loglikelihood_fn` so blackjax's vmap-over-num_delete sees a
   function that itself uses `lax.map` internally on batched input.
   Doesn't work straightforwardly because blackjax expects a scalar
   per-particle likelihood and vmaps it; if `loglikelihood_fn` is
   already batched, the vmap broadcasts incorrectly. Would require
   replacing `algo.step` entirely. Reject this path unless 1 + 2
   are infeasible.

Path 1 is the right answer.

## Test plan

- Unit test in `blackjax`: bit-identical results between
  `chunk_size=num_delete` (full vmap) and `chunk_size=1`
  (sequential), on a 5-parameter Gaussian-likelihood NSS run.
- Memory test: `jax.jit.lower(...).compile().memory_analysis()`
  before/after on a synthetic 20,000-pixel mapping-matrix
  likelihood; confirm peak temp size drops from
  `num_delete × bytes_per_replica` to
  `chunk_size × bytes_per_replica`.
- End-to-end: re-submit `autolens_profiling`'s
  `searches/nss/imaging/delaunay × hst × fp64` A100 cell with
  `chunk_size=4` (or whatever the probe says) and confirm the search
  completes. Compare timing against the Nautilus baseline
  (`results/searches/nautilus/imaging/delaunay/hst/hpc_a100_fp64.json`:
  84.8 ms/eval, 45 min wall).
- Convergence test: confirm log_Z and max log L agree between
  `chunk_size=None` (current) and `chunk_size=k` on the HST MGE cell
  (where both fit on A100 today).

## Affected callers / interaction surface

- **`af.NSS`** — gains a `chunk_size: Optional[int] = None` kwarg.
  Default `None` preserves current behaviour. Documented as a
  GPU-memory knob; CPU runs can leave it unset.
- **`autolens_profiling/searches/_samplers.py:build_nss`** — wires
  `chunk_size = vmap_batch_for_cell(...)` the same way it currently
  sets `num_delete`.
- **`yallup/nss`** — `run_nested_sampling(..., chunk_size=None)`
  forwards to blackjax.
- **`handley-lab/blackjax`** — `nss.from_mcmc.build_kernel` does the
  actual lax.map(batch_size=...) switch.

## Why this matters

`af.NSS`'s headline win on HST MGE (1.6 ms/eval vs Nautilus 12.1
ms/eval, ~7.5× speedup) makes it the obvious recommendation for
small-likelihood lensing models. Inversion-based source models
(pixelization, Delaunay, datacube — i.e. the SLaM `source_pix[1/2]`
phases that every production-quality PyAutoLens fit ends in) are
exactly the cells where users would want NSS most: each likelihood
eval is expensive enough that a sampler-side 7× speedup compounds
into real wall-time savings. As things stand we can't profile NSS
on those cells at all on an 80 GB A100 — A100 80 GB is the standard
HPC accelerator, not a niche, so this is a real production gap.

## Out of scope

- Tuning `num_inner_steps` for memory (a separate axis that doesn't
  fix the fundamental vmap-too-wide problem).
- Replacing slice-MCMC with HMC / NUTS for better mixing (separate
  upstream concern; the gradient probe at
  `autolens_workspace_developer/searches_minimal/probe_grad.py` flagged
  NaN gradients on this likelihood years ago anyway).
- A100-vs-H100 / multi-GPU sharding (`jax.shard_map`). Single-GPU
  chunked vmap is the cheapest fix for the immediate gap.

## Cross-references

- PyAutoFit:autofit/non_linear/search/nest/nss/search.py:289 (af.NSS._fit)
- blackjax/ns/from_mcmc.py:85-86 (the offending vmap)
- yallup/nss:ns.py:run_nested_sampling
- autolens_profiling/searches/_samplers.py:build_nss
- autolens_profiling/vram/config.py:VMAP_BATCH
- autolens_profiling A100 evidence: jobs 322592, 322596, 322600, 322602
  all OOM in `mapping_matrix_from`; job 322590 (MGE) completed fine
  because MGE has no mapping-matrix scatter.
