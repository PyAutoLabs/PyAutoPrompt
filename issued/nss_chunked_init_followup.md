# `af.NSS`: chunked `algo.init` so n_live initial particles don't OOM

## Context

Follow-up to PyAutoFit#1303 ("feat(nss): chunked vmap for inversion-heavy
A100 likelihoods"). That PR added a `chunk_size` kwarg that replaces
`blackjax.ns.from_mcmc.update_with_mcmc_take_last`'s inner
`jax.vmap(num_delete)` with `jax.lax.map(batch_size=chunk_size)`. The
PyAutoFit unit tests + a 5D Gaussian smoke confirmed bit-identical
log_Z between the unchunked and chunked paths.

A100 validation re-runs of the cells the first PR was supposed to
unblock (jobs 322605 NSS pixelization × HST × fp64, 322606 NSS delaunay
× HST × fp64) **OOM at the same allocations as before** (28.05 GB
pixelization, 27.67 GB delaunay):

```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED:
  Out of memory while trying to allocate 28055330400 bytes.
  in PyAutoArray:autoarray/inversion/mappers/mapper_util.py:315
    mat = mat.at[flat_parent, flat_pixidx].add(flat_contrib_out)
```

Decisive evidence the chunked-update path isn't the right seam: the
NSS configuration INFO log line (which lives after `algo.init` and
before the sampling loop) **never appears**. The OOM fires inside
`algo.init(initial_samples)`, not inside the per-iteration `algo.step`
that the first PR fixed.

## The actual root cause

`blackjax.ns.nss.as_top_level_api` constructs `algo.init` as:

```python
def init_fn(position, rng_key=None):
    return init(
        position,
        init_state_fn=jax.vmap(init_state_fn),  # ← hardcoded, no kwarg seam
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
```

(blackjax/ns/nss.py:223-230, handley-lab fork at SHA `ef45acd2`).

The `jax.vmap(init_state_fn)` is hardcoded inline — there is **no
kwarg seam** to inject chunking through `blackjax.nss(...)`. Initial
particle state for all `n_live=150` particles is computed in one
parallel JAX call. With the
`PyAutoArray:autoarray/inversion/mappers/mapper_util.py:315` scatter
allocating ~184 MB per particle (15,361 image pixels × 1,500 source
pixels × 8 bytes fp64) **plus** XLA scatter temp buffers, 150 ×
~184 MB ≈ 27.6 GB matches the observed OOM exactly (the ratio
`28,055,330,400 / 184,332,000 = 152.2` is conspicuously close to
n_live=150).

The chunked **update_strategy** the first PR added only fires inside
`algo.step`, which runs after `algo.init` returns successfully. For
inversion-heavy lensing cells we never get there.

## Desired fix

Replicate `blackjax.nss.as_top_level_api` locally in PyAutoFit (it's
~30 lines) so we control both:

1. The chunked update_strategy (already done by PyAutoFit#1303 via
   `make_chunked_update_strategy`).
2. The chunked init_fn (this PR).

Sketch (new module `autofit/non_linear/search/nest/nss/_chunked_nss.py`):

```python
import jax
from functools import partial
from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import init as ns_init
from blackjax.ns.base import init_state_strategy
from blackjax.ns.nss import (
    build_kernel,
    update_inner_kernel_params,
)
from ._chunked_update import make_chunked_update_strategy


def build_chunked_nss_algorithm(
    *, logprior_fn, loglikelihood_fn,
    num_inner_steps, num_delete, chunk_size,
):
    """Local replica of ``blackjax.nss(as_top_level_api)`` with
    chunked init AND chunked update.

    ``chunk_size`` controls both the inner-vmap inside the per-
    iteration MCMC step (matches PyAutoFit#1303) and the n_live-wide
    vmap inside the algorithm's ``init``. When ``chunk_size`` is
    None or >= max(n_live, num_delete) both paths fall back to
    bit-identical upstream behaviour (plain ``jax.vmap``).
    """
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        update_strategy=make_chunked_update_strategy(chunk_size),
    )

    def init_fn(position, rng_key=None):
        if chunk_size is None:
            init_batcher = jax.vmap(init_state_fn)
        else:
            init_batcher = lambda p: jax.lax.map(
                init_state_fn, p, batch_size=chunk_size
            )
        return ns_init(
            position,
            init_state_fn=init_batcher,
            update_inner_kernel_params_fn=update_inner_kernel_params,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
```

Then in `af.NSS._fit`, when `chunk_size` is set and chunking would
actually help (e.g. `chunk_size < max(n_live, num_delete)`), use
`build_chunked_nss_algorithm` instead of `_blackjax.nss(...)`.
`chunk_size=None` keeps using `blackjax.nss(...)` unchanged for
bit-identical fallback.

## Test plan

- Unit test (test_autofit, no JAX): factory builds a `SamplingAlgorithm`
  with the expected `(init, step)` shape; kwarg plumbing on `af.NSS`
  threads `chunk_size` through.
- Workspace-test (JAX-traced): bit-identical log_Z on a 5D Gaussian
  between `_blackjax.nss(...).init` (full vmap) and our
  `build_chunked_nss_algorithm(chunk_size=2).init` on the same seed,
  with `n_live=20`. (Same shape as the PyAutoFit#1303 smoke; that
  smoke only exercised the step seam, not the init seam.)
- A100 end-to-end: resubmit `autolens_profiling`'s
  `searches/nss/imaging/{pixelization,delaunay} × hst × fp64`;
  `chunk_size=16` set automatically by
  `autolens_profiling/searches/_samplers.py:build_nss`. Confirm
  completion (was OOMing as 322605/606 after PyAutoFit#1303 landed,
  same allocations as 322602/604 before it).

## Affected callers / interaction surface

- **`af.NSS`** — `_fit` switches from `_blackjax.nss(...)` to
  `build_chunked_nss_algorithm(...)` when `chunk_size` is set and
  smaller than the wider of n_live / num_delete.
- **`autolens_profiling`** — no change needed. `build_nss` already
  sets `chunk_size=vmap_batch_for_cell(...)`; the PyAutoFit-side
  change is transparent.
- **`handley-lab/blackjax`** — still no patch needed for this fix.
  Long-term cleanup: file an upstream PR adding `init_batcher` (or
  similar) as a kwarg on `blackjax.nss(as_top_level_api)` so our
  local replica can shrink to a thin forwarder. Lower priority since
  the PyAutoFit shim works.

## Why this matters

PyAutoFit#1303 was the right partial fix — per-iteration vmap chunking
is necessary, just not sufficient. Without this follow-up, NSS still
can't profile or run on the production lensing cells (SLaM
`source_pix[1/2]` Delaunay / pixelization phases) at A100 80 GB,
which were exactly the cells the original profile sweep was supposed
to compare against Nautilus.

The same A100 evidence as PyAutoFit#1301 applies; the next round
(322605 NSS pix, 322606 NSS delaunay) confirms the bug is in the
init path. Nautilus baselines for the comparison still apply:
pixelization 46.5 ms/eval / 46 min (322603), delaunay 84.8 ms/eval /
45 min (322601), NSS MGE 1.6 ms/eval / 11 min (322590).

## Out of scope

- Replacing slice-MCMC with HMC / NUTS for better mixing — separate
  upstream concern.
- Multi-GPU sharding via `jax.shard_map` — single-GPU chunked init is
  the cheapest fix for the immediate gap.

## Cross-references

- PyAutoFit#1303 — first chunked-vmap PR, fixed the per-iteration path
- PyAutoFit#1301 — original issue with chunked-vmap framing
- autolens_profiling#43 — workspace consumer wiring of chunk_size
- `PyAutoPrompt/autoarray/delaunay_interpolator_pure_callback_vmap_memory.md`
  — separate efficiency follow-up (already shown not to be the OOM cause)
- blackjax/ns/nss.py:223-230 (the hardcoded `jax.vmap(init_state_fn)`)
- A100 evidence: 322605 (NSS pix), 322606 (NSS delaunay) — both OOM
  at the same byte counts as 322604 / 322602 from before
  PyAutoFit#1303
