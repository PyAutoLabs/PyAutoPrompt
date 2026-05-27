# Unify JAX visualization with likelihood JIT path

Remove `use_jax_for_visualization` and have quick-update visualization
reuse the same JIT-compiled function as the likelihood evaluation. The
current architecture compiles visualization separately, paying a 20–30s
penalty on the first quick update that the user perceives as "the quick
update is slow."

## Problem statement

When `use_jax=True` and `use_jax_for_visualization=False` (the current
default), the quick-update visualization path calls `fit_for_visualization`
which runs `analysis.fit_from(instance)` through **plain Python**. But
the profile methods (Sersic, Isothermal, etc.) still dispatch to JAX
internally because `analysis._use_jax=True`. Each profile evaluation
triggers its **own** small `jax.jit` compilation via the
`@aa.grid_dec.transform` / `@aa.grid_dec.to_array` decorator chain.

**cProfile of the first `model_data` access** (15k masked pixels, HST):

```
234 calls to jax._src.pjit.cache_miss  → 12.97s
216 calls to backend_compile_and_load  →  9.59s
```

This is 234 individual XLA compilations instead of one composed graph.
Subsequent quick updates are fast (~0.5s) because the per-function JIT
cache is warm — but only for the **same parameter shapes**. The first
quick update in every process pays the full 20s+ cost.

Setting `use_jax_for_visualization=True` compiles a single
`jax.jit(analysis.fit_from)` — better in principle, but:

1. It creates its **own** JIT function, separate from the search's
   `Fitness._vmap = jax.vmap(jax.jit(self.call))`. No cache sharing.
2. First compile takes ~31s (one big graph vs 234 small ones).
3. Subsequent calls are ~5s, not sub-second — suggests JIT cache misses
   (possibly pytree structure changes between calls).
4. Requires `register_model(model)` from `autofit.jax.pytrees` before
   use — the `ModelInstance` / `Galaxy` types must be pytree-registered
   or `jax.jit` rejects them.

## Profiling numbers (HST, 15k pixels, CPU, no GPU)

| Scenario | First call | Subsequent |
|---|---|---|
| `use_jax_for_visualization=False` (current default) | 22s (234 JIT compiles) | 0.5s |
| `use_jax_for_visualization=True` (separate JIT) | 31s (1 compile) | 5–6s |
| **Target: reuse search's JIT** | 0s (already compiled) | <1s |

The 35s matplotlib rendering cost (`subplot_fit` with 12 panels) is a
separate issue being tracked independently.

## Proposed fix

### Phase 1 — Remove `use_jax_for_visualization`

Delete the flag from:
- `Analysis.__init__` (`autofit/non_linear/analysis/analysis.py`)
- `Analysis.fit_for_visualization` — remove the `_use_jax_for_visualization`
  branch; `fit_for_visualization` should always go through the fast path
  when `use_jax=True`.
- `AnalysisImaging.__init__` signature in PyAutoGalaxy / PyAutoLens
- All workspace scripts that pass it
- Config defaults

Visualization follows `use_jax` — if the search uses JAX, visualization
does too.

### Phase 2 — Have visualization reuse the search's compiled function

The search's `Fitness` class compiles `jax.vmap(jax.jit(self.call))` as
`self._vmap`. This wraps the full chain:

```
Fitness.call(parameters: vector)
  → model.instance_from_vector(vector, xp=jnp)
  → analysis.log_likelihood_function(instance)
    → analysis.fit_from(instance)
      → FitImaging(dataset, tracer, ...)
    → fit.figure_of_merit
  → scalar
```

The JIT trace sees a flat float vector in and a scalar out. For
visualization we need the intermediate `FitImaging` — the scalar isn't
enough.

**Key question:** can we cache `jax.jit(analysis.fit_from)` on the
`Fitness` instance (alongside `_vmap`) so that:

1. The **search** uses `_vmap` (vector → scalar) for likelihood evaluation.
2. The **quick update** uses a cached `jax.jit(analysis.fit_from)` to go
   instance → FitImaging, then hands the FitImaging to the visualizer.
3. Both share the same XLA compilation cache (same sub-expressions get
   deduplicated by XLA).

This cached `_jit_fit_from` would replace `fit_for_visualization`'s own
`jax.jit(self.fit_from)`. It lives on `Fitness` (which has access to
the model for `instance_from_vector`) rather than on `Analysis`.

**Alternatively:** compile `fit_from` during `Fitness.__init__` warmup
by calling it once with `model.instance_from_prior_medians()`. The
compiled function is then warm for the first quick update. This is
effectively the warmup idea from the earlier investigation, but now it's
the same function the visualization path will use.

### Phase 3 — Investigate the 5s steady-state cost

With `use_jax_for_visualization=True`, subsequent calls to
`jax.jit(analysis.fit_from)(instance)` take 5–6s instead of sub-second.
This suggests JIT cache misses — the compiled function is being
retraced on each call. Possible causes:

- **Pytree structure changes** — if `ModelInstance` or `Galaxy` objects
  have a different set of attributes between calls, JAX sees a different
  pytree structure and recompiles. Check: do all instances produced by
  `model.instance_from_vector` have identical pytree structure?
- **Python-float vs jax.Array leaves** — if some profile parameters are
  Python floats and others are `jax.Array`, the dispatch changes. Check:
  does `instance_from_vector(vector, xp=jnp)` consistently produce
  `jax.Array` for all leaf values?
- **Side effects in `fit_from`** — any Python-level branching (`if`,
  `hasattr`, `isinstance`) that depends on traced values will cause
  retracing. Audit `fit_from` → `tracer_via_instance_from` →
  `FitImaging.__init__` for such branches.
- **`register_model` called too late** — pytree registration must happen
  before the first JIT trace. If it's called after, JAX may cache a
  non-pytree version.

**Profiling approach:** add `jax.make_jaxpr(analysis.fit_from)(instance)`
before and after the second call — if it succeeds without recompilation,
the issue is in the execution path, not the tracing.

### Phase 4 — Wire up in `Fitness.manage_quick_update`

Currently `manage_quick_update` calls:

```python
instance = self.model.instance_from_vector(
    vector=self.quick_update_max_lh_parameters, xp=self._xp
)
self.analysis.perform_quick_update(self.paths, instance)
```

After this change, when `use_jax=True`:

```python
instance = self.model.instance_from_vector(
    vector=self.quick_update_max_lh_parameters, xp=self._xp
)
fit = self._jit_fit_from(instance)  # reuses the cached JIT
self.analysis.visualize_quick_update(self.paths, fit)
```

Note the API change: instead of passing `instance` to
`perform_quick_update` (which calls `fit_for_visualization` internally),
pass the already-computed `FitImaging` to a new
`visualize_quick_update(paths, fit)` method that only does the
visualization part (critical curves + subplot_fit render). This
separates "compute the fit" from "render the fit" and lets the Fitness
own the JIT-cached computation.

## Key files

| File | Role |
|---|---|
| `PyAutoFit/autofit/non_linear/fitness.py` | `Fitness` class — `_vmap`, `_jit`, `manage_quick_update` |
| `PyAutoFit/autofit/non_linear/analysis/analysis.py` | `fit_for_visualization`, `_use_jax_for_visualization` flag |
| `PyAutoFit/autofit/jax/pytrees.py` | `register_model` — pytree registration for ModelInstance/Galaxy |
| `PyAutoGalaxy/autogalaxy/analysis/analysis/analysis.py` | `perform_quick_update` — calls `fit_for_visualization` + `Visualizer` |
| `PyAutoLens/autolens/imaging/model/visualizer.py` | `Visualizer.visualize` — calls `fit_for_visualization` + subplot_fit |
| `PyAutoLens/autolens/imaging/model/analysis.py` | `AnalysisImaging.fit_from` — builds FitImaging |
| `PyAutoLens/autolens/imaging/model/analysis.py:169` | `_register_fit_imaging_pytrees` — pytree wiring |
| `autolens_profiling/quick_update/imaging.py` | Profiling script for this work |

## Profiling script

`autolens_profiling/quick_update/imaging.py` already exists and profiles
the current quick-update path. Extend it to also profile:

1. The unified JIT path (Phase 2) — `jax.jit(analysis.fit_from)` reused
   across calls, with `register_model` called upfront.
2. Steady-state JIT cache hit rate — is the compiled function reused or
   retraced on each call?
3. Comparison table: current (234 compiles) vs unified (1 compile, reused)
   vs target (0 compiles, search already compiled it).

## What this unblocks

Once visualization shares the search's JIT:

- **First quick update** costs ~0s for the fit (already compiled by the
  search's first likelihood eval). Only the matplotlib render and
  critical curves remain.
- **`live_visual_update=True`** becomes genuinely "live" — sub-second
  fit computation + whatever render time we achieve after the matplotlib
  optimization.
- **`use_jax_for_visualization` flag goes away** — one less knob for
  users to worry about. If `use_jax=True`, everything is fast.
- **Phase 2 of the JAX viz roadmap** (default `use_jax_for_visualization`
  to follow `use_jax`) becomes unnecessary — there's no separate flag.

## Relationship to other work

- `PyAutoPrompt/z_features/complete/jax_visualization.md` — the JAX viz
  roadmap. Phases 0–1 (pytree registration, workspace_test coverage)
  are shipped. Phase 2 (default the flag) is superseded by this task.
  Phases 3–5 (workspace adoption, subprocess viz, live Jupyter) are
  orthogonal.
- `live_visual_update` flag (PR #1293, shipped) — controls the display
  surface (matplotlib window / Jupyter cell). Orthogonal to this task;
  the flag stays.
- matplotlib rendering optimization — the 35s `subplot_fit` cost is
  separate and will be addressed after this lands.
