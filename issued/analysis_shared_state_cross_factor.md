# Cross-`Analysis` shared-state mechanism for `FactorGraphModel`

A large new PyAutoFit feature: let the per-factor `Analysis` objects in a
`FactorGraphModel` share **per-evaluation, model-dependent precomputed state**
across each other, so that work which is identical for every factor at a given
point in parameter space is computed **once** and reused by all factors —
instead of every factor recomputing it independently.

Primary repo: **@PyAutoFit** (the mechanism). Consumer/proof: **@PyAutoLens** +
**@autolens_workspace** (the ALMA datacube likelihood that motivates it).

## Hard constraint (read first)

**PyAutoFit must not depend on PyAutoArray / PyAutoGalaxy / PyAutoLens** (see
`PyAutoFit/CLAUDE.md` — "PyAutoFit does NOT depend on..."). Therefore:

- The mechanism PyAutoFit ships must be **completely domain-agnostic**: it knows
  nothing about lensing, inversions, mappers, or visibilities. It only knows
  "factors may want to compute a shared object once per evaluation and have all
  factors see it."
- All lensing-specific logic (what to share, how to build the mapper once, how
  each channel consumes it) lives in **PyAutoLens** (the `AnalysisInterferometer`
  side) and is wired up in **autolens_workspace** datacube scripts.

So the deliverable is a *generic shared-state protocol* in PyAutoFit plus a
*lensing consumer* in PyAutoLens that proves it on the datacube.

## Existing hooks to build on (do not reinvent)

PyAutoFit already has two precedents for injecting state into an `Analysis`
around a fit — study both before designing:

1. **`EPAnalysisFactor`** (`autofit/graphical/declarative/factor/analysis.py:257+`)
   attaches a per-iteration `_cavity_mean_field` onto its wrapped `Analysis`
   immediately before optimisation, so the user's `log_likelihood_function` can
   read shared cross-factor messages. This is *exactly* the shape of mechanism we
   want — state computed at the graph level and attached to each factor's
   Analysis — except EP attaches it once per EP outer-iteration, whereas the
   datacube needs it recomputed **once per likelihood evaluation** (the lens
   parameters change every sample).

2. **`Analysis.modify_before_fit`** (`autofit/non_linear/analysis/analysis.py:320`)
   is the existing per-`Analysis` pre-fit hook. It is per-analysis and runs once
   before sampling, so it cannot host per-evaluation shared state, but its
   docstring ("alter the `Analysis` in ways that can speed up the fitting") is
   the precedent for "precompute-then-reuse" and the new hook should read as its
   per-evaluation, cross-factor sibling.

## Design (to refine in the issue, present options)

The core need: at each call to `FactorGraphModel.log_likelihood_function(instance)`,
**before** the per-factor loop, optionally compute a shared object from the
instance, then make it available to every factor's `log_likelihood_function`.

Sketch of the target loop in `collection.py`:

```
def log_likelihood_function(self, instance):
    shared = self.compute_shared(instance)   # None unless a shared-state provider is set
    log_likelihood = 0
    for model_factor, instance_ in zip(self.model_factors, instance):
        log_likelihood += model_factor.log_likelihood_function(instance_, shared=shared)
    return shared_aware_sum(...)
```

Design questions the issue must resolve (present 2-3 concrete options, pick one):

1. **Who computes the shared object?** Options:
   - A `shared_state_provider` callable/object set on the `FactorGraphModel`
     (domain-agnostic: it takes the instance, returns an opaque object).
   - A designated "lead" factor whose Analysis exposes a
     `compute_shared(instance)` method; remaining factors receive its output.
   - A new optional `Analysis.shared_state_from(instance)` protocol method
     (default returns `None`) so any Analysis can opt in.
   Favour whichever keeps PyAutoFit domain-blind and makes the lensing side a
   thin consumer.

2. **How does a factor receive it?** Options:
   - New optional kwarg `log_likelihood_function(self, instance, shared=None)`
     with a default so every existing Analysis keeps working unchanged
     (back-compat is mandatory — hundreds of Analyses exist).
   - Attribute injection like `EPAnalysisFactor` (`analysis._shared_state = ...`)
     set/cleared around the loop.
   The kwarg is cleaner and JIT-friendlier; the attribute path matches the EP
   precedent. Decide explicitly and justify.

3. **JAX / pytree correctness.** The datacube path is JIT-compiled
   (`use_jax=True`, `register_model` pytrees). The shared object will contain
   traced arrays (mapper triplets, mapping matrix, curvature). It must:
   - be threadable through `jax.jit` as a normal pytree (no Python-side caching
     that cache-busts — see `feedback_jax_closure_cache_busts`);
   - be recomputed inside the jitted region each eval (it depends on the traced
     lens parameters), not memoised across evals on the instance;
   - not break the single-factor / non-cube path (shared is `None` → identical
     behaviour and identical numbers).

4. **Correctness + ordering.** The shared object is only valid when the relevant
   parameters really are shared across factors. The mechanism must not silently
   produce wrong likelihoods if a user wires up factors whose "shared" inputs
   actually differ. Decide whether to (a) trust the provider, (b) assert
   structural equality of the relevant sub-instance across factors, or (c)
   document the contract and leave it to the consumer. Note the physical caveat
   from `alma_datacube.md`: sharing is only valid when `uv_wavelengths` and
   `noise_map` are ~channel-invariant (narrow-emission-line regime); outside it,
   the consumer must fall back to per-factor compute.

## Why this is needed (the motivating problem)

The ALMA **datacube** likelihood (autolens_workspace#120 and its roadmap, all
shipped: see `complete.md` "datacube roadmap") fits an N-channel spectral cube
as **N independent `AnalysisInterferometer` objects sharing one lens model**,
wired together with `af.FactorGraphModel`. The FactorGraph routes the shared
lens parameters to every per-channel `AnalysisInterferometer.log_likelihood_function`
and sums the results:

```
# autofit/graphical/declarative/collection.py:89-107
def log_likelihood_function(self, instance):
    log_likelihood = 0
    for model_factor, instance_ in zip(self.model_factors, instance):
        log_likelihood += model_factor.log_likelihood_function(instance_)
    return log_likelihood
```

`AnalysisFactor` just forwards to the wrapped analysis:

```
# autofit/graphical/declarative/factor/analysis.py:253-254
def log_likelihood_function(self, instance):
    return self.analysis.log_likelihood_function(instance)
```

**The problem:** because the lens model is shared across all channels, a large
fraction of each channel's likelihood is *identical work*. Profiling
(`autolens_profiling/likelihood_breakdown/datacube/delaunay.py`, results in
`autolens_profiling/likelihood_runtime/OPTIMIZATION_NOTES.md`) shows that for a
34-channel cube the step-by-step CPU cost is ~170-205 s/eval, of which:

- **~78%** is the per-channel "inversion setup" — ray-tracing the shared lens
  model, then building the source-plane **mapper** (Delaunay triangulation,
  neighbours, pixel weights) and the **mapping matrix L**;
- **~17-19%** is the curvature matrix `F = Lᵀ W̃ L`;
- only **~5%** (data vector `D`, NNLS reconstruction, log-evidence) is genuinely
  per-channel (it depends on each channel's distinct visibilities).

In the **sparse / w̃ inversion route that production actually uses** (this is the
important subtlety — see `PyAutoPrompt/issued/alma_datacube.md` and the
investigation note in `complete.md` about the transformer-free per-likelihood
path), the expensive NUFFT is precomputed once at dataset load, so the
shareable per-eval work is the **traced grids + Delaunay mapper + mapping matrix
L + curvature F** — all pure functions of the shared lens model + shared source
mesh, currently rebuilt N times. The data vector and reconstruction are the only
irreducibly per-channel parts.

This is "Aris's deferred shared-`Lᵀ W̃ L` optimisation" (autolens_workspace#120).
A decomposition of the dominant inversion-setup step
(`autolens_profiling/likelihood_breakdown/datacube/inversion_setup_decompose.py`,
SMA / CPU, sparse route) confirms **`Lᵀ W̃ L` is exactly the right thing to
share** and sizes the win:

| inversion-setup sub-step              | per-call | shareable? |
|---------------------------------------|----------|------------|
| ray-trace                             | ~0.001 s | ✅ invariant |
| Delaunay mapper + mapping matrix L    | ~0.19 s  | ✅ invariant |
| **curvature F = `Lᵀ W̃ L`**           | **~1.57 s** | ✅ invariant |
| data vector D = `Lᵀ·dirty_image`      | ~0.06 s  | ❌ per-channel |

So **~97% of the per-channel inversion work is channel-invariant**, and the
curvature `F` alone is **~86% of it** — not the mapper as an earlier hypothesis
assumed. Sharing the invariant block collapses the per-channel inversion total
from `N × ~1.81 s` to `~1.81 s + (N-1) × ~0.06 s` — roughly a **17× reduction on
the inversion-setup block** for a 34-channel cube (≈60 s → ≈3.5 s). The
remaining per-channel cost is just the `Lᵀ·dirty_image` matmul + NNLS + log-ev.

(Absolute seconds are SMA-scale on a contended laptop CPU and provisional — the
*ratios* are the robust deliverable; re-measure at ALMA scale on a quiet A100 to
pin the cube-level number. The old `shared_lwl_savings_estimate ≈ 17%` field in
the breakdown JSON under-counts because it credits `F` against the full ~170 s
cube rather than against the inversion-setup block `F` actually dominates.)

**The blocker is purely architectural, and it lives in PyAutoFit, not in
PyAutoLens.** As the design note in `PyAutoPrompt/autoarray/datacube.md` states:

> "The problem here is the analysis list API does not currently share
> information across likelihood functions or analysis objects. We therefore
> either need to make a DataCube data class, Inversion object and add bespoke
> source code, or we need to have AnalysisCombined objects be able to share
> information in their likelihood functions."

This prompt is the **second, general** option: give `FactorGraphModel` a way for
its factors to share per-evaluation state. The bespoke-`DataCube`-class option is
explicitly *not* what we want — it would solve only lensing cubes and bake
domain logic into a one-off path.


## Plan

### Phase 1 — PyAutoFit: the generic mechanism
- Add the shared-state protocol (chosen option from Design Q1/Q2) to
  `FactorGraphModel` (`collection.py`) and `AnalysisFactor`
  (`declarative/factor/analysis.py`), with a default that is a no-op so all
  existing graphs are byte-for-byte unchanged.
- Add the opt-in surface to `Analysis` (`non_linear/analysis/analysis.py`) —
  default `shared_state_from(instance) -> None` (or equivalent), mirroring the
  `modify_before_fit` precedent.
- Thread `shared=` through `log_likelihood_function` signatures with a defaulted
  kwarg; keep `EPAnalysisFactor` working.
- Unit tests in `test_autofit/graphical/`: a 3-factor mock graph where the
  shared object is a counter proving `compute_shared` runs **once** per eval (not
  N times), the sum is correct, and a graph with no provider is unchanged.

### Phase 2 — PyAutoLens: the datacube consumer
- On the interferometer datacube path, implement the lensing-specific
  `compute_shared`: ray-trace the shared lens model once, build the Delaunay
  mapper + mapping matrix L (and, where `uv`/`noise` are channel-invariant, the
  curvature `F`) once, and hand it to every channel's
  `AnalysisInterferometer.log_likelihood_function` to consume in place of its own
  rebuild.
- Per-channel work that remains: data vector `D` (channel visibilities),
  NNLS reconstruction, log-evidence.
- Fall back to the current per-channel path when the shared-invariance precondition
  doesn't hold.

### Phase 3 — autolens_workspace + profiling
- Update the datacube modeling/likelihood scripts to opt into the shared path.
- Re-run `autolens_profiling/likelihood_breakdown/datacube/delaunay.py` (which now
  carries the inversion-setup sub-decomposition as a permanent step) and record
  the new cube cost. Per the decomposition above, ~97% of the per-channel
  inversion work is shareable, so the inversion-setup block should drop ~17× for
  a 34-channel cube (≈60 s → ≈3.5 s); the cube total drops from ~170 s toward the
  per-channel residual (data-vector matmul + NNLS + log-ev, a few seconds) plus
  one shared mapper+L+F build. Compare against the
  `inversion_setup_decompose_*.json` artifact for the channel-invariant/variant
  split that sets the ceiling.

## Critical files

PyAutoFit (modify):
- `autofit/graphical/declarative/collection.py` — `FactorGraphModel.log_likelihood_function`, the per-factor sum loop
- `autofit/graphical/declarative/factor/analysis.py` — `AnalysisFactor.log_likelihood_function`, and the `EPAnalysisFactor` precedent
- `autofit/non_linear/analysis/analysis.py` — `Analysis` base: new opt-in protocol method, `modify_before_fit` sibling
- `test_autofit/graphical/` — new tests

PyAutoFit (reference, do not modify):
- `EPAnalysisFactor` (`declarative/factor/analysis.py:257+`) — the attach-state-to-analysis precedent
- `autofit/non_linear/analysis/model_analysis.py`, `visualize.py` — other Analysis wrappers that must keep working

PyAutoLens (consumer, Phase 2):
- the `AnalysisInterferometer` likelihood path + interferometer `Inversion`/mapper construction
- `autolens_workspace/scripts/interferometer/features/datacube/{likelihood_function,modeling,delaunay}.py`

Profiling (Phase 3):
- `autolens_profiling/likelihood_breakdown/datacube/delaunay.py`
- `autolens_profiling/likelihood_runtime/OPTIMIZATION_NOTES.md`

## Out of scope
- A bespoke `DataCube` data class / cube-specific `Inversion` (the rejected option).
- The dense-route variant (production uses sparse; dense is not the target).
- Generalising shared state to arbitrary cross-factor *gradients* — likelihood
  value only for now.

## Cross-references
- autolens_workspace#120 — Aris's shared-`Lᵀ W̃ L` optimisation, the origin
- `PyAutoPrompt/autoarray/datacube.md` — the "analysis list API does not share
  information" problem statement
- `PyAutoPrompt/issued/alma_datacube.md` — Aris's Slack design + the channel-
  invariance caveat (lines 24, 30, 34, 53, 207)
- `complete.md` datacube roadmap entries (Phases 1-4, all shipped)
- the paired decomposition note in `autolens_profiling` splitting the 78%
  "inversion setup" block into mapper vs mapping-matrix vs data-vector, which
  quantifies the real ceiling of this optimisation
