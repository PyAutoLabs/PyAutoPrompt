# Datacube consumer for the cross-`Analysis` shared-state mechanism

**Sub-task B of the `analysis_shared_state` epic** (see
`PyAutoPrompt/z_features/analysis_shared_state.md`). Do **not** `/start_dev` this
until sub-task A (the PyAutoFit mechanism + 1D Gaussian toy + autofit workspace
tutorial/tests, PyAutoFit#1307) is close to shipping — the mechanism it consumes
is delivered there.

Primary repo: **@PyAutoLens**. Consumers/proof: **@autolens_workspace**,
**@autolens_workspace_test**, **@autolens_profiling**.

## Depends on (delivered by sub-task A)

PyAutoFit ships the generic, domain-agnostic protocol:

- `Analysis.shared_state_from(instance) -> None` (opt-in, default `None`), the
  per-evaluation cross-factor sibling of `modify_before_fit`.
- `log_likelihood_function(self, instance, shared=None, ...)` — defaulted kwarg.
- `FactorGraphModel.log_likelihood_function` computes the shared object **once**
  from the lead factor before the per-factor loop and forwards it to each factor
  **only when non-`None`** (so non-cube graphs are byte-for-byte unchanged).

The 1D Gaussian toy in `af.ex` + `autofit_workspace` + `autofit_workspace_test`
is the worked, tested reference for how a consumer implements `shared_state_from`
and a `shared`-aware `log_likelihood_function`. **Mirror it.**

## Scope (Phases 4-5 of the original prompt)

### Phase 4 — PyAutoLens: the datacube consumer
- On the interferometer datacube path, implement the lensing-specific
  `shared_state_from`: ray-trace the shared lens model once, build the Delaunay
  mapper + mapping matrix `L` (and, where `uv_wavelengths`/`noise_map` are
  channel-invariant, the curvature `F = LᵀW̃L`) once, returning them as a normal
  **JAX pytree** of traced arrays (recomputed inside the jitted region each eval,
  never memoised on the instance — see `feedback_jax_closure_cache_busts`).
- Make `AnalysisInterferometer.log_likelihood_function(self, instance, shared=None)`
  consume the shared object in place of its own rebuild; the `shared is None`
  fallback rebuilds everything so the single-channel path is unchanged.
- Per-channel work that remains: data vector `D = Lᵀ·dirty_image` (channel
  visibilities), NNLS reconstruction, log-evidence.
- **Fall back to the current per-channel path** when the channel-invariance
  precondition does not hold (`uv_wavelengths`/`noise_map` not ~channel-invariant —
  i.e. outside the narrow-emission-line regime). The consumer owns this guard;
  PyAutoFit trusts the provider.

### Phase 5 — autolens_workspace + autolens_workspace_test + profiling
- Update the datacube modeling/likelihood scripts
  (`autolens_workspace/scripts/interferometer/features/datacube/{likelihood_function,modeling,delaunay}.py`)
  to opt into the shared path.
- Add a **fast-assert datacube script** to `autolens_workspace_test` mirroring the
  autofit_workspace_test pattern: prove `shared_state_from` runs once per eval
  (counter), shared-vs-unshared likelihood equality, and a tiny end-to-end run.
- Re-run `autolens_profiling/likelihood_breakdown/datacube/delaunay.py` (carrying
  the inversion-setup sub-decomposition step) and record the new cube cost. Per
  the decomposition, ~97% of the per-channel inversion work is shareable, so the
  inversion-setup block should drop ~17× for a 34-channel cube (≈60 s → ≈3.5 s);
  the cube total drops from ~170 s toward the per-channel residual (data-vector
  matmul + NNLS + log-ev) plus one shared mapper+L+F build. Compare against the
  `inversion_setup_decompose_*.json` artifact for the channel-invariant/variant
  split that sets the ceiling. Re-measure at ALMA scale on a quiet A100 to pin the
  cube-level number (laptop SMA seconds are provisional; the ratios are robust).

## Critical files

PyAutoLens (Phase 4):
- the `AnalysisInterferometer` likelihood path + interferometer `Inversion`/mapper construction

Workspace / test / profiling (Phase 5):
- `autolens_workspace/scripts/interferometer/features/datacube/{likelihood_function,modeling,delaunay}.py`
- `autolens_workspace_test/` — new fast-assert datacube script
- `autolens_profiling/likelihood_breakdown/datacube/delaunay.py`
- `autolens_profiling/likelihood_runtime/OPTIMIZATION_NOTES.md`

## Out of scope
- A bespoke `DataCube` data class / cube-specific `Inversion` (the rejected option).
- The dense-route variant (production uses the sparse / w̃ route; dense is not the target).
- Cross-factor *gradients* — likelihood value only.

## Cross-references
- `PyAutoPrompt/z_features/analysis_shared_state.md` — the epic tracker
- PyAutoFit#1307 — sub-task A (the mechanism + toy this consumes)
- autolens_workspace#120 — Aris's shared-`Lᵀ W̃ L` optimisation, the origin
- `PyAutoPrompt/issued/alma_datacube.md` — Aris's Slack design + channel-invariance caveat
- `PyAutoPrompt/autoarray/datacube.md` — the "analysis list API does not share information" problem statement
