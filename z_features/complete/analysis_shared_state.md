# Epic: cross-`Analysis` shared per-evaluation state for `FactorGraphModel`

**FEATURE COMPLETE — sub-task A merged 2026-06-07, sub-task B merged 2026-06-08.**
Both "Done when" conditions below are satisfied: the generic mechanism + toy
tutorial + fast tests are live, and the lensing datacube consumer uses the shared
path with the cube-level profiling win recorded.

Merged work:
- **A (autofit):** PyAutoFit#1307 / [PR#1308](https://github.com/PyAutoLabs/PyAutoFit/pull/1308),
  [autofit_workspace#69](https://github.com/PyAutoLabs/autofit_workspace/pull/69),
  [autofit_workspace_test#31](https://github.com/PyAutoLabs/autofit_workspace_test/pull/31).
- **B (lensing):** PyAutoLens#565,
  [PyAutoArray#344](https://github.com/PyAutoLabs/PyAutoArray/pull/344),
  [PyAutoLens#566](https://github.com/PyAutoLabs/PyAutoLens/pull/566),
  [autolens_workspace#218](https://github.com/PyAutoLabs/autolens_workspace/pull/218),
  [autolens_workspace_test#138](https://github.com/PyAutoLabs/autolens_workspace_test/pull/138),
  [autolens_profiling#47](https://github.com/PyAutoLabs/autolens_profiling/pull/47).

---

Umbrella tracker for a multi-repo feature: give `FactorGraphModel` a
**domain-agnostic** way for its per-factor `Analysis` objects to compute a shared,
model-dependent object **once per likelihood evaluation** and have every factor
reuse it — instead of every factor recomputing identical work.

The motivating consumer is the ALMA datacube likelihood (N
`AnalysisInterferometer` objects sharing one lens model), where ~97% of the
per-channel inversion-setup work — ray-trace + Delaunay mapper + mapping matrix
`L` + curvature `F = LᵀW̃L` — is channel-invariant and currently rebuilt N times.

**Hard constraint:** PyAutoFit must not depend on PyAutoArray/PyAutoGalaxy/
PyAutoLens. PyAutoFit ships only the generic protocol; all lensing logic lives in
the consumer.

## Strategy — toy-first, then lensing

Build and prove the mechanism on a **1D Gaussian toy** (with thorough, fast tests)
before touching the more complicated lensing path. The toy's shared state — a
shared Gaussian component's `model_data` array computed once — is the structural
analog of the lensing mapper + L + curvature `F`.

## Locked design decisions

- **Q1 who computes it:** opt-in `Analysis.shared_state_from(instance) -> None`
  (default `None`, sibling of `modify_before_fit`); `FactorGraphModel` calls it on
  the **lead factor** (first returning non-`None`; all `None` → no sharing).
- **Q2 how factors receive it:** defaulted kwarg
  `log_likelihood_function(self, instance, shared=None)`; forwarded **only when
  non-`None`** → existing graphs byte-for-byte unchanged.
- **Q3 JAX:** shared object is a normal pytree of traced arrays, recomputed inside
  the jitted region each eval, no Python-side memoisation on the instance.
- **Q4 correctness:** PyAutoFit trusts the provider + documents the contract; the
  lensing consumer owns the channel-invariance precondition and falls back to
  per-channel compute when it fails.
- **Toy code home:** the shared-aware example `Analysis` lives in PyAutoFit
  `autofit/example/analysis.py` (`af.ex`), reused by both the autofit_workspace
  tutorial and the autofit_workspace_test script.

## Sub-tasks

### A — autofit deliverable (mechanism + toy + tutorial + fast tests) — ✅ merged 2026-06-07
- Prompt: [analysis_shared_state_cross_factor.md](../../issued/analysis_shared_state_cross_factor.md)
- Issue: PyAutoFit#1307 — merged via [PR#1308](https://github.com/PyAutoLabs/PyAutoFit/pull/1308) (task `analysis-shared-state`)
- Repos: PyAutoFit, autofit_workspace, autofit_workspace_test
- Phases:
  1. PyAutoFit mechanism (`shared_state_from` + `shared=` kwarg, lead-factor
     forwarding) + shared-aware `af.ex` Analysis + `test_autofit/graphical/` unit tests.
  2. autofit_workspace tutorial `scripts/features/shared_analysis_state.py` + notebook + docs.
  3. autofit_workspace_test fast-assert script(s): counter proves compute-once,
     exact shared-vs-unshared likelihood equality, no-provider graph unchanged,
     tiny end-to-end `DynestyStatic`, JAX pytree-threading variant.

### B — lensing deliverable (datacube consumer + workspace + profiling) — ✅ merged 2026-06-08
- Prompt: [datacube_shared_state_consumer.md](../../issued/datacube_shared_state_consumer.md)
- Issue: PyAutoLens#565 — merged via PyAutoArray#344 + [PyAutoLens#566](https://github.com/PyAutoLabs/PyAutoLens/pull/566) (task `datacube-shared-state`)
- Repos: PyAutoArray, PyAutoLens, autolens_workspace, autolens_workspace_test, autolens_profiling
- Phases:
  4. PyAutoLens lensing `shared_state_from` (ray-trace + mapper + L + F) +
     `shared`-aware `AnalysisInterferometer` + per-channel fallback.
  5. autolens_workspace datacube scripts opt in; autolens_workspace_test
     fast-assert datacube script; autolens_profiling re-measure (~17× on the
     inversion-setup block for a 34-channel cube, at ALMA scale on a quiet A100).

## Done when
- [x] A merged (mechanism live, toy tutorial published, fast tests green in CI).
- [x] B merged (datacube uses the shared path, profiling records the cube-level win).

## Cross-references
- autolens_workspace#120 — Aris's shared-`Lᵀ W̃ L` optimisation, the origin
- `PyAutoPrompt/issued/alma_datacube.md` — Aris's Slack design + channel-invariance caveat
- `PyAutoPrompt/autoarray/datacube.md` — the "analysis list API does not share information" problem statement
- `complete.md` datacube roadmap entries (Phases 1-4, all shipped)
