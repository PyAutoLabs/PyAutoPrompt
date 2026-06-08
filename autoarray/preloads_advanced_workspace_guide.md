# Advanced workspace guide: `Preloads` (PyAutoArray)

Write an **advanced** guide, in both `@autolens_workspace` and `@autogalaxy_workspace`, on the
`Preloads` API (`aa.PreloadsInterferometer`, `aa.AbstractPreloads`, and `PreloadsImaging` if added)
— what preloads are, when they are safe to use, and how to use them.

Primary repos: **@autolens_workspace**, **@autogalaxy_workspace** (workspace docs). Depends on the
`Preloads` API landing in **@PyAutoArray** (the `datacube-shared-state` task, PyAutoLens#565 /
sub-task B of the `analysis_shared_state` epic).

## Why this guide is needed

`Preloads` let a caller compute an invariant fit/inversion quantity once (e.g. the curvature matrix
`F = LᵀW̃L`) and inject it so repeated evaluations reuse it instead of rebuilding it. An earlier
preload system was **removed** because it was bug-prone (preloads set up incorrectly), hard to
maintain, and — most importantly — it was hard to know *when* a model could safely use a preload
(i.e. when a quantity genuinely does not change). Preloading a quantity that in fact changes
silently corrupts the result.

The API is being reintroduced because the **shared / combined likelihood** context (the datacube
`FactorGraphModel`, where the lens model is identical for every spectral channel) makes invariance
**explicit and easy to verify** — it is obvious exactly which quantities are channel-invariant. So
preloads are now an *advanced, opt-in* tool, not something applied as standard. The guide must make
this framing unmistakable so users don't reintroduce the old footguns.

## Scope

- **What preloads are**: the `AbstractPreloads` / `PreloadsInterferometer` containers, the optional
  fields (starting with `curvature_matrix`), and how an `Inversion` / fit reuses a populated field
  and falls back to the standard computation when a field is `None`.
- **When they are safe**: the invariance contract. Lead with the shared/combined-likelihood case
  (datacube) where invariance is explicit. Explicitly caution that the caller is responsible for
  ensuring a preloaded quantity is genuinely invariant, and that getting this wrong silently
  produces a wrong likelihood (the reason the old system was removed and the reason this is an
  advanced feature).
- **How to use them**: worked example(s) building a `PreloadsInterferometer` and passing it through
  a fit, ideally tying back to the datacube shared-state example so readers see the real motivation
  (compute the channel-invariant `F` once, reuse across channels).
- **When NOT to use them**: the general single-fit "preload across a search" pattern that was
  bug-prone; note it is possible but deliberately not the default.

## Placement

Advanced features directory of each workspace (e.g. `scripts/.../features/` or an `advanced/`
subfolder, matching where other advanced/“here be dragons” guides live). Mirror the existing
interferometer datacube feature scripts so the preloads guide sits alongside them.

## Cross-references
- `PyAutoArray/autoarray/preloads/abstract.py` — the `AbstractPreloads` docstring already records the
  history and the safe-use contract; the guide should expand on it for users.
- `PyAutoPrompt/issued/datacube_shared_state_consumer.md` — sub-task B (the consumer that motivates
  preloads).
- `PyAutoPrompt/z_features/analysis_shared_state.md` — the epic.
