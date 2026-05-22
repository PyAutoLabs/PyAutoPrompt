# Phase D.2.b.ii — `ellipse/visualization_jax.py` + `ellipse/modeling_visualization_jit.py` for autogalaxy_workspace_test

Authors the missing JAX-backed visualization integration scripts for
the `ellipse` dataset type. `autogalaxy_workspace_test/scripts/ellipse/`
currently has only the non-JAX `visualization.py`; both
`visualization_jax.py` (single-shot) and `modeling_visualization_jit.py`
(JIT-cached + live Nautilus) are absent.

Phase D.2.b.ii of `z_features/fast_visualization.md`. With the quantity
package archived (D.2.b.i scope was dropped), this and D.2.b.iii (weak
lensing) are the remaining gaps. Per session decision, only **ellipse**
is in scope here; weak lensing is parked indefinitely (needs library
work first to author `AnalysisWeak`).

## Inherited infrastructure

- `AnalysisEllipse.__init__` already accepts `use_jax=True`. It passes
  `**kwargs` to `super().__init__(...)` (the parent `af.Analysis`
  init), so `use_jax_for_visualization=True` flows through to the
  parent — no library change required *unless* `fit_for_visualization`
  fails on a `FitEllipseSummed` return type (the JIT path requires the
  return type to be pytree-registered).
- `VisualizerEllipse` exists at `autogalaxy/ellipse/model/visualizer.py`
  with `visualize_before_fit` and `visualize` methods following the
  imaging variant's signature.
- Existing `ellipse/visualization.py` provides the dataset
  (`dataset/imaging/jax_test` — pre-built from
  `jax_likelihood_functions/imaging/simulator.py`), mask construction
  (`mask_generous`, `mask_tight`), and the model shapes
  (`Ellipse` + `EllipseMultipole` + `EllipseMultipoleScaled`).

## What to author

### 1. `autogalaxy_workspace_test/scripts/ellipse/visualization_jax.py`

Single-shot JAX-backed visualization pilot. Mirror the structure of
`autogalaxy_workspace_test/scripts/imaging/visualization_jax.py`
(PR #54 / #55 lineage):

- Same dataset path and mask as `ellipse/visualization.py` — single
  `mask_generous` scenario.
- Simpler model: a single `af.Model(ag.Ellipse)` (no multipoles for
  the pilot — multipoles are exercised by `visualization.py`).
- `analysis = ag.AnalysisEllipse(dataset=dataset, use_jax=True, use_jax_for_visualization=True, title_prefix="JAX_PILOT")`.
- `VisualizerEllipse.visualize(analysis=..., paths=..., instance=..., during_analysis=False)`.
- Assert the expected ellipse PNG lands on disk (verify the actual
  artifact name at impl time — `fit_ellipse.png` is plausible per the
  existing `visualization.py`).
- Append the autogalaxy non-lensing Sanity block from Phase D.2.a
  (PR #55) — `fit.figure_of_merit` finite. **Skip `fit.model_data`
  assertion** — `FitEllipseSummed` may not expose a `model_data`
  attribute; verify at impl time and either use it or document why
  it's omitted.

### 2. `autogalaxy_workspace_test/scripts/ellipse/modeling_visualization_jit.py`

JIT-cached + live Nautilus pattern. Mirror the structure of
`autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py`
(PR #54):

- Part 1 — caching probe. Build `model_mge` (here: a single
  `ag.Ellipse` model — name kept for parity with the imaging variant's
  `_mge` naming). Call `analysis_mge.fit_for_visualization(instance_mge)`
  twice; assert `cached_time < compile_time * 0.5`. **Note:** the
  `cached < 0.5 * compile` assertion was fragile on the autogalaxy
  imaging pixelization variants but stable on autogalaxy imaging
  parametric. Ellipse is parametric — should be stable. If local
  timings show the assertion fails on CPU with these small datasets,
  loosen to `< compile_time` (any speedup at all) and document.
- Sanity block — same shape as the autogalaxy non-lensing template
  used in `visualization_jax.py` above, on the cached `fit_2`.
- Part 2 — live Nautilus quick-update. Build a similar single-ellipse
  model and run `af.Nautilus(..., n_live=50, n_like_max=1500, iterations_per_quick_update=500)`.
  Assert `fit_ellipse.png` (or whichever artifact is produced)
  lands under `output/scripts/ellipse/images/modeling_visualization_jit/...`.
  Mirror the autogalaxy imaging variant's `rglob` + `len > 0` shape.

## Possible library snags

If `analysis.fit_for_visualization(instance)` raises under
`use_jax_for_visualization=True`:

1. **`FitEllipseSummed` not pytree-registered** — most likely cause.
   `jax.jit` needs every return type to be a registered pytree.
   `register_instance_pytree(FitEllipseSummed, no_flatten=[...])` in
   PyAutoGalaxy (similar to the existing `FitImaging` registration)
   would fix it. Fork a follow-up library prompt if so; ship the
   workspace scripts with `use_jax=True, use_jax_for_visualization=False`
   in that case and document the gap inline.
2. **`fit_from` not JAX-traceable for ellipse** — the docstring at
   `ellipse/model/analysis.py:115` claims it is. If it isn't, same
   follow-up.

These are out of scope for this prompt — surface as follow-up rather
than fixing inline.

## Verification

1. **Local end-to-end.** Run both scripts directly with
   `python scripts/ellipse/visualization_jax.py` and
   `python scripts/ellipse/modeling_visualization_jit.py` from the
   worktree. Confirm assertions pass and the printed Sanity values
   are sensible.
2. **Workspace smoke.** `/smoke_test autogalaxy_workspace_test` —
   `visualization_jax*.py` and `modeling_visualization_jit*.py`
   scripts are not in the smoke list (build-server only via
   `run_all_scripts.sh`), so smoke covers other scripts for regression.
3. **Pattern conformance.** Spot-check the new scripts diff-cleanly
   against the autogalaxy imaging variants modulo dataset/model swap.

## Out of scope

- **Weak lensing.** `autolens/weak/` has no `model/analysis.py` /
  `AnalysisWeak` — entire modeling layer absent. Tracked separately
  as a parked roadmap item; not in this PR.
- **Library changes** beyond surfacing pytree-registration follow-ups
  if `fit_for_visualization` fails.

## References

- `autogalaxy_workspace_test/scripts/imaging/visualization_jax.py` and
  `imaging/modeling_visualization_jit.py` — primary templates (PR #54,
  PR #55).
- `autogalaxy_workspace_test/scripts/ellipse/visualization.py` —
  dataset + mask + model template for the ellipse-specific bits.
- `complete.md::viz-sanity-rollout-jit-scripts` (PR #54) and
  `complete.md::viz-sanity-rollout-jax-scripts` (PR #55) — establish
  the Sanity-block patterns used here.
- `z_features/fast_visualization.md` — parent tracker; declare Phase D
  shipped after this lands.
