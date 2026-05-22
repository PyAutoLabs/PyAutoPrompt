# Phase D.2.b.i — `quantity/modeling_visualization_jit.py` for autogalaxy_workspace_test

Authors the missing JIT-cached visualization integration script for the
`quantity` dataset type. autogalaxy_workspace_test currently has only
`quantity/visualization_jax.py` (single-shot, no Nautilus); the
`modeling_visualization_jit.py` JIT-cached + live-Nautilus variant is
absent.

Phase D.2.b.i of `z_features/fast_visualization.md`. Phase D.2.b.ii
(ellipse) and D.2.b.iii (weak lensing) are separate follow-ups.

## What to author

`autogalaxy_workspace_test/scripts/quantity/modeling_visualization_jit.py`
mirroring the structure of
`autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py`
(merged in Phase D.1 PR #54). Same Part-1 / Part-2 split, same
boilerplate around `enable_pytrees()` / `register_model()`, same
`assert cached_time < compile_time * 0.5` caching probe in Part 1,
same Sanity block + live-Nautilus search in Part 2 — only the dataset,
model, and assertions specific to `FitQuantity` differ.

### Dataset

Use the existing quantity dataset that `quantity/visualization_jax.py`
already points to. Inspect that file first to identify the dataset path
and any pre-processing (mask, simulator dependency).

### Model

Single galaxy with linear MGE light profile, matching the imaging
variant's `analysis_mge` setup but using `ag.AnalysisQuantity` instead
of `ag.AnalysisImaging`. Use the same model shape `model_mge` /
`analysis_mge` / `instance_mge` so the existing Sanity-block snippets
copy-paste cleanly.

### Part 1 (caching probe)

Identical structure to the imaging variant:
- `fit_1 = analysis_mge.fit_for_visualization(instance_mge)` first call.
- `fit_2 = analysis_mge.fit_for_visualization(instance_mge)` second.
- Assert `cached_time < compile_time * 0.5` (the standard caching ratio
  guard; on pixelization variants this assertion was fragile locally,
  but on parametric MGE it consistently holds).
- Assert `analysis_mge._jitted_fit_from is not None`.

### Visualization Sanity block

Apply the **non-lensing template** from Phase D.1 (no Tracer / no
lensing-side checks for autogalaxy):

```python
"""
__Visualization Sanity__

Phase D.2.b.i — autogalaxy quantity variant. No Tracer / no lensing
latents, so the imaging-pilot's SIE assertions don't transfer. Asserts
fit.model_data + figure_of_merit finite/non-zero on the cached fit_2
from Part 1.
"""
_md = np.asarray(fit_2.model_data)
assert np.isfinite(_md).all(), (
    "fit.model_data has nan/inf — JAX-trace mismatch on quantity helpers"
)
assert float(np.abs(_md).sum()) > 0.0, (
    "fit.model_data all-zero — quantity model field collapsed"
)
_fom = float(fit_2.figure_of_merit)
assert np.isfinite(_fom), f"figure_of_merit = {_fom} — chi² nan/inf"
print(
    f"  PASS Visualization Sanity (autogalaxy quantity): "
    f"|model_data|.sum() = {float(np.abs(_md).sum()):.4f}, "
    f"figure_of_merit = {_fom:.4f}"
)
```

`FitQuantity.model_data` is confirmed at
`autogalaxy/quantity/fit_quantity.py:71-72`.

### Part 2 (live Nautilus quick-update)

Match the imaging variant: small Nautilus search with
`iterations_per_quick_update`, asserts `fit.png` lands on disk under
the output path. Use the same model_mge constructed above (or a
slightly simpler variant if Part-2 timings call for it — match what
imaging uses for ergonomics).

## Out of scope

- **No library changes.** All work is in workspace_test scripts.
- **No new `__Visualization Sanity__` patterns.** Use the established
  non-lensing template from Phase D.1 verbatim.
- **No edits to the existing `quantity/visualization_jax.py`** — it's
  already covered in D.2.a (PR #55).
- **No ellipse / weak lensing work** — those are D.2.b.ii and D.2.b.iii.

## Verification

1. **Local end-to-end run.** Execute the new script directly with
   `python scripts/quantity/modeling_visualization_jit.py` (no
   `PYAUTO_TEST_MODE`). Confirm Part 1 caching probe passes, Sanity
   block prints `|model_data|.sum() > 0` + finite FoM, Part 2 produces
   `fit.png` under the output path.
2. **Workspace smoke list.** `/smoke_test autogalaxy_workspace_test` —
   modeling_visualization_jit*.py scripts are not in smoke (build-server
   only via `run_all_scripts.sh`), so smoke should pass without
   exercising the new script directly. Catches any broader regression.
3. **Pattern conformance.** Spot-check that the new script's structure
   (variable names, prose section headers, comment style) matches the
   imaging variant. Should diff-cleanly side-by-side modulo the
   AnalysisImaging → AnalysisQuantity swap and dataset construction.

## References

- `autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py`
  — primary template for the Part-1 / Sanity / Part-2 structure.
- `autogalaxy_workspace_test/scripts/quantity/visualization_jax.py` —
  source of the dataset path, mask, and model_mge composition for the
  quantity type.
- `complete.md::viz-sanity-rollout-jit-scripts` (PR #54) — establishes
  the autogalaxy non-lensing Sanity template.
- `complete.md::viz-sanity-rollout-jax-scripts` (PR #55) — confirms
  `FitQuantity.model_data` API.
