# Migrate the Euclid pipeline to library latent variables

## Context

Parent epic: [`PyAutoPrompt/z_features/latent_refactor.md`](../z_features/latent_refactor.md).
Depends on [`autogalaxy/latent_module.md`](../autogalaxy/latent_module.md) and [`autolens/latent_module.md`](../autolens/latent_module.md) having shipped.

`euclid_strong_lens_modeling_pipeline/util.py:306-490` currently houses a bespoke `AnalysisImaging` subclass with a hand-written `LATENT_KEYS` list and `compute_latent_variables` body. This is the de facto reference implementation that the library spine in sub-prompts #1 and #2 was designed to absorb. With those landed, the euclid pipeline should now just inherit `al.AnalysisImaging` and let the library do the work.

## Task

1. **Migrate the library-shipped latents** out of `util.py:306-490` — remove the entries that PyAutoGalaxy #441 and PyAutoLens #533 now ship (`total_lens_flux_mujy`, `total_source_flux_mujy`, `total_lensed_source_flux_mujy`, `magnification`, `effective_einstein_radius`). These come from the library registry once the subclass inherits `al.AnalysisImaging` and the corresponding keys are enabled in the workspace's `config/latent.yaml`.
2. **KEEP the FWHM aperture-flux latents Euclid-specific** — `total_lens_flux_1_fwhm` … `_4_fwhm` (`util.py:421-441` + the `aperture_flux_from` helper). These require the per-band `psf_lowest_resolution` / `psf_lowest_resolution_fwhm` kwargs which are Euclid-pipeline-specific, so they MUST NOT be promoted to PyAutoLens library code. The euclid `AnalysisImaging` subclass keeps a small custom `LATENT_KEYS` containing just the aperture keys, and a custom `compute_latent_variables` that **calls `super().compute_latent_variables(parameters, model)`** to get the library tuple, then appends the aperture values. Maintain `LATENT_KEYS` and the tuple in matching positional order (autofit zips positionally at `autofit/non_linear/analysis/analysis.py:285`).
3. **Simplify the rest of the subclass** to keep only Euclid-pipeline-specific behaviour (RGB visualizer, magzero handling). The library-coverage latent machinery is gone.
3. **Add `euclid_strong_lens_modeling_pipeline/config/latent.yaml`** — toggle list, default-true, mirroring the workspace pattern. If the user wants a Euclid-specific subset (e.g. always include aperture fluxes), set their defaults here.
4. **Run `tests/test_compute_latent_variable.py`** — it should still pass against the library-driven path. Update the test if (and only if) the new yaml-driven defaults change the latent set in a way the test was asserting against; the test should now exercise the library API, not a custom subclass.
5. **Update CLAUDE.md** if it mentions the custom latent subclass anywhere.

## Where to look

- **What to delete:** `euclid_strong_lens_modeling_pipeline/util.py:299-490` — the `AnalysisImaging` subclass (keep the visualizer wiring, drop the latent guts).
- **What to keep / verify:** `euclid_strong_lens_modeling_pipeline/tests/test_compute_latent_variable.py` — confirm the test now goes through `al.AnalysisImaging` cleanly. Line 216 calls `analysis.compute_latent_variables(...)` — that call site should still work, just via the library now.
- **New config:** mirror `euclid_strong_lens_modeling_pipeline/config/output.yaml`.

## Verification

```bash
cd /home/jammy/Code/PyAutoLabs/euclid_strong_lens_modeling_pipeline
pytest tests/test_compute_latent_variable.py -x -v

# Smoke test the full pipeline
PYAUTO_TEST_MODE=1 python start_here.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley
```

Inspect `output/.../latent/latent_summary.json` — confirm all the original Euclid latents are still produced (total lens flux + aperture fluxes, total lensed source flux, total source flux, magnification, effective Einstein radius).

## Affected repos

- euclid_strong_lens_modeling_pipeline (primary)

## Suggested branch

`feature/euclid-latent-migration`

## Notes

- Per CLAUDE.md in this repo: this is "code-heavy, doc-light" pipeline glue → **Sonnet model** is fine for execution. Tutorial-style prose isn't part of this task.
- Per memory `feedback_euclid_pipeline_not_z_projects`: the pipeline workspace at `euclid_strong_lens_modeling_pipeline/` is the target, not `z_projects/euclid` (which is live science work).
- If the test was relying on the custom `magzero` kwarg behaviour, double-check the library now raises the same loud error when `magzero` is missing (per memory `feedback_no_silent_guards`).
