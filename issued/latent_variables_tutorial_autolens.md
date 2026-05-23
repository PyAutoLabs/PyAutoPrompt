# Add latent-variables workspace tutorial to autolens_workspace

## Context

Parent epic: [`PyAutoPrompt/z_features/latent_refactor.md`](../z_features/latent_refactor.md).
Depends on the library spine in [`autolens/latent_module.md`](../autolens/latent_module.md) and the foundational docs from [`autofit_workspace/latent_variables_tutorial_expand.md`](../autofit_workspace/latent_variables_tutorial_expand.md).

Lensing-specific mirror of [`autogalaxy_workspace/latent_variables_tutorial.md`](../autogalaxy_workspace/latent_variables_tutorial.md), focused on the latents that matter for lensing science: **magnification, effective Einstein radius, lensed source flux, total source flux, lens light apertures**.

## Task

1. **Add a new tutorial** at `autolens_workspace/scripts/results/<filename>.py` + `.ipynb`. Match the autogalaxy companion's naming.

2. **Content sections:**
   - **What is a latent variable?** — short, link out to the autofit_workspace foundational tutorial (#4).
   - **Default lensing latents in PyAutoLens** — list the curated set from `autolens/config/latent.yaml`. Explain the science of each:
     - Magnification — what it means, why source-plane errors propagate non-linearly.
     - Effective Einstein radius — via `LensCalc.einstein_radius_jit_from`; what "effective" means here (zero-contour of the deflection-angle field).
     - Lensed source flux / total source flux — image-plane vs source-plane integration.
     - Aperture fluxes — FWHM-defined apertures, what magzero conversion does.
   - **Toggling latents** — same pattern as the autogalaxy tutorial; show the yaml.
   - **Loading latent results** — open a `output/.../latent/` directory, read csv + json, plot a 1D posterior of (say) effective Einstein radius.
   - **Extending with a custom latent** — subclass `al.AnalysisImaging`, override `LATENT_KEYS` + `compute_latent_variables`, run a fit, show the new column. Encourage upstream contributions.

3. **Workspace `config/latent.yaml`** — add under `autolens_workspace/config/`, mirroring the library default. Per memory `feedback_workspace_config_default_true`, workspace values shadow library values.

4. **Update the ad-hoc latent demos** at `autolens_workspace/scripts/guides/results/workflow/csv_make.py:175`, `png_make.py:175`, `fits_make.py:175` if their custom subclass pattern is now redundant (i.e. the library yaml gives the same default set). If still useful as "custom latent" demos, simplify them and link to the new tutorial.

## Where to look

- Library default: `PyAutoLens/autolens/config/latent.yaml` (exists after sub-prompt #2).
- Tutorial style reference: `autolens_workspace/scripts/results/` — pick a comparable results tutorial for prose density.
- Existing ad-hoc latents to audit: `autolens_workspace/scripts/guides/results/workflow/{csv,png,fits}_make.py`.
- Foundational autofit tutorial: URL produced by sub-prompt #4.

## Verification

```bash
source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh
cd autolens_workspace
python scripts/results/latent_variables.py
/smoke_test
```

Manual review: lensing-aware reader (knows what an Einstein radius is) should follow it cold.

## Affected repos

- autolens_workspace (primary)

## Suggested branch

`feature/latent-tutorial-autolens`

## Notes

- Tutorial prose → **Opus** (memory `feedback_tutorial_prose_opus`).
- Per memory `feedback_smoke_tests_small_subset`: do not mass-promote into `smoke_tests.txt`. Smoke coverage lives in sub-prompt #7.
- If `effective_einstein_radius` plots look noisy in 1D posteriors, that may be a downstream signal worth flagging — but **fixing** it is out of scope for this task. Note observations for the next planning round.
- For scientific framing of magnification / Einstein radius / source-plane vs image-plane integration, the lensing wiki at `../PyAutoPaper/lensing_wiki/concepts/` is the authoritative cross-repo source (do NOT cite the wiki from the tutorial itself — see memory `feedback_pyautopaper_personal_repo`).
