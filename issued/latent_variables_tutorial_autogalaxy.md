# Add latent-variables workspace tutorial to autogalaxy_workspace

## Context

Parent epic: [`PyAutoPrompt/z_features/latent_refactor.md`](../z_features/latent_refactor.md).
Depends on the library spine in [`autogalaxy/latent_module.md`](../autogalaxy/latent_module.md) and the foundational docs from [`autofit_workspace/latent_variables_tutorial_expand.md`](../autofit_workspace/latent_variables_tutorial_expand.md).

Once PyAutoGalaxy has first-class latent variables (sub-prompt #1), users need a workspace tutorial that:
- Explains what galaxy latents are scientifically (errors on derived quantities, posterior draws of structural params).
- Shows the curated yaml-driven default set and how to toggle individual latents.
- Demonstrates loading and using latent results from a completed search.
- Shows how to **extend** `AnalysisImaging` with a user-defined latent, and encourages contributing useful ones upstream.

## Task

1. **Add a new tutorial** at `autogalaxy_workspace/scripts/results/<filename>.py` (and the matching `.ipynb`). Suggested name: `latent_variables.py` or `latent.py` — confirm against the existing results-tutorial naming convention.

2. **Content sections:**
   - **What is a latent variable?** — short, link out to the autofit_workspace foundational tutorial (#4).
   - **Default latents in PyAutoGalaxy** — list the curated set from `autogalaxy/config/latent.yaml`, explain each.
   - **Toggling latents** — show editing `latent.yaml` (or the workspace override of it, per memory `feedback_workspace_config_default_true`).
   - **Loading latent results** — open a search output, read `latent/samples.csv` + `latent/latent_summary.json`, plot a posterior of one latent.
   - **Extending with a custom latent** — subclass `ag.AnalysisImaging`, override `LATENT_KEYS` and `compute_latent_variables`, run a fit, show the new column appears. End with a paragraph encouraging the user to submit useful custom latents back upstream.

3. **Workspace `config/latent.yaml`** — add it under `autogalaxy_workspace/config/`, mirroring whatever the library default file looks like. Per memory `feedback_workspace_config_default_true`, the workspace value shadows the library, so any latent on by default must also be on in the workspace yaml.

## Where to look

- Library default to mirror: `PyAutoGalaxy/autogalaxy/config/latent.yaml` (will exist after sub-prompt #1).
- Existing workspace-tutorial style references: `autogalaxy_workspace/scripts/results/` — pick a comparable tutorial (e.g. errors/posterior-draws content) and match prose density.
- Existing ad-hoc latent usage to replace: `autogalaxy_workspace/scripts/guides/results/workflow/csv_make.py:140-160`, `png_make.py`, `fits_make.py` — these subclass `ag.AnalysisImaging` to add `bulge.sersic_index`. After sub-prompt #1, this pattern is the "custom latent" path — but the default `sersic_index` etc. may live in the library yaml. Update these scripts only if the boilerplate is no longer needed.
- Foundational autofit tutorial: the URL produced by sub-prompt #4.

## Verification

```bash
source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh
cd autogalaxy_workspace
python scripts/results/latent_variables.py    # or whatever filename
/smoke_test
```

Manual review: a reader who has run `autogalaxy_workspace/scripts/imaging/modeling/start_here.py` once and has a basic understanding of Bayesian fitting should be able to follow this tutorial cold.

## Affected repos

- autogalaxy_workspace (primary)

## Suggested branch

`feature/latent-tutorial-autogalaxy`

## Notes

- Tutorial prose → **Opus** per CLAUDE.md model split (memory `feedback_tutorial_prose_opus`).
- Per memory `feedback_smoke_tests_small_subset`: do **not** mass-promote this new script into `smoke_tests.txt`. The smoke coverage for library latents is sub-prompt #7's job, in `autolens_workspace_test`.
- The `.ipynb` should be generated from the `.py` via the standard workspace nb-build path, not hand-edited.
