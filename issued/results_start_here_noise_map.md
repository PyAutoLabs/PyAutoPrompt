# Autogalaxy results start-here noise-map failure

## Original Request

continue

## Context

After rerunning the full PyAutoBuild release-prep checks on
`2026-06-09T18-03-42Z`, one active failure is in
`autogalaxy_workspace/scripts/guides/results/start_here.py`.

## Current Failure

Primary repo: @autogalaxy_workspace

- `@autogalaxy_workspace/scripts/guides/results/start_here.py`
  - Fails when loading `image/dataset.fits` from the saved result folder.
  - `ag.Imaging.from_fits(...)` rejects the saved noise-map HDU because every
    value is zero.
  - PyAutoBuild correlates this failure with the recent
    `fix: reload saved result dataset with unchecked noise map` workspace PR.

## Proposed Scope

Fix the results guide so the simple-loading section can load the saved dataset
artifact from `output/` without tripping strict noise-map validation, then rerun
the target script. Treat this as workspace-only unless investigation shows the
saved FITS output itself should change in a library.
