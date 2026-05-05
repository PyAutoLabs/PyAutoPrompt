Create `scripts/model_composition/` in @autogalaxy_workspace_test with an autogalaxy version of
the autolens `multi_galaxy_mge.py` script.

__Reference__

@autolens_workspace_test/scripts/model_composition/multi_galaxy_mge.py

That script exercises PyAutoFit's model-composition machinery on an autolens lens-plus-source
model with MGE bases. Strip it to an autogalaxy equivalent — multiple `al.Galaxy` objects in a
single plane (no lens/source split, no ray-tracing), each composed from an MGE light basis.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/model_composition/__init__.py`
2. `autogalaxy_workspace_test/scripts/model_composition/multi_galaxy_mge.py` — ported script. Use
   `ag.Galaxy`, `ag.Galaxies`, `ag.ImagingAnalysis` (not `al.` / Tracer). Pick a small dataset
   from an existing autogalaxy workspace example and model two galaxies each with an MGE.
3. Append `model_composition/multi_galaxy_mge.py` to `smoke_tests.txt`.
4. Verify locally with `PYAUTOFIT_TEST_MODE=2 PYAUTO_WORKSPACE_SMALL_DATASETS=1 python
   scripts/model_composition/multi_galaxy_mge.py`.

__Depends on__

Task 1 (CI) must have merged so the new script runs in GitHub Actions. Review failure is cheaper
on CI than locally once CI is wired up.

__Umbrella issue__

Task 2/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
