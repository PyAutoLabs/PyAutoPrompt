The autogalaxy imaging `features/extra_galaxies` example needs adapting to interferometer.

Adapt it to the interferometer context in
`autogalaxy_workspace/scripts/interferometer/features/extra_galaxies/`. The `autolens_workspace`
already has an interferometer port at `scripts/interferometer/features/extra_galaxies/` — use it as
a structural template for `modeling.py` and `simulator.py`, stripped of the lens-mass aspects since
autogalaxy is for non-lensing morphology fits.

Modeling extra (perturber / line-of-sight companion) galaxies in the field works identically for
imaging and visibility data once light profile transforms are fast — which they now are thanks to
nufftax. The script should explain the autogalaxy use case (multiple galaxies in a field of view,
not lensing) and how the visibility-domain fit proceeds.
