The imaging shapelets example needs improving and padding out before adapting to interferometer.
Source paths differ between repos: `autolens_workspace/scripts/imaging/features/advanced/shapelets/`
and `autogalaxy_workspace/scripts/imaging/features/shapelets/`.

Once the imaging versions are more complete, adapt to interferometer in **both** repos at the
matching paths: `autolens_workspace/scripts/interferometer/features/advanced/shapelets/` and
`autogalaxy_workspace/scripts/interferometer/features/shapelets/`.

Shapelets are a polar / Gauss-Hermite basis for galaxy morphology that previously was prohibitively
slow against visibilities (each basis component needs its own Fourier transform per iteration).
With nufftax, the full shapelet basis can be transformed in batches on GPU, making this feature
practical for interferometer modeling. The script should explain the basis, the visibility-domain
fit, and credit nufftax for the performance shift.
