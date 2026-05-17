The imaging `features/multi_gaussian_expansion` example needs reviewing before adapting to interferometer.

Once the imaging version is in good shape, adapt it to the interferometer context in
`scripts/interferometer/features/multi_gaussian_expansion/` for **both** `autolens_workspace` and
`autogalaxy_workspace`.

Multi-Gaussian Expansion (MGE) decomposes a galaxy's light into many Gaussian components — until
recently infeasible against visibilities because each Gaussian required its own Fourier transform
per iteration. With nufftax (point to its GitHub and credit it), the full MGE basis is transformed
quickly on GPU, so MGE fits to interferometer data are now practical even with millions of
visibilities. The script should mirror the imaging API explanation and call out the nufftax-enabled
performance shift.
