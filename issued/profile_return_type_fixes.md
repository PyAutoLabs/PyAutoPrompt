Two library bugs surfaced while writing
`autolens_workspace/scripts/guides/profiles/light.py` (#86 / #176) and
`mass.py` (#178 / #179):

1. **`Basis.image_2d_from` returns a raw `numpy.ndarray`** instead of
   an `Array2D` when the basis is composed of `LightProfileLinear`
   constituents. The light.py guide had to wrap the basis in a Galaxy
   and plot via the galaxy's image to make `aplt.plot_array` work.

   Root cause: `Basis.image_2d_list_from` returns `xp.zeros((grid.shape[0],))`
   for `LightProfileLinear` profiles (because their intensity will be
   solved by the inversion later). Summing those zero ndarrays in
   `image_2d_from` produces a raw ndarray rather than an `Array2D`.

2. **`dPIEPotential.convergence_2d_from` returns a `VectorYX2D`**
   instead of an `Array2D`. The mass.py guide had to skip plotting
   convergence for `dPIEPotential` and use `dPIEPotentialSph` in the
   walkthrough instead.

   Root cause: a typo / copy-paste of the deflections decorator —
   `@aa.decorators.to_vector_yx` is used where `@aa.decorators.to_array`
   is correct. The function body returns a scalar field
   (`kappa_circ * (1 - asymm_term) + (alpha_circ / grid_radii) * asymm_term`)
   which is not a `(y, x)` vector.

Both fixes are tiny library-code changes with regression tests.
