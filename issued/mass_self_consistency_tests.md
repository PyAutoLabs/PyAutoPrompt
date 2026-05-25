Build a comprehensive self-consistency test suite for every mass profile in PyAutoGalaxy.

## Goal

Create `scripts/mass/` in @autolens_workspace_test with scripts that verify the internal mathematical consistency of every mass profile independently of the source code implementation. This is the safety net for the broader mass profiles refactoring effort (z_features/mass_profiles_refactor.md).

## What to Build

One script per mass profile category under `scripts/mass/`:
- `total.py` — Isothermal, IsothermalCore, PowerLaw, PowerLawCore, PowerLawBroken, PowerLawMultipole, PIEMass, dPIEMass, dPIEPotential (+ Sph variants)
- `dark.py` — NFW, gNFW, cNFW, NFWTruncated (+ MCR/Scatter/Virial variants, + Sph variants)
- `stellar.py` — Sersic, SersicCore, SersicGradient, Gaussian, GaussianGradient, Exponential, DevVaucouleurs, Chameleon (+ Sph variants)
- `sheets.py` — ExternalShear, ExternalPotential, MassSheet
- `point.py` — PointMass, SMBH, SMBHBinary

Each script should:

1. Instantiate every profile in its category with physically representative parameters.

2. For each profile, compute convergence, potential, and deflections via the profile's own methods on a sufficiently large, fine grid (e.g. 100x100, pixel_scales=0.05).

3. Verify internal consistency using generic lensing relations computed via numerical differentiation (independent of source code):
   - `div(alpha) = 2*kappa` — divergence of deflections equals twice the convergence
   - `grad(psi) = alpha` — gradient of potential equals deflections
   - `laplacian(psi) = 2*kappa` — Laplacian of potential equals twice the convergence
   - Shear from Hessian of potential vs. from deflection field derivatives

4. Use `np.gradient` on the native 2D grid for numerical differentiation. Compare against the profile's own methods with tolerances appropriate for finite differences (e.g. `rtol=1e-2` for central regions, excluding edge pixels).

5. For profiles that currently return zeros for convergence or potential (see the 16 profiles listed in z_features/mass_profiles_refactor.md), **document the expected failure** with a clear comment and `try/except` or conditional skip — do not let known-zero methods cause the script to crash. Print a summary table at the end showing PASS/FAIL/SKIP per profile per relation.

6. Cross-check against `mass_via_integral/` reference implementations where they exist (NFW, Sersic, Gaussian, gNFW, SersicGradient, gNFWVirialMassConc).

7. Support `PYAUTO_MASS_FAST` environment variable: when set, use smaller grids (e.g. 40x40) and looser tolerances (e.g. `rtol=5e-2`) for quick CI-style validation.

## Design Notes

- Use `ag.Grid2D.uniform(shape_native=(...), pixel_scales=0.05)` for uniform grids.
- Access the raw numpy array via `.native` for 2D reshaping before `np.gradient`.
- Exclude boundary pixels (2-pixel border) from comparisons — finite differences are unreliable at edges.
- For vector quantities (deflections), compare component-wise (y and x separately).
- Each script should be runnable standalone: `python scripts/mass/total.py`.
- Follow the workspace script docstring style (triple-quoted prose blocks with `__Section Name__` headers).

## Repos

- @autolens_workspace_test (primary — new scripts go here)
- @PyAutoGalaxy (read-only — reference for profile classes and parameters)
