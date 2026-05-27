Fix the elliptical MGE potential in `potential_2d_via_mge_from`.

## Problem

The MGE potential works perfectly for spherical profiles (errors 1e-4 to 1e-6) but
fails for elliptical variants:

- Sersic/SersicCore/Exponential: ~4% median error on grad(psi)=alpha
- DevVaucouleurs: ~3% median
- Chameleon: ~1% median
- Gaussian/GaussianGradient: ~1100% median (catastrophic)
- cNFW (elliptical): ~5% median

The circular E1 potential formula with sigma rescaling (the approach used for
convergence and deflections) does not give the correct potential for elliptical
profiles.

## Root Cause

The potential of an elliptical Gaussian convergence does NOT have a simple
closed-form like the circular case. The Jax-Lensing-Profiles repo (Herculens)
uses numerical quadrature of a line integral involving the sigma function
(Faddeeva-based, from Shajib 2019 Eq. 4.15) for the elliptical case.

Our `convergence_2d_via_mge_from` and `deflections_2d_via_mge_from` work for
elliptical profiles because the Faddeeva function approach handles ellipticity
natively. But `potential_2d_via_mge_from` uses the circular E1 formula with
sigma rescaling, which doesn't correctly capture the elliptical potential.

## Approach

Two options:

1. **Numerical quadrature** (like Jax-Lensing-Profiles): integrate the deflection
   sigma function along a line from (0,0) to (x,y) in the profile frame. This is
   exact but expensive and harder to make JAX-compatible.

2. **Compute potential from deflections numerically**: since we have correct
   elliptical deflections via MGE, we can integrate alpha along radial lines to
   get the potential. This reuses existing correct code.

Either way, consult `@PyAutoPaper/lensing_wiki/` for the Shajib 2019 potential
formula and the Jax-Lensing-Profiles implementation at
https://github.com/Herculens/Jax-Lensing-Profiles (gaussian_ellipse_kappa.py,
multi_gaussian_ellipse_kappa.py).

## Verification

Run the Phase 1 self-consistency test suite (scripts/mass/*.py in
@autolens_workspace_test). All elliptical profiles should show PASS for
grad(psi)=alpha and laplacian(psi)=2kappa.

## Repos

- @PyAutoGalaxy (primary — mge.py)
