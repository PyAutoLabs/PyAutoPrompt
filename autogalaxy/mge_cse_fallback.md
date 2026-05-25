Implement MGE/CSE fallback for mass profiles that currently return zeros for convergence or potential.

## Goal

16 mass profiles in PyAutoGalaxy currently return `np.zeros` / `xp.zeros` for their `convergence_2d_from` or `potential_2d_from` methods. These should instead fall back to computing the quantity via MGE (Multi-Gaussian Expansion) or CSE (Cored Steep Ellipsoid) decomposition.

## Design

The fallback hierarchy for any lensing quantity is:
1. **Analytic** — if the profile has a direct implementation, use it (current behavior for non-zero profiles)
2. **MGE** — decompose the profile's convergence into Gaussians, compute the quantity from the Gaussian sum (preferred — already JAX-ready)
3. **CSE** — decompose via cored steep ellipsoids (Oguri 2021), compute from CSE sum

### Implementation Approach

Add fallback methods to the `MassProfile` base class (`@PyAutoGalaxy/autogalaxy/profiles/mass/abstract/abstract.py`):

- `convergence_2d_via_mge_from(grid)` — uses MGEDecomposer to decompose `self.convergence_func` into Gaussians, then evaluates convergence from the decomposition
- `potential_2d_via_mge_from(grid)` — same, but for potential (integrate the MGE convergence to get potential)
- `convergence_2d_via_cse_from(grid)` — uses CSE decomposition (for profiles that inherit MassProfileCSE)
- `potential_2d_via_cse_from(grid)` — same for potential via CSE

Then replace each zero-returning method with a call to the appropriate fallback. The choice of MGE vs CSE depends on:
- Whether the profile has a `convergence_func` (needed for MGE decomposition) — most do
- Whether the profile already mixes in `MassProfileCSE` — dark matter profiles typically do

### Profiles to Fix

**convergence_2d_from returning zeros:**
- `ExternalShear` — convergence is physically zero for a pure shear field; this is CORRECT, do NOT add fallback
- `PowerLawMultipole` — convergence is physically zero for a pure multipole perturbation; this is CORRECT, do NOT add fallback
- `cNFW`, `cNFWSph` — needs MGE or CSE fallback

**potential_2d_from returning zeros:**
- `PointMass` — analytic potential exists (ψ = R_E² ln(r)); implement directly
- `NFWTruncatedSph` — use CSE fallback
- `cNFW`, `cNFWSph` — use CSE or MGE fallback
- `MassSheet` — analytic potential exists (ψ = ½κ_ext r²); implement directly
- `Chameleon` — use MGE fallback
- `Gaussian` — use MGE fallback (Gaussian is itself an MGE basis function)
- `Sersic` (AbstractSersic) — use MGE fallback
- `dPIEPotential`, `dPIEPotentialSph` — check if analytic form exists in literature; otherwise MGE fallback
- `PowerLawBroken` — use MGE fallback
- `dPIEMass` — check if analytic form exists; otherwise MGE fallback

### Key Decisions

- For `ExternalShear` and `PowerLawMultipole`, the zero convergence is physically correct — document this clearly but do NOT replace with a fallback.
- For `PointMass` and `MassSheet`, implement the analytic formula directly rather than using decomposition — these have trivial closed-form potentials.
- For everything else, prefer MGE (it's JAX-ready and more general).

## Verification

Run the Phase 1 self-consistency test suite after implementation. All profiles that previously showed SKIP/FAIL for potential or convergence should now show PASS (except ExternalShear and PowerLawMultipole convergence, which remain correctly zero).

## Repos

- @PyAutoGalaxy (primary)
