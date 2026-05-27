Investigate and fix the NFWSph potential-deflection mismatch.

## Problem

NFWSph `grad(psi) = alpha` fails at 11% median error despite
`laplacian(psi) = 2kappa` passing at 1%. This means the analytic potential and
deflection implementations are slightly inconsistent with each other, even though
each is individually self-consistent with the convergence.

Found in Phase 1 (autolens_workspace_test#124) and persists after all Phase 3
fixes. This is independent of the MGE work — it affects the existing analytic
NFWSph potential implementation.

## Diagnostic

The potential satisfies laplacian(psi)=2kappa to 1%, so the potential is
consistent with the convergence. The deflections also satisfy div(alpha)=2kappa.
But grad(psi) != alpha at 11%. This suggests a constant scaling factor or
additive offset difference between the potential and deflection implementations.

## Approach

1. Compare the analytic NFWSph potential formula against the integral-based
   reference in `@autolens_workspace_test/scripts/mass_via_integral/nfw.py`
2. Check for missing factors of 2, pi, or axis ratio in the potential formula
3. Check the `potential_func` integral kernel used by `potential_2d_from` against
   the `deflection_func` used by `deflections_yx_2d_from`
4. The Jax-Lensing-Profiles (Herculens) NFW implementation may serve as a
   cross-reference

## Repos

- @PyAutoGalaxy (primary — dark/nfw.py, dark/abstract.py)
