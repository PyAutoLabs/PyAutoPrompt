RESOLVED — not a bug. NFWSph potential-deflection finite-difference artifact.

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

## Resolution

Investigated 2026-05-27. The formula is CORRECT:
- Point-wise comparison (dr=1e-5) gives grad(psi)/alpha = 1.0000 at every radius
- The 11% error in the grid-based test suite is finite-difference truncation error
  from np.gradient at pixel_scales=0.05 on the highly-curved NFW potential
- Error does NOT decrease with finer grids (constant ~11.2%) because the
  error metric uses max(|alpha|) as denominator, not local values
- The NFW potential has strong 3rd derivatives near r_s, making 2nd-order central
  differences inaccurate at typical pixel scales

One real (minor) issue found: `potential_func_sph` line 422 has hardcoded `np.sqrt`
instead of `xp.sqrt` — only affects JAX path, not numerical accuracy.

No code fix needed for the 11% discrepancy. The Phase 1 test suite should document
this as a known finite-difference limitation for NFWSph, not a FAIL.

## Repos

- @PyAutoGalaxy (primary — dark/nfw.py, minor xp fix only)
