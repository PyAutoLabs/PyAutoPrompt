# Mass Profiles Follow-Up

Follow-up issues surfaced by the Phase 1 self-consistency test suite after the
Phase 3 MGE fallback landed. Three independent problems, each a separate task.

## Issue 1: Elliptical MGE Potential Sigma Convention

- [mge_potential_elliptical_fix.md](../autogalaxy/mge_potential_elliptical_fix.md)
- Repos: PyAutoGalaxy
- The `potential_2d_via_mge_from` method works perfectly for spherical profiles
  (errors 1e-4 to 1e-6) but fails for elliptical variants (errors 1–4% for Sersic,
  1100% for Gaussian). The sigma convention or eccentric radii interaction with the
  E1 potential formula needs a different treatment for elliptical profiles — likely
  requiring the numerical quadrature approach used by Jax-Lensing-Profiles (Shajib
  2019 line integral of the sigma function) rather than the circular formula with
  sigma rescaling.
- Affects: Sersic, SersicCore, Exponential, DevVaucouleurs, Chameleon, Gaussian,
  GaussianGradient, cNFW (all elliptical variants)

## Issue 2: Missing xp Threading in convergence_func

- [convergence_func_xp_threading.md](../autogalaxy/convergence_func_xp_threading.md)
- Repos: PyAutoGalaxy
- Several profiles' `convergence_func` methods don't accept the `xp=np` keyword,
  causing the MGE decomposer to crash with TypeError when computing potential.
  Currently caught and SKIPped in the test suite. Need to add `xp=np` to these
  methods and thread xp through.
- Affects: PowerLawBroken, dPIEMass, dPIEPotential, SersicGradient, and
  potentially the abstract `MassProfile.convergence_func` base method

## Issue 3: NFWSph Potential-Deflection Mismatch

- [nfw_sph_potential_mismatch.md](../autogalaxy/nfw_sph_potential_mismatch.md)
- Repos: PyAutoGalaxy
- NFWSph `grad(psi) = alpha` fails at 11% median error despite `laplacian(psi) = 2kappa`
  passing at 1%. The analytic potential and deflection implementations are slightly
  inconsistent. This was found in Phase 1 and persists — it is independent of the
  MGE work and affects the existing analytic NFWSph potential.

## Test Suite Status (after all fixes)

```
78 PASS / 16 FAIL / 32 SKIP (was 56P/1F/69S before Phase 3)
```

16 FAILs: elliptical MGE potential (Issue 1)
32 SKIPs: missing xp (Issue 2), point mass convergence (physical), ExternalShear convergence (physical)
