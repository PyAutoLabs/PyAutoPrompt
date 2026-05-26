# Mass Profiles Refactor

Epic refactoring of `autogalaxy/profiles/mass/` — self-consistency testing, CSE JAX port,
MGE/CSE fallback for zero-returning profiles, spring clean, and LaTeX documentation.

## Phases

### Phase 1: Self-Consistency Test Suite
- [mass_self_consistency_tests.md](../autolens_workspace_test/mass_self_consistency_tests.md)
- Repos: autolens_workspace_test (primary), PyAutoGalaxy (reference)
- Build `scripts/mass/` with per-category scripts verifying lensing relations
  (div(alpha)=2kappa, grad(psi)=alpha, laplacian(psi)=2kappa) for every profile

### Phase 2: CSE Module JAX Port
- [cse_jax_port.md](../autogalaxy/cse_jax_port.md)
- Repos: PyAutoGalaxy
- Thread `xp=np` through all CSE methods, replace scipy.lstsq with JAX-compatible path
- Depends on: Phase 1

### Phase 3: MGE/CSE Fallback Mechanism
- [mge_cse_fallback.md](../autogalaxy/mge_cse_fallback.md)
- Repos: PyAutoGalaxy
- Replace 16 zero-returning convergence/potential methods with MGE/CSE decomposition fallbacks
- Depends on: Phase 2

### Phase 4: Spring Clean
- [mass_profiles_spring_clean.md](../autogalaxy/mass_profiles_spring_clean.md)
- Repos: PyAutoGalaxy
- xp threading audit, dead code removal, decorator consistency, parameter naming
- Depends on: Phase 3

### Phase 5: LaTeX Documentation
- [mass_profiles_documentation.md](../autogalaxy/mass_profiles_documentation.md)
- Repos: PyAutoGalaxy
- First-class docstrings with math, paper references, and parameter units for all profiles
- Update docs/api/mass.rst for completeness
- Depends on: Phase 4

## Zero-Returning Profiles (Phase 3 targets)

| Profile | convergence_2d_from | potential_2d_from |
|---------|:-------------------:|:-----------------:|
| ExternalShear | zeros (correct) | — |
| PowerLawMultipole | zeros (correct) | — |
| cNFW / cNFWSph | zeros | zeros |
| NFWTruncatedSph | — | zeros |
| PointMass | — | zeros |
| MassSheet | — | zeros |
| Chameleon | — | zeros |
| Gaussian | — | zeros |
| Sersic (AbstractSersic) | — | zeros |
| dPIEPotential / Sph | — | zeros |
| PowerLawBroken | — | zeros |
| dPIEMass | — | zeros |
