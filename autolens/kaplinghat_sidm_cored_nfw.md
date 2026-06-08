# Kaplinghat SIDM cored NFW profile

Original user request, verbatim:

> can you come up with a plan for tackling this issue: https://github.com/PyAutoLabs/PyAutoLens/issues/564
>
> ok begin development, you know to use the start_dev workflow right

Implement the existing PyAutoLens issue:

- https://github.com/PyAutoLabs/PyAutoLens/issues/564
- Title: Add a Kaplinghat, Tulin & Yu (2016) isothermal-Jeans cored-NFW dark matter mass profile

Relevant repositories:

- @PyAutoGalaxy: mass-profile implementation and tests live under `autogalaxy/profiles/mass/dark/`.
- @PyAutoLens: validate export/use through `al.mp.*` and update substructure batching if required.
- @autolens_assistant: consult project/profile, dark-matter, substructure, mass-profile, and custom-profile context before finalising the development plan.

High-level goal:

Add a spherical SIDM-motivated Kaplinghat, Tulin & Yu 2016 cored-NFW mass profile that can be used in PyAutoLens substructure and dark-matter analyses. The profile should reduce to NFW in the zero-interaction limit and support the same core lensing methods expected of existing dark-matter profiles.

Important planning notes:

- Treat this as source-library work, likely PyAutoGalaxy first with PyAutoLens validation.
- Resolve the physical API before implementation: the proposed direct `kappa_s` / `scale_radius` constructor may be under-specified for `sigma_over_m` unless physical density and velocity scales are available. The MCR/Ludlow constructor may need to be the primary physically meaningful path.
- The profile is scientifically motivated by SIDM subhalo concentration/core tests, not merely by a phenomenological cored profile.
