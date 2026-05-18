Follow-up to mass_profiles.md (issue #178, shipped via #179): add a
`scripts/guides/profiles/light_and_mass_profiles.py` guide to
`autolens_workspace`, paired with the existing `light.py` (#86/#87) and
`mass.py` (#179) guides in the same folder.

Scope (per user direction):

- Stellar mass profiles in `al.mp.*` (Sersic, Chameleon, Gaussian,
  GaussianGradient, SersicCore, SersicGradient, DevVaucouleurs,
  Exponential + `*Sph` variants).
- Dark-matter mass profiles in `al.mp.*` (NFW family: NFW, gNFW, cNFW,
  NFWTruncated, plus their MCR / virial-mass / scatter variants).
- Combined light-and-mass profiles in `al.lmp.*` (one object emitting
  both `image_2d_from` and `convergence_2d_from` via a shared
  `mass_to_light_ratio`).
- Linear combined light-and-mass profiles in `al.lmp_linear.*` —
  include alongside the standard `lmp` profiles.

Section flow mirrors `light.py` / `mass.py` so the three guides read as
a coherent set.
