
workspaces/[interferometer.md](../workspaces/interferometer.md)
workspaces/[interferometer_linear_light_profiles.md](../workspaces/interferometer_linear_light_profiles.md)
workspaces/[interferometer_multi_gaussian_expansion.md](../workspaces/interferometer_multi_gaussian_expansion.md)
workspaces/[interferometer_extra_galaxies.md](../workspaces/interferometer_extra_galaxies.md)
workspaces/[interferometer_shapelets.md](../workspaces/interferometer_shapelets.md)

shipped (sibling work, not direct prompts above): `interferometer/features/pixelization`, `interferometer/features/subhalo` (autolens), `interferometer/features/extra_galaxies` (autolens), `interferometer/features/datacube` (autolens)

# Dropped sub-prompts (do not apply to interferometer)

The original tracker listed three additional sub-prompts that were dropped on 2026-05-17 without
being issued, because each has a structural reason it doesn't transfer to interferometer modeling:

- `interferometer_double_einstein_ring.md` — two-plane lens systems exist in principle for any data
  type, but no known mm/sub-mm-detected double Einstein ring exists. Not worth the workspace tutorial
  maintenance burden until a real ALMA double-ring is observed.
- `interferometer_mass_stellar_dark.md` — stellar + dark decomposition requires constraints on the
  lens-galaxy stellar light (M/L tie) to break degeneracies. Interferometer data does not detect the
  lens-galaxy light (the central interferometer convention this tracker keeps repeating), so the
  decomposition collapses to a single dark-matter total-mass component — which is what existing
  interferometer modeling scripts already use (`Isothermal` / `PowerLaw` / `NFW`).
- `interferometer_scaling_relation.md` — same problem as `mass_stellar_dark`. Scaling relations tie
  mass to luminosity; without lens-galaxy light, there's no luminosity to anchor the relation. Would
  require multi-wavelength HST+interferometer joint modeling, which is a separate feature.

# Tracker complete

5 shipped (linear_light_profiles, multi_gaussian_expansion, extra_galaxies, shapelets, and the
original `interferometer.md` nufftax-updates rollout) + 3 dropped (above) = all 8 sub-prompts in the
original epic resolved.
