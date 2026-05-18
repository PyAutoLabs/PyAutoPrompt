Workspace follow-up to PyAutoGalaxy #425
(`profile-return-type-fixes`): now that
`Basis.image_2d_from` and `dPIEPotential.convergence_2d_from` return
the correct wrapper types, the Galaxy-wrap and Sph-substitute
workarounds in the `scripts/guides/profiles/` guides can be removed.

While auditing the workaround removal, the Basis demo was also found
to plot an all-zeros map (an MGE of `ag.lp_linear.Gaussian` constituents
has no intensities yet — the inversion would solve those at fit time,
but in the standalone demo the image is just zeros). Switching the
demo to use standard `ag.lp.Gaussian` constituents with explicit
intensities produces a meaningful MGE plot, and a follow-on note
explains that you'd use `ag.lp_linear.Gaussian` in an actual fit.

Three small edits:

1. `autogalaxy_workspace/scripts/guides/profiles/light.py` — Basis
   section: swap `lp_linear.Gaussian` for `lp.Gaussian` with explicit
   intensities; drop the Galaxy wrap; plot `basis.image_2d_from(grid)`
   directly; update the prose to reflect the inversion-vs-explicit
   framing.
2. `autolens_workspace/scripts/guides/profiles/light.py` — same edit,
   `al.*` namespace.
3. `autolens_workspace/scripts/guides/profiles/mass.py` — Remaining
   Walkthrough: add or swap to `al.mp.dPIEPotential` (the elliptical
   variant) now that its `convergence_2d_from` returns `Array2D`.
