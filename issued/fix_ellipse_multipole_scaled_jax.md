User-reported bug: `EllipseMultipoleScaled` (`@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse_multipole.py:118-168`) is not JAX-traceable. HPC job using `ag.EllipseMultipoleScaled(m=3 or 4, scaled_multipole_comps=Prior, major_axis=...)` inside an `af.Model` fails when `AnalysisEllipse(use_jax=True)` (now default after #412) traces the model.

Root cause: `EllipseMultipoleScaled.__init__` does derivation work at construction time, calling `convert.multipole_k_m_and_phi_m_from(scaled_multipole_comps, m)` and `convert.multipole_comps_from(k_adjusted, phi, m)` **without `xp=xp`**. The convert helpers accept `xp=np` (fixed in PR #412) but `__init__` has no `xp` argument to thread, so the calls default to numpy. When `instance_from_vector(jax_array)` constructs the instance with JAX tracers in `scaled_multipole_comps`, the numpy calls raise `TracerArrayConversionError`.

Secondary issue: even if `xp` could be threaded into `__init__`, storing the derived `specific_multipole_comps` at construction time is wrong under `vmap`. The pytree machinery flattens `__init__`-stored attributes; under `vmap`, different batch elements need fresh derivations but get the cached one from the constructing tracer.

`EllipseMultipole` (non-scaled) is unaffected — its `__init__` just stores `multipole_comps` directly, no derivation.

The workspace_test JAX parity scripts in `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/multipoles.py` cover `EllipseMultipole` only, not `EllipseMultipoleScaled` — that's how this gap shipped silently.

Please:

1. Move the derivation out of `EllipseMultipoleScaled.__init__` and into `points_perturbed_from`. The `__init__` should just store `self.scaled_multipole_comps`, `self.major_axis`, `self.m` and skip the `multipole_k_m_and_phi_m_from` / `multipole_comps_from` calls. Don't call `super().__init__(m, specific_multipole_comps)` with a pre-derived value — derive on-the-fly inside `points_perturbed_from`.

2. In `points_perturbed_from`, the current code does a round-trip: `__init__` builds `specific_multipole_comps` from (k, phi); then `points_perturbed_from` calls `multipole_k_m_and_phi_m_from(specific_multipole_comps, ...)` to extract (k_orig, phi_orig) back. Collapse this round-trip by computing (k_adjusted, phi) directly from `scaled_multipole_comps` once at the top of `points_perturbed_from`:

   ```python
   def points_perturbed_from(self, pixel_scale, points, ellipse, n_i=0, xp=np):
       k_scaled, phi = multipole_k_m_and_phi_m_from(
           multipole_comps=self.scaled_multipole_comps, m=self.m, xp=xp
       )
       k = k_scaled * self.major_axis

       symmetry = 360.0 / self.m
       comps_adjusted = multipole_comps_from(
           k,
           symmetry - 2 * phi + (symmetry - (ellipse.angle(xp=xp) - phi)),
           self.m,
           xp=xp,
       )

       theta = xp.arctan2(points[:, 0], points[:, 1])
       delta_theta = self.m * (theta - ellipse.angle_radians(xp=xp))
       radial = comps_adjusted[1] * xp.cos(delta_theta) + comps_adjusted[0] * xp.sin(delta_theta)

       x = points[:, 1] + radial * xp.cos(theta)
       y = points[:, 0] + radial * xp.sin(theta)
       return xp.stack(arrays=(y, x), axis=-1)
   ```

3. Grep the repo for `.specific_multipole_comps` and `.multipole_comps` accessed on an `EllipseMultipoleScaled` instance. If anyone reads those, they need to either be turned into `@property`s that recompute on-the-fly (numpy-only, since they go to plotting/aggregation) OR have their callers updated to call the new derivation path. Likely no external readers — `EllipseMultipoleScaled` is internal.

4. Add a parity test in `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/`:
   - Either add a new `multipoles_scaled.py` covering the `EllipseMultipoleScaled` path with the full `fitness._vmap` + `jax.jit(fit_from)` round-trip blocks
   - Or extend the existing `multipoles.py` with a `__Scaled Multipoles__` section testing `EllipseMultipoleScaled` alongside `EllipseMultipole`
   The point is closing the gap: every multipole variant gets the vmap-validation bar going forward.

5. Test bar:
   - `pytest test_autogalaxy/ellipse/ -v` — 32/32 still pass (numpy semantics unchanged)
   - `pytest test_autogalaxy/ -x` — 870/870 still pass
   - The new workspace_test script(s) complete the vmap + JIT round-trip with `rtol=1e-4` parity
   - Reference numbers for the existing `multipoles.py` script byte-stable (no `EllipseMultipoleScaled` there, so unaffected)

This is a tight follow-up to PR #411/#412 (the keystone of `ellipse_fitting_jax`). The bug was technically present before the feature — `EllipseMultipoleScaled.__init__` always had the issue — but the feature flipping `use_jax=True` to default exposed it.

Out-of-scope: the `power_law_multipole.py` (mass-profile) call sites flagged in PR #411's session notes have the same `multipole_comps_from` without `xp` threading issue at their call sites; not in scope here, separate prompt when mass-multipole JAX support is needed.
