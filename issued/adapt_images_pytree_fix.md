Fix `AdaptImages.galaxy_image_dict` Galaxy-identity mismatch across `jax.jit` boundary in
@PyAutoGalaxy, and re-enable the three autogalaxy_workspace_test scripts that this blocks.

__Problem__

When `jax.jit(analysis.fit_from)(instance)` returns a `FitImaging` via the pytree registration
added in PyAutoGalaxy PR #364, accessing `fit.log_likelihood` post-unflatten fails for any model
that uses `AdaptImages`:

```
AttributeError: 'NoneType' object has no attribute 'array'
  File .../rectangular_adapt_image.py", line 93, in mesh_weight_map_from
    mesh_weight_map = adapt_data.array
```

__Root cause__

`FitImaging` is registered with `no_flatten=("dataset", "adapt_images", "settings")`, so
`adapt_images` rides across the pytree boundary as aux — its `galaxy_image_dict` keys are the
**trace-time** `ag.Galaxy` instances. `self.galaxies` is registered as a pytree (dynamic), so
post-unflatten it contains **fresh** `Galaxy` instances built via `autofit.Model.instance_unflatten`
→ `self.cls(*constructor_arguments)`, each with a new `.id`. `hash(galaxy)` returns `int(self.id)`,
so the fresh Galaxy doesn't match any key in `adapt_images.galaxy_image_dict`. The lookup at
`PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:555` raises `KeyError` → `adapt_galaxy_image = None`
→ `mesh.mesh_weight_map_from(adapt_data=None)` blows up.

The analogous fix on the autolens side that solved a similar dict-keyed-by-instance problem is
tracked at `@PyAutoPrompt/autolens/linear_light_profile_intensity_dict_pytree.md`.

Note that autolens's jax_likelihood_functions/imaging/rectangular.py currently passes in
autolens_workspace_test despite apparently having the same Galaxy-identity issue — worth checking
what autolens does differently (e.g. a shared fix in autoarray / autofit, or a subtly different
FitImaging inversion path that bypasses the `galaxy_image_dict` lookup for the adapt mesh). That
diff may reveal the minimal fix for autogalaxy, or a broader pattern that should be lifted into
autoarray.

__Scripts blocked__

From @autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/, these were deferred in
the initial task 3/9 ship (PyAutoGalaxy PR #364, workspace PR on autogalaxy_workspace_test):

- `rectangular_mge.py` — MGE bulge + `ag.mesh.RectangularAdaptImage` + `ag.reg.Adapt`
- `delaunay.py` — `ag.mesh.Delaunay` + `ag.image_mesh.Hilbert` (or `Overlay`, which still wires
  the image-plane mesh grid via `adapt_images.galaxy_name_image_plane_mesh_grid_dict`)
- `delaunay_mge.py` — MGE bulge + Delaunay

The initial ship used `ag.mesh.RectangularUniform` + `ag.reg.Constant` for `rectangular.py` (no
adapt dependency), which does pass. After this fix lands, re-port the three scripts above using
the proper adapt-image autolens references at
`@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/{rectangular,rectangular_mge,
delaunay,delaunay_mge}.py`.

__Deliverables__

1. **PyAutoGalaxy library fix** for the Galaxy-identity issue across the JIT boundary.
   Candidate approaches (pick whichever is cleanest):
   - Key `AdaptImages.galaxy_image_dict` by the galaxy's path-tuple (e.g. `('galaxies', 'galaxy')`)
     instead of the Galaxy instance — stable across unflatten.
   - Look up by `galaxy.id` via an identity map that is rebuilt during `fit_from` (not carried as
     aux).
   - Register `Galaxy` with a custom pytree that preserves `.id` through unflatten so hashes match.
   - Move the adapt-image lookup inside `fit_from` so it runs during tracing (before the pytree
     boundary) and stores `adapt_galaxy_image` on the mapper directly rather than looking it up
     lazily.
2. **Unit test** in `test_autogalaxy/` exercising the fix **without importing JAX** (follow the
   numpy-only unit test convention — cross-xp checks live in workspace_test).
3. **Re-port the three deferred scripts** into autogalaxy_workspace_test using the autolens
   references, and re-enable them in `smoke_tests.txt` alongside the existing jax_likelihood_
   functions/imaging/ entries.
4. Add `jax_likelihood_functions/imaging/delaunay_mge.py` commented out in `smoke_tests.txt` with
   the exact jax-0.7 regression comment from autolens's smoke_tests.txt.

__Dependencies__

- PyAutoGalaxy PR #364 (pytree registration scaffold) must be merged first.
- Cross-check: if the fix in autoarray/autofit applies to autolens too, include an autolens test
  update in the same library PR.

__Umbrella__

Follow-up from PyAutoLabs/autogalaxy_workspace_test#8 (epic #5 task 3/9). Same issue may re-surface
in task 4/9 (`jax_likelihood_interferometer`) and task 5/9 (`jax_likelihood_multi`) — either fix
once here, or carry the same deferral pattern into those tasks.
