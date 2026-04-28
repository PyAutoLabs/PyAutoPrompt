After PyAutoGalaxy `AnalysisImaging` shipped JAX pytree registration for
`FitImaging` + `DatasetModel` + `Galaxies` (PR #364, 2026-04-22), three sibling Analysis classes
were left without equivalent registration:

- `@PyAutoGalaxy/autogalaxy/interferometer/model/analysis.py::AnalysisInterferometer`
- `@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py::AnalysisEllipse`
- `@PyAutoGalaxy/autogalaxy/quantity/model/analysis.py::AnalysisQuantity`

Setting `use_jax_for_visualization=True` on any of these will fail the moment the visualizer
returns a `FitInterferometer` / `FitEllipse` / `FitQuantity` from a jitted `fit_from`, because
the return type isn't a registered JAX pytree.

__Reference patterns__

PyAutoLens has the equivalent registrations for the interferometer dataset type — mirror it on
the autogalaxy side:

- `@PyAutoLens/autolens/interferometer/model/analysis.py::AnalysisInterferometer._register_fit_interferometer_pytrees`
  (lines 177, 196–214) registers `FitInterferometer`, `DatasetModel`, `Tracer (no_flatten=("cosmology",))`.
  The PyAutoGalaxy version registers `Galaxies` instead of `Tracer`.

PyAutoGalaxy's existing `autogalaxy/imaging/model/analysis.py::_register_fit_imaging_pytrees`
(lines 169–207) is the closest in-repo template — it registers the `Galaxies` list container
with explicit `_flatten_galaxies` / `_unflatten_galaxies` helpers because
`register_instance_pytree` alone drops list contents. Any of the three new analyses that hold
a `Galaxies` will need the same helper — re-export from a shared location or duplicate.

__What to do__

For each of `AnalysisInterferometer`, `AnalysisEllipse`, `AnalysisQuantity`:

1. Add `_register_fit_*_pytrees()` to the class. Call it from `__init__` under the existing
   `use_jax` gate (mirror autogalaxy `AnalysisImaging` line 147).
2. Register the dataset's `Fit*` class via `register_instance_pytree`. For the inversion-bearing
   ones (Interferometer), check whether the inversion solver state needs `no_flatten=` — see
   the PyAutoLens reference.
3. Register `DatasetModel` via `register_instance_pytree(DatasetModel)`. Confirm the underlying
   helper is idempotent (PyAutoLens AnalysisImaging and AnalysisInterferometer both register it,
   so it is — but verify before assuming).
4. Register the model container the analysis fits with — `Galaxies` for Interferometer/Quantity,
   the appropriate isophote container for Ellipse.
5. Add a workspace_test pilot per dataset type:
   `autogalaxy_workspace_test/scripts/{interferometer,ellipse,quantity}/visualization_jax.py`
   matching the existing `imaging/visualization_jax.py` pattern. Each should pass
   `use_jax_for_visualization=True` and assert `fit.png` is produced.

__Spawn-off awareness__

If any profile / dataset class reachable from the new `Fit*` types isn't yet pytree-friendly,
follow the established spawn-off pattern: stop, open a per-class registration issue, ship that
PR first. Do **not** paper over with ad-hoc `register_pytree_node` calls inside the workspace
script. See `issued/autogalaxy_workspace_test_jax_likelihood_imaging.md` for the pattern.

`AnalysisEllipse` may surface deeper blockers — its `FitEllipse` doesn't carry the same
`Galaxies` structure as the imaging/interferometer analyses. If registration turns out
non-trivial, scope it out into its own issue rather than forcing it into this task.

__Scope boundary__

- Eager-JAX path only (Path C in the original `issued/fit_imaging_pytree.md` framing). Do not
  attempt full `jax.jit` wrapping — that's gated on the Path A feasibility study.
- No production workspace adoption — the test workspace pilots are sufficient verification.
- This task assumes the visualizer dispatch fix from
  `autogalaxy/visualizer_fit_for_visualization_dispatch.md` has shipped first; otherwise the
  pilot scripts will pass without actually exercising the jit path.

__Verification__

- `JAX_ENABLE_X64=True python autogalaxy_workspace_test/scripts/{type}/visualization_jax.py`
  for each new pilot.
- Add the three new pilots to `autogalaxy_workspace_test/smoke_tests.txt`.

__Background__

Original feature: `complete.md` entries `jax-visualization` and `mge-jit-visualization`
(both 2026-04-19). Imaging pytree registration on autogalaxy: `complete.md` entry for the
autogalaxy_workspace_test imaging port (2026-04-22). The `_register_fit_imaging_pytrees`
spawn-off was originally flagged in `issued/autogalaxy_workspace_test_jax_likelihood_imaging.md`
under "Pytree prerequisite — likely blocker".
