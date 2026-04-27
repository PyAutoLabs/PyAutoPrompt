Create `scripts/jax_likelihood_functions/imaging/` in @autogalaxy_workspace_test with autogalaxy
ports of every autolens JAX-likelihood imaging script, **excluding** the `*_dspl.py` double-source-
plane variants (lens-specific, no autogalaxy analogue).

__Scripts to port__

From @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/:

- `simulator.py`
- `lp.py` (parametric light profile)
- `mge.py` (MGE basis)
- `mge_group.py`
- `rectangular.py` (rectangular pixelization source)
- `rectangular_mge.py`
- `delaunay.py`
- `delaunay_mge.py` — currently disabled in autolens smoke suite (jax 0.7 regression,
  see PyAutoPrompt/autobuild/smoke_workspace_fixes.md). Ship it but disable with the same
  comment in `smoke_tests.txt`.

**Skip**: `rectangular_dspl.py`, `simulator_dspl.py`.

__Pytree prerequisite — likely blocker__

`autolens/imaging/model/analysis.py::AnalysisImaging._register_fit_imaging_pytrees` registers
`FitImaging`, `DatasetModel`, and `Tracer` with `autoarray.abstract_ndarray.register_instance_pytree`.
**@PyAutoGalaxy/autogalaxy/imaging/model/analysis.py has no such method today.** Before the
JAX likelihood scripts will JIT, you will need a library PR on PyAutoGalaxy that:

1. Adds `_register_fit_imaging_pytrees` to `autogalaxy.imaging.model.analysis.AnalysisImaging`.
2. Registers the autogalaxy equivalents: `FitImaging` (autogalaxy's), `DatasetModel`, `Galaxies`
   (with `no_flatten=("cosmology",)` if it holds cosmology the way `Tracer` does — check first).
3. Calls it from `__init__` under the same `use_jax` gate autolens uses.

Treat this as a **spawn-off library task** if it surfaces: stop, open a PyAutoGalaxy issue via
`/start_dev`, ship the library PR first, then resume. Do not paper over missing registrations
with ad-hoc `register_pytree_node` calls inside the workspace script.

Other known spawn-offs if they surface during porting:

- **Linear light profile** models need `linear_light_profile_intensity_dict_pytree` fixed — see
  @PyAutoPrompt/autolens/linear_light_profile_intensity_dict_pytree.md for the lens-side
  counterpart. Only blocks scripts that use `ag.lp_linear.*` or MGE bases via
  `fit_for_visualization`, not the scalar `fit_from` round-trip.
- Any autogalaxy profile that isn't pytree-registered (follow the per-profile pattern in
  @PyAutoPrompt/autolens/fit_imaging_pytree_*.md).

__Three-step JAX pattern__

Each script mirrors the autolens reference: NumPy baseline → `jax.jit`-wrapped `analysis.fit_from`
→ scalar `log_likelihood` match. The reference file `mge_pytree.py` in autolens is the gold
standard for this pattern (see @PyAutoPrompt/autolens/fit_imaging_pytree_lp.md for background).

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_likelihood_functions/__init__.py`
2. `autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/__init__.py`
3. Each ported script prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
4. Scripts appended to `smoke_tests.txt` (delaunay_mge commented out with the jax-0.7 comment).
5. Any PyAutoGalaxy library PRs needed for pytree registration (shipped first, merged before
   this workspace PR).
6. Verify locally with `JAX_ENABLE_X64=True python scripts/jax_likelihood_functions/imaging/<name>.py`.

__Umbrella issue__

Task 3/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
