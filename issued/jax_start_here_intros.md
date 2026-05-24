# Phase 1 — `__JAX__` sections in top-level start_here scripts

Add a new `__JAX__` section to both `autolens_workspace/start_here.py` and
`autogalaxy_workspace/start_here.py`, immediately after the existing
`__Tracer__` / `__Galaxies__` sections. This is Phase 1 of the
`z_features/jax_user_intro.md` series.

**Authoritative design doc:** `admin_jammy/notes/jax_interface.md` (shipped
2026-05-24, admin_jammy main `f381393`). All prose in this phase must be
consistent with that doc — cite §3.1 (user-facing principles) and §3.6
(per-phase mapping) when in doubt.

**Run in Opus, not Sonnet** — workspace tutorial prose per
[[feedback_tutorial_prose_opus]]. The `__JAX__` section is reader-facing
narrative; it teaches the science user when JAX runs implicitly, when they
need `@jax.jit` themselves, and what that buys them. Sonnet drifts toward
generic ML-textbook phrasing here.

## Scope

**In scope:**
- One `__JAX__` section added to each top-level `start_here.py`.
- Code-light. At most one short snippet showing the `AnalysisImaging(use_jax=True)`
  default and one short snippet of the `__JAX Variant__` shape the per-dataset
  `simulator.py` files will adopt in Phase 3.
- Cover principles 1, 3, 4 from `admin_jammy/notes/jax_interface.md` §3.1:
  - **(1)** JAX is enabled by default (opt-out, not opt-in).
  - **(3)** User writes `@jax.jit` only at the outer call; inside an Analysis
    the search driver handles it.
  - **(4)** `use_jax=True` changes the return-type contract — outputs come
    back JAX-backed, plotting/`.fits` writers transfer transparently.
- Cite the per-dataset-type docs (Phase 3, 4) and the `lens_calc.py` guide
  (Phase 5d) for runnable examples.

**Out of scope:**
- **Do not** dive into pytree registration (`register_instance_pytree`,
  `register_model`, `enable_pytrees`). That's principle 2 in the design doc
  and lives in Phase 5a (`data_structures.py` guide).
- **Do not** show `xp=jnp` in any example code. That's Phase 5d's
  (`lens_calc.py`) territory for advanced users.
- **Do not** edit any per-dataset-type `start_here.py` (`imaging/start_here.py`,
  `interferometer/start_here.py`, etc.) — those are Phase 3.
- **Do not** edit `simulator.py`, `fit.py`, or `likelihood_function.py` in
  any subdirectory — those are Phase 3 / Phase 4.
- **Do not** edit any guide (`data_structures.py`, `galaxies.py`, `tracer.py`,
  `lens_calc.py`) — those are Phase 5.

The discipline this phase enforces: the **top-level** start_here is the
user's first encounter with JAX in either workspace. Everything else points
back to here.

## Files

### 1. `autolens_workspace/start_here.py`

- **Placement:** after `__Tracer__` (currently line 199 — ends around line
  214 with the `tracer.image_2d_from(grid)` plot), before `__Units__`
  (currently line 215).
- **Length:** ~60-80 lines of prose + at most one short code snippet.
- **Cross-reference:** mention that `__Lens Modeling__` (line 293) and the
  per-dataset-type pages will both show JAX in action.

### 2. `autogalaxy_workspace/start_here.py`

- **Placement:** after `__Galaxies__` (currently line 153), before
  `__Units__` (currently line 176).
- **Length:** similar to autolens, ~50-70 lines (slightly lighter — no
  lens-modeling-specific framing).
- **Cross-reference:** `__Galaxy Modeling__` (line 242) and the
  per-dataset-type pages.

## Draft prose (autolens_workspace/start_here.py)

Use this as a starting point and refine. The shape is fixed; the wording
can be polished.

```python
"""
__JAX__

`PyAutoLens` uses [JAX](https://github.com/google/jax) under the hood to
make model-fitting fast — typically 10-100× faster than pure NumPy for
realistic lens models, with GPU acceleration if one is available.

You do not have to do anything to opt in. If you installed `autolens` with
the `[jax]` extra (`pip install autolens[jax]`, Python ≥ 3.11), every
model-fit you launch is already JAX-accelerated. The `AnalysisImaging`,
`AnalysisInterferometer`, and `AnalysisPoint` classes you'll meet in the
`__Lens Modeling__` section all default to `use_jax=True`; the non-linear
search driver (Nautilus, dynesty, emcee, etc.) handles the JAX wrapping
internally. You will see a log line like `JAX: Applying vmap and jit to
likelihood function -- may take a few seconds.` the first time a search
starts. That's the JIT compilation kicking in; subsequent likelihood
evaluations reuse the compiled trace.

If JAX is not installed (older Python, or you didn't pick up the `[jax]`
extra), `AnalysisImaging` warns once and falls back to NumPy automatically.
You can also force the NumPy path with `al.AnalysisImaging(dataset=dataset,
use_jax=False)` or by setting `PYAUTO_DISABLE_JAX=1` — useful for
debugging, where NumPy stack traces are easier to read than JAX traces.

__When you do write `@jax.jit` yourself__

Two situations call for it:

1. **Custom simulations.** Wrap a `SimulatorImaging` call (or
   `SimulatorInterferometer`, or `PointSolver`) in `@jax.jit` when you want
   to render many datasets quickly — parameter sweeps, mock-data studies,
   batch figure generation. The per-dataset-type `simulator.py` files
   (`scripts/imaging/simulator.py`, `scripts/interferometer/simulator.py`,
   `scripts/point_source/simulator.py`) each show the canonical pattern in
   their `__JAX Variant__` section:

   ```python
   import jax
   simulator = al.SimulatorImaging(
       exposure_time=300.0, psf=psf, background_sky_level=0.1, use_jax=True
   )

   @jax.jit
   def simulate(tracer):
       return simulator.via_tracer_from(tracer=tracer, grid=grid)

   dataset = simulate(tracer)
   ```

   One `@jax.jit`, one `use_jax=True`. No pytree registration, no
   `xp=jnp` threading.

2. **Custom likelihood functions** that you assemble by hand rather than
   reaching for `AnalysisImaging`. Same shape: `@jax.jit` around your own
   `def log_likelihood(instance):` that builds a `Tracer` and a `FitImaging`
   and returns `fit.log_likelihood`. The per-dataset-type
   `likelihood_function.py` files show the recipe.

For directly JIT-ing library methods (`tracer.image_2d_from`,
`LensCalc.magnification_2d_via_hessian_from`, etc.) without going through
a `Simulator` or `Analysis`, see the `lens_calc.py` guide
(`scripts/guides/lensing/lens_calc.py`) — that's the advanced "JIT-it-
yourself" path for users building custom forward models.

__Return-type contract__

When `use_jax=True`, the data structures you get back (`Imaging`,
`FitImaging`, `Tracer.image_2d_from(...)` results, etc.) carry `jax.Array`
data inside instead of `numpy.ndarray`. For nearly everything you'd do in
a workspace — plotting, saving to `.fits`, comparing fit residuals —
this is transparent: the plotters and the FITS writers call
`numpy.asarray()` internally and you see the same images and numbers you
would on the NumPy path.

What changes:

- Arithmetic on JAX arrays stays on the JAX path. Direct calls into NumPy
  (`np.sqrt(fit.residual_map.array)`) will host-transfer the array off the
  GPU; not wrong, but slower than `jnp.sqrt(...)` if you're inside a hot
  loop. For one-off analysis code, don't worry about it.
- The `.array` property of `aa.Array2D` etc. is the raw backing array — a
  `numpy.ndarray` on the NumPy path, a `jax.Array` on the JAX path.

The `data_structures.py` guide
(`scripts/guides/api/data_structures.py`) covers the wrapper-vs-raw-array
distinction in detail.
"""
```

## Draft prose (autogalaxy_workspace/start_here.py)

Same structure, lighter touch — drop the lens-specific framing
(`__Lens Modeling__`, `PointSolver`, multi-plane). Reference
`__Galaxy Modeling__` instead. Reference the autogalaxy guides
(`scripts/guides/api/data_structures.py`, `scripts/guides/api/galaxies.py`).
Three dataset types only: imaging, interferometer, multi.

The key adjustment: where the autolens version says "custom simulations
(SimulatorImaging, SimulatorInterferometer, PointSolver)", the autogalaxy
version drops PointSolver and only mentions the imaging and interferometer
simulators.

## Validation

After editing both files:

1. **Run them.** Each `start_here.py` should still execute end-to-end on
   NumPy:
   ```bash
   NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
   python autolens_workspace/start_here.py
   NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
   python autogalaxy_workspace/start_here.py
   ```
   The added `__JAX__` sections are prose only (the snippet inside the
   docstring is illustrative, not executed). If the script runs and the
   plots come up, the edit is structurally sound.

2. **Check `scripts/check_sizes.sh`** in `autolens_workspace` — the
   start_here size should grow, not shrink. If the snapshot complains,
   refresh per the workspace's `Bulk-edit safety` rules.

3. **Notebooks regenerate.** Both workspaces produce `.ipynb` from
   `.py` via the `/generate_and_merge` skill — that'll happen on the
   `ship_workspace` pass, no action needed here beyond confirming the
   `.py` is valid.

## References

- **Phase 0 design doc** (the source of truth): `admin_jammy/notes/jax_interface.md`
- **Z-features tracker**: `PyAutoPrompt/z_features/jax_user_intro.md`
- **Related (cite, don't absorb)**: `PyAutoPrompt/autofit/on_the_fly_docs.md`
  — workspace doc updates around background quick-update. Mention in passing
  if the prose touches `iterations_per_quick_update`; otherwise skip.
- **Sibling phase tracker entry**: when Phase 1 ships, mark it in
  `PyAutoPrompt/z_features/jax_user_intro.md` and the design doc's §3.6
  Phase 1 line — both currently mark Phase 1 as "unblocked, TBA".

## Out-of-band notes

- **`PYAUTO_DISABLE_JAX=1` is workspace-relevant.** It's mentioned in the
  draft prose above. Confirm it still works as written in `autofit`'s
  `Analysis.__init__` before shipping (it should — `analysis.py:64-67`).
- **Don't add the `from autoconf import jax_wrapper` import** to the
  top-level `start_here.py`. It's not needed for the prose; the per-dataset
  `simulator.py` files already include it where required (Phase 3
  refreshes those).
- **The cluster/group/multi `start_here.py` paths are out of scope** for
  Phase 1. The user reframe on 2026-05-24 anchors design on
  imaging/interferometer/point_source + guides; the cluster/group/multi
  refresh happens in Phase 3c/3e/3f if at all.
