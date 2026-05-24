# Phase 2 — `Simulator.use_jax=True` + `PointSolver.use_jax=True`

The library deliverable of `z_features/jax_user_intro.md`. Adds `use_jax=True`
constructor flags to the three classes the Phase 0 audit identified as needing
them, with internal pytree registration so workspace users never import
`register_instance_pytree` or `register_model`.

**Authoritative design doc:** `admin_jammy/notes/jax_interface.md` (admin_jammy
main `f381393`). Recommendation 2 is the contract:

> Add `Simulator.use_jax=True` (imaging + interferometer first) and
> `PointSolver.use_jax=True` (point_source). **Bounded to these three classes.**
> Do NOT proliferate `use_jax=True` onto `Tracer`, `Galaxy`, `LightProfile`,
> `MassProfile`. Those are reached via the user-written `@jax.jit + xp=jnp`
> path (the `lens_calc.py` advanced guide), not via constructor flags.

This phase ships the library changes. Phases 3a/3b/3d consume them with new
`__JAX Variant__` workspace examples.

**Run in Opus for design judgement; Sonnet via subagent is fine for the
mechanical test additions and parity scripts.** See CLAUDE.md model
delegation rules — the prompt-author / planner is Opus; per-file test
implementations can route to Sonnet.

## Scope

**In scope:**

1. `SimulatorImaging.__init__(..., use_jax=False)` — PyAutoArray.
2. `SimulatorInterferometer.__init__(..., use_jax=False)` — PyAutoArray. Also
   fix the existing signature inconsistency: `via_image_from` does not accept
   `xp` today. Add `xp=np` parameter symmetric with `SimulatorImaging`.
3. `PointSolver` (PyAutoLens) — add `use_jax=False`. Threaded into
   `solver.solve(...)`. Defaults `remove_infinities=False` on the JAX path.
4. **Pytree registration helper** — a new function (location TBD; see
   "Open API decisions" below) that walks a concrete `Tracer` and registers
   each class it carries (Galaxy, light profiles, mass profiles, Point) as
   a JAX pytree. The existing `register_instance_pytree(cls, no_flatten=())`
   helper in `PyAutoArray/autoarray/abstract_ndarray.py:84-131` is the building
   block — this phase wraps it for the simulator surface.
5. **The `xp=np + jnp-backed grid` mismatch error** (design doc §4.8). Add a
   loud `ValueError` at `AbstractMaker.__init__` (or equivalent entry point)
   when `xp is np` but `grid.use_jax` is True. Helps users discover the
   `@jax.jit + xp=jnp` pairing rule.

**Out of scope:**

- Adding `use_jax=True` to `Tracer`, `Galaxy`, `LightProfile`, `MassProfile`.
  Per recommendation 2's bound: no proliferation.
- Editing any workspace script. That's Phase 3 (3a/3b/3d for the new
  `__JAX Variant__` blocks).
- Migrating `autolens_workspace/scripts/cluster/simulator.py` to use the new
  API. Could be done as a worked example, but cluster is in-development per
  the design's scope anchor — defer until cluster is stable and Phase 3f is
  authored.
- Gradients (`jax.grad`). Explicitly deferred to a future series per the
  z_features tracker.
- A user-facing `al.jax.enable_for_modeling()` helper. Open question 4.1 in
  the design doc — flag here as deferred to a separate (future) prompt;
  Phase 2 does not introduce it.

## Implementation

### 1. `SimulatorImaging.use_jax=True`

File: `PyAutoArray/autoarray/dataset/imaging/simulator.py`.

Signature change:

```python
class SimulatorImaging:
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        subtract_background_sky: bool = True,
        psf: Convolver = None,
        use_real_space_convolution: bool = True,
        normalize_psf: bool = True,
        add_poisson_noise_to_data: bool = True,
        include_poisson_noise_in_noise_map: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
        use_jax: bool = False,
    ):
        ...
        self.use_jax = use_jax
        self._pytrees_registered = False
```

Internal `_xp` property (mirror `Analysis._xp`):

```python
@property
def _xp(self):
    if self.use_jax:
        import jax.numpy as jnp
        return jnp
    return np
```

`via_image_from(image, over_sample_size=None, xp=None)`: when `xp is None`,
default to `self._xp`. Keep the explicit `xp=` parameter for library-author
flexibility but make it optional for users.

The PyAutoLens override at `PyAutoLens/autolens/imaging/simulator.py` —
`SimulatorImaging.via_tracer_from(self, tracer, grid)` — adds the pytree
registration step on first call:

```python
def via_tracer_from(self, tracer, grid):
    if self.use_jax and not self._pytrees_registered:
        self._register_simulator_pytrees(tracer)
        self._pytrees_registered = True
    xp = self._xp
    ...
```

Where `_register_simulator_pytrees(tracer)` walks the tracer once and calls
`register_instance_pytree(...)` for `Tracer`, `Imaging`, and every concrete
class reachable from the tracer (`Galaxy`, each `LightProfile` subclass,
each `MassProfile` subclass, `Point`). The autoarray-side `AbstractNDArray`
self-registers on construction when `xp=jnp` is passed, so the walker
doesn't need to handle `Array2D` / `Grid2D` etc.

### 2. `SimulatorInterferometer.use_jax=True`

File: `PyAutoArray/autoarray/dataset/interferometer/simulator.py`.

Today `via_image_from(self, image)` does NOT accept `xp`. Two fixes here:

- Add `xp=None` parameter symmetric with `SimulatorImaging`.
- Replace the hardcoded `np.isnan(noise_map).any()` with `xp.isnan(...)`
  and the `data_with_complex_gaussian_noise_added` helper with a
  JAX-compatible Poisson-/complex-Gaussian-noise variant. Today's helper
  is numpy-only; needs an xp-aware path. If a JAX-safe RNG (`jax.random`)
  is needed, accept a `key=` argument when `use_jax=True`.

The PyAutoLens override (`autolens/interferometer/simulator.py`) gains the
same pytree-registration hook as `SimulatorImaging`. Also: confirm
`Interferometer` itself needs pytree registration (it carries `Convolver`
and `Transformer*` which may not be pytrees) — design doc §4.2 flags this
as an open question that Phase 2 implementation resolves.

### 3. `PointSolver.use_jax=True`

File: `PyAutoLens/autolens/point/solver.py` (or wherever `PointSolver` lives;
the class is exported as `al.PointSolver`).

Signature change on `__init__` and `for_grid` factory: accept `use_jax=False`.
Thread into `solver.solve(...)`:

- When `self.use_jax`, set `xp=jnp` internally.
- Default `remove_infinities=False` on the JAX path (output is padded with
  `inf` sentinels for static-shape compatibility).
- Register `Tracer` as a pytree on first `solve()` call.

Return type: `Grid2DIrregular` should be returned wrapped on both paths
(today the cluster simulator does `.array` unwrap because `Grid2DIrregular`
isn't pytree-registered for the return path). If the autoarray-side
`AbstractNDArray._register_as_pytree` covers `Grid2DIrregular` reliably
when constructed inside the solver with `xp=jnp`, the user can return the
wrapped object directly. **Verify this** during Phase 2 implementation — if
the autoarray pytree registration is unreliable for return paths, the
solver returns raw arrays on the JAX path and users do the rewrap outside
the jit (matches the design doc §3.4.1 pattern).

### 4. The `xp=np` + jnp-grid mismatch error (design doc §4.8)

File: `PyAutoArray/autoarray/structures/decorators/abstract.py`,
`AbstractMaker.__init__`.

Add at the top of `__init__`:

```python
if xp is np and getattr(grid, "use_jax", False):
    raise ValueError(
        f"Called {func.__qualname__} with xp=np but the input grid is "
        f"JAX-backed (grid.use_jax=True). Inside @jax.jit, pass xp=jnp "
        f"explicitly. See the lens_calc.py guide for the JIT-it-yourself "
        f"pattern."
    )
```

Default-on, opt-out via a `_strict_xp=False` kwarg on the call (for library
internals that legitimately want the host-transfer NumPy fallback path on
a jnp grid). Add a similar guard to `AbstractNDArray.__init__` if there's a
parallel construction site that could trip it.

## Tests

PyAutoArray tests stay numpy-only per [[feedback_no_jax_in_unit_tests]].
Cross-xp validation lives in `autolens_workspace_test/` / `autogalaxy_workspace_test/`.

For PyAutoArray:

- `test_autoarray/dataset/imaging/test_simulator.py` — new test:
  `SimulatorImaging(use_jax=True)` constructs cleanly; `via_image_from` with
  `use_jax=True` returns an `Imaging` whose `.data.array` is a `jax.Array`.
  Use the autoconf JAX fixture if needed.
- `test_autoarray/structures/decorators/test_abstract.py` — new test for
  the `xp=np` + jnp-grid `ValueError`.

For PyAutoLens:

- `test_autolens/imaging/test_simulator.py` — `SimulatorImaging(use_jax=True)`
  + `via_tracer_from` end-to-end; assert `dataset.data.array` is jnp-backed.
- `test_autolens/point/test_solver.py` — `PointSolver(use_jax=True).solve(...)`
  returns positions; assert backing array is `jax.Array`.

For `autolens_workspace_test/scripts/`:

- Add a parity script:
  `autolens_workspace_test/scripts/imaging/simulator_use_jax_parity.py` that
  asserts `SimulatorImaging(use_jax=True)` produces a dataset numerically
  equal to `SimulatorImaging(use_jax=False)` (within reasonable tolerance —
  noise seeds differ between RNG implementations; consider asserting
  noise-free or zero-seed paths).
- Add `autolens_workspace_test/scripts/point_source/solver_use_jax_parity.py`
  for the `PointSolver` change.

These are workspace_test scripts, not unit tests — they exercise the
imaging/point-source pipelines end-to-end with both backends and confirm
parity, matching the existing pattern in
`autolens_workspace_test/scripts/jax_likelihood_functions/`.

## Open API decisions

The implementing PR needs to make explicit choices on these:

1. **Where does the pytree-walker live?** Options:
   - `PyAutoArray/autoarray/abstract_ndarray.py` — alongside
     `register_instance_pytree`. Library-internal, no `autolens` import.
     Requires the walker to be tracer-shape-agnostic (walks any object's
     `__dict__` recursively and registers each class).
   - `PyAutoLens/autolens/lens/tracer.py` — as a `Tracer` method. Cleaner
     because it's tracer-specific, but couples the pytree registration to
     the autolens import path. `Simulator.via_tracer_from` would call
     `tracer._register_for_jax()`.
   - `PyAutoLens/autolens/jax/registration.py` — a new module. Most
     discoverable for users who go looking for "where does the JAX
     registration code live?"
   - Recommendation: option 3 (`autolens/jax/registration.py` or
     `autogalaxy/jax/registration.py` mirror). Matches PyAutoFit's
     `autofit/jax/pytrees.py` layout. Keeps autoarray clean.

2. **`Tracer` and `Galaxy` registration: walk-on-call or preemptive?**
   - Walk-on-call: the simulator walks `tracer` on its first
     `via_tracer_from` invocation and registers each class it sees. Cheap
     because it only fires once. Won't catch classes the user adds later.
   - Preemptive: `enable_jax_for_modeling()` (still deferred per scope) walks
     all known PyAutoLens classes at import time. Heavier startup, more
     thorough.
   - Recommendation: walk-on-call. Aligns with the existing
     `AnalysisImaging._register_fit_imaging_pytrees()` style.

3. **`Interferometer` pytree registration**: design doc §4.2 flags this.
   Confirm during implementation: does `register_instance_pytree(Interferometer,
   no_flatten=("transformer_class", "real_space_mask"))` round-trip cleanly
   through `jax.jit`? If yes, register it. If not, document why and have
   the user do `.data` unwrap on the JIT boundary.

4. **`Imaging` pytree registration**: same question for the imaging dataset.
   `no_flatten` candidates: `psf` (Convolver), `mask` (Mask2D). Verify and
   register.

5. **Error message wording for §4.8**: cite the `lens_calc.py` guide
   directly. Once Phase 5d ships, the error message should link to it.

## Validation

After the library changes land:

1. **PyAutoArray tests pass** with the new helpers; no JAX leakage into
   the numpy unit test path.
2. **PyAutoLens tests pass** including the new use_jax parity tests.
3. **Workspace_test parity scripts pass** — both numpy and JAX paths
   produce numerically agreeing simulations.
4. **The cluster simulator can be optionally migrated** as a worked example
   showing the API collapse (8 steps → 1-2 user-visible lines). Optional;
   cluster is in-development per scope anchor.
5. **The Phase 1 top-level start_here `__JAX__` section** still reads
   correctly — its prose references "use_jax=True" on Simulator/PointSolver
   as a thing that exists.

## References

- **Phase 0 design doc:** `admin_jammy/notes/jax_interface.md` (sections
  2.2, 3.4, 3.5, 4.2, 4.8 are most relevant).
- **Z-features tracker:** `PyAutoPrompt/z_features/jax_user_intro.md`.
- **Phase 1 (sibling, ships in parallel):**
  `PyAutoPrompt/workspaces/jax_start_here_intros.md` — top-level start_here
  `__JAX__` sections.
- **Phase 3 (downstream consumers):** `autolens_workspace/jax_docs_imaging.md`,
  `..._interferometer.md`, `..._point_source.md` — these use the new
  `Simulator.use_jax=True` / `PointSolver.use_jax=True` API in their
  `__JAX Variant__` blocks.
- **Existing model templates:**
  - `PyAutoLens/autolens/imaging/model/analysis.py:168-188` —
    `AnalysisImaging._register_fit_imaging_pytrees()` is the canonical
    auto-registration pattern this phase mirrors.
  - `PyAutoLens/autolens/point/model/analysis.py:172-208` —
    `AnalysisPoint._register_fit_point_pytrees()` for the point-source case.
  - `PyAutoArray/autoarray/abstract_ndarray.py:84-131` —
    `register_instance_pytree(cls, no_flatten=())` building block.

## Out-of-band notes

- Library tests stay numpy-only per [[feedback_no_jax_in_unit_tests]].
- Smoke tests are a curated subset per [[feedback_smoke_tests_small_subset]];
  do not mass-promote the new use_jax parity scripts into `smoke_tests.txt`.
- `[[feedback_jax_validation_vmap_not_jit]]` — when validating that
  `use_jax=True` works end-to-end, use `fitness._vmap(jnp.array(parameters))`,
  not `jax.jit(fn)(concrete)`.
