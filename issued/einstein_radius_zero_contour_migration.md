# Fast visualization Phase B — Euclid effective_einstein_radius latent via zero_contour

Re-enable the `effective_einstein_radius` latent variable in the Euclid
pipeline workspace by routing it through the JAX-traceable
`einstein_radius_via_zero_contour_from()` rather than the legacy
marching-squares `einstein_radius_from(grid=...)` that forced the latent
off-JAX.

This is Phase B of `z_features/fast_visualization.md`. Phase A′ (PR
PyAutoGalaxy#434, PR PyAutoLens#527, PR autolens_workspace_test#111)
landed the perf cache + safety net that makes this migration viable
without paying the ~10s ZeroSolver compile cost on every Nautilus
sample.

## Background

The `effective_einstein_radius` latent was commented out in
`euclid_strong_lens_modeling_pipeline/util.py` because the only
available method, `tracer.einstein_radius_from(grid=...)`, routes
through `skimage.measure.find_contours` — not JAX-traceable. With
`use_jax=True` (the default for Euclid pipeline fits) the latent
computation either had to flip `self._use_jax = False` per sample
(slow workaround used in `z_projects/euclid`) or be disabled outright
(approach taken in `euclid_strong_lens_modeling_pipeline`).

After PR `Jammy2211/PyAutoGalaxy#434`, the
`einstein_radius_via_zero_contour_from()` path is fast on repeat calls
(~68 ms warm on CPU on an SIE) because `LensCalc` now caches its
`(f, ZeroSolver)` pair. That makes it viable as the standard latent
computation: the first sample pays the one-time ~10 s ZeroSolver JIT
compile; every subsequent sample reuses the cached compile.

`z_projects/euclid/scripts/util.py` is live science work for the DR1
catalogue paper and is **not** touched by this migration; the science
branch picks up the change on its own cadence.

## What to change

### 1. `LATENT_KEYS` list

`@euclid_strong_lens_modeling_pipeline/util.py` around line 305-315

Uncomment the commented-out `latent_effective_einstein_radius` entry
and **rename to `latent.effective_einstein_radius`** so the dotted
naming matches the surrounding entries (`latent.total_lens_flux`,
`latent.magnification`, etc.). The underscore form is a typo from the
original commented-out version.

```python
LATENT_KEYS = [
    "latent.total_lens_flux",
    "latent.total_lens_flux_1_fwhm",
    "latent.total_lens_flux_2_fwhm",
    "latent.total_lens_flux_3_fwhm",
    "latent.total_lens_flux_4_fwhm",
    "latent.total_lensed_source_flux",
    "latent.total_source_flux",
    "latent.magnification",
    "latent.effective_einstein_radius",
]
```

### 2. `compute_latent_variables` body — Einstein radius computation

`@euclid_strong_lens_modeling_pipeline/util.py` around line 488-495

Replace the commented-out block with a dispatch that respects the
Analysis's `use_jax` setting:

```python
# EFFECTIVE EINSTEIN RADIUS

try:
    if self._use_jax:
        effective_einstein_radius = tracer.einstein_radius_via_zero_contour_from()
    else:
        effective_einstein_radius = tracer.einstein_radius_from(
            grid=self.dataset.grids.lp,
        )
except ValueError:
    # No tangential critical curve found (degenerate model — e.g. very
    # weak lens with no eigenvalue zero-crossing). The latent is
    # undefined for this sample; record as NaN.
    effective_einstein_radius = xp.nan
```

Notes:

- **Branch on `self._use_jax`.** `_via_zero_contour_from()` always
  imports `jax.numpy` and calls `ZeroSolver` internally — it does not
  consult the caller's `xp`. Routing a `use_jax=False` user through it
  would silently pull JAX into their critical path and raise
  `ModuleNotFoundError` if they don't have `jax_zero_contour`
  installed. The dispatch above keeps the legacy marching-squares
  numpy path for users who explicitly opted out of JAX, and routes
  JAX users through the fast traceable path that PR #434 makes viable.
- **Use `except ValueError:` specifically**, not bare
  `except Exception:`. Both `_init_guess_from_coarse_grid` (zero_contour
  path) and `find_contours`-based fallbacks raise `ValueError` for
  "no zero crossings"; that's the expected recoverable failure.
  Other exceptions (e.g. `ModuleNotFoundError` if `jax_zero_contour`
  is uninstalled despite `use_jax=True`, or a `TypeError` from a JAX
  trace mismatch) must propagate loudly per
  `feedback_no_silent_guards`.
- **No `grid=` argument on the zero_contour branch.** That method
  traces the curve directly without a dense grid. The legacy
  `tracer.einstein_radius_from(grid=...)` call still needs
  `self.dataset.grids.lp`.
- **Do NOT flip `self._use_jax = False`** to coerce JAX users back to
  the numpy path. The point of the migration is that JAX users get
  the JAX path end-to-end; the workaround that `z_projects/euclid`
  carries is no longer needed.

### 3. Returned tuple at line 497-507

Uncomment the `effective_einstein_radius` entry so the returned tuple
matches the `LATENT_KEYS` length:

```python
return (
    total_lens_flux_muJy,
    total_lens_flux_muJy_aperture_list[0],
    total_lens_flux_muJy_aperture_list[1],
    total_lens_flux_muJy_aperture_list[2],
    total_lens_flux_muJy_aperture_list[3],
    total_lensed_source_flux_muJy,
    total_source_flux_muJy,
    magnification,
    effective_einstein_radius,
)
```

Verify the tuple length matches `len(LATENT_KEYS)` (now 9).

## Out of scope

- **No `z_projects/euclid` edits.** Live science work for the DR1
  catalogue paper. The science branch picks up the change on its own
  cadence after this lands.
- **No library changes.** This is purely a workspace edit; the library
  API needed (`einstein_radius_via_zero_contour_from()`) already exists
  and shipped fast in PyAutoGalaxy #434.
- **No tracker / `LATENT_KEYS` extension** beyond Einstein radius. If
  you want additional latents (e.g. critical-curve area, magnification
  at a specific position), file a separate prompt.

## Verification

1. **Pipeline smoke run:**
   `/smoke_test euclid_strong_lens_modeling_pipeline` — all 6 scripts
   pass with the updated latent code path. Set
   `PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1` in the env prefix per the
   pattern used during Phase A′ smoke (the workspace's
   `config/general.yaml` doesn't pin `workspace_version`, which causes
   the version check to mismatch the installed library version — this
   is a routine workspace-vs-library drift, not a regression of this
   PR).

2. **Latent output inspection (use_jax=True path).** Run one of the
   pipeline scripts with the latent computation enabled
   (`PYAUTO_SKIP_FIT_OUTPUT=0`, `PYAUTO_SKIP_VISUALIZATION=0`,
   `PYAUTO_TEST_MODE=1` to get a small real fit). Inspect the resulting
   `output/.../latent/latent_summary.json` and confirm
   `latent.effective_einstein_radius` is present and finite. The value
   should be in the expected range for the lens (typically 0.5–2.0
   arcsec for a typical Euclid strong lens).

3. **Both `use_jax` paths exercised.** The dispatch branches on
   `self._use_jax`, so both branches need a smoke. The smoke list runs
   under `PYAUTO_DISABLE_JAX=1` (forces `use_jax=False`) per the
   workspace's `config/build/env_vars.yaml` — that already covers the
   numpy branch. For the JAX branch, run **one** pipeline script
   manually without `PYAUTO_DISABLE_JAX`:

   ```bash
   PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 PYAUTO_TEST_MODE=1 \
     python start_here.py \
       --dataset=102018665_NEG570040238507752998 \
       --sample=q1_walsmley
   ```

   Confirm `latent.effective_einstein_radius` is finite in the output
   `latent_summary.json` and that no `ModuleNotFoundError` /
   `falling back to numpy` warning fires during latent computation.

4. **No silent `_use_jax = False` flip.** Skim the script logs for
   any "JAX backend disabled" / "falling back to numpy" warning during
   the latent step. If something flips backends silently mid-search,
   the migration hasn't actually landed the JAX path.

## References

- `z_features/fast_visualization.md` — parent tracker, Phase B section.
- PyAutoGalaxy PR #434 (merged 2026-05-21) — `(f, ZeroSolver)` cache
  fix that makes warm `_via_zero_contour_from()` calls ~68 ms instead
  of ~10 s. The migration depends on this PR shipping in an installed
  library version.
- PyAutoLens PR #527 (merged 2026-05-21) — broad `except` tighten in
  the visualizer; relevant because the migration's `except ValueError`
  uses the same "tight catch, loud on unexpected" pattern.
- `feedback_no_silent_guards` (memory) — codebase rule against bare
  `except Exception: return nan` patterns.
- `feedback_euclid_pipeline_not_z_projects` (memory) — `z_projects/euclid`
  is off-limits; this prompt targets the pipeline workspace only.
- `feedback_jax_closure_cache_busts` (memory) — the JIT cache identity
  bug that PR #434 fixed; relevant if the warm-call latency on Euclid
  ever regresses past ~100 ms.
