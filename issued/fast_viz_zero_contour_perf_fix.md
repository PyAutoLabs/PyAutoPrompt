# Fast visualization Phase A′ — zero_contour perf fix + safety net

This task is the **prerequisite for any future Phase A flip / Phase B
latent migration** of `z_features/fast_visualization.md`. Two coupled
bugs prevent `zero_contour`-based critical curves and Einstein radii
from being viable defaults today; this task fixes both and lands the
first regression-net assertion to catch any future regression.

## Background

Two earlier attempts to put `zero_contour` on the default visualization
path were reverted because the silent-failure mode looked successful
from outside:

- **2026-04-18 → 2026-04-19** — PyAutoGalaxy commit `aea3bc95` flipped
  the YAML default in `autogalaxy/config/visualize/general.yaml` from
  `marching_squares` to `zero_contour`, reverted in `abd7b717` because
  `ZeroSolver` raised inside model-fits and the exception was swallowed
  by the broad `except Exception: return None, None, None, None` at
  `PyAutoLens/autolens/imaging/plot/fit_imaging_plots.py:52`. Critical
  curves silently vanished on supercomputer runs.
- **2026-05-16 → 2026-05-17** — same failure shape on the Euclid DR1
  pipeline (PR #1280 reverted `use_jax_for_visualization=True` default).
  Source-plane FITS files wrote all-zero, Einstein-radius posteriors
  collapsed to the full prior across every tile, none of which raised.

The 2026-05-21 perf benchmark on an SIE + circular source revealed a
third, independent issue: `_critical_curve_list_via_zero_contour` at
`PyAutoGalaxy/autogalaxy/operate/lens_calc.py:1167-1170` builds a fresh
`f = self._make_eigen_fn(...)` and `solver = ZeroSolver(...)` on every
invocation. Because JAX's compiled function cache is keyed on callable
identity, every call rebuilds the JIT cache and pays the full
~10-second compile cost. Measured on CPU:

| Method | First call | Warm call |
|---|---|---|
| `marching_squares` | 32 ms | 32 ms |
| `zero_contour` (current code) | 10300 ms | 10300 ms |
| `zero_contour` (reused `f` / solver) | 10679 ms | **66 ms** |

With the closure cached, `zero_contour` is fast enough on warm calls to
be a sensible default for any JIT'd likelihood function. The compile
cost still applies on the first call in a process — that's the reason
`marching_squares` stays the plotter's YAML default for one-shot plotting,
while JIT'd callers explicitly route via `_via_zero_contour_from()`.

## What to change

### 1. PyAutoGalaxy — cache `(f, solver)` in `LensCalc`

`@PyAutoGalaxy/autogalaxy/operate/lens_calc.py`

In `_critical_curve_list_via_zero_contour` (around line 1121), cache the
`(f, solver)` tuple on the `LensCalc` instance keyed on
`(kind, pixel_scales, tol, max_newton)`. Suggested shape:

```python
def _critical_curve_list_via_zero_contour(self, kind, ...):
    cache_key = (kind, pixel_scales, tol, max_newton)
    cached = getattr(self, "_zero_contour_cache", {}).get(cache_key)
    if cached is None:
        f = self._make_eigen_fn(kind=kind, pixel_scales=pixel_scales)
        solver = ZeroSolver(tol=tol, max_newton=max_newton)
        self._zero_contour_cache = getattr(self, "_zero_contour_cache", {})
        self._zero_contour_cache[cache_key] = (f, solver)
    else:
        f, solver = cached
    ...
```

Pick whichever idiomatic-Python shape is cleanest (`functools.cached_property`
won't work because the key is parameterised; a plain dict on the instance
or a `functools.lru_cache`-decorated helper that returns `(f, solver)` are
both fine).

**Verification:** the regression-net script in step 3 below asserts that
the second call from a fresh `LensCalc` is under 100 ms on CPU.

### 2. PyAutoLens — tighten the broad `except`

`@PyAutoLens/autolens/imaging/plot/fit_imaging_plots.py:52`

Replace:

```python
try:
    tan_cc, rad_cc = _critical_curves_from(tracer, grid)
    tan_ca, rad_ca = _caustics_from(tracer, grid)
    ...
    return image_plane_lines, ..., source_plane_line_colors
except Exception:
    return None, None, None, None
```

with specific catches for the *known recoverable* failure modes plus a
loud warning for anything else:

```python
import logging
logger = logging.getLogger(__name__)

try:
    tan_cc, rad_cc = _critical_curves_from(tracer, grid)
    tan_ca, rad_ca = _caustics_from(tracer, grid)
    ...
    return image_plane_lines, ..., source_plane_line_colors
except ModuleNotFoundError:
    # jax_zero_contour missing in this environment — already handled
    # upstream in plot_utils._critical_curves_method() with a warning.
    return None, None, None, None
except ValueError:
    # No zero crossings in the eigenvalue grid (e.g. slope >= 2
    # isothermal where lambda_r > 0 everywhere). Curves don't exist
    # for this model.
    return None, None, None, None
except Exception:
    logger.warning(
        "Critical-curve computation failed unexpectedly; rendering "
        "without overlays. Investigate — this used to be silent.",
        exc_info=True,
    )
    return None, None, None, None
```

The unit test for this change should re-broaden the bare `except` to
`except Exception:` in a test fixture, raise a synthetic exception
inside `_critical_curves_from`, and assert that:

1. The original broad-except returns `(None, None, None, None)` silently.
2. The new tightened-except logs at `WARNING` level with traceback info.

### 3. autolens_workspace_test — first `__Visualization Sanity__` block

`@autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py`

Append a `__Visualization Sanity__` block following the same prose-block
style as the existing `__Likelihood Sanity__` block in the same file
(see line ~170). The block sits inline before the Nautilus search,
builds the prior-median instance, and asserts:

```python
import time
import numpy as np
from autogalaxy.operate.lens_calc import LensCalc

# Build the LensCalc from the prior-median tracer.
instance = model.instance_from_prior_medians()
tracer = analysis.tracer_via_instance_from(instance=instance)
od = LensCalc.from_tracer(tracer)

# Correctness: zero_contour produces a non-empty tangential critical curve
# and a finite, positive Einstein radius.
tc = od.tangential_critical_curve_list_via_zero_contour_from()
assert len(tc) > 0, "no tangential critical curves (zero_contour regression)"
er = od.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "Einstein radius unconstrained (zero_contour regression)"

# Perf regression net: with the (f, solver) cache fix, the second call
# from the SAME LensCalc must be under 100 ms on CPU. The first call
# pays the ~10 s ZeroSolver compile.
od.tangential_critical_curve_list_via_zero_contour_from()  # warm the cache
t0 = time.perf_counter()
od.tangential_critical_curve_list_via_zero_contour_from()
dt = time.perf_counter() - t0
assert dt < 0.1, (
    f"zero_contour warm call took {dt*1000:.1f} ms — closure-cache-busting bug "
    "may have regressed (see fast_viz_zero_contour_perf_fix)"
)
```

Also append the **silent-zero source-plane** assertion from the tracker's
imaging template (line 163), so the block catches both failure modes
(perf regression *and* algorithmic collapse) in one place.

Do NOT propagate this pattern to other dataset types in this task — that's
Phase D rollout, gated on this pilot landing.

## Out of scope

- **No config flip.** `autogalaxy/config/visualize/general.yaml:8` stays
  `marching_squares`. The config-flip / context-aware-dispatch design
  is Phase A's follow-up sub-prompt.
- **No `z_projects/euclid` edits.** That tree is live science work. The
  Euclid latent migration (Phase B) targets
  `euclid_strong_lens_modeling_pipeline/util.py:491` and is a separate
  sub-prompt to author after this lands.
- **No new `__Visualization Sanity__` blocks** beyond
  `modeling_visualization_jit.py`'s. Rollout across other scripts is
  Phase D.
- **No IPython display wiring.** `BackgroundQuickUpdate` already exists;
  the `update_display(fig, display_id=...)` work is Phase C.

## Verification

1. **Library unit tests:**
   - `pytest test_autogalaxy/operate/test_lens_calc.py` (cache behaviour
     + correctness of the cached path).
   - `pytest test_autolens/imaging/plot/` (broad-except tightening, log
     assertion).
2. **Workspace smoke:**
   - `python autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py`
     runs cleanly under `use_jax=True`, the `__Visualization Sanity__`
     block passes (curve count, Einstein radius, < 100 ms warm).
3. **Pipeline smoke:**
   - `/smoke_test` against `euclid_strong_lens_modeling_pipeline` after
     library changes — confirms the perf fix didn't regress the pipeline
     scripts. Catches any latent / lens_calc API drift.
4. **Benchmark re-run:**
   - `python /tmp/bench_critical_curves.py` on the fix branch — warm
     `zero_contour` calls should drop from ~10300 ms to under 100 ms.

## References

- `z_features/fast_visualization.md` — parent tracker, Phase A′ section.
- `complete.md::jax-viz-default-broken` — 2026-05-17 revert of the same
  failure shape.
- PyAutoGalaxy commit `abd7b717` — 2026-04-19 revert of the YAML default
  flip; the message describes the broad-except silent-zero failure mode.
- `feedback_no_silent_guards` (memory) — codebase rule against silent
  catch-and-degrade.
- `feedback_euclid_pipeline_not_z_projects` (memory) — `z_projects/euclid`
  is off-limits; pipeline workspace is the target for related changes.
