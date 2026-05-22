# Fast Visualization — Sequenced Roadmap

## Status: COMPLETE (2026-05-22) — archived

Phase D is shipped. The user-visible goal — `use_jax=True` searches
render visualization fast enough to live-update in a Jupyter / Colab
cell, with a regression net in every `_test` workspace — is achieved
for every dataset type that has the necessary library infrastructure.

### Shipped phases

- **Phase A′** — `LensCalc._zero_contour_cache` closure caching +
  `fit_imaging_plots.py:52` broad-except tightening + first
  `__Visualization Sanity__` block (PRs: PyAutoGalaxy #434, PyAutoLens
  #527, autolens_workspace_test #111).
- **Phase B** — Euclid latent migration via new
  `LensCalc.einstein_radius_jit_from(init_guess, ...)` + PyAutoFit
  `LATENT_BATCH_MODE` class attribute (PRs: PyAutoGalaxy #435,
  PyAutoFit #1288, euclid_strong_lens_modeling_pipeline #15).
- **Phase C** — `BackgroundQuickUpdate` + IPython `update_display(...,
  display_id=...)` live-cell wiring + autofit_workspace cookbook
  section.
- **Phase D.1** — `__Visualization Sanity__` rollout across
  `modeling_visualization_jit*.py` (PRs: autolens_workspace_test #113,
  autogalaxy_workspace_test #54).
- **Phase D.2.a** — same rollout across existing `visualization_jax*.py`
  (PRs: autolens_workspace_test #115, autogalaxy_workspace_test #55).
- **Phase D.2.b.ii (ellipse)** — authored
  `ellipse/visualization_jax.py` +
  `ellipse/modeling_visualization_jit.py` (PR:
  autogalaxy_workspace_test #60).

### Dropped / parked / superseded

- **Phase A** (config flip) — superseded by Phase A′'s context-aware
  dispatch approach; the YAML default stays `marching_squares` and
  callers self-route.
- **Phase D.2.b.i (quantity)** — DROPPED. `quantity` package archived
  in PyAutoGalaxy #437; the JAX-trace bug at `geometry_profiles.py:168`
  (PyAutoGalaxy #436, closed won't-fix) no longer matters because the
  only caller is gone.
- **Phase D.2.b.iii (weak lensing)** — PARKED indefinitely.
  `autolens/weak/` has only `dataset.py`, `fit.py`, `simulator.py`,
  `plot/` — no `model/analysis.py`, no `AnalysisWeak`. The Phase D
  pattern can't be applied until the weak modeling layer is built;
  that's a separate epic.
- **Phase E (`ModelInstance` pytree cascade)** — deferred indefinitely.
  Long-term enabler for `jax.jit(fit_from)(instance)` end-to-end; not
  blocking any current dataset type's visualization use case.
- **Phase F (subprocess viz)** — obsolete. `BackgroundQuickUpdate` +
  IPython live-cells solved the live-update problem in-process; no
  subprocess complexity needed.

### Standing follow-ups still useful

- `autogalaxy/einstein_radius_jit_native_seed_finder.md` (in
  PyAutoPrompt/autogalaxy/) — parked enhancement to replace the static
  `init_guess` argument on `einstein_radius_jit_from` with a JAX-native
  seed finder. Not blocking; pick up when cluster-scale geometry needs
  it.

---

End goal: when a search runs with `use_jax=True`, visualization is fast enough
that **a Jupyter / Colab cell can update in place during the fit**, with no
subprocess complexity. Plus a regression net in every `_test` workspace so the
next "default flipped, no test exercises the new path, ships broken to HPC"
class of bug (see 2026-05-16 Euclid all-zero-source-plane regression) gets
caught locally.

This z_feature **supersedes** the subprocess-visualization approach that was
investigated under issue #1279 / task `viz-subprocess-feasibility`. See
`z_features/complete/visualization_subprocess_feasibility.md` for the
banked spike findings (`FitImaging` round-trips cleanly through stdlib
`pickle` — design parked, not lost).

## Why pivoted

During #1279 we discovered the JAX-jittable critical-curve / Einstein-radius
infrastructure already exists in the codebase:

- **`jax_zero_contour`** (external package) is already installed in the venv
  and gives a `ZeroSolver` with JIT-compatible level-curve tracing.
- **PyAutoGalaxy `autogalaxy/operate/lens_calc.py:1001-1500ish`** has
  `..._via_zero_contour_from()` implementations for tangential / radial
  critical curves, caustics, and Einstein radius.
- **PyAutoGalaxy `autogalaxy/plot/plot_utils.py`** routes between
  `zero_contour` and `marching_squares` via the config switch
  `visualize.general.critical_curves_method`.
- The config default at
  **`autogalaxy/config/visualize/general.yaml:8`** is currently
  `marching_squares` (the legacy `skimage.measure.find_contours` path,
  not JAX-traceable).

So the unlock for "JAX-jit'd visualization" is mostly **enablement** of code
that already exists, plus a regression net to prevent the next default flip
from silently shipping broken on dataset types that have no end-to-end test.

## Update — what we discovered after writing this tracker (2026-05-21)

Two findings shift the sequencing meaningfully.

**1. The April 2026 zero_contour revert.** A separate, earlier attempt to
flip `critical_curves_method` to `zero_contour` shipped on **2026-04-18**
(PyAutoGalaxy commit `aea3bc95`) and was **reverted on 2026-04-19**
(commit `abd7b717`). The revert message describes the *same shape* of
silent-zero failure as the 2026-05-16 Euclid regression that prompted this
tracker — `ZeroSolver` raising inside model-fits and the exception getting
swallowed by `_compute_critical_curve_lines` at
`PyAutoLens/autolens/imaging/plot/fit_imaging_plots.py:52`
(`except Exception: return None, None, None, None`). So Phase D's
"regression net" is not optional polish — it is the **prerequisite for any
future Phase A flip to be safe**.

**2. `zero_contour` has a cache-busting performance bug.** A CPU benchmark
on an SIE + circular source shows:

| Method | First call | Warm calls |
|---|---|---|
| marching_squares | 32 ms | 32 ms |
| zero_contour (current code, closure busts cache) | 10300 ms | ~10300 ms |
| zero_contour (reused `f`/solver) | 10679 ms | **66 ms** |
| zero_contour, JIT disabled (`JAX_DISABLE_JIT=1`) | >10 min (timeout) | n/a |

`_critical_curve_list_via_zero_contour` at
`PyAutoGalaxy/autogalaxy/operate/lens_calc.py:1167-1170` builds a fresh
`f = self._make_eigen_fn(...)` and `solver = ZeroSolver(...)` on every call,
so JAX cannot reuse its compiled function cache and every call pays the
full ~10 s compile cost. With `(f, solver)` cached, warm calls drop to
~66 ms — *under* the 100 ms threshold needed for it to be a sensible
default for any JIT'd likelihood function. **This is the real reason the
April default flip failed**, not just the broad-except / silent-zero issue.

This also tells us where `zero_contour` belongs and where it doesn't:
JIT'd likelihood / latent code (Phase B's call sites) gets ZeroSolver
baked into the outer XLA graph so per-iteration cost is negligible. A
plain one-shot plotting call still has to pay the 10 s compile, then
warm runs at ~66 ms — so `marching_squares` stays the natural default
for non-JIT, single-call plotting, and `zero_contour` becomes the default
for any JIT'd context. Phase A's config flip is therefore not an
unconditional switch but a re-statement of this rule: the config switch
controls the *plotter's* default (stays `marching_squares` for non-JIT
ergonomics), while JIT'd callers route via the explicit
`_via_zero_contour_from()` methods regardless of config.

**3. `BackgroundQuickUpdate` already shipped.** Phase C's "no subprocess
complexity" preference is already realised — PyAutoFit commit `1fee93174`
adds `autofit/non_linear/quick_update.py::BackgroundQuickUpdate`, a daemon
`threading.Thread` with latest-only drop backpressure, wired into the
Nautilus sampler at `search/nest/nautilus/search.py:196,216` via the
`background_quick_update` kwarg. So Phase C reduces to "wire
`IPython.display.update_display(fig, display_id=...)` into the existing
`perform_quick_update` path" — no new subprocess / IPC architecture
needed. Phase F (subprocess viz) is now likely obsolete.

These findings produce a new **Phase A′** (below) as a hard prerequisite
for Phase A, and a small retargeting of Phase B.

## Phase A′ — Fix `zero_contour` perf + safety so it can be a default

**Scope:** address both findings above so the eventual Phase A config flip
can ship safely. No config flip in this phase — that's Phase A.

**Prompt file:** `autogalaxy/fast_viz_zero_contour_perf_fix.md`

Three coupled changes, one PR pair:

1. **PyAutoGalaxy** `autogalaxy/operate/lens_calc.py` — cache `(f, solver)`
   inside `LensCalc` keyed on `(kind, pixel_scales, tol, max_newton)` so
   subsequent calls reuse JAX's compiled function cache. Target: warm
   call < 100 ms on CPU.

2. **PyAutoLens** `autolens/imaging/plot/fit_imaging_plots.py:52` — replace
   the bare `except Exception: return None, None, None, None` with
   specific recoverable catches (`ModuleNotFoundError` for missing
   `jax_zero_contour`, `ValueError` for "no zero crossings" from
   `_init_guess_from_coarse_grid`). Anything else is logged with
   `logger.warning(..., exc_info=True)` and still returns the no-overlay
   tuple — silent collapse becomes a loud warning.

3. **autolens_workspace_test** `scripts/imaging/modeling_visualization_jit.py`
   — add the first `__Visualization Sanity__` block per the Phase D
   template, plus an explicit perf assertion that re-running
   `tangential_critical_curve_list_via_zero_contour_from()` on the same
   `LensCalc` stays under 100 ms on the second call. Regression net both
   for the silent-zero failure mode *and* for any future code change that
   re-introduces the closure-cache-busting bug.

Verification step: `/smoke_test` against
`euclid_strong_lens_modeling_pipeline` after the library changes land,
to confirm the pipeline scripts still run cleanly.

## Phase A — Superseded by Phase B's findings

> *Originally scoped (2026-05-20) as "flip YAML default to zero_contour",
> then revised on 2026-05-21 to "context-aware dispatch". Both versions
> are now superseded.*

**Status (2026-05-21):** Not actionable as previously scoped. The 2026-05-21
Phase B work (PyAutoGalaxy #435, PyAutoFit #1288) established that:

- The plotter dispatch at `autogalaxy/plot/plot_utils.py::_critical_curves_from`
  is **always called from plain Python**, never from inside a `jax.jit`
  trace. (`fit_for_visualization` is jit'd, but the plotter consumes its
  output in Python afterwards.) So "context-aware dispatch" has no actual
  caller to detect — the JIT case the dispatch was meant to handle doesn't
  exist.
- The PR #434 `(f, ZeroSolver)` cache is **per-LensCalc-instance**.
  Notebook workflows that build many tracers (one per model exploration)
  would pay a fresh ~10 s ZeroSolver compile on the first plot of each
  tracer. Flipping the YAML default to `zero_contour` would be a
  noticeable regression for interactive exploration. `marching_squares`
  stays a strict win as the plotter default for galaxy-scale lenses.
- The legitimate use case where `zero_contour` is a clear win (cluster-
  scale fits where marching-squares on a 1000×1000 grid is slow) can be
  served by users overriding `critical_curves_method: zero_contour` in
  their workspace's `config/visualize/general.yaml` — i.e. it's already
  a per-workspace knob.

**No library change required.** Document the trade-off in the next docs
pass (workspace tutorials should mention how to override the config for
cluster fits). Not gated on a sub-prompt; folded into the next
documentation refresh.

If we later observe a real need for JIT-aware dispatch (e.g. a future
analysis that calls `_critical_curves_from` directly from inside a JIT
trace — none today), reopen this with the implementation pattern from
the 2026-05-21 revised scope above.

## Phase B — Migrate latent-variable call sites (shipped 2026-05-21)

**Status:** SHIPPED via three coordinated PRs:

- PyAutoFit #1288 — `Analysis.LATENT_BATCH_MODE` switch ("vmap" default,
  "jit" opt-in). Required because the original premise (drop in
  `_via_zero_contour_from()` inside `jax.vmap`) was invalid — upstream
  `jax_zero_contour.ZeroSolver.zero_contour_finder` uses `lax.cond` /
  `lax.while_loop` for early termination, documented as vmap-incompatible.
- PyAutoGalaxy #435 — new `LensCalc.einstein_radius_jit_from(init_guess, ...)`
  helper (jit-friendly, bypasses skimage seed search and `path_reduce`)
  plus `AnalysisDataset.LATENT_BATCH_MODE = "jit"` override.
- euclid_strong_lens_modeling_pipeline #15 — workspace dispatch on
  `self._use_jax` → new helper (JAX) or legacy `einstein_radius_from(grid=...)`
  (numpy).

Final scope grew beyond the original drop-in swap because the upstream
ZeroSolver vmap caveat is fundamental. Verified end-to-end on
dataset 102018665_NEG570040238507752998: `latent.effective_einstein_radius
= 2.1002 arcsec` on the prior-median MGE tracer.

Follow-up authored but not yet issued: `autogalaxy/einstein_radius_jit_native_seed_finder.md`
— replace the required `init_guess` argument with a JAX-native seed finder
(`jnp.argmin` on a coarse `|eigen_values|` grid) so callers don't need to
know lens position upfront.

## Phase C — Live Jupyter cell rendering via `IPython.display.update_display`

**Scope reduction (2026-05-21):** the "render off the main thread" half
of this phase already shipped in PyAutoFit commit `1fee93174`. That
commit added `autofit/non_linear/quick_update.py::BackgroundQuickUpdate`
— a daemon `threading.Thread` with a latest-only drop backpressure
policy and a `_convert_jax_to_numpy(instance)` step so the worker thread
never touches JAX/GPU state — and wired it into the Nautilus sampler at
`search/nest/nautilus/search.py:196,216` behind the `background_quick_update`
kwarg. No subprocess, no pickle, no IPC machinery needed. So Phase C
reduces to the IPython display layer.

**Remaining scope:** in `perform_quick_update` (the function the background
thread calls), additionally call
`IPython.display.update_display(fig, display_id="fit_progress")` when
running inside a Jupyter kernel. Cell updates in place during the fit;
falls back to writing PNG only when not in a kernel.

**Prompt file:** `autofit/quick_update_display_id.md` (to author after Phase A′ ships)

Touches:
- `PyAutoFit/autofit/non_linear/quick_update.py::BackgroundQuickUpdate._process_pending`
  (the call site that invokes `analysis.perform_quick_update`) — detect
  IPython kernel via `get_ipython()`, set a stable `display_id` on first
  call, `update_display` on subsequent calls.
- The visualizer call sites that produce the figure (`subplot_fit.png`)
  — return the `matplotlib.figure.Figure` alongside the save so the
  display layer can hand it to `IPython.display`.
- Falls back gracefully outside Jupyter (still writes PNG to disk).
- Smoke test: a notebook-style script that runs a tiny Nautilus fit and
  observes the display message stream.

Depends on **Phase A′** landing first (so the rendered figure's critical
curves / latents are JAX-jit fast enough to be worth watching).

## Phase D — Per-dataset end-to-end JAX-jit visualization tests

**The regression net.** The 2026-05-16 all-zero-source-plane Euclid bug snuck
through because the existing `modeling_visualization_jit*.py` scripts in
`autolens_workspace_test` only assert that *a fit completes*. None of them
assert that the **rendered images are non-trivial**. Adding three asserts
per script would have caught the regression locally before HPC.

**Block style — mirror `__Likelihood Sanity__`.** The project adopted the
inline `__Likelihood Sanity__` regression-net pattern on 2026-05-18
(`autolens_workspace_test#102/103`, `autogalaxy_workspace_test#50`). Each
guard block sits inline in the script before the Nautilus search, builds the
prior-median instance, calls the relevant analysis/fit methods, asserts the
expected invariant, and runs on **both** numpy and JAX backends. Phase D
adds parallel `__Visualization Sanity__` blocks following the same shape —
same file, same style, sibling header. The two block families catch
complementary bug classes: `__Likelihood Sanity__` catches
`figure_of_merit` vs `log_likelihood` regressions (PR #504 family);
`__Visualization Sanity__` catches all-zero-source / collapsed-source /
unconstrained-latent regressions (2026-05-16 Euclid family).

**Sequencing note.** Phase D's pilot (one `__Visualization Sanity__` block
on the imaging case, paired with the Phase A config flip) should ship
**before** the Phase D rollout sweep across all scripts. That way the
regression net is live in at least one place when the config flips, and a
single failure surfaces immediately.

**Prompt files:**
- `autolens_workspace_test/end_to_end_jax_viz_pilot.md` (paired with Phase A library PR — to author)
- `autolens_workspace_test/end_to_end_jax_viz_rollout.md` (the sweep — to author after pilot ships)

For every `modeling_visualization_jit*.py` and `visualization_jax*.py` across
both workspace_test repos, append assertions of the form:

```python
# Source plane reconstruction must be non-trivial.
src_image = result.max_log_likelihood_fit.galaxy_image_dict[
    "('galaxies', 'source')"
].array
assert float(src_image.sum()) > 0.0, "source-plane image is all-zero — viz regression"

# Critical curves overlay must produce at least one curve.
tc = result.max_log_likelihood_tracer.tangential_critical_curve_list_via_zero_contour_from()
assert len(tc) > 0, "no tangential critical curves — critical-curves regression"

# Einstein radius latent variable must be finite and positive.
er = result.max_log_likelihood_tracer.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "Einstein radius latent unconstrained — viz regression"
```

### Per-dataset assertion shapes

The imaging block above is the template. Each dataset has a slightly different
"the JIT viz path produced a non-trivial output" shape — characterised by what
fails when the viz path silently degenerates (i.e. the failure mode the
assertion is meant to catch).

#### Imaging (`AnalysisImaging` → `FitImaging`)

Failure mode caught: source-plane reconstruction = 0 (the 2026-05-16 Euclid
case).

```python
fit = result.max_log_likelihood_fit
src = fit.galaxy_image_dict[("galaxies", "source")].array
assert float(src.sum()) > 0.0, "source image all-zero (viz regression)"

tracer = result.max_log_likelihood_tracer
tc = tracer.tangential_critical_curve_list_via_zero_contour_from()
assert len(tc) > 0, "no tangential critical curves (zero_contour regression)"

er = tracer.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "einstein_radius latent unconstrained (zero_contour regression)"
```

#### Interferometer (`AnalysisInterferometer` → `FitInterferometer`)

**Identical lensing-side assertions as imaging** — the source-plane image and
critical-curve / Einstein-radius computations come from the same `Tracer`.
The 2026-05-16 all-zero-source-plane bug class applies here unchanged. Plus
one interferometer-specific assertion on visibilities:

```python
fit = result.max_log_likelihood_fit
src = fit.galaxy_image_dict[("galaxies", "source")].array
assert float(src.sum()) > 0.0, "source image all-zero (viz regression)"

tracer = result.max_log_likelihood_tracer
tc = tracer.tangential_critical_curve_list_via_zero_contour_from()
assert len(tc) > 0, "no tangential critical curves (zero_contour regression)"

er = tracer.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "einstein_radius latent unconstrained (zero_contour regression)"

# Interferometer-specific: model visibilities must not collapse to zero.
mv = np.asarray(fit.model_visibilities)
assert np.isfinite(mv).all() and np.abs(mv).sum() > 0.0, "model visibilities all-zero / nan"
```

#### Point source (`AnalysisPoint` → `FitPointDataset`)

No source-plane reconstructed *image* (the model is image-plane positions).
The failure-mode characteristic is "deflections evaluated as zero so source-
plane positions all land at (0, 0)" — the analogue of the imaging all-zero-
source bug.

```python
fit = result.max_log_likelihood_fit
sp = np.asarray(fit.positions_source_plane)
data_n = len(fit.positions)
assert len(sp) == data_n, "lost source-plane positions (deflection regression)"
assert np.isfinite(sp).all(), "non-finite source-plane positions"
assert float(np.max(np.linalg.norm(sp - sp.mean(axis=0), axis=1))) > 0.0, \
    "source-plane positions all coincident at one point (deflection collapse)"

# Lensing-side latents still apply — same Tracer machinery.
tracer = result.max_log_likelihood_tracer
er = tracer.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "einstein_radius latent unconstrained"
```

(Note: `FitPointDataset` has no `galaxy_image_dict` and no per-fit critical-
curve plot in its default subplot, so the imaging assertions don't transfer
directly. The point-source viz subplot is dominated by the
positions-on-image-plane scatter overlay; the position-collapse assertion is
its analogue of the source-plane-all-zero check.)

#### Quantity (`AnalysisQuantity` → `FitQuantity`)

No lensing. The viz subplot is the model field vs the target field with a
residual map. Failure mode: model field collapses to zeros (e.g. a JAX
tracer for the field passed through an `np.*` op silently returns zero).

```python
fit = result.max_log_likelihood_fit
model = np.asarray(fit.model.array)
data = np.asarray(fit.dataset.data.array)
assert np.isfinite(model).all(), "model field has nan/inf"
assert float(np.abs(model).sum()) > 0.0, "model field all-zero"
# Residual should not exceed pure-data RMS by an order of magnitude (lower
# bound on "fit did something").
rms_resid = float(np.sqrt(np.mean(np.asarray(fit.residual_map.array) ** 2)))
rms_data = float(np.sqrt(np.mean(data ** 2)))
assert rms_resid < 10.0 * rms_data, f"residual {rms_resid} >> data {rms_data} (fit collapsed)"
```

#### Ellipse (`AnalysisEllipse` → `FitEllipse`)

No lensing, no inversion — perimeter-sampled intensities along an ellipse.
Failure mode is the perimeter intensities being NaN (a JAX trace through the
ellipse-multipole helpers losing tracer values — exactly the 2026-05-15
`fix: vmap-blocker bugs in convert.py and FitEllipse cached_property` family).

```python
fit = result.max_log_likelihood_fit
intensities = np.asarray(fit.intensities)  # per-perimeter-sample
assert np.isfinite(intensities).all(), "ellipse perimeter intensities have nan/inf"
assert float(np.abs(intensities).sum()) > 0.0, "ellipse intensities all-zero"
# figure_of_merit must be finite (catches log_det / inversion collapse).
assert np.isfinite(float(fit.figure_of_merit)), "FoM nan/inf — fit collapsed"
```

(Exact attribute names — `fit.intensities` vs `fit.intensities_perimeter`
etc. — will be verified against the current `FitEllipse` API when the prompt
is authored; the principle is "the per-perimeter array that drives the chi²
is non-zero and finite.")

#### Weak lensing (`AnalysisWeak` → `FitWeak`)

**New dataset type added 2026-05-18** (PyAutoLens #523 / #525). The model is
a per-galaxy shear catalogue; the fit predicts (γ₁, γ₂) at each source-galaxy
position from the lens mass model. Failure modes: shear field collapses to
zero (mass model deflections evaluated as zero — analogue of the imaging
all-zero-source bug), or model shear is NaN (JAX trace through hessian-
derived shear losing tracer values).

```python
fit = result.max_log_likelihood_fit
gamma_pred = np.asarray(fit.model_shear)  # (N, 2) — (γ₁, γ₂) per source position
gamma_obs  = np.asarray(fit.dataset.shear)
assert gamma_pred.shape == gamma_obs.shape, "model shear shape mismatch"
assert np.isfinite(gamma_pred).all(), "model shear has nan/inf"
assert float(np.abs(gamma_pred).sum()) > 0.0, "model shear all-zero — deflections collapsed"

# Lensing-side latents still apply — same Tracer machinery.
tracer = result.max_log_likelihood_tracer
er = tracer.einstein_radius_via_zero_contour_from()
assert np.isfinite(er) and er > 0, "einstein_radius latent unconstrained"
```

(Exact attribute names — `fit.model_shear` vs `fit.shear_yx_2d` — will be
verified against the current `FitWeak` API at prompt-authoring time. The
principle: the predicted-shear array driving the chi² is non-zero, finite,
and shape-matches the observed catalogue.)

### Coverage audit (as of 2026-05-20)

| Dataset | autolens_workspace_test | autogalaxy_workspace_test | Gap |
|---|---|---|---|
| Imaging | ✓ `modeling_visualization_jit.py` + `_delaunay` + `_rectangular` + `visualization_jax.py` | ✓ `modeling_visualization_jit.py` + `visualization_jax.py` | needs `__Visualization Sanity__` |
| Interferometer | ✓ `modeling_visualization_jit.py` + `visualization_jax.py` | ✓ `modeling_visualization_jit.py` + `visualization_jax.py` | needs `__Visualization Sanity__` |
| Point source | ✓ `modeling_visualization_jit.py` + `visualization_jax.py` | n/a | needs `__Visualization Sanity__` |
| Quantity | n/a | ✓ `visualization_jax.py` only | **needs `modeling_visualization_jit.py`** + `__Visualization Sanity__` |
| Ellipse | n/a | **missing both** | **needs both scripts** + `__Visualization Sanity__` |
| **Weak lensing** | **missing both** *(new dataset type — PyAutoLens #523/#525, 2026-05-18)* | n/a (lens-only) | **needs both scripts** + `__Visualization Sanity__` |

Coverage gaps to author from scratch:
1. `autogalaxy_workspace_test/scripts/ellipse/modeling_visualization_jit.py` + `visualization_jax.py`
2. `autogalaxy_workspace_test/scripts/quantity/modeling_visualization_jit.py`
3. `autolens_workspace_test/scripts/weak/modeling_visualization_jit.py` + `visualization_jax.py`

All land in Phase D rollout alongside the assertions sweep on existing scripts.

## Phase E (longer-term, optional) — Pytree-register `ModelInstance` cascade

**Scope:** the real fix that would re-enable `use_jax_for_visualization=True`
end-to-end. Reverted PR #1278 failed because `ModelInstance` (the input to
`fit_from`) and the `Galaxy` / `LightProfile` / `MassProfile` types it carries
aren't pytree-registered, so `jax.jit(self.fit_from)(instance)` raises
`TypeError: ModelInstance not a valid JAX type`.

**Prompt file:** `autofit/model_instance_pytree_cascade.md` (to author when prior phases ship)

Cascade:
- `ModelInstance` (PyAutoFit `autofit/mapper/model.py:385`)
- `Collection` (PyAutoFit)
- `Galaxy` (PyAutoGalaxy `autogalaxy/galaxy/galaxy.py:31`)
- All `LightProfile` subclasses (`lp.*`, `lp_linear.*`, `lp_basis.*`, `lp_operated.*`)
- All `MassProfile` subclasses (`mp.*`)
- All `Pixelization` mesh classes (`Delaunay`, `RectangularUniform`,
  `KNNBarycentric`, etc.)

Scaffolding exists: `autoarray.abstract_ndarray.register_instance_pytree(cls, no_flatten=...)` is already used for `FitImaging`, `Tracer`,
`DatasetModel`, `AdaptImages`, `FitPoint`, `FitQuantity`. Extending to the
input side is mechanical — the design decision per class is which fields are
"dynamic" (parameter values → leaves) vs "static" (class identity, prior
metadata → aux). Roughly 100-200 lines + per-class tests.

Once this lands, `use_jax_for_visualization=True` default can be re-attempted
(the original PR #1278 intent). At that point, `fit_for_visualization` is
fully fused under XLA and the visualization is potentially 10-50× faster than
the current JAX-eager path.

## Phase F (deferred) — Subprocess visualization for failure isolation

The original `viz-subprocess-feasibility` task (issue #1279, closed). Banked
the picklability finding (`FitImaging` round-trips cleanly through stdlib
`pickle` on every tested model — see closed-issue comment for the spike
table).

Reasons it's not in the critical path:

- Phase A+B+C should make in-process viz fast enough for live Jupyter cells.
- Subprocess viz would still be useful for **failure isolation** (a viz bug
  doesn't take the search down) and for the long-tail of cases where viz
  somehow remains slow despite Phases A-E. If those needs become acute,
  re-enter the design from the spike findings (`mp.Process` + `Queue` with
  drop backpressure — the picklability finding makes the simplest design
  viable).

Not currently authored as a prompt. Keep the door open; do not build until
needed.

## Background

- Predecessor roadmap: `z_features/complete/jax_visualization.md` (Phases
  0-3 shipped; Phase 4 became `viz-subprocess-feasibility`).
- The 2026-05-16 Euclid all-zero-source-plane regression that triggered this
  pivot: `complete.md::jax-viz-default-broken`.
- Closed issue #1279 carries the picklability spike findings in its closing
  comment for the archaeological record.
