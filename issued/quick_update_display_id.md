# Phase C — Live Jupyter cell rendering via `IPython.display.update_display`

Adds in-cell live updating of the quick-update visualization when a
PyAutoFit search runs inside a Jupyter / Colab kernel. The background
threading + JAX-fast figure generation already exist (Phase A′ +
`BackgroundQuickUpdate`, Phase B latents); this phase wires
`IPython.display` so the cell *displays and refreshes in place* during
the fit, instead of just writing PNG files to disk.

## What already exists (don't re-implement)

- **Background thread:** `PyAutoFit/autofit/non_linear/quick_update.py`
  ships `BackgroundQuickUpdate` — a daemon `threading.Thread` with a
  latest-only drop backpressure policy. It calls
  `analysis.perform_quick_update(paths, instance)` off the search hot
  path. Wired into the Nautilus sampler at
  `search/nest/nautilus/search.py:196,216` via the
  `background_quick_update` kwarg on `Fitness`.
- **JAX-fast visualization:** PR #1278 / PR #434 / PR #435 mean
  `analysis.perform_quick_update` runs the JIT-cached `fit_for_visualization`
  + zero_contour critical curves with warm-call latency in the ~10s of ms.
- **Output to disk:** the visualizer writes `subplot_fit.png` (and
  related) under `paths.image_path` on every quick update.

What's missing: when a notebook user runs a search, the PNGs land on
disk but the user has to open them externally. The user-facing goal from
the original tracker is:

> "Visualization … would happen … with Jupyter Notebooks update the
> quick visuals on the fly during modeling … as a cell which updates."

## What to change

### 1. Add IPython display layer to `BackgroundQuickUpdate._process_pending`

`@PyAutoFit/autofit/non_linear/quick_update.py`

After `analysis.perform_quick_update(paths, instance)` returns, the
worker should:

1. Detect whether we're running inside an IPython kernel (Jupyter / Colab).
   Use `IPython.get_ipython()` — returns a kernel instance when inside
   one, `None` otherwise.
2. If yes: locate the `subplot_fit.png` (or configured primary plot)
   that `perform_quick_update` just wrote.
3. Use `IPython.display.update_display` with a **stable `display_id`**
   so subsequent updates *replace* the previous cell output rather than
   appending. The first call uses `display(...)` with `display_id=True`
   to initialise; subsequent calls use `update_display(...)` with the
   same id.

Suggested shape:

```python
class BackgroundQuickUpdate:
    def __init__(self, convert_jax: bool = False, display_id: str = "pyauto_fit_progress"):
        self._convert_jax = convert_jax
        self._display_id = display_id
        self._display_initialised = False
        # ... existing state ...

    def _is_ipython_kernel(self) -> bool:
        try:
            from IPython import get_ipython
        except ImportError:
            return False
        ipy = get_ipython()
        return ipy is not None and "IPKernelApp" in getattr(ipy, "config", {})

    def _push_to_ipython(self, paths):
        """Display or update the primary subplot in the active IPython cell."""
        png_path = Path(paths.image_path) / "subplot_fit.png"
        if not png_path.exists():
            return  # nothing to show — visualizer didn't write that frame
        try:
            from IPython.display import Image, display, update_display
        except ImportError:
            return

        img = Image(filename=str(png_path))
        if not self._display_initialised:
            display(img, display_id=self._display_id)
            self._display_initialised = True
        else:
            update_display(img, display_id=self._display_id)

    def _process_pending(self):
        # ... existing worker body that calls analysis.perform_quick_update ...
        try:
            analysis.perform_quick_update(paths, instance)
        except NotImplementedError:
            return
        except Exception:
            logger.exception("Background quick-update raised (ignored).")
            return

        # NEW: push to active IPython cell if we're in one.
        if self._is_ipython_kernel():
            try:
                self._push_to_ipython(paths)
            except Exception:
                logger.exception("IPython display update raised (ignored).")
```

**Key points:**

- **Graceful fallback** outside IPython kernels — `_is_ipython_kernel()`
  returns False, no display call fires. Script users get the existing
  PNG-on-disk behaviour, nothing changes for them.
- **`Image(filename=path)`, not `Figure`.** Reading the PNG from disk
  avoids cross-thread matplotlib figure handling — matplotlib figures
  are not thread-safe and `_process_pending` runs on the
  `BackgroundQuickUpdate` daemon worker. PNG → bytes → IPython.display
  is thread-safe.
- **Display ID stays stable across the search** so the cell output is
  one continuously updating image, not a wall of stacked frames.
- **Quiet on failure** — log and swallow; do not crash the search.

### 2. Locate the right image path

The plot file paths come from the configured `paths.image_path`. For
imaging the canonical entry is `subplot_fit.png`; for interferometer
`subplot_fit.png` (per the visualizer naming convention). Check both
exist before picking. If neither, no-op (script wrote no quick-update
plots — e.g. `PYAUTO_FAST_PLOTS=1`).

Worth supporting a small ordered list (`subplot_fit.png`, `fit.png`,
`subplot_tracer.png`) so that whichever the analysis class wrote is
displayed.

### 3. Optional: opt-out env var

For users who don't want the IPython display behaviour even in a notebook
(e.g. they're scripting via `papermill` and don't want display side
effects), add `PYAUTO_DISABLE_IPYTHON_DISPLAY=1` as an opt-out. Default
behaviour: display when in a kernel.

### 4. Unit tests

`@PyAutoFit/test_autofit/non_linear/test_quick_update.py`

Three tests:

- `_is_ipython_kernel` returns False when not inside IPython (i.e.
  during normal pytest execution). Verifies the script-mode fallback.
- `_push_to_ipython` is a no-op when no PNG exists at the expected
  path (e.g. fast-plots-disabled run).
- With a mock IPython environment + a synthetic PNG, the first call
  uses `display(... display_id="pyauto_fit_progress")` and subsequent
  calls use `update_display(...)` with the same id. Mock the IPython
  imports so the test runs without a live kernel.

## Smoke validation

After implementation, create a tiny notebook-style script (Python `.py`
that imitates a notebook kernel by importing `IPython.display`) and
verify it produces `display_data` / `update_display_data` messages on
the kernel side. Alternatively, run an actual `jupyter nbconvert --execute`
on a notebook that fits a small Nautilus search and inspect the cell
output for the updating image.

## Out of scope

- Subprocess visualization. `BackgroundQuickUpdate` (threading) is the
  shipped approach; this phase doesn't revisit the deferred Phase F
  subprocess design.
- Matplotlib figure capture / cross-thread figure handling. We avoid
  this entirely by reading the PNG from disk after `perform_quick_update`
  writes it.
- Colab-specific tweaks. The IPython kernel detection should cover
  Colab automatically (it runs an IPython kernel). If Colab-specific
  display quirks emerge, follow up in a separate prompt.
- Multi-figure layouts (residual / source plane shown side-by-side in
  the cell). For now, a single primary subplot per cell is sufficient.

## Verification

1. **Unit tests:** `pytest test_autofit/non_linear/test_quick_update.py` —
   all three new tests pass; existing tests unchanged.
2. **Script mode:** run any existing PyAutoFit search from a plain
   Python script. Confirm no behaviour change (PNG still lands on disk;
   no IPython side effects; no warnings about missing IPython).
3. **Notebook smoke:** create a minimal notebook that runs a small
   `n_live=25, n_like_max=200` Nautilus fit on the `autofit_workspace`
   Gaussian example with `iterations_per_quick_update=50` and
   `background_quick_update=True`. After execution, the cell should
   show a single image (not multiple stacked frames) reflecting the
   final iteration's `subplot_fit.png`.
4. **End-to-end:** rerun the Euclid pipeline's
   `start_here.py` (no `PYAUTO_DISABLE_JAX`, `PYAUTO_TEST_MODE=1`)
   *inside* a jupyter kernel (e.g. `jupyter nbconvert --execute` on a
   wrapper notebook) and confirm the cell shows a live-updating
   `subplot_fit` image during the fit.

## References

- `z_features/fast_visualization.md` — parent tracker, Phase C section.
- PyAutoFit commit `1fee93174` — added `BackgroundQuickUpdate`. The
  worker hook for the IPython display layer goes inside
  `_process_pending`.
- PyAutoFit `autofit/non_linear/fitness.py` — call sites that drive
  `BackgroundQuickUpdate.submit(...)`. No changes needed there; the
  display layer is entirely inside the worker.
- `IPython.display.update_display` docs:
  https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
