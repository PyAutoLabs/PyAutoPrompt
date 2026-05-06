# JAX JIT Profiling: Imaging Delaunay (update existing script)

## Context

`autolens_workspace_developer/jax_profiling/imaging/delaunay.py`
already exists (≈1000 lines) and profiles a Delaunay-pixelization
source under `jax.jit` + eager + vmap. It was written earlier in
the JAX migration, before three pytree-readiness pieces landed on
`main`, so its model construction / JAX-leaf plumbing does not
necessarily reflect the current "full pytree" approach used by
`mge.py` and `pixelization.py`.

Companion scripts that *do* use the current pytree approach:

- `imaging/mge.py`
- `imaging/pixelization.py` (RectangularAdaptDensity)

These two should be treated as the reference structure.

## Pytree infrastructure (already shipped — align `delaunay.py` with it)

Three library pieces have landed on `main` since the current
`delaunay.py` was written:

- **PyAutoFit#1222** — `TuplePrior` registered as a JAX pytree. On
  a typical Isothermal+Shear+Delaunay model this lifts live JAX-leaf
  count from 3 to O(100+), so
  `jax.jit(AnalysisImaging.log_likelihood)` flows through the full
  model rather than freezing most of it as constants.
- **PyAutoArray#279** — Jacobi preconditioning of the NNLS
  curvature matrix.
- **PyAutoArray#282** — `nnls_target_kappa=1.0e-2` config default.

The aim of this task is to bring `imaging/delaunay.py` in line with
`mge.py` / `pixelization.py` so all three imaging profiling scripts
share a common structure and pytree-readiness baseline.

## Task

Update `autolens_workspace_developer/jax_profiling/imaging/delaunay.py`:

1. **Audit the current model construction** — confirm all priors
   flow as pytree leaves (not frozen Python constants) by counting
   JAX leaves on the model pytree and printing the count. It should
   be comparable to `mge.py` / `pixelization.py` on an equivalent
   lens setup, not a small handful.
2. **JIT path parity with `mge.py`**: `jax.jit` wraps the
   `Fitness.call` / `AnalysisImaging.log_likelihood` equivalent,
   measures first-call (compile) time + N steady-state repeats,
   reports mean / median / stdev.
3. **Eager baseline**: matching `FitImaging` figure-of-merit /
   log-likelihood, asserted numerically equal to the JIT path
   within `rtol`.
4. **vmap path**: batch `batch_size` parameter vectors through
   `fitness._vmap`, report per-likelihood cost.
5. **Results artefact**: JSON + PNG into
   `jax_profiling/imaging/results/` with the same schema as
   `mge.py` / `pixelization.py`.

## Expected output

`delaunay.py` runs end-to-end via:

```bash
cd jax_profiling/imaging
python delaunay.py
```

Producing:

- JIT vs eager timing comparison
- Numerical-agreement assertion PASS
- vmap batch throughput measurement
- JAX leaf count printed at model-construction time, matching the
  order of magnitude seen in `mge.py` / `pixelization.py`

## Likely blockers to raise if encountered

- If `delaunay.py` currently relies on model construction that
  freezes most priors as constants, rewriting it to use the pytree
  approach may expose a JAX-tracing issue in the Delaunay mapper
  path (e.g. `delaunay_2d_interpolation` gradient behaviour). If
  so, file a separate issue and park this one until the library
  issue is addressed — do not work around it with a "partial pytree"
  hack.
- Delaunay construction itself (`scipy.spatial.Delaunay`) is not
  JAX-traceable. The existing script already handles this by
  building the triangulation outside the JIT boundary — preserve
  that boundary when refactoring.
