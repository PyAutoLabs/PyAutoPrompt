# Visualization Profiling: Cluster Simulator

## Background

The cluster-scale simulator at `autolens_workspace/scripts/cluster/simulator.py`
used to take well over five minutes end-to-end. The natural assumption was
that the `PointSolver` was the culprit — solving the lens equation is iterative
and slow in numpy. Issue #89 JAX-jitted the solver and dropped its cost to
about 22 seconds (most of which is the one-off JIT compile), but the
simulator still takes around three minutes on a warm cache.

When I instrumented each phase, the new picture was clear:

| Phase                                | Time |
|--------------------------------------|------|
| Imports + pytree registration        |  ~3s |
| Solver compile + first solve         | ~19s |
| Solver second solve (cached)         |  ~2s |
| `SimulatorImaging.via_tracer_from`   | ~92s |
| `aplt.subplot_imaging_dataset`       |  ~1s |
| `aplt.subplot_tracer`                | ~51s |
| `aplt.subplot_galaxies_images`       |  ~2s |

Almost all of the remaining runtime sits in two places — multi-plane image
rendering and the multi-panel `subplot_tracer` plotter. Neither is JAX-jitted,
and both scale with grid pixel count, the depth of over-sampling near galaxy
centres, and the number of internal panels rendered. This task is about
understanding those two costs well enough to make an informed call on what
(if anything) to do about them.

## Why this matters

Cluster-scale examples are not just one simulator. Group-scale and the
upcoming scaling-relation cluster examples will share the same multi-plane
ray-tracing path through `via_tracer_from` and the same plotter calls at the
end of every script. If we can JAX-accelerate either piece, every
cluster-style example will get the same speedup for free. If we can't,
that is itself a useful answer — it tells us the right place to push back
on the imaging grid resolution, or to swap `subplot_tracer` for a leaner
plotter inside cluster examples.

## What's already in place

A profiling script lives at:

`autolens_workspace_developer/visualization_profiling/imaging/cluster.py`

It rebuilds the cluster geometry from the simulator (2 main lens galaxies +
NFW host halo + 2 multi-plane sources at z=1.0 and z=2.0), then independently
times:

1. Image rendering across four `imaging_grid` variants — full resolution
   (1000x1000 with sub_size=[32,8,2]), half resolution, half resolution with
   lighter over-sampling, and full resolution with no over-sampling. The aim
   is to separate the cost of pixel count from the cost of sub-sampling.
2. `subplot_imaging_dataset` once on a fast pre-rendered dataset.
3. `subplot_tracer` and `subplot_galaxies_images` across four viz grid
   resolutions (50x50 through 500x500) all spanning the same 100" field.

It writes nothing to disk and prints a per-section table at the end.

## Task

Run the script as-is, look at the table, and answer:

- **For image rendering**: of the difference between the 1000x1000-with-heavy-
  over-sampling baseline and the 500x500-with-lighter-over-sampling variant,
  how much is pixel count and how much is over-sampling? If over-sampling
  is the bigger lever, is the cluster simulator's choice of `sub_size=32`
  near each centre actually justified — i.e. would `sub_size=8` give a
  visibly worse simulated image?
- **For `subplot_tracer`**: how does cost scale with viz grid resolution? It
  is roughly linear in pixel count, or is there a fixed per-panel overhead
  that dominates at small grids? Counting the panels and timing each in
  isolation would clarify this.
- **JAX feasibility**: `via_tracer_from` and the plotter ray-tracing both
  ultimately call `Tracer.image_2d_from` (or its multi-plane equivalent).
  How much of that path is already pytree-friendly? If `Tracer` is registered
  as a pytree (it is, when an `AnalysisPoint` constructs it with
  `use_jax=True`) does `jax.jit(simulator.via_tracer_from)` succeed, or does
  it bail out somewhere — and if so, where exactly?

The end goal is a short report inside the script's docstring or a sibling
markdown file in the same folder, recommending one of:

- **JAX-jit the renderer** — if the bottleneck is amenable and a clean
  pytree path exists. Sketch the API surface that would change.
- **Drop default grid sizes / over-sample levels in cluster examples** — if
  JAX is not feasible. State which numbers, with reasoning grounded in the
  measurements.
- **Live with it** — if the costs are fundamental and the cluster examples
  just have to be slow at this scale.

## Constraints and notes

- The cluster simulator itself is in good shape after issue #89 — the solver
  is JIT-compiled, the script outputs sensible multi-image positions for
  both sources, and the 3-minute total is acceptable while we figure out
  the rest. So this is **not blocking the cluster work**; it is a follow-up
  whose output should feed back into the cluster scripts in a later PR.
- The profiling script is intended to be run, edited, re-run iteratively.
  Add new variants, drop old ones, instrument deeper into `via_tracer_from`
  if needed.
- Do not modify `autolens_workspace/scripts/cluster/simulator.py` from this
  task. Any user-visible changes (e.g. lowering default grid resolution)
  should land as a separate cluster-workspace PR once the profiling story
  is settled.

## Companion scripts to copy structure from

The file is laid out like
`autolens_workspace_developer/jax_profiling/imaging/mge.py` — same
`Timer.section` context manager, same end-of-run summary table. Mirror that
style if you add new instrumentation files alongside it (e.g.
`subplot_tracer_panel_breakdown.py`).
