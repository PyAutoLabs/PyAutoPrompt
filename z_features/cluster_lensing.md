__Outstanding__

- [2_modeling_cluster.md](../issued/2_modeling_cluster.md) — minimal first-pass rewrite of
  `autolens_workspace/scripts/cluster/modeling.py` to pair with the current simulator
  (CSV-loaded `PointDataset`s, per-source redshift linkage, 2-main + halo + 2-source model,
  drop the stale `extra_galaxies` block). `start_here.py` stays parked until this lands.

__Shipped__

- 0_simulator_cluster — `cluster-simulator` (autolens_workspace#77, PyAutoLens#465, 2026-04-20)
  and v2 `cluster-simulator-jax-multiplane` (autolens_workspace#91, 2026-04-27). Built the
  small multi-plane cluster simulator (2 main lens galaxies + standalone NFW host halo + 2
  sources at z=1.0/2.0) and JAX-jitted the PointSolver. Outputs `point_datasets.csv` with a
  per-source `redshift` column as the canonical hand-editable cluster input.
- 1_visualization — `cluster-viz-prototype` (autolens_workspace_test#75, 2026-05-07). Landed
  at `autolens_workspace_test/scripts/imaging/visualization_cluster.py` (not under
  `scripts/cluster/`). Produces three reference PNGs (overlaid positions, per-source grid,
  cluster-tuned critical curves) into `autolens_workspace/dataset/cluster/simple/`. Library-
  side `aplt` promotion is deferred — see __Deferred__ below.

__Deferred (future prompts, after `2_modeling_cluster` lands)__

- Lens/source CSV API — `lens_galaxies.csv` + `source_galaxies.csv` mirroring
  `al.galaxy_table_from_csv` from `imaging/features/scaling_relation/`. Only worth doing once
  the rewritten cluster `modeling.py` has matured.
- `cluster/start_here.py` rewrite — currently parked in `no_run.yaml`; needs to follow
  whatever shape `modeling.py` settles into.
- Scaling-relation cluster members — simulator extension that emits a member galaxy
  population on a luminosity-mass scaling relation, plus matching modeling support.
- `aplt` plotter promotion — promote the most useful patterns from the visualization
  prototype (per-source colouring, per-image-group zoom grid, cluster-tuned critical-curve
  overlay) into the library `aplt`/`Visuals2D`/`Include2D` interfaces.
