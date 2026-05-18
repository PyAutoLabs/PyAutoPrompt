__Outstanding__ (sequenced)

1. [cluster/2_scaling_relation.md](../cluster/2_scaling_relation.md) —
   make scaling-relation members the default in the three
   `autolens_workspace/scripts/cluster/` scripts (10 low-mass members on a
   luminosity–mass scaling relation, always via the CSV interface mirroring
   `imaging/features/scaling_relation/`). Simulator extension must still
   produce multiple images in the right positions.
2. [cluster/3_test_workspace.md](../cluster/3_test_workspace.md) —
   stand up `autolens_workspace_test/scripts/cluster/`:
   - `simulator.py` re-using the workspace simulator (so scaling galaxies
     flow through into every downstream test).
   - Move `scripts/imaging/visualization_cluster.py` →
     `scripts/cluster/visualization.py`.
   - Likelihood sanity check: source-plane chi² (`FitPositionsSource`) and
     image-plane chi² (`FitPositionsImagePair`) against perturbed mass-model
     inputs (0.1% / 1% / 5%) — chi² ≈ 0 at the truth, log-likelihood drops
     monotonically as perturbation grows.
   - Multi-redshift variant: max-likelihood only when per-source redshifts
     are correct.
   - `scripts/jax_likelihood_functions/cluster/single_plane.py` +
     `multi_plane.py` with numerical assertions against the perturbation
     test case.
3. [cluster/4_likelihood_function.md](../cluster/4_likelihood_function.md) —
   `autolens_workspace/scripts/cluster/likelihood_function.py` step-by-step
   walkthrough: source-plane chi² first, then image-plane chi². Comment
   density and structure should match the other workspace
   `likelihood_function.py` scripts.
4. [cluster/5_profiling.md](../cluster/5_profiling.md) — two scripts in
   `autolens_profiling/likelihood/cluster/` that time the source-plane and
   image-plane likelihood paths, following the per-model breakdown style of
   the rest of `autolens_profiling/likelihood/`. Depends on (3) for the
   reference step-by-step decomposition.

__Shipped__

- 0_simulator_cluster — `cluster-simulator` (autolens_workspace#77, PyAutoLens#465, 2026-04-20)
  and v2 `cluster-simulator-jax-multiplane` (autolens_workspace#91, 2026-04-27). Built the
  small multi-plane cluster simulator (2 main lens galaxies + standalone NFW host halo + 2
  sources at z=1.0/2.0) and JAX-jitted the PointSolver. Outputs `point_datasets.csv` with a
  per-source `redshift` column as the canonical hand-editable cluster input.
- 1_visualization — `cluster-viz-prototype` (autolens_workspace_test#75, 2026-05-07). Landed
  at `autolens_workspace_test/scripts/imaging/visualization_cluster.py` (not under
  `scripts/cluster/`; the move to `scripts/cluster/visualization.py` is folded into Outstanding #2).
  Produces three reference PNGs (overlaid positions, per-source grid, cluster-tuned
  critical curves) into `autolens_workspace/dataset/cluster/simple/`. Library-side `aplt`
  promotion is deferred — see __Deferred__ below.
- 2_modeling_cluster — `cluster-modeling-v2` (autolens_workspace#174, PR #175, 2026-05-18).
  Full rewrite of `scripts/cluster/modeling.py` against the multi-plane simulator:
  `al.list_from_csv` for per-source redshifts, JSON centres for main lenses + host halo,
  2×`dPIEMassSph` + 1×`NFWMCRLudlowSph` host halo + 2×`Point` source galaxies, factor-graph
  `search.fit` returning `result_list`. Auto-sim guard tightened to check `data.fits`.
  Smoke 7/7 green.

__Deferred (future prompts, after Outstanding chain lands)__

- `cluster/start_here.py` rewrite — currently parked in `no_run.yaml`; needs to follow
  whatever shape `modeling.py` settles into once scaling-relation members are in.
- Lens/source CSV API — `lens_galaxies.csv` + `source_galaxies.csv` mirroring
  `al.galaxy_table_from_csv` from `imaging/features/scaling_relation/`. Only worth doing
  once the rewritten cluster `modeling.py` has matured through Outstanding #1.
- `aplt` plotter promotion — promote the most useful patterns from the visualization
  prototype (per-source colouring, per-image-group zoom grid, cluster-tuned critical-curve
  overlay) into the library `aplt` / `Visuals2D` / `Include2D` interfaces.
