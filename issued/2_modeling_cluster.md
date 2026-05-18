Cluster modeling: minimal first-pass rewrite of `modeling.py`.

The cluster simulator at `@autolens_workspace/scripts/cluster/simulator.py` has long since been
rewritten to a small multi-plane cluster (2 main lens galaxies + standalone NFW host halo + 2
sources at distinct redshifts z=1.0 and z=2.0, with point-source positions written to a combined
`point_datasets.csv` carrying a per-source `redshift` column). The paired
`@autolens_workspace/scripts/cluster/modeling.py` was **never updated** to match — it still loads
five separate `point_dataset_{i}.json` files, hardcodes source `redshift=1.0`, and composes the
pre-rewrite "10 extra galaxies on a scaling relation + 5 sources" model. `cluster/modeling` and
`cluster/start_here` are both parked in `autolens_workspace/config/build/no_run.yaml`.

This prompt is a **minimal first pass**: rewrite `modeling.py` to actually pair to the current
simulator. `start_here.py` stays parked; a CSV-driven lens/source-galaxy interface is deferred to a
later prompt.

__Required changes (autolens_workspace/scripts/cluster/modeling.py)__

1. **CSV-load the point datasets.** Replace the `for i in range(5)` JSON loop (and the docstring
   example block that currently *describes* CSV loading) with the real call:

   ```python
   dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
   ```

   Each `PointDataset` exposes `.positions`, `.positions_noise_map`, and `.redshift` — use all
   three downstream. Do not retain a JSON fallback.

2. **Load the centre files the simulator actually writes.** Replace
   `extra_galaxies_centre_list.json` + `extra_galaxies_luminosities.json` (these no longer exist)
   with:

   - `main_lens_centres.json` — `Grid2DIrregular` of the 2 main lens galaxy centres.
   - `host_halo_centre.json` — `Grid2DIrregular` of the 1 host halo centre.

   See `@autolens_workspace/scripts/cluster/simulator.py` lines ~511–522 for the exact filenames
   and how they're written.

3. **Compose the lens model paired to the simulator.** The truth model has three categories. Each
   maps to a model component as follows:

   - **2 main lens galaxies** at `redshift_lens = 0.5`. Each uses `al.mp.dPIEMassSph` with centre
     fixed from `main_lens_centres.json[i]`. Free parameters per galaxy: `ra`, `rs`, `b0`. See
     `simulator.py` lines ~72–86 for parameter meaning and physical ranges — anchor priors there
     (e.g. `ra` ~ 0.05–0.1", `rs` ~ 10–30", `b0` log-uniform around the truth Einstein-scale).

   - **1 host halo galaxy** at `redshift = 0.5`, with `al.mp.NFWMCRLudlowSph` mass, centre fixed
     from `host_halo_centre.json[0]`, `mass_at_200` a free `LogUniformPrior` bracketing `10**15.3`.
     The `redshift_object` and `redshift_source` plumbing must match the simulator —
     `redshift_object = 0.5`, `redshift_source = max(source_redshifts)` so the concentration is
     anchored to the furthest source plane.

   - **2 source galaxies**, one per `PointDataset` in `dataset_list`. The key change is that the
     **source redshift comes from the dataset, not a literal**:

     ```python
     for i, dataset in enumerate(dataset_list):
         point = af.Model(al.ps.Point)
         positions = np.atleast_2d(dataset.positions)
         point.centre_0 = af.GaussianPrior(mean=float(np.mean(positions[:, 0])), sigma=3.0)
         point.centre_1 = af.GaussianPrior(mean=float(np.mean(positions[:, 1])), sigma=3.0)
         source = af.Model(al.Galaxy, redshift=dataset.redshift, **{f"point_{i}": point})
     ```

4. **Drop the entire `extra_galaxies` + scaling-relation block** (the old 10-satellite
   `dPIEMassSph` scaling block + `extra_galaxies=af.Collection(**extra_galaxies_dict)` wiring).
   The current simulator does not emit a scaling-relation member population; that's a follow-up.

5. **Rewrite the `__Model__` docstring.** It currently claims "ten extra lens galaxies with
   `DPIEPotentialSph`", "five source galaxies", and "N=22" — all wrong. Replace with an accurate
   description of the 2-main + halo + 2-source model and its actual free-parameter count.

6. **Reactivate the script.** Remove `cluster/modeling` from
   `autolens_workspace/config/build/no_run.yaml`. Leave `cluster/start_here` parked (a separate
   prompt will rewrite it later).

__Verification__

Before opening the PR, confirm the script runs end-to-end under test mode from
`autolens_workspace`:

```bash
PYAUTO_TEST_MODE=2 PYAUTO_SKIP_FIT_OUTPUT=1 PYAUTO_SKIP_VISUALIZATION=1 \
    python scripts/cluster/modeling.py
```

It should compose the model, call the analysis log-likelihood once, and exit cleanly. The
canonical workspace smoke run (`/smoke_test` over the modeling.py path) must also pass.

__Out of scope (do NOT touch in this prompt)__

- **No new lens/source-galaxy CSV API.** A `lens_galaxies.csv` / `source_galaxies.csv` interface
  mirroring `al.galaxy_table_from_csv` is the natural next step but is *deferred* — for 2 main
  galaxies the existing JSON+per-galaxy code path is fine and lets us see the rewritten modeling
  work in isolation. The reference pattern (when we eventually do this work) lives in
  `@autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` lines ~180–215.
- **No `start_here.py` rewrite.** Stays in `no_run.yaml`.
- **No library changes.** All edits land in the workspace.
- **No `extra_galaxies` scaling-relation work.** Adding scaling-relation satellite members back
  in is its own follow-up prompt, paired with a simulator extension that emits them.

__Reference files__

- `@autolens_workspace/scripts/cluster/simulator.py` — truth model, parameter ranges, redshift
  conventions, JSON+CSV output filenames.
- `@autolens_workspace/scripts/point_source/start_here.py` — current galaxy-scale point-source
  modeling baseline (priors, `PointSolver` setup, `AnalysisPoint` wiring).
- `@autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — reference for the
  deferred lens/source CSV API. Do not adopt it here; just preserve as a pointer for next round.
