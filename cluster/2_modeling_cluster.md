The cluster simulator at `@autolens_workspace/scripts/cluster/simulator.py` now emits a combined
`point_datasets.csv` with a `redshift` column — one row per observed multiple image, grouped by source `name`,
with every row in a group carrying the same source redshift. `al.list_from_csv` round-trips this into a
`List[PointDataset]` where each dataset exposes `dataset.redshift`.

In the modeling script, the redshift of the Galaxy objects, currently hardcoded as redshift=1.0, is not paired
to the redshfits in the dataset loaded via `al.list_from_csv`. Thus, you need to link the redshift of the Galaxys
to the redshift in the point dataset.

We need to update the lens model to be paired to cluster/simulator.py, which has 2 main galaxies, a host halo, 
galaxies on a scaling relaiton and 2 sources at different redshifts. The current modelingpy scfript assumes a 
completely different moddel, with only one main galaxies and it uses the extra_galaxies API which the main example
now does not use.

However, we need to update the modeling API to use a csv API through, for example both main lens galaxies
should be in a .csv file which is loaded. To be honest, we need to think long and hard about how we
map lens galaxies to a csv, source galaixes to a csv. Note that an exsmple for csv loading of scaling
galaxies is given in autolens_Workspace/scripts/imaging/features/scaling_relation.py

So this run can be a first go at the csv interface and then we'll interact and improve on it with a future promtpo
