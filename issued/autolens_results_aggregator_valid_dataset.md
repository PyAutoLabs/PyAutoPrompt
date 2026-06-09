# Fix autolens results aggregator dataset reload

Original user request:

> continue

Release report context:

The PyAutoBuild release run reports two failures in `autolens_workspace`:

- `scripts/guides/results/aggregator/data_fitting.py`
- `scripts/guides/results/aggregator/models.py`

Both fail when `al.agg.ImagingAgg(...).dataset_gen_from()` attempts to reconstruct an imaging dataset:

```text
TypeError: 'NoneType' object is not subscriptable
  PyAutoGalaxy/autogalaxy/aggregator/agg_util.py:101
  header = aa.Header(header_sci_obj=fit.value(name=name)[0].header)
```

Reproduction on current `autolens_workspace/main` using the PyAutoBuild environment confirms that the aggregator finds fits under `output/results_folder`, but `agg.values("dataset.mask")` returns `[None, None]` for stale or incompatible results. The output tree can contain a completed fit without `image/dataset.fits`, while the aggregator tutorials require that FITS artifact.

Fix the workspace results aggregator flow so the affected scripts only scrape reusable helper results that contain `image/dataset.fits`, while preserving the normal tutorial behavior and PyAutoBuild test-mode path handling.
