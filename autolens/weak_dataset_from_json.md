Fix `al.from_json` round-trip for `WeakDataset` / `ShearYX2DIrregular`.

The simulator script at `@autolens_workspace/scripts/weak/simulator.py` writes the simulated `WeakDataset` to
`dataset/weak/simple/dataset.json` via `al.output_to_json`, but `al.from_json(file_path=...)` cannot read it
back. The traceback is:

    TypeError: VectorYX2DIrregular.__init__() missing 1 required positional argument: 'values'

at `PyAutoFit/autofit/mapper/model_object.py:223` (`return cls_(...)`), called from
`PyAutoConf/autoconf/dictable.py:316`.

The bug surfaced when writing `@autolens_workspace/scripts/weak/fit.py` (PR PyAutoLens #525 / workspace #188,
step 3 of the weak-lensing series). That tutorial works around it by rebuilding the dataset inline using
`SimulatorShearYX(seed=1)` rather than loading from disk — but that's a real limitation: any downstream
consumer wanting to load a serialised shear catalogue is blocked.

## Root cause hypothesis (inspect to confirm)

`ShearYX2DIrregular` inherits from `aa.VectorYX2DIrregular` (see `@PyAutoGalaxy/autogalaxy/util/shear_field.py`).
The `from_dict` round-trip for irregular vector fields likely needs the underlying `values` and `grid` arrays
keyed correctly in the serialised dict. The generic `dictable` mechanism in `PyAutoConf` probably doesn't
know about the `values` / `grid` constructor convention used by `aa.VectorYX2DIrregular`.

## Suggested approach

1. Look at how `aa.Grid2DIrregular` round-trips (it must, since `dataset.positions` is just a property on
   `dataset.shear_yx.grid`). If `Grid2DIrregular` has a custom `__init_subclass__`, `to_dict`, or `from_dict`
   hook, mirror that pattern for `VectorYX2DIrregular` and `ShearYX2DIrregular`.
2. Alternatively: add a custom serializer for `WeakDataset` in `@PyAutoLens/autolens/weak/dataset.py` that
   handles its three components (`shear_yx`, `noise_map`, `name`) by their public types.
3. Add a regression test under `@PyAutoLens/test_autolens/weak/test_dataset.py`:
   - build a `WeakDataset` via `SimulatorShearYX`
   - `al.output_to_json(obj=dataset, file_path=tmp_path / "dataset.json")`
   - `loaded = al.from_json(file_path=tmp_path / "dataset.json")`
   - assert `loaded.shear_yx == dataset.shear_yx`, `loaded.noise_map == dataset.noise_map`,
     `loaded.name == dataset.name`

## Workspace impact

Once fixed, the `@autolens_workspace/scripts/weak/fit.py` tutorial can be simplified to load from disk
rather than rebuilding the dataset inline. That migration is a small workspace follow-up after the library
fix lands.

## Out of scope

- Pytree registration of `WeakDataset` for JAX compatibility — a separate effort.
- Reworking the simulator output format — `dataset.json` is the right artifact; only the loader is broken.
