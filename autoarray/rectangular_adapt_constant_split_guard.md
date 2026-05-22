Users keep combining `RectangularAdaptDensity` meshes with `ConstantSplit`
regularization in pixelized source models, for example:

```python
pixelization = af.Model(
    al.Pixelization,
    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=(28, 28)),
    regularization=al.reg.ConstantSplit,
)
```

This must be flagged as an error before the fit runs. The combination is
currently invalid and can produce a pathological all-zero pixelized source in
the output products instead of failing clearly.

Concrete repro:

- Data/script context: `@z_help/jacob/HerBS-28.ms.shifted_fits/jax_modelling_template.py`
- Verbatim capped reproduction: `@z_help/jacob/HerBS-28.ms.shifted_fits/jax_modelling_template_verbatim_nlike1000.py`
- Pixelization output path from the repro:
  `output/interferometer/verbatim_nlike1000/pixelization/01c465702e769ad81c9c0e35b3d232d0`

Observed behavior from that run:

```text
source_plane_images.fits HDU 0: sum_abs=0.0 nonzero=0
source_plane_images.fits HDU 1: sum_abs=0.0 nonzero=0
fit_dirty_images.fits DIRTY_MODEL_IMAGE: sum_abs=0.0 nonzero=0
Maximum Log Likelihood = -12395933.858
```

Rebuilding the reported max-likelihood fit eagerly and accessing the inversion
reconstruction crashes in the split regularization path:

```text
IndexError: index 4 is out of bounds for axis 0 with size 4
```

The exception occurs in:

```text
@PyAutoArray/autoarray/inversion/regularization/regularization_util.py
reg_split_np_from()
```

called from:

```text
@PyAutoArray/autoarray/inversion/regularization/constant_split.py
ConstantSplit.regularization_matrix_from()
```

Root cause:

`ConstantSplit` assumes split-point interpolation rows have enough padded
mapping slots to append the parent pixel when the parent pixel is not already
present. With `RectangularAdaptDensity`, the split mappings can have only four
slots. If all four are already occupied and the parent pixel is absent,
`reg_split_np_from()` tries to write to slot `j + 1 == 4`, which is out of
bounds for an array with size 4. In JAX/vectorized modeling this can surface as
a silently bad inversion / zero source output rather than a clear exception.

Desired fix:

Add an explicit validation guard which rejects this unsupported configuration
early, with an error message that tells users what to do instead.

The guard should catch both concrete and model-composition forms:

```python
al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(...),
    regularization=al.reg.ConstantSplit(...),
)
```

and:

```python
af.Model(
    al.Pixelization,
    mesh=af.Model(al.mesh.RectangularAdaptDensity, ...),
    regularization=al.reg.ConstantSplit,
)
```

Suggested error text:

```text
ConstantSplit regularization is not supported with RectangularAdaptDensity
meshes. This combination can produce invalid split regularization stencils and
all-zero pixelized source outputs. Use al.reg.Constant with
RectangularAdaptDensity, or use ConstantSplit with a Delaunay mesh.
```

Likely implementation locations to assess:

- `@PyAutoArray/autoarray/inversion/pixelization.py` if concrete
  `Pixelization` construction can validate mesh / regularization types.
- The inversion / mapper construction path if concrete construction is too
  early or incomplete.
- The AutoFit model-analysis pre-fit validation path may be needed to catch
  `af.Model(al.Pixelization, ...)` before sampling begins. If the validation
  naturally belongs outside PyAutoArray for `af.Model` objects, add the concrete
  PyAutoArray guard first and add a companion AutoLens / analysis guard where
  the prior model can be inspected.

Verification:

1. Add unit tests for concrete `Pixelization` construction or first use:
   `RectangularAdaptDensity + ConstantSplit` must raise the clear error.
2. Add a model-composition test, or an analysis pre-fit test, showing that
   `af.Model(al.Pixelization, mesh=af.Model(al.mesh.RectangularAdaptDensity),
   regularization=al.reg.ConstantSplit)` fails before Nautilus starts.
3. Confirm allowed combinations still work:
   - `RectangularAdaptDensity + Constant`
   - `Delaunay + ConstantSplit`
4. If touching the split regularization utility itself, add a low-level
   regression test for a split mapping row where the parent pixel is absent and
   all four slots are occupied, so the code cannot silently write out of bounds
   or produce invalid stencils.

Do not paper over this by clipping indices or returning an all-zero
reconstruction. This should be a hard, user-facing configuration error until
the split-regularization stencil code is redesigned to support rectangular
adaptive meshes.
