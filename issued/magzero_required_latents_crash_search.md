# Flux-mujy latents crash `search.fit()` when `magzero` is missing from the Analysis

## Context

Surfaced by the first first-class A100 search run from `autolens_profiling`
(`searches/nautilus/imaging/mge.py`, HPC job 322548). Nautilus converged
cleanly (`log_Z=+31690.50`, 65500 evals, 11m40s) but the post-fit latent
computation crashed inside `SearchUpdater._compute_latent_samples`:

```
ValueError: magzero must be passed to the Analysis via kwargs to compute
the 'total_lens_flux_mujy' latent. Disable it in config/latent.yaml or
pass magzero=<value>.
```

The raise lives at
`PyAutoLens:autolens/analysis/latent.py:31-37`:

```python
def _require_magzero(magzero, name):
    if magzero is None:
        raise ValueError(
            f"magzero must be passed to the Analysis via kwargs to compute "
            f"the '{name}' latent. Disable it in config/latent.yaml or "
            f"pass magzero=<value>."
        )
```

and is the gate for three sibling latents:

- `total_lens_flux_mujy` (line 40)
- `total_lensed_source_flux_mujy` (line 60)
- `total_source_flux_mujy` (line 77)

## The bug

`af.Nautilus(...).fit(model, analysis)` does latent computation as part of
its standard update loop (via `SearchUpdater._compute_latent_samples`).
The set of latents to compute comes from `config/latent.yaml` on the
autoconf search path. A workspace that ships a `latent.yaml` enabling
`total_lens_flux_mujy` (e.g. `autolens_profiling/config/latent.yaml`, or
any workspace that opts into the full lensing latent catalogue) will
crash every `search.fit()` that constructs `AnalysisImaging` without
`magzero=...` — which is the standard path in every workspace tutorial
and SLaM pipeline today.

This is a default-config-crashes-by-default bug. The user has to know
to either:

- Pass `magzero=<value>` on every `AnalysisImaging(...)` construction
  (and learn what value is appropriate for their instrument), or
- Disable the offending latents in a project `config/latent.yaml`
  override, or
- Set `PYAUTO_SKIP_LATENTS=1` (the workaround applied in
  `autolens_profiling/hpc/batch_gpu/submit_imaging_mge_a100_hst_fp64`
  in PR #30).

None of these are discoverable from the failure mode — the search runs
to convergence, then dies in post-fit bookkeeping with no metric output.

## Desired fix

The three flux-mujy latent functions already return `xp.nan` when the
galaxy image isn't available (lines 51-52, 67-69, etc.):

```python
try:
    image = fit.galaxy_image_dict[fit.tracer.galaxies[0]]
except (AttributeError, KeyError, IndexError):
    return xp.nan
```

The same graceful path applies for missing `magzero`: the flux conversion
is meaningless without a zero-point, so the latent value is genuinely
unknown. Replace the raise with a soft NaN return + a one-time warning
per process:

```python
def _maybe_magzero_warn(magzero, name):
    if magzero is None:
        if name not in _MAGZERO_WARNED:
            logger.warning(
                "magzero not set on Analysis; '%s' latent will be NaN. "
                "Pass magzero=<value> to AnalysisImaging to enable it, "
                "or disable in config/latent.yaml to silence this warning.",
                name,
            )
            _MAGZERO_WARNED.add(name)
        return True
    return False
```

then at the top of each affected latent:

```python
if _maybe_magzero_warn(magzero, "total_lens_flux_mujy"):
    return xp.nan
```

This matches the existing "galaxy image missing → return NaN" pattern,
keeps default configs runnable, and surfaces the issue once in the
process log without killing a long search.

## Test plan

- Unit test in `test_autolens/analysis/test_latent.py` (likely
  already exists for `total_lens_flux_mujy`): assert calling with
  `magzero=None` now returns a NaN of the expected dtype instead of
  raising, and that a logger warning is emitted on first call only.
- End-to-end: construct `AnalysisImaging(dataset=..., use_jax=True)`
  (no `magzero`), run `af.Nautilus(...).fit(...)` in `PYAUTO_TEST_MODE=1`
  with `total_lens_flux_mujy` enabled in latent config. Confirm the
  search completes, the JSON metric write happens, and `latent.csv`
  has NaN values in the flux-mujy columns.
- Smoke check: re-run an `autolens_profiling/searches/nautilus/imaging/mge.py`
  cell without `PYAUTO_SKIP_LATENTS=1` and confirm no crash.

## Affected callers / interaction surface

- Every workspace SLaM pipeline (`source_lp[1]`, `source_pix[1]`, ...) that
  builds `AnalysisImaging` without `magzero` and runs with the full
  lensing latent catalogue enabled in `config/latent.yaml`.
- `autolens_profiling/config/latent.yaml` explicitly enables this set —
  any first-class search profile in `searches/` would hit this if the
  workaround env var were forgotten.
- Any downstream tool reading `latent.csv` should already handle NaN
  values (the galaxy-image-missing path produces them today).

## Why not a workspace config fix?

The workaround "disable these latents in every workspace `latent.yaml`"
is a one-by-one chase that's already out of date as soon as a new
workspace appears or an existing one opts into the full catalogue for
a different reason (e.g. `autolens_profiling/latent/imaging/` profiling
scripts). Library code should not depend on every consumer remembering
to defang it; the default behaviour should be runnable.

## Out of scope

- Auto-defaulting `magzero` to anything (`0.0`, instrument-specific):
  the flux value would be silently wrong rather than absent. NaN is
  the honest answer when the user hasn't supplied a zero-point.
- Reworking the `_require_*` pattern across other PyAutoLens latents
  (e.g. cosmology-dependent ones). Scope this PR to the magzero family
  only.
