# Optional-dep latents should soft-fail to NaN, not re-raise ModuleNotFoundError

## Context

Discovered while validating the PyAutoLens #557 magzero fix on the
first-class A100 search profile (`autolens_profiling` job 322552). The
magzero `_mujy` latents now soft-fail correctly, but a second post-fit
latent crashed the same metric-JSON write path with a parallel bug:

```
File "PyAutoGalaxy/autogalaxy/operate/lens_calc.py", line 1567,
   in einstein_radius_jit_from
    from jax_zero_contour import ZeroSolver
ModuleNotFoundError: No module named 'jax_zero_contour'

The above exception was the direct cause of the following exception:

  File "PyAutoLens/autolens/analysis/latent.py", line 225,
     in effective_einstein_radius
    return lens_calc.einstein_radius_jit_from(init_guess=init_guess)
  raise ModuleNotFoundError(
    "jax_zero_contour is required for einstein_radius_jit_from. "
    "Install it with: pip install jax_zero_contour"
  ) from exc
```

The raise lives at `PyAutoGalaxy:autogalaxy/operate/lens_calc.py:1566-1572`:

```python
try:
    from jax_zero_contour import ZeroSolver
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "jax_zero_contour is required for einstein_radius_jit_from. "
        "Install it with: pip install jax_zero_contour"
    ) from exc
```

This is the same default-config-crashes-by-default pattern as the
magzero family (fixed in PyAutoLens #557 with `_maybe_magzero_warn`),
just keyed on an optional dependency instead of an Analysis kwarg.

## The bug

`effective_einstein_radius` is enabled by default in
`PyAutoLens:autolens/config/latent.yaml`, and is computed during every
`SearchUpdater._compute_latent_samples` call as part of the standard
post-fit pipeline. When `jax_zero_contour` isn't installed (it isn't
in PyAuto's default `pyproject.toml` deps — it's a JAX-only optional
extra), every `search.fit()` that reaches the post-fit step crashes
with the message above. The search itself converged; the metric-JSON
write never happens.

This is the same failure shape as
`PyAutoPrompt/autolens/magzero_required_latents_crash_search.md`
(landed as PyAutoLens #557). The PyAutoLens fix only addressed the
magzero family of latents — the optional-dependency family was not
touched, and surfaces with the same symptoms.

## Desired fix

Mirror PyAutoLens #557's `_maybe_magzero_warn`: replace the re-raise
with a per-process warning + `xp.nan` return. The "module is missing"
case is structurally identical to the "user kwarg is missing" case —
both mean the latent value is unknown, and the right behaviour is
return-NaN-and-warn, not kill the search.

Sketch (in `PyAutoGalaxy:autogalaxy/operate/lens_calc.py` or wherever
the latent function lives — `PyAutoLens:autolens/analysis/latent.py:225`
calls into this):

```python
_OPTIONAL_DEP_WARNED: set[str] = set()

def _maybe_optional_dep_warn(import_name: str, name: str) -> bool:
    """Return True (and warn once) if the optional dependency is missing."""
    try:
        importlib.import_module(import_name)
        return False
    except ModuleNotFoundError:
        if name not in _OPTIONAL_DEP_WARNED:
            logger.warning(
                "Optional dependency '%s' not installed; '%s' latent will "
                "be NaN. pip install %s to enable it, or disable in "
                "config/latent.yaml to silence this warning.",
                import_name, name, import_name,
            )
            _OPTIONAL_DEP_WARNED.add(name)
        return True
```

`einstein_radius_jit_from` (and any sibling that does the same
`try/except ModuleNotFoundError → re-raise` pattern) becomes:

```python
def einstein_radius_jit_from(self, ...):
    if _maybe_optional_dep_warn("jax_zero_contour", "effective_einstein_radius"):
        return xp.nan
    from jax_zero_contour import ZeroSolver
    ...
```

## Test plan

- Unit test (mirror `test_autolens/analysis/test_latent.py`'s magzero
  cases): mock `jax_zero_contour` import to raise, assert the latent
  returns NaN and emits one warning per process.
- End-to-end on a node WITHOUT `jax_zero_contour` installed: construct
  `AnalysisImaging(..., use_jax=True)` with default
  `config/latent.yaml`, run `af.Nautilus(...).fit(...)` in
  `PYAUTO_TEST_MODE=1`. Confirm search completes, JSON write happens,
  `latent.csv` has NaN values for the affected columns.
- Smoke check: re-run `autolens_profiling/searches/nautilus/imaging/mge.py`
  on a clean venv (no `jax_zero_contour`) and confirm no crash.

## Affected callers / interaction surface

- Every PyAutoLens search using default latents on a venv without
  `jax_zero_contour` — the HPC `PyAutoNSS` venv on Euclid SAAS in
  particular, where `jax_zero_contour` was not installed at venv
  creation time (added manually post-hoc to unblock job 322552).
- Sibling functions in `PyAutoGalaxy:autogalaxy/operate/lens_calc.py`
  that do the same `try / except ModuleNotFoundError → raise` pattern:
  grep for `raise ModuleNotFoundError` in lens_calc.py and adjacent
  modules; any that gate optional-dep imports for default-on latents
  should switch to the soft-fail pattern.

## Why not just add `jax_zero_contour` as a hard dep?

That's the pragmatic short-term fix (it's what unblocked job 322552
on the HPC). But it makes a JAX-only utility a required dep of every
PyAutoGalaxy install, including non-JAX numpy-only users — wider blast
radius than this fix needs. Soft-fail keeps the optional dep optional
while making default configs runnable.

## Out of scope

- Promoting `jax_zero_contour` to a hard dep (alternative; not the
  preferred fix).
- Auditing every optional-dep code path in PyAutoGalaxy. Scope to
  the latent-computation surface where default configs hit them.
- Re-reviewing PyAutoLens #557 — that fix is correct for what it
  covered; this is a parallel-but-independent surface.

## Cross-references

- PyAutoLens #557 — fixed the magzero family with the same soft-fail
  pattern this prompt proposes for optional deps.
- PyAutoPrompt/autolens/magzero_required_latents_crash_search.md —
  the upstream prompt for #557.
- autolens_profiling HPC job 322552 — the surfacing run; sampling
  completed (64,200 evals, log_Z=+31690.49) but post-fit latent
  crashed before metric JSON write.
