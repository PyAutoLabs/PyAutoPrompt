# Test-mode fake sample count for latent robustness

## Original user request

ok back to the release list

## Context

The release PyAutoBuild run `2026-06-08T16-10-15Z` failed three latent robustness guards:

- `autofit_workspace_test/scripts/features/latent_nan_robustness.py`
- `autogalaxy_workspace_test/scripts/latent/latent_nan_robustness.py`
- `autolens_workspace_test/scripts/latent/latent_nan_robustness.py`

All three fail before exercising their intended latent masking assertion because the search result has only two samples:

```text
AssertionError: Need >3 samples for a multi-batch latent run; got 2.
```

The common cause appears to be PyAutoFit test-mode bypass. `autofit/non_linear/search/abstract_search.py::_build_fake_samples` always creates exactly two fake samples when `PYAUTO_TEST_MODE=2` or `PYAUTO_TEST_MODE=3`, which is too few for structural downstream guards that need multiple latent batches.

## Goal

Update PyAutoFit test-mode bypass so fake samples are still cheap but numerous enough for multi-batch downstream structural checks. Preserve existing bypass semantics: no real sampling, deterministic fake samples, and valid `SamplesPDF` summary behavior.

## Suggested verification

- Add / update PyAutoFit unit coverage for `_build_fake_samples` or bypass-mode fitting to assert at least four fake samples are returned.
- Run the relevant PyAutoFit unit tests.
- Run the three downstream latent robustness scripts, or at least confirm they progress past the sample-count assertion.
