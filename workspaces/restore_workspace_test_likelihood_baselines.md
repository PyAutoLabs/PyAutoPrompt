Restore absolute NumPy likelihood regression baselines in the `_workspace_test`
repositories.

## Original Request

"ok, double check they didnt end up in autolens_proifiling or anything and if not, write a prompt in PyAutoPrompt to put them back in (single vlaues against final likelihood on numpy impplementation, noting that numpy and JAX also test one another)"

## Context

Older `autolens_workspace_test/scripts/jax_likelihood_functions/*` scripts used
to include fixed literal assertions against eager NumPy `fit.figure_of_merit` /
`fit.log_likelihood` values. PR #10 in `autolens_workspace_test`
(`d8e5b1b`, "Unify JAX likelihood tests: fixed seeds, single dataset,
simplified scripts") removed several of these as part of deleting
"post-assertion code" and plotting/memory-analysis blocks.

The absolute baselines were not removed because they lacked value. They appear
to have been lost as cleanup collateral while simplifying scripts and moving to
fixed seeded datasets.

There are analogous fixed absolute baselines in profiling/developer areas, for
example:

- `autolens_profiling/likelihood_runtime/imaging/*.py`
- `autolens_profiling/likelihood_breakdown/imaging/*.py`
- `autolens_profiling/likelihood_runtime/interferometer/*.py`
- `autolens_workspace_developer/jax_profiling/jit/**/*.py`

Those are useful, but they do not replace release-facing `_workspace_test`
coverage. The `_workspace_test` repositories should again pin the final NumPy
likelihood / log-evidence values for representative seeded datasets.

## Goal

Add back single-value, fixed numerical regression assertions against the final
likelihood value computed by the NumPy implementation in the workspace-test
scripts.

These checks should complement, not replace, the existing parity checks where
NumPy and JAX test one another. The desired coverage shape is:

1. NumPy eager implementation computes the reference final likelihood /
   figure-of-merit / log-evidence from the prior-median instance.
2. A literal `EXPECTED_*` value asserts that NumPy reference has not drifted.
3. JAX / vmap / jit paths continue to assert they match the NumPy reference
   where those scripts already exercise JAX.

This gives two layers of protection:

- absolute baseline: catches silent drift where all code paths move together
- parity baseline: catches CPU/JAX disagreement or tracing-specific regressions

## Scope

Primary repositories:

- `autolens_workspace_test`
- `autogalaxy_workspace_test`

Candidate script families:

- `scripts/jax_likelihood_functions/imaging/*.py`
- `scripts/jax_likelihood_functions/interferometer/*.py`
- `scripts/jax_likelihood_functions/multi/*.py`
- `scripts/jax_likelihood_functions/datacube/*.py`
- `scripts/jax_likelihood_functions/point_source/*.py`
- any related script-style tests that currently compute a NumPy likelihood and
  then only compare JAX to NumPy without pinning the NumPy value

Do not add these assertions to user-facing `autolens_workspace` or
`autogalaxy_workspace` tutorials.

## Implementation Notes

- Prefer a short self-contained `__NumPy Likelihood Baseline__` or similarly
  named block near the existing likelihood sanity / JAX parity block.
- Build the prior-median instance using the same model used by the script.
- Compute the final NumPy value through the production path the script is meant
  to guard:
  - pixelized imaging/interferometer: `fit.figure_of_merit` / log-evidence
  - pure parametric fits: `fit.log_likelihood` where this is the final objective
  - factor graph / multi scripts: the summed NumPy `log_likelihood_function`
    value if that is the final objective exposed to the search
- Name constants explicitly, e.g. `EXPECTED_NUMPY_LOG_EVIDENCE`,
  `EXPECTED_NUMPY_LOG_LIKELIHOOD`, or
  `EXPECTED_NUMPY_FACTOR_GRAPH_LOG_LIKELIHOOD`.
- Use `np.testing.assert_allclose(..., rtol=1e-4)` unless an existing script has
  a documented tighter or looser tolerance requirement.
- Keep existing NumPy-vs-JAX assertions in place. Where a script currently does
  `assert_allclose(float(log_l_jit), log_l_np, ...)`, keep that parity check and
  add a separate literal check for `log_l_np`.
- Avoid reintroducing plotting, memory-analysis blocks, or long post-assertion
  output. This task is only about restoring the high-value absolute numerical
  baselines.

## Validation

For every modified script:

1. Run the script once to capture the NumPy baseline.
2. Insert the `EXPECTED_*` literal and assertion.
3. Re-run the script and confirm both the literal NumPy assertion and the
   existing NumPy/JAX parity assertions pass.

Use writable caches where needed:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python <script>
```

## Acceptance Criteria

- Representative `_workspace_test` JAX likelihood scripts again contain fixed
  absolute NumPy final-likelihood assertions.
- Existing NumPy/JAX parity checks remain present and passing.
- The new assertions are scoped to `_workspace_test` test infrastructure, not
  public workspace tutorials.
- The final PR summary lists which scripts gained absolute baselines and which
  objective each baseline pins (`log_likelihood`, `figure_of_merit`, or
  factor-graph summed likelihood).
