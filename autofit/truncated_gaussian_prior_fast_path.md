# `TruncatedGaussianPrior.value_for` — direct-erf fast path

Replace the `scipy.stats.norm.cdf` / `jax.scipy.stats.norm.cdf` calls inside
`@PyAutoFit/autofit/mapper/prior/truncated_gaussian.py:136(value_for)` with
direct `erf` / `erfinv` calls from `scipy.special` (numpy) and
`jax.scipy.special` (JAX). The current implementation routes through
`scipy.stats._distn_infrastructure.cdf`, which has substantial Python-side
wrapper overhead per call (arg validation, broadcasting setup, dispatch
chain) that dwarfs the actual `erf` computation.

This is the #1 cProfile hotspot identified by the
`graphical-ep-scale-up` scoping pass — see
`PyAutoPrompt/graphical_ep/graphical_scoping.md` and
`PyAutoPrompt/graphical_ep/ep_scoping.md`.

## Motivation

cProfile attribution at N=10 from
`autofit_workspace_developer/{graphical,ep}/profiles/N10_hotspots.txt`:

| Package | Total | `value_for` cumtime | `scipy.stats..cdf` cumtime | % of total |
|---------|-------|---------------------|-----------------------------|------------|
| graphical | 60 s | 22.7 s | 19.5 s (184 200 calls) | **33%** |
| ep | 724 s | not isolated | 116.4 s (1 599 610 calls) | **16%** |

Almost all of `value_for`'s cumtime is the scipy.stats wrapper — the actual
`erf` math is fast. `GaussianPrior.value_for` (the non-truncated variant)
already uses the direct-erfinv approach on its JAX path
(`gaussian.py:117`); this prompt extends that pattern to the truncated
variant on both numpy and JAX.

## Numerics-preserving identity

```
norm.cdf(z)  ≡  0.5 * (1 + erf(z / sqrt(2)))
             ≡  0.5 * erfc(-z / sqrt(2))         (preferred when z << 0)

norm.ppf(p)  ≡  sqrt(2) * erfinv(2*p - 1)
             ≡  -sqrt(2) * erfcinv(2*p)          (preferred when p close to 1)
             ≡   sqrt(2) * erfcinv(2*(1-p))      (preferred when p close to 0)
```

Both forms are mathematically equivalent to `scipy.stats.norm.cdf`/`ppf`
and produce the same float64 values to within ULPs. The `erfc`/`erfcinv`
forms are used when the argument is in the tail to avoid catastrophic
cancellation.

For `TruncatedGaussianPrior`:

```
a = (lower_limit - mean) / sigma
b = (upper_limit - mean) / sigma
Phi_a = 0.5 * (1 + erf(a / sqrt(2)))   # or erfc form if a << 0
Phi_b = 0.5 * (1 + erf(b / sqrt(2)))   # or erfc form if b >> 0
p     = Phi_a + unit * (Phi_b - Phi_a)
x     = sqrt(2) * erfinv(2*p - 1)       # or erfcinv forms if p near 0/1
return mean + sigma * x
```

## Plan

1. **Add a new helper module** (or inline functions in the same file)
   that exposes `truncated_normal_value_for(unit, mean, sigma, lower, upper, xp)`
   built on `xp.special.erf`/`erfinv` (importing `scipy.special` for numpy
   and `jax.scipy.special` for jax). Centralise the tail-safe `erfc`/
   `erfcinv` branching so both the prior and the message classes can call
   into it.
2. **Replace the bodies** of:
   - `autofit/mapper/prior/truncated_gaussian.py:136(value_for)`
   - The corresponding code path in
     `autofit/messages/truncated_normal.py` (if `TruncatedNormalMessage`
     also uses `scipy.stats.norm.cdf` — verify; reuse the helper if so).
3. **Numerical equivalence test** in
   `test_autofit/mapper/prior/test_truncated_gaussian.py`: compare the
   new `value_for` against `scipy.stats.truncnorm.ppf` (the
   library-agnostic ground truth, not the old code path) on a grid of
   `(unit, mean, sigma, lower, upper)` including extreme truncations
   (a=10/b=20, a=-20/b=-10, narrow [0.499, 0.501] bracket). Tolerance:
   `1e-12` relative error for moderate cases, `1e-9` in the deep tails.
4. **Benchmark gate.** Run
   `autofit_workspace_developer/graphical/fit.py --total_datasets={3,10,30}`
   and `ep/fit.py --total_datasets={3,10,30}` from a clean checkout
   and compare `profiles/baseline.json` against the pre-fix baseline
   committed by issue #16. Expected wall-time reduction:
   - Graphical: ~30% (prior transforms drop from 38% of total to <5%)
   - EP: ~17% (same fix, smaller share of total runtime)
   - Sanity-check max log L and recovered posteriors must match the
     pre-fix baseline to within `1e-6` relative tolerance — this is the
     correctness gate, the speed gain is only meaningful if numerics
     don't drift.
5. **Run the full PyAutoFit prior test suite**: `pytest test_autofit/mapper/prior/`
   and any tests under `test_autofit/non_linear/` that exercise the
   prior transform. They must all pass unchanged.

## Affected files

- `@PyAutoFit/autofit/mapper/prior/truncated_gaussian.py` — `value_for` body.
- `@PyAutoFit/autofit/messages/truncated_normal.py` — verify and update if it
  also uses `scipy.stats.norm.cdf`/`ppf`.
- `@PyAutoFit/autofit/mapper/prior/abstract.py` or a new file in
  `autofit/mapper/prior/_erf_helpers.py` — the shared helper.
- `@PyAutoFit/test_autofit/mapper/prior/test_truncated_gaussian.py` — numerical
  equivalence + extreme-truncation tests.
- (Verification only, no changes expected) `@PyAutoFit/autofit/mapper/prior/gaussian.py`
  — already uses the direct-erfinv pattern on JAX; cross-check the new
  helper is consistent.

## Out of scope

- Replacing `scipy.special.erf` with a hand-rolled rational approximation —
  scipy already calls into the system `libm` `erf`, which is what we want.
- Other priors (`LogGaussianPrior`, `LogUniformPrior`): they don't show
  up in the cProfile data and are not on the scaling critical path. If
  the helper extracts cleanly they can re-use it later — out of scope
  for this issue.
- JAX vs numpy parity audit beyond `TruncatedGaussianPrior`: separate
  scope.

## Success criteria

- All existing PyAutoFit prior tests pass.
- New numerical equivalence test passes at `1e-12` rel tolerance.
- `autofit_workspace_developer/graphical/fit.py --total_datasets={3,10,30}`
  posterior means and max log L match the pre-fix `baseline.json` to
  `1e-6` rel tolerance.
- Graphical wall time at N=10 drops by ≥20% (target ~30%); EP wall time
  at N=10 drops by ≥10% (target ~17%).
- Updated baseline.json files are committed in a follow-up
  autofit_workspace_developer PR after the library change merges.
