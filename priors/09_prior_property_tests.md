# `@PyAutoFit` Add property-based correctness tests for every `Prior` subclass

Found during the priors/messages audit (see
`PyAutoPrompt/autofit/priors_and_messages_math_audit.md`, finding C3).

> **Prerequisite:** Prompts 01-08 should be acked (and ideally merged)
> first. The point of these tests is to lock in the fixes — adding them
> before the fixes would just produce a long list of red tests with no
> clear action.

## Problem

The audit found three real bugs (`LogGaussianPrior.with_limits` crash,
`GammaMessage.from_mode` wrong formula, `TruncatedNormalMessage` pdf
not normalised) that would each have been caught by a single
property-based test. None of them were caught because the existing
test suite is hand-rolled per-class and covers each method in
isolation.

The original `LogUniformPrior` sign-convention bug (`e95295b83`) is
in the same category — it would have been caught by a single test:
"for every prior, the analytic gradient of `log_prior_from_value`
matches a finite-difference gradient".

## Wider context — what exists already

`@PyAutoFit/test_autofit/graphical/functionality/test_messages.py`:

- `check_dist_norm(dist)` — uses `scipy.integrate.quad` to verify
  `pdf` integrates to 1. **Run on `NormalMessage`, `BetaMessage`,
  `GammaMessage`, `LogNormalMessage` only.** Not run on
  `TruncatedNormalMessage`, `LogGaussianPrior`, any `Prior` subclass,
  or any `TransformedMessage`-wrapped distribution. That's why
  prompt 04's bug survived.

- `check_log_normalisation(ms)` — verifies the product-of-messages
  log normalisation matches numerical integration.

- `check_numerical_gradient_hessians(message, x=None)` — verifies
  analytic gradient and Hessian match finite differences. Same limited
  coverage as above.

So the patterns are there; they just don't sweep every prior.

## Python reproducer — sketch of the missing tests

```python
# Sketch only — not a runnable file yet. This is what would go into
# test_autofit/mapper/prior/test_prior_properties.py
import inspect
import numpy as np
import pytest
from scipy.integrate import quad

import autofit as af
from autofit.mapper.prior.abstract import Prior

# Enumerate every Prior subclass that should obey the contract.
# Skipped: TuplePrior (container, not a density), Constant (point mass).
def all_priors():
    return [
        af.UniformPrior(0.0, 1.0),
        af.UniformPrior(-3.0, 7.5),
        af.GaussianPrior(0.0, 1.0),
        af.GaussianPrior(2.5, 0.3),
        af.LogUniformPrior(0.01, 100.0),
        af.LogGaussianPrior(0.0, 1.0),
        af.TruncatedGaussianPrior(0.0, 1.0, -2.0, 2.0),
        af.TruncatedGaussianPrior(1.0, 0.5, 0.0, 3.0),
    ]


# === Property 1: value_for is the inverse CDF ===
# i.e. cdf(value_for(u)) ≈ u for u in (0, 1).
@pytest.mark.parametrize("prior", all_priors())
@pytest.mark.parametrize("u", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_value_for_is_inverse_cdf(prior, u):
    x = prior.value_for(u)
    u_recovered = prior.message.cdf(x)
    assert u_recovered == pytest.approx(u, abs=1e-6)


# === Property 2: pdf integrates to 1 over support ===
# (Decided by prompt 07: this should hold under Option B and may need
# a fixed normaliser hook under Option A.)
@pytest.mark.parametrize("prior", all_priors())
def test_pdf_integrates_to_one(prior):
    lo, hi = prior.limits
    # Truncate infinite supports to a wide enough window for quad
    if not np.isfinite(lo):
        lo = prior.value_for(1e-9)
    if not np.isfinite(hi):
        hi = prior.value_for(1 - 1e-9)

    def pdf(x):
        return float(np.exp(prior.message.logpdf(np.asarray(float(x)))))

    integral, err = quad(pdf, lo, hi, limit=200)
    assert integral == pytest.approx(1.0, abs=max(err, 1e-3))


# === Property 3: analytic log_prior_from_value gradient matches finite-difference ===
# This is exactly the check that would have caught the original LogUniform
# sign-convention bug.
@pytest.mark.parametrize("prior", all_priors())
@pytest.mark.parametrize("u", [0.3, 0.5, 0.7])
def test_log_prior_gradient_sign(prior, u):
    x = prior.value_for(u)
    eps = 1e-6
    lp0 = prior.log_prior_from_value(x)
    lp1 = prior.log_prior_from_value(x + eps)
    finite_diff = (lp1 - lp0) / eps

    # For maximands (Emcee, Zeus, MLE-Drawer), grad(log_prior) should
    # have the SAME sign as the analytic density-form gradient — i.e.
    # negative when x is above the mode, positive when below.
    # For Uniform priors the gradient is 0, which we exempt.
    if isinstance(prior, af.UniformPrior):
        assert abs(finite_diff) < 1e-3
    else:
        # Compare to JAX-traced gradient if available, else just check sign
        # against the analytic mode location.
        ...  # see issue body for fully fleshed version


# === Property 4: with_limits round-trip ===
# Catches the LogGaussianPrior.with_limits crash from prompt 01.
@pytest.mark.parametrize(
    "cls,limits",
    [
        (af.UniformPrior, (0.0, 1.0)),
        (af.GaussianPrior, (-1.0, 1.0)),
        (af.LogUniformPrior, (0.01, 10.0)),
        (af.LogGaussianPrior, (0.5, 2.0)),
        (af.TruncatedGaussianPrior, (-2.0, 2.0)),
    ],
)
def test_with_limits_constructs(cls, limits):
    """with_limits should not crash — catches prompt 01-style bugs."""
    prior = cls.with_limits(*limits)
    assert prior is not None


# === Property 5: from_mode invariants for messages ===
# Catches the GammaMessage.from_mode wrong-formula bug from prompt 03.
@pytest.mark.parametrize("cls", [...])  # NormalMessage, GammaMessage, ...
def test_from_mode_matches_documented_invariant(cls):
    # Spec depends on each class's documented contract (mean-match vs
    # mode-match). See prompt 03 for the decision.
    ...
```

## Proposed scope

Add `test_autofit/mapper/prior/test_prior_properties.py` (and/or
`test_autofit/messages/test_message_properties.py`) with the five
properties above. Parametrise over every concrete subclass.

For each property, pick the tolerance carefully:

- Integrals: `abs=1e-3` is enough to catch all the audit findings;
  tighter would slow CI.
- Inverse-CDF round-trip: `abs=1e-6` works for double precision.
- Gradient finite-difference: `rtol=1e-2, atol=1e-3` (existing
  `check_numerical_gradient_hessians` uses these).

## What the agent picking this up should do

1. Read `@PyAutoFit/test_autofit/graphical/functionality/test_messages.py`
   to understand the existing patterns and reuse them.
2. Read every `Prior` subclass to confirm which can be parametrised
   over (skip `TuplePrior`, `Constant`, `DeferredArgument`).
3. Sketch the tests as a draft commit in a scratch checkout. Run
   `pytest test_autofit/mapper/prior/test_prior_properties.py`.
4. **Expect the tests to PASS after prompts 01-08 land.** If any fail
   on top of the fixed code, that's a new finding — file it as a
   separate issue and do not bundle it here.
5. File the GitHub issue via
   `/create_issue priors/09_prior_property_tests.md`.
6. **In the issue body, ask the reviewer to confirm the tolerance and
   property choices.** The five properties above are the audit's
   recommendation but the reviewer may want stricter ones (e.g.
   `jax.grad` cross-checks, sample-mean/variance moment checks).
7. **Stop. Do not implement until prompts 01-08 are at least acked
   in their own issues.** This prompt is the safety net for the fixes,
   not a substitute for them.
