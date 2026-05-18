# `@PyAutoFit` Refactor: each density should live in one place, not three

Found during the priors/messages audit (see
`PyAutoPrompt/autofit/priors_and_messages_math_audit.md`, finding C1).

> **Prerequisites:** Prompts 01-09 should be acked before this one
> begins. Without those, the refactor would just paper over the bugs;
> with them, the refactor codifies the corrected definitions.
>
> **Scope warning:** this is multi-week effort. Open the issue early
> for design discussion, but do not start implementation until the
> earlier prompts are merged.

## Problem

Each prior's density is currently encoded in *three* places:

| Where | What | Example for Gaussian |
|---|---|---|
| `Prior.value_for(unit)` | inverse CDF | `mean + sigma·√2·erfinv(2u-1)` |
| `Message.logpdf(x)` via the generic exponential-family path | natural-params + log_partition | computed from η, T(x), A(η) |
| `Prior.log_prior_from_value(x)` | density-form quadratic for inference | `-(x-μ)²/(2σ²)` |

Each has its own conventions (dropped constants, JAX dispatch, error
handling). Nothing forces the three to agree.

This is exactly why the LogUniform sign convention bug existed: the
hand-written `log_prior_from_value` returned `+log p(x)` while the
other two paths were silent. It's also why prompt 04 (`TruncatedNormalMessage`
pdf doesn't integrate to 1) survived: the generic-interface path is
wrong but the direct paths are right, so nobody noticed.

## Wider context — what the existing architecture buys us

The three-place encoding is not gratuitous:

- `value_for` exists because nested sampling (`Dynesty`, `Nautilus`)
  takes a unit-cube transform, not a logpdf. This path is the
  inference hot loop for nested samplers.
- `Message.logpdf` exists because the graphical / EP machinery wants
  the exponential-family representation (natural params, sufficient
  stats, log_partition) for message arithmetic.
- `Prior.log_prior_from_value` exists because MCMC / MLE searches add
  it directly to `log_likelihood`, and the hot loop wants a cheap
  hand-tuned scalar — not the full exponential-family expansion.

So the three paths have legitimate users. The refactor must keep all
three callable but make them *derivative* of one definition.

## Python reproducer — not a bug, a code-smell exposition

```python
# Reproducer: single_source_density.py
# Demonstrate that the three code paths are independent — and how a
# convention change has to be made in three places to stay consistent.

import inspect

from autofit.mapper.prior.uniform import UniformPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior
from autofit.mapper.prior.gaussian import GaussianPrior
from autofit.mapper.prior.log_gaussian import LogGaussianPrior
from autofit.messages.normal import NormalMessage

for cls in [UniformPrior, LogUniformPrior, GaussianPrior,
            LogGaussianPrior, NormalMessage]:
    print(f"=== {cls.__name__} ===")
    for method_name in ["value_for", "log_prior_from_value", "logpdf"]:
        meth = getattr(cls, method_name, None)
        if meth is None or not callable(meth):
            print(f"  {method_name}: <inherited via __getattr__>")
            continue
        src_file = inspect.getsourcefile(meth)
        try:
            lineno = inspect.getsourcelines(meth)[1]
        except (TypeError, OSError):
            lineno = "?"
        print(f"  {method_name}: {src_file}:{lineno}")
    print()
```

The output lists three different file:line locations per class, often
in separate files (e.g. `uniform.py` for `value_for` and
`log_prior_from_value`, but `normal.py` / inherited for `logpdf`). The
LogUniform fix `e95295b83` had to touch the prior file and the message
file; the same fix replicated four times across the families.

## Proposed direction (sketch)

Reduce to one `log_density(x, params)` function per family, and derive
the others:

```python
class Distribution:
    """Single source of truth for a parametric family."""

    def log_density(self, x, xp=np):
        """log p(x) — the canonical normalised density. The one place
        the math lives."""
        raise NotImplementedError

    def value_for(self, unit, xp=np):
        """Inverse CDF. Each subclass implements this — there's no
        closed-form way to derive it from log_density in general."""
        raise NotImplementedError

    def logpdf(self, x, xp=np):
        return self.log_density(x, xp=xp)

    def log_prior_from_value(self, x, xp=np):
        """Identical to log_density. The historical name is kept for
        Fitness._call compatibility but it's just an alias."""
        return self.log_density(x, xp=xp)
```

The constants-dropping convention from prompt 07 would either:

- Be `Distribution.log_density` itself (Option A), or
- Be a separate `Distribution.log_density_normaliser()` method that
  `log_density` adds back (Option B).

Either way, **two** functions per family, not three (`value_for` and
`log_density`), with `logpdf` and `log_prior_from_value` as one-line
aliases.

For JAX support, `value_for` and `log_density` both take an `xp`
kwarg as today; everything else inherits.

## What the agent picking this up should do

1. Read every prior and message file in `@PyAutoFit/autofit/mapper/prior/`
   and `@PyAutoFit/autofit/messages/` end-to-end. Confirm the
   three-place encoding pattern.
2. Read the inference call sites that consume each path:
   - `Fitness._call` — uses `log_prior_from_value`.
   - Nested-sampler `prior_transform` — uses `value_for`.
   - Graphical / EP — uses `Message.logpdf` and natural-parameter
     machinery.
3. Sketch the refactor as a design doc in this prompt's GitHub issue
   (not as code). Include:
   - Class hierarchy proposal.
   - Migration plan for each subclass (one PR per family of priors?).
   - Backwards-compatibility plan (deprecated aliases for the old
     method names).
4. File the GitHub issue via `/create_issue priors/12_single_source_density_refactor.md`.
5. **In the issue body, frame this as a design discussion, not a code
   change.** Tag the reviewer for input on the class-hierarchy
   proposal. Ask whether `Distribution` should also subsume
   `Message` (prompt 13) or whether the two should remain separate
   layers in the unified hierarchy.
6. **Stop. Implementation only after a design review.** The likely
   path is multiple sub-PRs landing over weeks; do not bundle them
   into one issue.
