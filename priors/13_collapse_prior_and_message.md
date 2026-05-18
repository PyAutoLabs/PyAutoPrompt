# `@PyAutoFit` Refactor: collapse the `Prior` / `Message` two-layer hierarchy

Found during the priors/messages audit (see
`PyAutoPrompt/autofit/priors_and_messages_math_audit.md`, finding C4).

> **Prerequisites:** Prompts 01-09 should be acked and ideally merged.
> Prompt 12 (single source of truth) is closely related; the reviewer
> may decide to bundle 12 + 13 into one design discussion.
>
> **Scope warning:** very large refactor. Open an issue for design
> discussion; do not start coding before the design is approved.

## Problem

`autofit` currently has a two-layer hierarchy:

- `Prior` (in `@PyAutoFit/autofit/mapper/prior/`) — user-facing, lives
  in the model mapper, has an `id_`, knows about `width_modifier`,
  and **delegates almost everything to its message via `__getattr__`**.

- `Message` (in `@PyAutoFit/autofit/messages/`) — internal-ish, holds
  the mathematical machinery (natural parameters, log-partition,
  sufficient stats), used by the graphical / EP backend.

Each `Prior` subclass wraps a corresponding `Message`:

| Prior | Wrapped message |
|---|---|
| `UniformPrior` | `TransformedMessage(UniformNormalMessage, LinearShiftTransform)` |
| `LogUniformPrior` | `TransformedMessage(UniformNormalMessage, LinearShiftTransform, log_10_transform)` |
| `GaussianPrior` | `NormalMessage` |
| `LogGaussianPrior` | `TransformedMessage(NormalMessage, log_transform)` |
| `TruncatedGaussianPrior` | `TruncatedNormalMessage` |

`Prior.__getattr__` (`@PyAutoFit/autofit/mapper/prior/abstract.py:201-204`)
forwards any attribute to the message:

```python
def __getattr__(self, item):
    if item in ("__setstate__", "__getstate__"):
        raise AttributeError(item)
    return getattr(self.message, item)
```

So `gaussian_prior.log_prior_from_value(x)` "inherits" from
`NormalMessage.log_prior_from_value(x)` via `__getattr__` — invisible
to anyone reading the `GaussianPrior` class file.

## Wider context — why the split exists, and what it costs

Historical reason: priors and messages were originally distinct
concerns. Priors were the user-facing model-builder API; messages
were a separate Bayesian-network internal that grew out of the
graphical / EP work. They share math but the lifecycles diverged.

What the split costs today:

1. **Bugs hide across the boundary.** The recent LogUniform fix had to
   touch both `autofit/mapper/prior/log_uniform.py` *and*
   `autofit/messages/normal.py` (the `UniformNormalMessage`
   composition there). The bug span was hidden because the prior
   delegates to the message via `__getattr__` — invisible unless you
   know to look.

2. **Doubled surface area.** Each family has both `Prior` and `Message`
   classes (plus `TransformedMessage` and `NaturalNormal` /
   `TruncatedNaturalNormal` variants). A new probability family
   requires touching ~4 files.

3. **Conventions can drift.** Prompts 05 (constants), 06 (sigma
   check), 07 (normalisation) all show conventions that were applied
   inconsistently between the two layers.

4. **`__getattr__` indirection is a debugging trap.** Stack traces
   land in `NormalMessage.log_prior_from_value` when the user thought
   they were calling a `GaussianPrior` method.

## What this prompt is *not*

This is **not** about deleting `Message`. The graphical / EP machinery
genuinely needs the exponential-family abstraction (natural parameters,
sum-natural-parameters, KL, project, etc.). Those operations are
specific to *messages in factor graphs*, not to priors in a model
mapper.

This prompt *is* about merging the two responsibilities into one class
hierarchy where the EP-specific operations live on the same class as
the user-facing prior API — eliminating the duplication while keeping
the math operations available where they're needed.

## Proposed direction (sketch)

```
Distribution             (abstract, replaces both Prior and Message)
├── Univariate           (provides value_for / log_density / __getattr__-free)
│   ├── Uniform
│   ├── LogUniform
│   ├── Gaussian         (mode 'natural' / 'mean-sigma' parameterisation)
│   ├── LogGaussian
│   ├── TruncatedGaussian
│   ├── Beta
│   └── Gamma
└── EP-mixin             (natural_parameters, log_partition, project, ...)
    — applied to any Univariate subclass that participates in EP.
```

The `width_modifier`, `id_`, model-mapper integration that used to live
on `Prior` move to the relevant `Univariate` subclasses. The EP
operations that used to live on `Message` move to the EP mixin and
are only active on `Distribution` instances used inside a factor
graph.

`prior_passing`, `with_limits`, `value_for_unit` continue to work as
today — the user-visible API is preserved.

## Python reproducer — exposition, not a bug

```python
# Reproducer: collapse_prior_and_message.py
# Demonstrate the __getattr__ delegation surface that this refactor
# would eliminate.

import autofit as af

p = af.GaussianPrior(mean=0.0, sigma=1.0)

# log_prior_from_value is "inherited" from NormalMessage via __getattr__
# — but not visible on the GaussianPrior class:
print("On the GaussianPrior class itself:")
print(f"  has log_prior_from_value? {'log_prior_from_value' in type(p).__dict__}")
print(f"  has cdf?                 {'cdf' in type(p).__dict__}")
print(f"  has logpdf?              {'logpdf' in type(p).__dict__}")
print(f"  has mean attribute?      {'mean' in type(p).__dict__}")
print()
print("But all of these work via __getattr__:")
print(f"  p.log_prior_from_value(0.0) = {p.log_prior_from_value(0.0)}")
print(f"  p.cdf(0.0) = {p.cdf(0.0)}")
print(f"  p.mean = {p.mean}")
print()
print("They land in NormalMessage, not GaussianPrior:")
print(f"  p.log_prior_from_value belongs to module: "
      f"{p.log_prior_from_value.__module__}")
print(f"  p.cdf belongs to module:                  "
      f"{p.cdf.__module__}")
```

This isn't a numerical bug — it's an architectural surface that makes
the codebase ~half as legible as it could be.

## What the agent picking this up should do

1. Read `@PyAutoFit/autofit/mapper/prior/abstract.py`,
   `@PyAutoFit/autofit/mapper/prior/*.py`, and
   `@PyAutoFit/autofit/messages/*.py` end-to-end. List every method
   that exists on a `Prior` subclass and every method that exists on
   the wrapped `Message`. Identify overlap.
2. Read `@PyAutoFit/autofit/graphical/` to understand which message
   operations are *only* used inside the factor graph (so they can
   live on the EP mixin) vs. which are user-facing.
3. Run the reproducer to see the `__getattr__` surface concretely.
4. File the GitHub issue via `/create_issue priors/13_collapse_prior_and_message.md`.
5. **In the issue body, propose the class hierarchy sketch above and
   ask the reviewer to comment.** Be explicit that this is a months-
   long migration, not a single PR. Suggest staged delivery: introduce
   `Distribution` as a sibling layer first, migrate one family at a
   time, deprecate `Prior` / `Message` last.
6. Coordinate with prompt 12 in the same conversation — the reviewer
   may want a combined design rather than two separate issues.
7. **Stop. No code changes until a design is approved.**
