# `@PyAutoFit` `TransformedMessage` reversal convention is undocumented foot-gun

Found during the priors/messages audit (see
`PyAutoPrompt/autofit/priors_and_messages_math_audit.md`, finding C6).

## Problem

`@PyAutoFit/autofit/messages/composed_transform.py:60-219` implements
`TransformedMessage`, the composition machinery that turns
`UniformNormalMessage` into `UniformPrior(lo, hi)` etc. Two sharp
edges that contributed to the LogUniform sign bug:

### 1. Reversal convention is asymmetric and undocumented

```python
# composed_transform.py:185-198
def _transform(self, x):
    for _transform in reversed(self.transforms):
        x = _transform.transform(x)
    return x

def _inverse_transform(self, x):
    for _transform in self.transforms:
        x = _transform.inv_transform(x)
    return x
```

`_transform` reverses, `_inverse_transform` does not. The convention is
correct but only obvious once you understand the underlying message
lives in the base distribution's space and the transforms map from
base → physical. Anyone adding a new transform reaches for "for
_transform in self.transforms: ..." in both methods and gets the order
wrong.

### 2. `LinearShiftTransform` stores the inverse of the intuitive scale

`@PyAutoFit/autofit/messages/transform.py:171-186`:

```python
class LinearShiftTransform(LinearTransform):
    def __init__(self, shift: float = 0, scale: float = 1):
        self.shift = float(shift)
        self.scale = float(scale)
        super().__init__(DiagonalMatrix(np.reciprocal(self.scale)))
                                        # ^^^^^^^^^^^^ Jacobian = 1/scale

    def transform(self, x):  # physical → base
        return (x - self.shift) / self.scale

    def inv_transform(self, x):  # base → physical
        return x * self.scale + self.shift

    def log_det(self, x):
        return -np.log(self.scale) * np.ones_like(x)
```

The user-facing kwargs (`shift`, `scale`) describe the *physical*-space
parameters, but the Jacobian stored on the `LinearTransform` parent is
`1/scale` — because the transform goes physical → base. The
off-by-reciprocal got someone in the LogUniform bug (the historical
sign issue compounded with confusion about which direction `log_det`
was computing).

## Wider context — why now

The audit's verdict is that `TransformedMessage` is *correct* but the
class is one or two lines of documentation away from being safe to
extend. Fixing 12 (single source of truth) and 14 (replace with
bijectors) would dissolve this problem, but those are big efforts;
this prompt is the cheap improvement in the meantime.

## Python reproducer — not a bug, an exposition

There is no numerical failure to reproduce. The "bug" is a teaching
gap. The clearest evidence is the historical record:

```python
# Reproducer: transformed_message_semantics_doc.py
# Show that the convention IS correct end-to-end, but only by tracing
# through three levels of indirection.

import numpy as np
import autofit as af

# UniformPrior(0, 2) composes:
#   base = NormalMessage(0, 1)
#   transforms = (phi_transform, LinearShiftTransform(shift=0, scale=2))
# value_for(0.5) flow (via _inverse_transform — forward order):
#   1) NormalMessage(0, 1).value_for(0.5) = 0
#   2) phi_transform.inv_transform(0) = Φ(0) = 0.5
#   3) LinearShiftTransform.inv_transform(0.5) = 0.5*2 + 0 = 1.0
# So value_for(0.5) = 1.0 ✓

p = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
print(f"UniformPrior(0, 2).value_for(0.5) = {p.value_for(0.5)} (expect 1.0)")

# Tracing the same call manually:
from autofit.messages.normal import NormalMessage
from autofit.messages.transform import phi_transform, LinearShiftTransform
from scipy.special import ndtr  # standard normal CDF

step1 = NormalMessage(0, 1).value_for(0.5)
print(f"  step 1 (base NormalMessage.value_for): {step1}")
step2 = phi_transform.inv_transform(np.asarray(step1))
print(f"  step 2 (phi_transform.inv_transform = ndtr): {step2}")
step3 = LinearShiftTransform(shift=0, scale=2).inv_transform(step2)
print(f"  step 3 (LinearShiftTransform.inv_transform): {step3}")

# Now the gotcha: if a new contributor wrote `_inverse_transform` as
# `reversed(self.transforms)` (matching `_transform`), step 2 and step 3
# would swap order:
manual_buggy = LinearShiftTransform(shift=0, scale=2).inv_transform(np.asarray(step1))
print()
print(f"BUG-shaped call: forgetting forward-order, scale before phi:")
print(f"  LinearShift on raw normal sample = {manual_buggy}")
print(f"  → if then fed through ndtr: {ndtr(manual_buggy)} ≠ 1.0")
```

This shows the convention is correct but only by manually tracing
through three layers. A four-line docstring on `_transform` /
`_inverse_transform` describing "transforms map base → physical when
inverse, physical → base when forward, hence the asymmetric reversal"
would have made this self-evident.

## Proposed change — docs + one rename

1. Add a comprehensive docstring on `TransformedMessage._transform`
   and `_inverse_transform`:

   ```
   Transforms are stored in physical → base composition order. To go
   physical → base we apply them in REVERSE because composition unwinds
   the outermost transform first. To go base → physical we apply them
   in FORWARD order, rebuilding the composition from the inside out.

   Example: UniformPrior(0, 2) wraps NormalMessage(0, 1) with
   (phi_transform, LinearShift(shift=0, scale=2)). Going base →
   physical (i.e. value_for): NormalMessage sample → phi.inv (ndtr) →
   LinearShift.inv. Going physical → base (i.e. logpdf evaluation):
   reverse — LinearShift.transform → phi.transform.
   ```

2. Rename `LinearShiftTransform(scale=...)` to `LinearShiftTransform(physical_scale=...)`
   so the user-facing name doesn't collide with the inverse stored on
   the parent class. Keep `scale=` as a deprecated alias with a
   `DeprecationWarning`.

3. Add a small ASCII diagram in the module docstring of
   `composed_transform.py` showing the two flows.

## What the agent picking this up should do

1. Read `@PyAutoFit/autofit/messages/composed_transform.py` and
   `@PyAutoFit/autofit/messages/transform.py` end-to-end.
2. Run the reproducer to confirm the convention is correct as currently
   coded.
3. Draft the docstring updates and the rename in a scratch checkout.
4. File the GitHub issue via
   `/create_issue priors/11_transformed_message_semantics_doc.md`.
5. **In the issue body, ask the reviewer whether the rename is worth
   the backwards-compatibility cost or whether docs alone are enough.**
   The audit's leaning is "docs alone are fine if prompt 14 is in the
   roadmap".
6. **Stop. Do not implement until acked.** This is a doc / API change
   so the bar for review is mostly "is the explanation correct?"
