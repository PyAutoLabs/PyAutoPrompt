# `@PyAutoFit` `FixedMessage.logpdf_cache` is an unbounded class-level dict

Found during the priors/messages audit (see
`PyAutoPrompt/autofit/priors_and_messages_math_audit.md`, finding A8).

## Problem

`@PyAutoFit/autofit/messages/fixed.py:57-62`:

```python
class FixedMessage(AbstractMessage):
    ...
    logpdf_cache = {}   # class-level mutable dict, lives forever

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        if x.shape not in FixedMessage.logpdf_cache:
            FixedMessage.logpdf_cache[x.shape] = np.zeros_like(x)
        return FixedMessage.logpdf_cache[x.shape]
```

Every distinct shape ever seen during the process lifetime is memoised
on the class. In a long-running fit that evaluates `logpdf` on arrays
of varying shape (per-iteration sample chunks, varying batch sizes,
JAX traces, etc.) the cache grows unbounded.

There is also a subtle aliasing risk: the cached zero-array is
returned by reference, so any caller that mutates the result mutates
the shared cache entry.

## Wider context — how `FixedMessage` is used

`FixedMessage` represents a delta / point-mass / fixed-value
"distribution" used as a placeholder in EP graphs when a variable is
clamped. Its logpdf is mathematically undefined (a delta is not a
density), so the class returns `0` as a no-op that doesn't perturb
the EP message-passing arithmetic.

The class is rarely the bottleneck of a fit, so the cache hasn't
caused user-visible problems. But:

- Long-running services (e.g. always-on EP solver) leak.
- Test suites that run thousands of fits in one process leak.
- Returning a shared mutable array is the kind of latent bug that
  shows up as "fix in this place produces a regression in that place".

Not a math bug. A correctness-of-Python bug with low blast radius.

## Python reproducer

```python
# Reproducer: fixed_message_cache_growth.py
import numpy as np
from autofit.messages.fixed import FixedMessage

msg = FixedMessage(value=1.0)

# Before
print(f"cache size before: {len(FixedMessage.logpdf_cache)}")

# Generate a logpdf call for many distinct shapes
for n in range(1, 1001):
    _ = msg.logpdf(np.zeros(n))

print(f"cache size after 1000 distinct shapes: {len(FixedMessage.logpdf_cache)}")
print()

# Aliasing demonstration
a = msg.logpdf(np.zeros(5))
b = msg.logpdf(np.zeros(5))
print(f"Same shape returns the SAME object? {a is b}")
a[0] = 99.0   # mutates the shared cache entry
print(f"After mutating a: b[0] = {b[0]}  (should be 0.0 — but shared cache)")
```

Expected (buggy) output: cache grows to 1000 entries; `a is b` is
True; mutating `a[0]` corrupts `b[0]`.

## Proposed fix

Either:

1. **Compute on demand** (simplest, no cache):

   ```python
   def logpdf(self, x):
       return np.zeros_like(x)
   ```

2. **Compute on demand, return a fresh array** to also kill the
   aliasing — same as (1), the cache was the only thing aliasing.

3. **Keep a tiny LRU cache** if profiling actually shows allocation
   pressure (unlikely for `np.zeros_like`):

   ```python
   from functools import lru_cache
   @staticmethod
   @lru_cache(maxsize=32)
   def _zeros(shape, dtype):
       return np.zeros(shape, dtype=dtype)
   def logpdf(self, x):
       return self._zeros(x.shape, x.dtype).copy()
   ```

The audit recommends option 1: `np.zeros_like(x)` is cheap, the cache
optimises nothing important, and removing it eliminates both the leak
and the aliasing.

## What the agent picking this up should do

1. Read `@PyAutoFit/autofit/messages/fixed.py` end-to-end.
2. Grep for `FixedMessage` usage to confirm no caller relies on the
   "shared mutable zero array" behaviour. If any test or code does, it
   was almost certainly accidental; flag for the reviewer.
3. Run the reproducer. Confirm cache growth and aliasing.
4. Sketch the fix (option 1) in a scratch checkout. Re-run the
   reproducer. Confirm `a is b` is now False and the cache is empty.
5. File the GitHub issue via `/create_issue priors/10_fixed_message_cache_growth.md`.
6. **In the issue body, ask the reviewer whether the cache was ever
   intentional** (i.e. is there a known hot path that benefits from
   it?). If not, option 1 is the right answer.
7. **Stop. Do not implement until acked.**
