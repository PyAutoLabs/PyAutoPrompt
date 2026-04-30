# Make autofit.messages safe under JAX jit trace (xp.array → xp.stack)

## Problem

Several `messages/` files build small fixed-length arrays from individual
scalars / 0-d arrays using `xp.array([a, b])`. That works fine for numpy
but breaks under JAX tracing: `jnp.array` of a Python list of traced 0-d
arrays goes through `__array__` coercion, which raises
`TracerArrayConversionError`.

The reproducer is the toy hierarchical fit in `z_projects/concr` once
JAX is enabled on its analyses + factor graph. Crash:

```
File "PyAutoFit/autofit/messages/normal.py", line 187, in calc_natural_parameters
    return xp.array([mu * precision, -precision / 2])
jax.errors.TracerArrayConversionError: The numpy.ndarray conversion method
__array__() was called on traced array with shape float64[]
```

The path that triggers it: `HierarchicalFactor` evaluates the per-dataset
parent-distribution likelihood inside the jit'd factor function, which in
turn calls `NormalMessage.natural_parameters → calc_natural_parameters`
with traced `mu` and `sigma`. The other three toy scripts (`one_by_one`,
`graphical`, `ep`) don't reach this code path under jit, so the bug is
invisible there.

## Requested change

Replace `xp.array([scalar_a, scalar_b, ...])` with `xp.stack([...])`
everywhere in `autofit/messages/` where the inputs are 0-d arrays /
scalars and the result is a small fixed-shape array. `xp.stack` is the
JAX-trace-safe primitive for this — it works on numpy too, identical
output.

### Hits (verified via `grep -nF 'xp.array([' autofit/messages/`)

| File | Line | Current |
|---|---|---|
| `messages/normal.py` | 187 | `return xp.array([mu * precision, -precision / 2])` |
| `messages/normal.py` | 226 | `return xp.array([x, x**2])` |
| `messages/normal.py` | 571 | `return xp.array([eta1, eta2])` |
| `messages/truncated_normal.py` | 192 | `return xp.array([mu * precision, -precision / 2])` |
| `messages/truncated_normal.py` | 236 | `return xp.array([x, x**2])` |
| `messages/truncated_normal.py` | 694 | `return xp.array([eta1, eta2])` |
| `messages/beta.py` | 221 | `return xp.array([alpha - 1, beta - 1])` |
| `messages/beta.py` | 274 | `return xp.array([xp.log(x), xp.log1p(-x)])` |
| `messages/gamma.py` | 44 | `return xp.array([alpha - 1, -beta])` |
| `messages/gamma.py` | 53 | `return xp.array([np.log(x), x])` |

For `gamma.py:53`: the `np.log(x)` should also be `xp.log(x)` for trace
correctness when `xp` is jax — fix that too while you're there.

For each call site, simply swap to `xp.stack([...])`. No semantic change
intended; under numpy `np.array([scalar, scalar])` and `np.stack([scalar, scalar])`
return identical 1-d arrays. Under JAX the difference is that
`jnp.stack` keeps tracers happy.

### Things to double-check

1. **Shape compatibility**: in numpy `np.array([1, 2])` and `np.stack([1, 2])`
   both return shape `(2,)`. But for batched inputs (arrays of shape
   `(N,)`) the two diverge: `np.array([a, b])` is `(2, N)`, `np.stack([a, b])`
   is also `(2, N)` — matches. So we're safe for both scalar and
   batched cases.

2. **`messages/normal.py:226` (`to_canonical_form`)** is a `@staticmethod`
   used to compute sufficient statistics; the inputs `x` could plausibly
   be batched already. Verify the existing tests in
   `test_autofit/messages/` still pass.

3. **`gamma.py:53`** mixes `np.log(x)` with `xp.array(...)` — the `np.log`
   bypasses jax tracing and would coerce traced inputs. Switch to
   `xp.log(x)`.

## Verification

### Library tests
```bash
cd PyAutoFit
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib pytest test_autofit/messages -x
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib pytest test_autofit -x
```

### Integration reproducer
The concr hierarchical fit currently has `use_jax=True` disabled with a
comment pointing at this issue. Re-enable JAX and run:

```bash
cd /home/jammy/Code/PyAutoLabs/z_projects/concr
```

Edit `scripts/toy/hierarchical.py`:

1. Add `use_jax=True` back to `af.ex.Analysis(...)` (around line 47).
2. Add `use_jax=True` back to `af.FactorGraphModel(...)` (around line 75).
3. Remove the multi-line "JAX disabled:" comment above the analysis list.

Then run:

```bash
PYAUTO_TEST_MODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
  python3 scripts/toy/hierarchical.py --sample=toy__gaussian_x1__low_snr --total_datasets=5
```

Expected: no `TracerArrayConversionError`, full result block printed,
`Parent distribution mean = ...` and `Parent distribution scatter = ...`
final lines present. Compare per-dataset numbers against a non-JAX run
to confirm parity (test mode rough numbers — exact match not expected,
but order-of-magnitude should be the same).

### Look around for the same pattern outside `messages/`

After fixing the listed sites, run

```bash
grep -rnF 'xp.array([' autofit/
```

and skim other hits — same diagnosis applies if the input list is
made of 0-d traced arrays. The graphical / mean-field code may have
analogous spots that just weren't on the hierarchical likelihood path.
Treat that as a stretch goal; the primary fix is the 10 sites above.

## Out of scope

- Don't touch the toy concr scripts. The verification step at the end
  will flip the JAX flag back on once the library fix lands.
- No need to add a JAX-specific test file; the existing JAX-aware tests
  under `test_autofit/messages/` will exercise the new code paths once
  the integration reproducer passes.

## Branch / PR

Single PR against PyAutoFit `main`. Suggested branch:
`feature/messages-xp-stack-jax-trace`. Title:
`Use xp.stack in autofit.messages so JAX jit can trace small fixed arrays`.
