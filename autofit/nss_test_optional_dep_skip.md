# Skip `af.NSS` unit tests when the optional `nss` dependency is absent

Primary repo: **@PyAutoFit** (test-only change).

## Problem

13 tests under `test_autofit/non_linear/search/nest/nss/` fail on a clean checkout of
`main` in any environment that does **not** have the optional `nss` extra installed:

```
test_autofit/non_linear/search/nest/nss/test_checkpoint.py  (4 failures)
test_autofit/non_linear/search/nest/nss/test_search.py      (9 failures)
```

All 13 share one root cause (confirmed by reproducing on clean `main`, no local changes):

```
ImportError: af.NSS requires the optional `nss` package and the matching
`handley-lab/blackjax` fork. Install via:
    pip install autofit[nss]
```

raised at `autofit/non_linear/search/nest/nss/search.py:254` inside `af.NSS.__init__`
(guarded by the module's `_HAS_NSS` flag).

This is **not a code regression** — it is a test-hygiene gap. The tests guard on JAX being
present (`jax = pytest.importorskip("jax")` at the top of `test_checkpoint.py`) but do **not**
guard on the `nss` optional dependency. So on a machine with `jax` but without the `[nss]`
extra (the default dev/CI environment here), every test that constructs `af.NSS` hard-fails
at import/instantiation instead of skipping.

## Fix (resolve in the issue — confirm the surgical vs blanket choice)

Add a skip guard for the optional `nss` dependency, mirroring the existing
`pytest.importorskip("jax")` pattern. Options:

1. **Module-level skip** on both files:
   ```python
   from autofit.non_linear.search.nest.nss.search import _HAS_NSS
   pytestmark = pytest.mark.skipif(not _HAS_NSS, reason="requires optional `nss` extra")
   ```
   (or `pytest.importorskip("nss")` / the blackjax-fork module at module top).

2. **Surgical**: only guard the tests that actually instantiate `af.NSS`. Note that
   `test_checkpoint.py` imports `_save_checkpoint` / `_load_checkpoint` at module top and
   these import fine without `nss` — so the pure-serialization round-trip tests may be able
   to run **without** the extra (they use synthetic pytree states, per the file's docstring:
   "No real `nss.ns.run_nested_sampling` calls"). Check whether those serialization tests pass
   once the `af.NSS`-instantiating tests are skipped; if so, prefer keeping them running and
   only skip the instantiation/config tests.

Decide between (1) and (2) in the issue. Favour (2) if the serialization helpers genuinely
work without `nss` — it keeps real coverage on this machine; fall back to (1) if they don't.

## Constraints / context

- Library policy: **no JAX in library unit tests** — these files already respect that
  (`test_search.py` docstring states it, `test_checkpoint.py` uses `importorskip("jax")`).
  The fix must not introduce a hard JAX/nss import at collection time.
- The heavy end-to-end nss runs live in
  `autolens_workspace_developer/searches_minimal/nss_*.py` (per the test docstrings) — out of
  scope here.
- Verify the fix against the full `test_autofit/` run: after the change, `test_autofit/`
  should report **0 failures** in an environment without the `[nss]` extra (the 13 become
  skips), and still pass when the extra IS installed.

## Critical files
- `test_autofit/non_linear/search/nest/nss/test_checkpoint.py`
- `test_autofit/non_linear/search/nest/nss/test_search.py`
- `autofit/non_linear/search/nest/nss/search.py` (reference: `_HAS_NSS` flag + the
  `__init__` ImportError guard — do not change the guard itself)

## Out of scope
- Installing the `nss` / blackjax-fork extra in the default environment.
- Any change to `af.NSS` runtime behaviour or the `_HAS_NSS` import guard.
