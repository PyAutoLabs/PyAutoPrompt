# Investigate FactorGraphModel instance shape on Python 3.14

## Problem

`autofit_workspace/scripts/overview/overview_1_the_basics.py` runs cleanly
on Python 3.9–3.13 but fails on Python 3.14 with:

```
TypeError: 'Gaussian' object is not iterable
```

This was surfaced by the `python_matrix.yml` evidence run (issue
`PyAutoLabs/PyAutoBuild#74`, run 25208343008). Same script, same library
code — only the Python version differs.

3.14 was dropped from advertised support (classifiers + `python_matrix.yml`
matrix) until this is understood. `requires-python` still allows `>=3.9`,
so users who install on 3.14 anyway will see the import-time banner warn
that 3.14 isn't first-class.

## Reproducer

Run on Python 3.14 in a venv with the libraries installed editable:

```bash
python3.14 -m venv /tmp/py314
/tmp/py314/bin/pip install -e PyAutoConf -e PyAutoArray -e PyAutoFit \
    -e PyAutoGalaxy -e "PyAutoLens[optional]"
PYAUTO_TEST_MODE=1 PYAUTO_SMALL_DATASETS=1 \
    /tmp/py314/bin/python autofit_workspace/scripts/overview/overview_1_the_basics.py
```

Reproduces immediately when the script reaches the `factor_graph =
af.FactorGraphModel(*analysis_factor_list)` block (~line 545).

## Stack trace (abridged)

```
File "autofit_workspace/scripts/overview/overview_1_the_basics.py", line 813
    [profile_1d.model_data_from(xvalues=xvalues) for profile_1d in instance]
                                                                   ^^^^^^^^
TypeError: 'Gaussian' object is not iterable

The above exception was the direct cause of the following exception:

File "autofit_workspace/scripts/overview/overview_1_the_basics.py", line 563
    result_list = search.fit(model=factor_graph.global_prior_model,
                             analysis=factor_graph)
File "PyAutoFit/autofit/non_linear/search/abstract_search.py", line 668
    search_internal, fitness = self._fit(...)
File "PyAutoFit/autofit/graphical/declarative/collection.py", line 105
    log_likelihood += model_factor.log_likelihood_function(instance_)
File "PyAutoFit/autofit/graphical/declarative/factor/analysis.py", line 189
    return self.analysis.log_likelihood_function(instance)
```

The workspace `Analysis.log_likelihood_function` expects `instance` to be
a `Collection` of profiles (since the model was built as
`af.Collection(gaussian=Gaussian(), exponential=Exponential())`). On
3.9–3.13 it receives a `Collection`; on 3.14 it receives a single
`Gaussian` object directly.

## What we know

- `model = af.Collection(gaussian=Gaussian(), exponential=Exponential())`
- Each `AnalysisFactor` is built with `model.copy()` — so each factor's
  `prior_model` is itself a Collection of two profiles.
- `FactorGraphModel(*analysis_factor_list).global_prior_model` is the
  combined model passed to the search.
- On 3.9–3.13: `zip(self.model_factors, instance)` in
  `collection.py:104` yields `(factor, sub_instance)` pairs where
  `sub_instance` is the per-factor `Collection` instance — iterable.
- On 3.14: the same iteration yields a single `Gaussian` instance — not
  iterable.

So either:

1. `ModelInstance.__iter__` (which falls back to `__getitem__` since
   `ModelInstance` has no explicit `__iter__`) yields different child
   types on 3.14, OR
2. `FactorGraphModel.global_prior_model` constructs a flatter structure
   on 3.14 (collapses nested Collections into scalars), OR
3. dynesty's multiprocessing pickling round-trips the model differently
   on 3.14 (the trace shows `multiprocessing.pool.RemoteTraceback`,
   suggesting the worker process saw a different structure than the
   main process).

## Where to start investigating

1. Print `instance` and `type(instance)` at the top of the workspace
   `Analysis.log_likelihood_function` on a 3.14 venv. Compare against
   3.13. Specifically:
   - Is `instance` a `ModelInstance`, a `Collection`, or a raw
     `Gaussian`?
   - What does `instance.__dict__` look like on each version?
   - What does `list(instance)` do?

2. `autofit/mapper/model.py:385 ModelInstance` has no explicit
   `__iter__` — Python falls back to the legacy sequence protocol via
   `__getitem__`. `ModelInstance.__getitem__(int)` returns
   `list(self.values())[item]`. On 3.14, check whether `values()` and
   the resulting iteration yield different types than on 3.13.

3. `FactorGraphModel.global_prior_model` — trace how the per-factor
   Collection structures get composed into the global model. If 3.14
   flattens `Collection -> [profile_1, profile_2]` into bare profiles
   (because the dict ordering or attribute lookup behaves differently),
   the per-factor `instance_` would be a single profile.

4. Check whether `dynesty.pool` (from the failure trace) round-trips
   the model object correctly on 3.14. The error is wrapped in a
   `multiprocessing.pool.RemoteTraceback`, so the failure is happening
   in a worker process. Try `dynesty(parallel=False)` to isolate.

5. Python 3.14 release notes worth scanning for relevant behavior
   changes:
   - PEP 768: safe external debugger interface
   - PEP 749: late-bound default values (annotations)
   - PEP 765: disallow `return`/`break`/`continue` in `finally`
   - Behavior changes around `dict` ordering, `__init_subclass__`,
     `__set_name__`, descriptor lookup

## Constraints when fixing

- Don't modify the workspace tutorial script just to paper over the
  symptom. The script worked on 3.9–3.13 because the library produced
  the right shape; it should produce that same shape on 3.14.
- Library unit tests must remain numpy-only — don't add jax-dependent
  tests for this.
- If the fix requires a workspace-side change too (e.g. a defensive
  helper), keep it minimal and add a comment pointing back to this
  prompt.

## Done when

- `python3.14 autofit_workspace/scripts/overview/overview_1_the_basics.py`
  runs cleanly under `PYAUTO_TEST_MODE=1`.
- 3.14 can be re-added to PyAutoBuild's `python_matrix.yml` matrix and
  to each library's `pyproject.toml` classifiers.
- The change is unit-tested in `test_autofit/` with a numpy-only
  factor-graph round-trip test that would have caught the 3.14 shape
  collapse on 3.13 too if the structure had been wrong there.
