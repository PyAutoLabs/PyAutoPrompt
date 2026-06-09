# Autolens JAX simulator release fixes

## Original Request

ok merge and on to the next fix

## Context

After merging the HowTo release fixes, continue the release failure list with the
Autolens JAX simulator API-drift group. Kaplinghat must remain outside this
release.

## Current Failures

Primary repo: @autolens_workspace_test

- `@autolens_workspace_test/scripts/imaging/simulator_use_jax_parity.py`
  - Eager JAX/NumPy parity passes.
  - Fails at `al.util.register_tracer_classes(tracer)` because
    `autolens.util.register_tracer_classes` no longer exists.
- `@autolens_workspace_test/scripts/interferometer/simulator_use_jax_parity.py`
  - Eager JAX/NumPy parity passes.
  - Fails at `al.util.register_tracer_classes(tracer)` for the same reason.
- `@autolens_workspace_test/scripts/cluster/simulator.py`
  - Fails under `jax.jit` because `tracer` contains `Galaxy` instances and is
    passed as a dynamic non-array argument to the jitted function.

## Proposed Scope

Fix the stale JAX usage in these workspace-test scripts, then rerun the three
target scripts. Treat this as workspace-only unless source inspection proves the
missing registration helper should be restored in @PyAutoLens.
