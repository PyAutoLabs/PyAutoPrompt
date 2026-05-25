Spring clean the mass profiles module in PyAutoGalaxy.

## Goal

Code quality sweep across `@PyAutoGalaxy/autogalaxy/profiles/mass/` after the MGE/CSE fallback mechanism is in place. Focus on JAX readiness, consistency, and removing dead code.

## Tasks

### 1. xp Threading Audit

Audit all utility files for hardcoded `np` that should be `xp`:
- `dark/nfw_hk24_util.py` — `small_f_1()`, `small_f_2()`, `small_f_3()` use bare `np.*`
- `dark/mcr_util.py` — some paths already have `xp` guards, audit for completeness
- `dark/ludlow16.py` — conditional JAX imports exist but some helpers may still hardcode `np`

For each, add `xp=np` parameter and replace `np.*` with `xp.*`. Thread `xp` from the calling profile method.

### 2. Dead Code Removal

- Remove any methods that are now superseded by the MGE/CSE fallback (e.g., old numerical integration paths that were kept "just in case")
- Remove commented-out code blocks
- Remove unused imports

### 3. Decorator Consistency

Verify all mass profile methods follow the canonical stacking order:
```python
@aa.decorators.to_array        # outermost
@aa.decorators.transform       # innermost
def convergence_2d_from(self, grid, xp=np, **kwargs):
```

And for vector methods:
```python
@aa.decorators.to_vector_yx
@aa.decorators.transform(rotate_back=True)
def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
```

Flag and fix any profiles that deviate.

### 4. Parameter Naming

Check for inconsistent parameter naming across profile families:
- `mass_to_light_ratio` vs `mass_to_light_ratio_base` (GaussianGradient vs SersicGradient)
- Einstein radius conventions across total profiles
- Scale radius conventions across dark profiles

Document any inconsistencies found; fix where safe (not a breaking API change).

### 5. Method Signature Consistency

Ensure all `convergence_2d_from`, `potential_2d_from`, `deflections_yx_2d_from` accept `xp=np` and `**kwargs` consistently across all profiles.

## Verification

- Run `pytest test_autogalaxy/profiles/mass/` — all existing unit tests pass
- Run Phase 1 self-consistency test suite — no regressions
- Run `autolens_workspace_test/scripts/profiles_jit.py` — JAX compilation still works

## Repos

- @PyAutoGalaxy (primary)
