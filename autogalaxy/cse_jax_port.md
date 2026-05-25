Port the CSE (Cored Steep Ellipsoid) module in PyAutoGalaxy to support JAX.

## Goal

Make `@PyAutoGalaxy/autogalaxy/profiles/mass/abstract/cse.py` JAX-compatible by threading the `xp=np` parameter through all methods, mirroring how the MGE module (`mge.py`) already supports both NumPy and JAX backends.

## What to Change

### cse.py Methods

1. `convergence_cse_1d_from(grid_radii, core_radius)` — static method
   - Currently pure NumPy. Add `xp=np` parameter, no numpy calls to replace (pure arithmetic), but signature must accept `xp` for consistency.

2. `deflections_via_cse_from(term1, term2, term3, term4, axis_ratio_squared, core_radius)` — static method
   - Replace `np.sqrt` → `xp.sqrt`, `np.vstack` → `xp.vstack`
   - Add `xp=np` parameter

3. `_deflections_2d_via_cse_from(self, grid, **kwargs)` — instance method
   - Thread `xp` through to `deflections_via_cse_from` calls
   - Replace any `np.*` with `xp.*` (grid operations use `.array` already)

4. `_convergence_2d_via_cse_from(self, grid_radii, **kwargs)` — instance method
   - Thread `xp` through to `convergence_cse_1d_from` calls

5. `_decompose_convergence_via_cse_from(self, func, radii_min, radii_max, ...)` — the decomposition solver
   - This uses `scipy.linalg.lstsq` which has no JAX equivalent that works inside JIT
   - **Design:** The decomposition (fitting amplitudes + core radii) is a one-time setup step, not part of the JIT-traced forward pass. Keep `scipy.linalg.lstsq` for the NumPy path. For JAX, add a `xp is not np` branch using `jnp.linalg.lstsq`. The decomposition results (amplitude_list, core_radius_list) should be cached on the profile instance so they're computed once and reused.
   - Replace `np.logspace`, `np.zeros`, `np.log10` with `xp.*` equivalents

### Callers

All profiles that inherit `MassProfileCSE` and call these methods must thread `xp=xp` through:
- `@PyAutoGalaxy/autogalaxy/profiles/mass/dark/nfw.py` (NFW uses CSE for deflections)
- Any other dark matter profiles that mix in `MassProfileCSE`

### Tests

- Add CSE-based profiles (NFW via CSE path) to `@autolens_workspace_test/scripts/profiles_jit.py` in the JAX three-step pattern (NumPy / JAX outer / JAX JIT).
- Verify the Phase 1 self-consistency test suite still passes after the port.

## Key Constraint

The CSE decomposition (`_decompose_convergence_via_cse_from`) must NOT be called inside a `jax.jit` trace. It is a setup computation. The forward methods (`_deflections_2d_via_cse_from`, `_convergence_2d_via_cse_from`) that consume the cached decomposition results ARE traced and must be pure `xp` code.

## Repos

- @PyAutoGalaxy (primary)
- @autolens_workspace_test (test additions)
