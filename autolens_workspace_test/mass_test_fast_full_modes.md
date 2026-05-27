Refactor the scripts/mass/ test suite into Fast and Full test modes.

## Goal

The current scripts/mass/ tests use a single grid size and one parameter set per
profile. Refactor into two modes that serve different purposes:

### Fast Mode (default)

Quick sanity check: "did my code changes break any mass profile?"

- Small grid (e.g. 40x40, pixel_scales=0.05) — same as current PYAUTO_MASS_FAST
- One representative parameter set per profile (the current defaults)
- Loose tolerances (rtol=5e-2 for finite-difference checks)
- Target runtime: under 30 seconds total for all 5 scripts
- Suitable for smoke tests, CI, and quick validation after library changes
- Triggered by default (no env var needed) or explicitly with PYAUTO_MASS_MODE=fast

### Full Mode

Stress-test: find subtle numerical issues before a release.

- Large grid (e.g. 200x200, pixel_scales=0.02) for higher finite-difference accuracy
- Multiple parameter sets per profile, sampling across:
  - Axis ratios from near-circular (q=0.95) to highly elliptical (q=0.3)
  - Scale parameters from very compact to very extended
  - Einstein radii / kappa_s across 2 orders of magnitude
  - Edge cases: core_radius near zero, slope near 2.0 (isothermal limit),
    break_radius near grid edge, f_c near the Penarrubia MCR limit (~0.18)
  - Profiles centred at origin AND offset from origin
- Tight tolerances (rtol=1e-2 for finite-difference, rtol=1e-3 for analytic comparisons)
- Target runtime: 5–10 minutes total (acceptable for pre-release)
- Triggered with PYAUTO_MASS_MODE=full

### Parameter Sweep Design

For each profile, define a list of parameter dictionaries. Fast mode uses index 0
only. Full mode iterates over all.

Example for Isothermal:
```python
ISOTHERMAL_PARAMS = [
    # Fast: representative
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2),
    # Full: near-circular
    dict(centre=(0.0, 0.0), ell_comps=(0.01, 0.005), einstein_radius=1.2),
    # Full: highly elliptical
    dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), einstein_radius=1.2),
    # Full: large einstein radius
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=5.0),
    # Full: small einstein radius
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=0.2),
    # Full: offset centre
    dict(centre=(0.5, -0.3), ell_comps=(0.1, 0.05), einstein_radius=1.2),
]
```

For dark matter profiles, push toward the boundaries that have caused issues:
- cNFW with f_c approaching the Penarrubia limit
- NFW with very large or very small scale_radius relative to the grid
- gNFW with inner_slope near 0 and near 3

For stellar profiles, test mass_to_light_ratio extremes and sersic_index from
0.5 (Gaussian-like) to 8.0 (very concentrated).

### Implementation

Refactor util.py:
- Replace FAST boolean with MODE = os.environ.get("PYAUTO_MASS_MODE", "fast")
- Add make_grid(mode) that returns appropriate grid for each mode
- Add get_tolerances(mode)
- Keep the existing run_all_checks / print_summary_table API unchanged

Each category script:
- Define parameter lists per profile
- In fast mode: run index 0 only (current behaviour, just faster grid)
- In full mode: iterate over all parameter sets, appending results with
  the parameter index (e.g. "Isothermal[0]", "Isothermal[1]", ...)
- Print a combined summary table at the end

### What This Catches

Fast mode catches:
- Outright crashes (TypeError, NotImplementedError)
- Sign flips and gross numerical errors
- Zero-returning regressions

Full mode additionally catches:
- Accuracy degradation at high ellipticity
- Edge-case parameter values that trigger NaN/Inf
- Boundary condition bugs (like the cNFWSph F_func issue at theta=r_s)
- MGE decomposition quality at parameter extremes
- Numerical instability in special functions (arctanh, arctan near singularities)

## Repos

- @autolens_workspace_test (primary — scripts/mass/)
