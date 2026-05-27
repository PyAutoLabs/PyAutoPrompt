Thread xp=np through convergence_func for profiles that currently lack it.

## Problem

Several profiles' `convergence_func` methods don't accept `xp=np`, causing
`MGEDecomposer.decompose_convergence_via_mge` to crash with TypeError when
the MGE potential calls it with `xp=xp`. Currently caught and SKIPped in the
test suite, meaning these profiles' MGE-based potential is silently unavailable.

## Affected Profiles

- `PowerLawBroken` — inherits abstract `MassProfile.convergence_func` which
  doesn't accept `xp`
- `dPIEMass` — same
- `dPIEPotential` — same
- `SersicGradient` — overrides `convergence_func` without `xp` parameter

## Fix

1. Add `xp=np` to `MassProfile.convergence_func` in `abstract/abstract.py`
2. Add `xp=np` to `SersicGradient.convergence_func` in `stellar/sersic_gradient.py`
3. Thread `xp=xp` in any internal calls within these methods
4. For profiles that override `convergence_func` (check all subclasses), ensure
   `xp=np` is accepted

After this fix, the MGE potential should work for PowerLawBroken, dPIEMass,
dPIEPotential, and SersicGradient — removing 14 SKIPs from the test suite
(though they may become FAILs due to Issue 1 if elliptical).

## Verification

Run `scripts/mass/total.py` and `scripts/mass/stellar.py` in
@autolens_workspace_test. Profiles that currently SKIP should now run
(PASS for spherical, FAIL for elliptical pending Issue 1 fix).

## Repos

- @PyAutoGalaxy (primary)
