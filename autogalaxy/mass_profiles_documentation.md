Add first-class LaTeX-readable documentation to all mass profile functions in PyAutoGalaxy.

## Goal

Every mass profile class and its key methods (`convergence_2d_from`, `potential_2d_from`, `deflections_yx_2d_from`) should have a docstring containing:
1. A brief description of the profile
2. The mathematical definition in LaTeX notation (convergence kappa, potential psi, deflection alpha formulas)
3. Reference to the source paper(s)
4. Parameter descriptions with physical units and typical ranges
5. Cross-references to related profiles (e.g. Sersic -> Exponential -> DevVaucouleurs as special cases)

## Scope

All classes under `@PyAutoGalaxy/autogalaxy/profiles/mass/`:
- `abstract/abstract.py` — MassProfile base class docstring explaining the interface contract
- `total/` — Isothermal, PowerLaw, dPIE families with Tessore 2015/2016, Eliasdottir 2007 references
- `dark/` — NFW, gNFW, cNFW families with Navarro+1996, Wyithe+2001, Oguri 2021 references
- `stellar/` — Sersic, Gaussian, Chameleon families with Sersic 1963, Dutton+2011 references
- `sheets/` — ExternalShear, MassSheet, ExternalPotential
- `point/` — PointMass, SMBH, SMBHBinary
- `abstract/mge.py` — MGE decomposition with Shajib 2019/2020 reference
- `abstract/cse.py` — CSE decomposition with Oguri 2021 reference

## LaTeX Format

Use raw docstrings and standard LaTeX math notation that renders in Sphinx/ReadTheDocs:

```python
def convergence_2d_from(self, grid, xp=np, **kwargs):
    r"""Projected surface mass density (convergence).

    .. math::

        \kappa(R) = \frac{\theta_E}{2R}

    where :math:`\theta_E` is the Einstein radius and :math:`R` is the
    elliptical radius.

    Parameters
    ----------
    grid : Grid2D or Grid2DIrregular
        2D coordinates at which to evaluate convergence.

    References
    ----------
    Kormann et al. (1994), A&A, 284, 285
    """
```

## Additional Task

Update `@PyAutoGalaxy/docs/api/mass.rst` to include all mass profiles. Currently missing:
- Several NFW variants (MCR, Scatter, Virial)
- ExternalPotential
- SMBH, SMBHBinary
- dPIEPotential family
- GaussianGradient

Cross-reference with `ag.mp.*`, `ag.lmp.*`, `ag.lmp_linear.*` namespaces to ensure completeness.

## Scientific Context

Consult `@PyAutoPaper/lensing_wiki/` for paper references and concept definitions. Key entries:
- `concepts/mass-models.md` — overview of parametric mass profiles
- `concepts/multipoles.md` — angular structure beyond elliptical
- `concepts/bulge-halo-decomposition.md` — stellar + dark decomposition

## Repos

- @PyAutoGalaxy (primary)
