JAX substructure forward simulator — multi-step feature adding a pure-array
`jax.jit` / `jax.vmap` path for multi-plane dark-matter substructure simulations.
Motivated by PyAutoLens issue #542 (mwiet, 2026-05-26): the user needs
`theta -> noisy image` evaluated ~10^6 times on GPU, with ~1000 LOS halos
across ~8 planes plus lens-plane subhalos (CDM/WDM NFWTruncated, SIDM cNFW).

This is a **separate pure-function path** alongside the existing OO Tracer,
not a rewrite of it. The existing `al.Tracer` with Python-loop unrolling is
the right approach for normal lens modeling (2-5 galaxies, fixed model, fast
compilation). The new path targets the substructure use case where variable N
and O(1000) halos make unrolling prohibitive. Both paths share the same
underlying deflection math from the profile classes.

__What's already done__ (no work needed):

- All 4 dark profiles JAX-traceable via xp: `NFWTruncatedSph`,
  `NFWTruncatedMCRLudlowSph`, `cNFWSph`, `cNFWMCRLudlowSph`
- cNFW deflection boundary bug fixed (PyAutoGalaxy #454, merged 2026-05-27)
- Ludlow16 MCR JAX-native (PyAutoGalaxy #403)
- Convolver: both FFT and real-space paths JAX-ready
- Poisson noise routes through `jax.random.poisson` when `xp=jnp`
  (`autoarray/dataset/preprocess.py`)
- Macro profiles: `PowerLaw` (with `jax.lax.scan` series expansion),
  `ExternalShear`, `SersicCore` all JAX-traceable
- CSE module ported to JAX (PyAutoGalaxy #447)
- Regular grid interpolator JAX path (PyAutoArray #306)
- Cosmology scaling factors fully xp-threaded
  (`autogalaxy/cosmology/model.py`)
- Simulator `use_jax=True` flag wired (PyAutoArray #334, PyAutoLens #538-540)

__Outstanding__ (sequenced):

1. [jax_substructure/1_vmap_subhalo_deflections.md](../jax_substructure/1_vmap_subhalo_deflections.md) —
   vectorized deflection path: represent N halos as `(max_N, n_params)` arrays,
   `jax.vmap` the profile deflection function, sum with mask. Integration test
   comparing against the existing Tracer Python-loop result.
2. [jax_substructure/2_tracer_lax_scan.md](../jax_substructure/2_tracer_lax_scan.md) —
   `jax.lax.scan` over planes: precomputed scaling-factor matrix, fixed-shape
   per-plane halo stacks, one scan op replaces the nested Python loops in
   `tracer_util.traced_grid_2d_list_from`.
3. [jax_substructure/3_simulator_jax_e2e.md](../jax_substructure/3_simulator_jax_e2e.md) —
   end-to-end `jax.jit(simulate)`: wire PSF convolution, add `prng_key` support
   for Poisson noise, thread xp through over-sampling, smoke test on a
   representative substructure configuration.
4. [jax_substructure/4_vmap_batched_simulation.md](../jax_substructure/4_vmap_batched_simulation.md) —
   stretch goal: `vmap(jit(simulate))(thetas, keys)` for ~1024 images per GPU
   launch. Depends on all three previous prompts.
