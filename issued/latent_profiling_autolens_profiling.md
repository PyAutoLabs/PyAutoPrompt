# Add a runtime-profiling `latent/` package to autolens_profiling

## Context

Parent epic: [`PyAutoPrompt/z_features/latent_refactor.md`](../z_features/latent_refactor.md).
Depends on [`autolens/latent_module.md`](../autolens/latent_module.md).

Once latent variables are first-class library API (sub-prompt #2), we need visibility into how long each latent costs at runtime — especially for the JAX-jit path through `LensCalc.einstein_radius_jit_from()` and the closure-caching behaviour at `PyAutoGalaxy/autogalaxy/operate/lens_calc.py:1580`. Mirror the existing `likelihood_runtime/` package style to produce a profiled, comparable, README-driven artefact.

## Task

1. **Create the package** at `autolens_profiling/latent/` with the same shape as `autolens_profiling/likelihood_runtime/`:
   ```
   latent/
   ├── README.md
   ├── sweep.py
   ├── aggregate.py
   └── <category>/<latent_name>.py    # one script per latent
   ```

2. **Per-latent scripts** — one file per default-enabled key in `autolens/config/latent.yaml`. Each script:
   - Sets up the minimum context needed to compute that latent (tracer, fit, image).
   - Times `compute_latent_variables` filtered to just that one key (`config/latent.yaml` toggles).
   - Outputs a JSON file with timing stats + a PNG (matching `likelihood_runtime` outputs).
   - Reports first-call vs cached-call timing separately (important for the einstein-radius cache verification).

3. **`sweep.py`** — driver that dispatches each per-latent script across the 6-config matrix (CPU/GPU × fp64/mixed-precision), same shape as `likelihood_runtime/sweep.py`.

4. **`aggregate.py`** — collates per-config JSON into a single comparison table, same shape as the likelihood_runtime counterpart.

5. **`README.md`** — sections (mirror likelihood_runtime/README.md):
   - latent
   - Methodology
   - The 6-config matrix
   - What mixed precision actually means
   - Scripts
   - Driving the matrix — `sweep.py` and `aggregate.py`
   - How to read the output
   - When the cache helps / hurts

## Where to look

- **Mirror target:** `autolens_profiling/likelihood_runtime/` — files `README.md`, `OPTIMIZATION_NOTES.md`, `sweep.py`, `aggregate.py` and the per-class subdirs.
- **Cache behaviour to characterize:** `PyAutoGalaxy/autogalaxy/operate/lens_calc.py:1520-1537` and the closure cache at 1580-1586. Memory `feedback_jax_closure_cache_busts` notes this is delicate.
- **Constant-folding gotcha:** per memory `feedback_jax_pure_callback_const_fold`, single-jit measurements can look 20-30× faster than vmap because `pure_callback` gets baked as XLA constant. Make the profiling **vmap-style** for honest numbers.
- **GPU prealloc cap:** per memory `feedback_jax_gpu_prealloc`, on the laptop GPU run with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` + renice for desktop usability. Document this in the README.
- **Output root convention:** `autolens_workspace_developer/jax_profiling/results/jit/` (where `likelihood_runtime` writes by default).

## Verification

```bash
source ~/Code/PyAutoLabs-wt/<task-name>/activate.sh
cd autolens_profiling

# Run a single latent script
python latent/<category>/<one_latent>.py --config-name=cpu_fp64 --output-dir=...

# Drive the full matrix
python latent/sweep.py
python latent/aggregate.py
```

Manual review: the README should be readable cold (someone new to the repo can run the sweep without reading source). The aggregated output table should make it easy to see which latent dominates and whether the einstein-radius cache is actually helping.

## Affected repos

- autolens_profiling (primary)

## Suggested branch

`feature/latent-profiling`

## Notes

- Per CLAUDE.md model split: dev/profiling scripts with short comments → **Sonnet** is fine for the per-latent script bodies. The README can stay Opus for the methodology framing.
- Per memory `feedback_jax_validation_vmap_not_jit`: use `fitness._vmap(jnp.array(params))` to force tracer propagation. `jax.jit(fn)(concrete_instance)` hides un-threaded xp sites.
- Output dirs must be `.gitignore`d (mirror what likelihood_runtime does). Per memory `feedback_ship_workspace_binary_leak`, check `.gitignore` covers new image dirs before shipping.
- The "first call vs cached call" split is the most interesting finding to surface — make it prominent in the aggregated table.
