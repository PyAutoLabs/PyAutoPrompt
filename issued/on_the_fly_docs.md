We just added background quick-update support to PyAutoFit (PR #1212) and PyAutoGalaxy (PR #350).
The feature allows on-the-fly visualization during model fits to run on a background thread so
sampling is not blocked.

The following workspace scripts and notebooks need updating with clear, extensive documentation
on how to use the new quick-update functionality:

1. In @autolens_workspace/scripts/guides/modeling, add or update a section in the relevant
   modeling guide(s) explaining:
   - What on-the-fly visualization is and why it's useful (seeing intermediate fit results
     while the sampler runs)
   - How to enable it: set `quick_update_background: true` in `config/general.yaml` under
     the `updates:` section
   - How to control update frequency: `iterations_per_quick_update` (how often the visualisation
     triggers) and `iterations_per_full_update` (how often all outputs including model.results
     are written)
   - The difference between quick updates (just the fit image) and full updates (all visuals,
     model results, search summary)
   - That background mode means sampling continues during visualization, giving ~1800x speedup
     on the update step

2. In @autogalaxy_workspace/scripts/guides/modeling, add the same documentation adapted for
   autogalaxy (same config, same mechanism, just autogalaxy Analysis objects).

3. In @autofit_workspace/scripts, add or update documentation explaining the general
   quick-update mechanism from the autofit perspective:
   - The `Fitness` class's `manage_quick_update` method
   - How Analysis subclasses can override `perform_quick_update` to define custom visualization
   - The `supports_background_update` property that Analysis subclasses should set to True
   - The `supports_jax_visualization` property (for future JAX-native visualization)

4. Ensure all workspace config/general.yaml files have the `quick_update_background: false`
   entry under `updates:` with a clear comment explaining what it does.

5. Add Jupyter notebook usage notes: in notebook environments, the quick update uses
   IPython.display.clear_output(wait=True) to refresh the visualization inline. This works
   automatically when running notebooks.

Also check whether there are any other undocumented features or config options from recent
PRs that workspace scripts should cover but currently don't.
