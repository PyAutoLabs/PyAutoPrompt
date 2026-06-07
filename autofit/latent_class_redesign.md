# Redesign latent variables: a first-class `Latent` class that separates definition / engine / output / config

## Motivation

Latent-variable handling is currently entangled inside `autofit.Analysis`, which has
made it bulky and hard to evolve (the last two bug-fix passes — #1310 per-batch
masking, #1311 degenerate edge cases — both had to surgically edit a ~180-line method
buried in `analysis.py`). The codebase is independently moving output concerns OUT of
`Analysis` into dedicated, user-subclassable classes — `Visualizer` and `Result` are
declared as `Analysis.Visualizer = ...` / `Analysis.Result = ...` class attributes, and
`SearchUpdater` was already extracted from `NonLinearSearch` (`be1bb75af`). Latents
should follow the same path: decouple latent *output* from the `Analysis` exactly as the
updater decouples search output, so concerns are separated and latent *output config*
(an active need in downstream projects, e.g. the Euclid pipeline) has a clean home.

This is a deliberate redesign + refactor, not a mechanical file move. Think it through in
Opus before coding; mechanical phases can be delegated to Sonnet.

## Current architecture (what exists today — read before designing)

Four concerns live on / around `Analysis`:

1. **Definition** — `Analysis.compute_latent_variables(self, parameters, model)` (abstract;
   user overrides). `autofit/non_linear/analysis/analysis.py`.
2. **Selection / config** — `Analysis.LATENT_KEYS` (autolens/autogalaxy make it a
   `@property` reading `latent_keys_enabled()`), backed by a module-level
   `LATENT_FUNCTIONS` registry + a `config/latent.yaml` `key: bool` toggle file
   (library defaults + workspace overrides; autoconf lowercases keys; unknown keys
   warn-and-drop). See `autolens/analysis/latent.py`, `autogalaxy/imaging/model/latent.py`.
3. **Batched evaluation engine** — `Analysis.compute_latent_samples(self, samples, batch_size)`:
   `LATENT_BATCH_MODE` ("vmap" default / "jit" for vmap-incompatible inner calls),
   per-sample NaN/inf masking (global col-then-row + greedy salvage), `inject_latent_nans`
   test hook, soft-fail. The bulk of the bloat.
4. **Output orchestration + config** — `SearchUpdater._compute_latent_samples`
   (`autofit/non_linear/search/updater.py`) reads `output.yaml` flags
   (`latent_during_fit`, `latent_after_fit`, `latent_draw_via_pdf`,
   `latent_draw_via_pdf_size`), draws via PDF, calls `analysis.compute_latent_samples`,
   and saves `latent/samples.csv` + `latent/latent_summary.json`.

The Visualizer precedent to mirror: base `Visualizer` in
`autofit/non_linear/analysis/visualize.py` (all `@staticmethod`, first arg `analysis`);
`Analysis.Visualizer = Visualizer` (analysis.py); `Analysis.__getattr__` forwards
`visualize*`/`should_visualize*` to `self.Visualizer`; `SearchUpdater.visualize` calls
`analysis.visualize(...)`; users subclass `af.Visualizer` (e.g. `VisualizerImaging`).

## Problems with the status quo

- `Analysis` owns definition + selection + the heavy engine — bloated, hard to test in
  isolation, inconsistent with the `Visualizer`/`Result` "declare a class" direction.
- Users extend latents by overriding a **method** (`compute_latent_variables`) + a
  property (`LATENT_KEYS`), not by subclassing a **class** — diverging from where
  visualization is heading (subclass a `Visualizer`).
- Latent **output config** is split across `config/latent.yaml` (which latents) and
  `output.yaml` (when/how) with no single owner — awkward as projects want richer
  per-latent output control.
- Documentation: the "define your own latent" guidance exists only buried inside the
  results tutorials (`autolens_workspace/scripts/guides/results/latent_variables.py:197-244`
  "Extending with a Custom Latent"; autogalaxy equivalent ~169-219; autofit cookbook
  `cookbooks/analysis.py:644-708`) and is method-override based — there is no standalone,
  discoverable guide, and these examples will need rewriting to whatever new API lands.

## Proposed design

Introduce a first-class `Latent` class, mirroring `Visualizer`/`Result`, that owns the
latent concerns and is declared on the analysis. New module
`autofit/non_linear/analysis/latent.py`:

- **`class Latent`** (base): the user-/library-subclassed extension point.
  - `keys(analysis) -> list[str]` — which latents are enabled (default: empty; a library
    subclass reads `config/latent.yaml` + its registry).
  - `variables(analysis, parameters, model) -> tuple | dict` — compute one sample's
    latents (replaces `compute_latent_variables`).
  - `BATCH_MODE = "vmap"` (replaces `LATENT_BATCH_MODE`).
  - Follow the `Visualizer` convention (`@staticmethod` taking `analysis`) unless a
    decision is made for instance methods (see open questions).
- **Engine as a module function** in `latent.py`:
  `latent_samples_from(latent, analysis, samples, batch_size)` — the batching, vmap/jit
  dispatch, masking and salvage moved verbatim out of `Analysis.compute_latent_samples`.
  Testable in isolation; reused by every package. This alone achieves the original
  "slim analysis.py" goal.
- **`Analysis.Latent = Latent`** class attribute (mirror `Visualizer`/`Result`).
  `SearchUpdater._compute_latent_samples` calls a single entry point
  (`analysis.latent_samples_from(samples, batch_size)` delegating to the engine with
  `self.Latent`, OR the updater calls `latent_samples_from(analysis.Latent, analysis, ...)`).
  Keep output orchestration (output.yaml flags, PDF draw, save csv/summary) in the updater.
- **Library latents become `Latent` subclasses**: e.g. `LensLatent(af.Latent)` in
  `autolens/analysis/latent.py`, `GalaxyLatent(af.Latent)` in autogalaxy — `keys()` reads
  the yaml via `latent_keys_enabled()`, `variables()` builds the `{fit, magzero, xp}`
  context and dispatches `LATENT_FUNCTIONS`. `AnalysisImaging` declares `Latent = LensLatent`.
  The registry + yaml toggles stay (good config story) but move inside the class.
- **Latent output config**: give the latent output flags a clear owner. Minimum: keep
  `output.yaml` latent_* read by the updater. Stretch: a small `LatentConfig` (draw mode,
  size, during/after, per-latent enable) consumed by the updater/`Latent`, so downstream
  projects configure output in one place. Decide scope (open question).

Net separation: **definition+selection** = `Latent` subclass; **engine** = `latent.py`
function; **output+config** = `SearchUpdater` (+ optional `LatentConfig`); `Analysis`
just declares `Latent = ...`.

## Backwards compatibility — phased, not big-bang

Many call sites override `compute_latent_variables` + `LATENT_KEYS`: autolens, autogalaxy,
the Euclid pipeline (`euclid_strong_lens_modeling_pipeline/util.py` aperture latents),
workspace tutorials, the autofit cookbook. A big-bang multi-repo break is risky.

- **Phase 1 (PyAutoFit):** add `Latent` + engine extraction + `Analysis.Latent`. Provide a
  **back-compat default `Latent`** that delegates to `analysis.compute_latent_variables` /
  `LATENT_KEYS` if those are still overridden, so existing subclasses keep working
  unchanged. Move latent engine tests into `test_autofit/non_linear/analysis/test_latent.py`.
  `analysis.py` slimmed. No behaviour change.
- **Phase 2 (PyAutoGalaxy + PyAutoLens):** ship `GalaxyLatent` / `LensLatent`, declare
  `Latent = ...` on `AnalysisImaging`; relocate the per-package latent tests. Keep the
  registry + yaml.
- **Phase 3 (workspaces + Euclid pipeline + docs):** migrate `util.py` aperture latents and
  the workspace tutorials to subclass `Latent`; write the standalone **"Define your own
  latent variable"** guide (see docs gap) against the new class API.
- **Phase 4 (optional):** deprecate then remove `compute_latent_variables` / `LATENT_KEYS`
  overrides once nothing depends on them.

Issue each phase as its own task when its predecessor is close to shipping (do not queue
all upfront).

## Open design decisions (resolve in Opus before coding)

1. **Static vs instance `Latent`.** Static mirrors `Visualizer` exactly and is simplest
   (state like `magzero` is reachable via the passed `analysis.kwargs`). Instance (like
   `Result`) allows per-fit caching. Recommend static for consistency unless a concrete
   need appears.
2. **Name.** `Latent` (noun, mirrors `Visualizer`/`Result`) vs `LatentMaker`/`LatentComputer`.
3. **Method names.** `keys()` / `variables()` vs keeping `compute_latent_variables` for
   continuity (affects the back-compat shim).
4. **How far to take output config.** Keep `output.yaml` flags only, or introduce a
   `LatentConfig` for richer per-latent output control.
5. **Shim+deprecate vs hard migrate.** Recommend shim (Phase 1) to de-risk.

## Documentation gap (close in Phase 3)

- Existing custom-latent guidance is buried and method-based (cite above). After the
  redesign it must be rewritten to subclassing `Latent`.
- Write a dedicated, discoverable **"Define your own latent variable"** example
  (autolens_workspace + autogalaxy_workspace, and an autofit_workspace cookbook entry):
  the quick local path (subclass `Latent`), the library-contribution path (add to the
  registry + yaml), when to choose which, magzero/xp/JAX threading, and how the latent
  reaches `latent.csv` / the aggregator. This was never written as a first-class tutorial.

## Acceptance / verification

- `analysis.py` no longer contains the latent batching/masking body; latent logic lives in
  `latent.py`; `analysis.py` measurably slimmer.
- `pytest test_autofit` green (excluding the pre-existing `nss` optional-dep ImportErrors).
- Phase 1 is a pure refactor: PyAutoGalaxy/PyAutoLens unchanged and still green; the three
  `*_workspace_test/latent_nan_robustness.py` integration guards and the
  `latent_variables_smoke.py` still pass.
- No behaviour change through Phases 1–3 (config-driven outputs identical); only the
  extension *mechanism* changes.

## Notes / context

Sequenced after the latent NaN-robustness fixes (PyAutoFit #1310, #1311 — merged).
Prior art: `SearchUpdater` extraction (`be1bb75af`), the `Visualizer`/`Result`
class-attribute extension pattern. Latent feature history for reference: first-class
latent APIs in autogalaxy (#441) and autolens (#534), `LATENT_BATCH_MODE` (#1288-era),
raw-flux + soft-fail magzero (#463/#557), global masking (#1310), degenerate edge cases
(#1311).
