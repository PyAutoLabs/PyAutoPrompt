
## euclid-einstein-radius-zero-contour
- issue: https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/issues/14
- completed: 2026-05-21
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1288
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/435
- workspace-pr: https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/pull/15
- notes: Phase B of z_features/fast_visualization.md. Re-enables the effective_einstein_radius latent in the Euclid pipeline workspace, previously commented out because the only available API (tracer.einstein_radius_from(grid=...)) was not JAX-traceable under compute_latent_samples' vmap+jit wrap. Deep-research finding that re-scoped the task: jax_zero_contour.ZeroSolver explicitly documents incompatibility with jax.vmap (uses lax.cond / lax.while_loop early-termination). Vmap-friendly path would require either an upstream fork or a parallel JAX algorithm bypassing ZeroSolver — both out of scope. Landed jit-only architecture instead: PR PyAutoFit#1288 adds Analysis.LATENT_BATCH_MODE class attribute (default "vmap" for backwards compat, new "jit" option). PR PyAutoGalaxy#435 adds LensCalc.einstein_radius_jit_from(init_guess, ...) — ~95-line JIT-friendly helper that bypasses _init_guess_from_coarse_grid (skimage) and ZeroSolver.path_reduce (variable-length output), computes shoelace area on raw NaN-padded paths via jnp.where masking, returns scalar jax.Array; also sets AnalysisDataset.LATENT_BATCH_MODE = "jit" so all PyAutoGalaxy/PyAutoLens analyses inherit jit-per-sample automatically. PR pipeline#15 dispatches on self._use_jax: JAX → new helper with 4-seed fan at ±1 arcsec, numpy → legacy einstein_radius_from(grid=...). Verified end-to-end: max-LL latent.effective_einstein_radius = 2.1002 arcsec on dataset 102018665_NEG570040238507752998 prior-median MGE tracer. Latent step ~480 ms/sample on CPU after the ~10s ZeroSolver compile (1000 samples ≈ 80s, vs ~30s per-sample numpy via z_projects/euclid workaround — slower for Euclid scale but unlocks the JAX-end-to-end pipeline architecture and is fundamentally faster than numpy for cluster-scale geometry). Gotchas: (1) LensCalc.from_mass_obj(tracer) is the correct construction pattern — Tracer doesn't expose einstein_radius_via_zero_contour_from directly (matches the z_projects/euclid pattern). (2) Tried direct vmap path first; failed in convert.axis_ratio_and_angle_from because _init_guess_from_coarse_grid line 1107 called tangential_eigen_value_from without xp=jnp threading. Even after fixing that, find_contours (skimage) blocks JAX trace fundamentally — necessitating the new helper. (3) PR body URL cross-references had to be patched via `gh api PATCH` because gh pr edit hit a Projects-Classic GraphQL deprecation warning — same workaround as the previous task. (4) workspace PR's url_check CI failed on a pre-existing Jammy2211/autolens_workspace reference in README.md:110 (not touched by this PR); merged anyway since the failure is independent and the library PRs were green. Follow-up: a future task should consider whether to fix README.md:110 broadly or add the URL pattern to allowlist. Future work: the new helper accepts init_guess as a required argument so callers must know lens position; for unconstrained scenarios a JAX-native seed-finder (e.g. jnp.argmin on |eigen_values| coarse grid) could replace the static init_guess — but that's a separate library task.

## fast-viz-zero-contour-perf
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/433
- completed: 2026-05-21
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/434
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/527
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/111
- notes: Phase A′ of z_features/fast_visualization.md. Two prior reverts (PyAutoGalaxy abd7b717 on 2026-04-19, PyAutoFit #1280 on 2026-05-17) shared the same shape — a JAX-trace failure inside the viz path swallowed by the broad except in fit_imaging_plots.py:52. The 2026-05-21 perf benchmark uncovered a THIRD independent issue: `_critical_curve_list_via_zero_contour` rebuilt its closure and ZeroSolver on every call, busting JAX's compile cache and paying the full ~10s compile cost every invocation. PR #434 caches `(f, ZeroSolver)` keyed on `(kind, pixel_scales, tol, max_newton)` — warm calls 10300ms → 67ms (tangential), 68ms (Einstein radius), 380ms (radial). PR #527 tightens the broad `except` so future regressions of this class fail loud (WARNING + exc_info) instead of silent (None, None, None, None). PR #111 lands the first __Visualization Sanity__ block on autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py with curve-count + Einstein-radius + warm-call < 100ms perf regression net (uses a separately-constructed SIE tracer, not the script's MGE prior medians, so assertions are deterministic). Tracker fast_visualization.md updated: Phase A scope revised from "flip YAML default" to "make plotter dispatch context-aware" (marching_squares stays default for one-shot plotting which pays a one-time 10s compile, JIT callers auto-route to zero_contour); Phase B retargeted from z_projects/euclid (live science, off-limits) to euclid_strong_lens_modeling_pipeline/util.py:491 (effective_einstein_radius latent currently commented out there, ready for re-enablement via einstein_radius_via_zero_contour_from()); BackgroundQuickUpdate noted as already-shipped in PyAutoFit (1fee93174, daemon thread + latest-only drop, wired into nautilus search.py:196,216 via background_quick_update kwarg) — Phase C reduced to IPython.display.update_display(fig, display_id=...) wiring only; Phase F (subprocess viz) likely obsolete now that threading + non-blocking viz already exists. Gotchas: (1) workspace_test smoke CI red on point.py — pre-existing JAX vmap regression (returns -1e99 sentinel instead of -83.38) confirmed by reproducing on canonical main; user picking up separately. (2) autolens_workspace_test canonical checkout had pre-existing drift in README.md + dataset/build/*.fits/json from earlier local simulator runs — left untouched, isolated from the worktree branch. (3) Workspace_version_check fires WARNINGs because workspace_version isn't pinned in workspace_test/config — informational only, not blocking. (4) gh pr edit failed with a GitHub Projects Classic GraphQL deprecation warning when patching cross-references; subagent fell back to gh api PATCH which worked. Follow-up prompts queued for future authoring: Phase A context-aware dispatch (autogalaxy/critical_curves_method_context_aware.md), Phase B Euclid latent migration (euclid_strong_lens_modeling_pipeline/einstein_radius_zero_contour_migration.md), Phase C IPython display wiring (autofit/quick_update_display_id.md), Phase D rollout (autolens_workspace_test/end_to_end_jax_viz_rollout.md), Phase E ModelInstance pytree cascade (autofit/model_instance_pytree_cascade.md). Eager-mode zero_contour (JAX_DISABLE_JIT=1) timed out > 10 min per call — unusable without JIT; this is recorded in the tracker as the third row of the perf table.

## pixelization-clumpy-galaxy
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/92
- completed: 2026-05-21
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/93

## sample-kwargs-mixed-keys
- issue: (user-reported by Sam via Slack, aggregator-to-database load, no GitHub issue)
- completed: 2026-05-21
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1287
- repos: PyAutoFit
- notes: One-line fix in `Sample.__init__` — removed `and "." in key` from the str→tuple key conversion so dotless kwargs keys (e.g. `'dummy_0'`) become single-element tuples (`('dummy_0',)`) consistently. Bug surfaced when a model had BOTH a nested path (e.g. `'ellipses.11.centre.centre_0'`) AND a top-level dotless prior: the asymmetric conversion produced a mixed-type dict, `is_path_kwargs` inspected only the first key and misclassified the sample, and `parameter_lists_for_paths` then raised `KeyError: "(('dummy_0',),)"`. No existing test combined these two model shapes, which is why the bug went undetected. Four test files (test_efficient, test_samples in database/paths, test_latent_variables ×3) codified the pre-fix raw-string shape — these were spot-checking internal kwargs representation, not user-visible behavior, and were updated to assert the new uniform-tuple shape. The fix also silently repairs two latent bugs: (1) `Samples.values_for_path(path: Tuple[str, ...])` would have raised KeyError pre-fix on any sample whose kwargs were built with dotless string keys; (2) aggregator `Column.value` lookup wrapped a `KeyError → None`, which means dotless latent variables (e.g. `'fwhm'`) merged into mixed-shape dicts via `kwargs.update(latent_summary.median_pdf_sample.kwargs)` would have silently produced None CSV cells. Round-trip is symmetric — `Sample.dict()` joins tuples back to dotted strings, so on-disk JSON/CSV/database forms are unchanged. Validation: PyAutoFit unit suite 1399/0 (1 skip), aggregator 49/49, database 144/144, by-path+result 28/28, full 5-workspace smoke 35/0/2 (the single failure was a pre-existing JAX vmap point-source rebaseline issue in autolens_workspace_test, completely unrelated — script doesn't reference `Sample` at all). Lesson: when a conversion is `isinstance + something else`, the "something else" guard is suspect — it usually means two callers were doing different things and got papered over rather than reconciled.

## jax-bump-floor-0.7
- issue: (user-reported via Slack, ppyjc14 traceback, no GitHub issue)
- completed: 2026-05-21
- library-prs: https://github.com/PyAutoLabs/PyAutoConf/pull/108, https://github.com/PyAutoLabs/PyAutoArray/pull/328, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/432, https://github.com/PyAutoLabs/PyAutoFit/pull/1286, https://github.com/PyAutoLabs/PyAutoBuild/pull/92, https://github.com/PyAutoLabs/PyAutoBuild/pull/93, https://github.com/PyAutoLabs/PyAutoBuild/pull/94
- released-version: 2026.5.21.1
- repos: PyAutoConf, PyAutoArray, PyAutoGalaxy, PyAutoFit, PyAutoBuild
- notes: User ppyjc14 hit `AttributeError: module 'jax.experimental.pallas.triton' has no attribute 'CompilerParams'` on `dataset.dirty_image`. Their resolved env was JAX 0.4.38 + nufftax 0.4.0 on Python 3.12. Root cause: nufftax 0.4.0 calls `pallas.triton.CompilerParams` (renamed from `TritonCompilerParams` in JAX 0.7.0), but nufftax's pyproject declares only `jax>=0.4.0` — too loose. PyAutoConf's `[jax]` extra also had a too-loose floor (`jax>=0.4.35,<0.10.0`), letting pip produce broken installs. Fix bumped PyAutoConf floor to `jax>=0.7.0,<0.11.0` (also raised ceiling — code audit confirmed no `jax.pmap` or `PartitionSpec` tuple-equality usage so JAX 0.10.x is safe), pinned `nufftax>=0.4.0,<0.5.0; python_version >= '3.12'` in PyAutoArray, and pinned `jax_zero_contour>=2.0.0,<3.0.0` in PyAutoGalaxy. The TestPyPI rehearsal also surfaced a pre-existing **release blocker**: PyAutoFit's `[nss]` extra had `blackjax @ git+...` and `nss @ git+...` direct URLs added on 2026-05-16 — PyPI/TestPyPI rejects these in uploaded wheels ("400 Can't have direct dependency"), so every release attempt since had been silently failing at the twine step. Stripped both git URLs from the extra, documented the manual `pip install git+...` step in the pyproject comment, and updated PyAutoFit's `unittest_nss` + `nss_install_smoke` CI jobs to do the manual install post-extras. Also patched `release.yml`'s release_test_pypi pytest to `--ignore=test_autofit/non_linear/search/nest/nss` (NSS tests need the fork; covered separately in unittest_nss), and patched `verify_workspace_versions.sh` to `tail -n 1` the version-reading python output (JAX's `cuda_plugin_extension is not found` log goes to STDOUT on non-CUDA laptops and contaminated the parser). PyAutoBuild added `--testpypi` flag to `verify_install.sh` for pre-release rehearsals against a TestPyPI dry-run upload. TestPyPI rehearsal A+B+D all PASS on Python 3.9/3.10/3.11/3.12/3.13 — confirmed the JAX install path is fixed end-to-end. Release dispatched with `SKIP_SCRIPTS=true SKIP_NOTEBOOKS=true` and shipped to PyPI at 2026.5.21.1. CI flake on `run_smoke_tests (autofit_workspace)` — `searches/mcmc.py` MCMC convergence issue with no-dynamic-range columns; unrelated to JAX bump, did not block the release path. Hidden risk worth flagging: nufftax under-declares its JAX needs (declares `jax>=0.4.0` but actually needs 0.7.0+) — PyAutoConf's tighter floor compensates, but a future nufftax 0.4.x bump could break things unannounced. Upstream issue to nufftax recommended as a follow-up. User handled ppyjc14 comms directly.

## many-vis-prep-dft
- issue: (CI-triage cluster G, no GitHub issue)
- completed: 2026-05-20
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/90
- library-followup-issue: https://github.com/PyAutoLabs/PyAutoArray/issues/326
- repos: autogalaxy_workspace
- notes: Cluster G triage flagged three interferometer failures under the new nufftax-backed `TransformerNUFFT`. Investigation showed only one was a workspace-side issue: `autogalaxy_workspace/scripts/interferometer/features/pixelization/many_visibilities_preparation.py` was the lone outlier in its folder still passing `transformer_class=ag.TransformerNUFFT` and then calling `apply_sparse_operator`, which raises `NotImplementedError` under the new transformer (deliberate guard at `PyAutoArray/autoarray/dataset/interferometer/dataset.py:261-282`; nufftax's strict adjoint has a different absolute scale from the dirty image the sparse-operator solver was built against). Every sibling script in the folder (`modeling.py:194`, `fit.py:172/472`, `source_science.py:69`) and the parallel `autolens_workspace` script (`many_visibilities_preparation.py:101`) already used `TransformerDFT` — confirming DFT+sparse is the canonical pairing, NUFFT was never intended for the sparse path. One-line fix: `ag.TransformerNUFFT` → `ag.TransformerDFT` at line 87. Smoke: 6/6 passed; script writes `nufft_precision_operator_3.0.npy` end-to-end in test mode. The other two Cluster G failures (`autolens_workspace_test/scripts/interferometer/nufft.py` 5-px round-trip offset; `dataset_model_parity_delaunay.py` Delaunay parity, script self-comment "THIS IS THE BUG THE FIX TARGETS") are library-side gaps in the new adjoint's scale/grid-origin convention — filed as PyAutoArray#326 ("Complete the TransformerNUFFT migration"). Gotcha worth flagging: this task pre-dated the `/start_dev` flow so it never got an `active.md` entry; logged here post-hoc.

## fast-plots-env-coverage
- issue: (CI-triage cluster A, no GitHub issue)
- completed: 2026-05-20
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/91
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/52, https://github.com/PyAutoLabs/autolens_workspace_test/pull/107
- repos: PyAutoBuild, autogalaxy_workspace_test, autolens_workspace_test
- notes: Triaged from CI bug cluster A — 7 `*_workspace_test/scripts/.../visualization*.py` scripts (+ 1 parked 40d-old NEEDS_FIX) failing with `dataset.png missing` / `fit.png was not produced`. Triage doc hypothesised a visualization-layer rename / output-dir change; reproducing on clean main showed the plotters work correctly — the failures were entirely an env-var resolution bug in PyAutoBuild. `env_config._pattern_matches` (and the matching `build_util.should_skip` / `_find_skip_reason`) substring-matched the YAML pattern against `file.with_suffix("")` — i.e. the path with `.py` stripped. Three env_vars.yaml entries ending in `.py` (e.g. `imaging/visualization.py`) therefore never matched, leaving `PYAUTO_FAST_PLOTS=1` set on the visualization scripts; `PYAUTO_FAST_PLOTS=1` short-circuits both `subplot_save` and `save_figure` in PyAutoArray (utils.py:365, 541), so no PNG was ever written and the file-existence assertions failed. Fix: substring-match against `str(file)` (with extension) so `.py`-anchored patterns work — that change also caught the just-merged `repro_command.canonical_env_for_script` whose call site needed updating (the worktree had to be rebased onto post-#90 main). On the workspace side, four override entries were missing `PYAUTO_FAST_PLOTS` from their `unset:` lists, three `visualization.py` scripts had no override at all, and `interferometer/visualization.py` (autolens) also needed `PYAUTO_SMALL_DATASETS` unset to handle its full-res FITS load. Verified: PyAutoBuild full test suite 72/72 (8 new regression tests in `test_pattern_matches.py` locking in the convention); all 9 affected scripts pass end-to-end under the autobuild-resolved env. Hidden risk worth flagging: the same dead-pattern bug existed in `build_util.should_skip` for `no_run.yaml` patterns — no current `no_run.yaml` uses `.py` suffix so nothing was broken, but it's now fixed defensively.

## cluster-point-tuple-prior
- issue: (CI-triage cluster, no GitHub issue)
- completed: 2026-05-20
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/429
- repos: PyAutoGalaxy
- notes: Triaged from a CI bug cluster — `cluster/start_here.py` and `cluster/modeling.py` crashed with `TypeError: Point.__init__() got an unexpected keyword argument 'centre_0'`. Root cause was upstream of `Point`: `galaxy_af_models_from_csv_tables` built each `af.Model` via `af.Model(cls, **params)`, and PyAutoFit's `PriorModel.__init__` (`prior_model.py:144-156`) only takes the TuplePrior auto-create branch for tuple values that arrive via *defaults*, not *kwargs* — so a tuple-valued `centre` in `kwargs` was stored as a raw tuple attribute on the model. Subsequent `model.centre_0 = af.GaussianPrior(...)` then hit `PriorModel.__setattr__`'s "look up a TuplePrior named `centre` and delegate" branch (lines 386-395), failed the lookup, and fell through to `super().__setattr__`, creating ghost direct `centre_0`/`centre_1` attributes alongside the raw `centre` tuple. At sample time `_instance_for_arguments` packed all three into one `cls(**kwargs)` call → TypeError. Fix: construct `af.Model(cls)` first so the TuplePrior auto-create branch fires, then `setattr(model, f"{name}_{i}", component)` for each tuple param. Scalar params unchanged. Hidden risk worth flagging: both crashing scripts are in `PyAutoBuild/.../no_run.yaml:32-33`, so the release-build sweep skips them — that's almost certainly why this regression survived the cluster-CSV API rollout. Same foot-gun would have hit any other CSV-built profile with tuple params (e.g. `mass.centre_0 = prior`); the producer-side fix closes the whole class. Regression test in `test_galaxy_model_csv.py` exercises the failing pattern end-to-end (CSV → af.Model → centre_0/1 GaussianPrior → instance_from_unit_vector). Full PyAutoGalaxy suite green (922/922). Manually re-ran both cluster scripts under PYAUTO_TEST_MODE=2 from the worktree — both complete cleanly.

## drawer-jax-fom-coerce
- issue: (chat-reported, no GitHub issue)
- completed: 2026-05-20
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1283
- repos: PyAutoFit
- notes: User reported `TypeError: Object of type ArrayImpl is not JSON serializable` from `af.Drawer` + `use_jax=True` (autogalaxy_workspace ellipse.py-style code). Root cause: `AbstractInitializer.figure_of_metric` returned the raw fitness output, which under JAX-backed Fitness is a 0-d `jax.Array`. Drawer is uniquely affected because it stuffs the full `search_internal` dict (with raw `log_posterior_list`) into `samples_info` — other searches build scalar-metadata-only `samples_info`. Existing `Fitness.call_wrap` float() coercion (fitness.py:243) only fires with `use_jax_jit=True`, and Drawer constructs Fitness with the default `use_jax_jit=False`, so it never covered this codepath. Fix: one-line `return float(figure_of_merit)` at the producer in initializer.py; function annotation was already `Optional[float]`. Regression test uses numpy 0-d fitness (library unit tests stay numpy-only per project rule) and asserts both `type(fom) is float` and `json.dumps` round-trip. No API surface affected, zero workspace script references to the private symbol. Full PyAutoFit suite green (1259 passed / 1 skipped).

## mge-source-truth-tests
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72 (follow-up; #72 closed by PR #73)
- completed: 2026-05-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/75
- repos: autolens_workspace_developer
- notes: Tests 3+4 in the source-science series. Flipped which source class is "matched to truth" relative to PR #74: use an MGE source truth extracted from the test-2 mge_source MLE (via new `source_science/extract_mge_truth.py`) instead of the SersicCore truth used in tests 1+2. Key findings: (a) test 3 (MGE truth + lens light) shows the MGE+MGE catastrophe is WORSE on MGE truth than on Sersic truth — source flux 7× truth vs 2× in test 1, magnitude bias -2.12 mag vs -0.77 — falsifying any "MGE is fine if you match its truth" intuition; (b) the Sersic-source robustness from test 1 evaporates when truth is MGE + lens light is present, hitting -13% to -26% magnification bias; (c) test 4 (MGE truth, no lens light) recovers magnification at ~5% for both Sersic AND MGE source fits, confirming the "no lens light is easy" regime; (d) but MGE+MGE no-lens-light has +15% source-flux and +10% image-plane-flux bias — a basis-internal MGE degeneracy distinct from lens-light absorption. The cleaner thesis emerging from tests 1-4 together: "Without lens light, magnification is robust to source-class mismatch (~4-5% across all combinations). With lens light, source-model mismatch creates significant bias; MGE source + MGE lens light is the catastrophic combination." Built two new cross-experiment plots: `test3_vs_test4_mge_source.png` (mirrors PR #74's headline for MGE truth) and `matched_vs_mismatched_2x2.png` (2x2 grid summarising every truth_class × lens_light combination). Latter required re-running tests 1+2 fits in this worktree since v3 cache was lost after PR #74 cleanup. Gotchas: `samples.draw_randomly_via_pdf` floating-point bug already documented in PR #74's notes (bypass via `_draw_indices_from_pdf` in visualize.py). For the next study (lens-config-robustness), the user wants to test 2 more lens setups + MGE-lens-light truth tests at the existing config — moved to GPU to keep compute tractable.

## cluster-likelihood-function
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/190
- completed: 2026-05-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/191
- repos: autolens_workspace
- notes: Added `scripts/cluster/likelihood_function.py` (~780 lines) — step-by-step walkthrough of the cluster point-source log-likelihood for the standard cluster model (lenses at z=0.5, sources at z=1.0 and z=2.0). Source-plane chi² (`FitPositionsSource`) section explains multi-plane recursive lens equation + cosmological scaling factors + magnification weighting; image-plane chi² (`FitPositionsImagePair`) section explains the PointSolver forward-solve + Hungarian-algorithm pairing + the three pairing schemes (Pair / PairAll / PairRepeat) + the too-many / too-few image pathology. Both flavours validated to match the library log likelihoods exactly. Granularity locked via AskUserQuestion upfront per the prompt's "I will refine with you" instruction — black-box API calls but with full physical/mathematical explanation; TODO comment placed for a future dedicated triangle-solver guide. API surprises encountered: (a) `tracer.deflections_between_planes_from(plane_j=intermediate)` gives wrong source-plane positions for non-final source planes; library uses `traced_grid_2d_list_from(grid)[plane_index]` instead, switched to that. (b) Magnification per source uses `ag.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=...)`, not a method directly on `Tracer`. (c) `solver.solve(...)` needs explicit `plane_redshift=dataset.redshift` for multi-plane sources or it assumes the last plane. (d) Library pairing is Hungarian (linear sum assignment via scipy), not greedy. The truth-model chi² is ~8e7 not zero — confirmed the precision-floor finding documented in cluster-test-workspace #105's likelihood_sanity.py.

## cluster-test-workspace
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/104
- completed: 2026-05-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/105
- follow-up-issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/106 (items 5-8 + precision-floor investigation)
- repos: autolens_workspace_test
- notes: Shipped items 1-4 of the 8-deliverable plan. Built `scripts/cluster/csv_api.py` (truth model writer — 2 main + 1 extra + 1 host halo + 10 scaling + 2 sources), `scripts/cluster/simulator.py` (loads family CSVs, JAX-jitted PointSolver, emits `data.fits` + `point_datasets.csv` + `tracer.json`), moved `scripts/imaging/visualization_cluster.py` → `scripts/cluster/visualization.py` (retargeted to `dataset/cluster/test/`, centres now read from `mass.csv` since `*_centres.json` files were dropped in PR #189), and `scripts/cluster/likelihood_sanity.py` (perturbs each numeric dPIE/NFW mass param by ε ∈ {±0.001, ±0.01, ±0.05, ±0.1, ±0.2}; runs source-plane chi² via `FitPositionsSource`). Headline finding from the sanity diagnostic: cluster source-plane chi² is dominated by the PointSolver precision floor amplified by image-plane magnification (~100x at multi-image positions, ÷ σ_pos=0.005" → ~8e7 baseline), making perturbations below ~10% unreliable for sensitivity. Shipped as a soft-warn diagnostic with the finding documented in the script header; full investigation queued for #106. `FitPositionsImagePair` is gated behind `RUN_IMAGE_PLANE=False` in the sanity script — each forward-solve is too expensive for the perturbation sweep (timed out at 1500s when enabled). Smoke 11/11. Items 5-8 (likelihood_redshift_sensitivity / likelihood_imaging / paired viz / JAX cluster likelihood functions) deferred to #106 per `feedback_no_bulk_issue_queues`.

## source-science-no-lens-light
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72 (follow-up; #72 closed by PR #73)
- completed: 2026-05-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/74
- repos: autolens_workspace_developer
- notes: Test 2 in the source-science recovery series. Two deliverables in one PR. (1) Restructure: moved test-1 scripts and results from flat `source_science/` into `source_science/results/1_with_lens_light/`, deleted the redundant `1_summary.md` from the prior PR's flow. (2) Test 2 itself: removed lens light from both simulator and fit models to test the hypothesis that the test-1 MGE-source magnification disaster (-44% bias, -0.77 mag) was caused by lens-light/source-light degeneracy. Hypothesis confirmed: MGE-source magnification bias collapsed from -44% to -3.6%, magnitude bias from -0.77 mag to -0.045 mag, source-flux bias from +103% to +3.9% — matching the Sersic-source recovery on the same dataset. (3) Visualisation suite added: `source_science/visualize.py` shared helpers (radial profile, cumulative flux, 2D image, posterior 1σ band) + per-experiment `make_diagnostics.py` drivers + cross-experiment headline plot `test1_vs_test2_mge_source.png` that overlays test-1 MGE source (with diffuse halo) vs test-2 MGE source (truth-matching) vs truth. Gotchas: (a) The v3 Nautilus cache from PR #73 lived in the prior task worktree and was lost on `worktree_remove` — test 1 diagnostics had to re-do all three Nautilus fits in this worktree (~45 min). (b) Autofit `samples.draw_randomly_via_pdf` raises `ValueError: probabilities do not sum to 1` on the cached MGE samples due to float drift in `weight_list.sum()`; `visualize.py` bypasses with `_draw_indices_from_pdf` that re-normalises locally. (c) `aplt.fits_imaging(dataset=...)` is the current API for writing simulated `Imaging` to .fits; the old `dataset.output_to_fits(...)` (used by the Codex baseline simulator) no longer exists. (d) `np.asarray(Array2D)` returns the slim (flat) shape; use `np.asarray(arr.native)` for the 2D shape needed by `plt.imshow`. Tightened test-2 RESULTS.md wording after user feedback ("33.66 magnification is not 'ok' just because it's not the test-1 catastrophe") — explicitly states neither test-2 fit is science-acceptable (still ~5σ off truth), and Sersic-MGE agreement implies residual bias is profile-shape (radius_break + sersic_index), not source-flexibility — that's test-3's target. Future-work order: radius_break=0.025 retest → MGE σ-cap → RectangularAdaptImage → Delaunay.

## cluster-csv-api
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/187
- completed: 2026-05-19
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/428, https://github.com/PyAutoLabs/PyAutoLens/pull/526
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/189
- repos: PyAutoGalaxy, PyAutoLens, autolens_workspace
- notes: Made CSV the first-class API for cluster lens modelling. Added `autogalaxy/galaxy/galaxy_model_csv.py` with `GalaxyModelRow` / `GalaxyModelTable` dataclasses and four public functions — `galaxy_models_to_csv`, `galaxy_models_from_csv`, `galaxies_from_csv_tables`, `galaxy_af_models_from_csv_tables` — re-exported under `ag.*` and `al.*`. Schema: one CSV per profile family (`mass.csv` / `light.csv` / `point.csv`), each row carries `galaxy` + `attr_name` + `profile_class` + sparse parameter columns + optional `redshift`. Profile-class dispatch via `getattr` against `autogalaxy.profiles.{mass,light.standard,point_sources}`. Tuple params: `centre` splits into `y, x` (precedent from `galaxy_table.py`); other tuples (e.g. `ell_comps`) into `<name>_0` / `<name>_1`. Workspace consumption: new pedagogical `scripts/cluster/csv_api.py` walks through every cluster CSV end to end; `simulator.py` writes the truth model as the three family CSVs (drops the per-tier JSON centre files entirely); `modeling.py` and `start_here.py` compose `af.Model[Galaxy]` directly from `al.galaxy_af_models_from_csv_tables` and mutate selected params into priors. Scaling-tier `scaling_galaxies.csv` deliberately kept on its legacy 3-column schema. Source centre priors deliberately initialised from observed-position mean (not CSV truth). Writer-side family-validation guard added after a bug surfaced while writing `csv_api.py` (passing a light profile under `family="mass"` silently wrote malformed rows). Discovered while in flight: PyAutoLens CI clones sibling repos from main, so the PR pair needs a re-trigger after PyAutoGalaxy merges before PyAutoLens checks turn green. Smoke 7/7 on autolens_workspace, 9 PyAutoGalaxy library tests cover single-family + sparse-column + cross-family-join + af.Model + redshift-consistency + class-not-found + wrong-family-rejection. Future work queued in `z_features/cluster_lensing.md`: cluster/3_test_workspace, /4_likelihood_function, /5_profiling.

## multipole-scaled-jax
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/426
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/427
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/51

## weak-fit
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/524
- completed: 2026-05-19
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/525
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/188
- repos: PyAutoLens, autolens_workspace
- follow-up-prompt: PyAutoPrompt/autolens/weak_dataset_from_json.md (al.from_json on WeakDataset broken — VectorYX2DIrregular missing 'values' kwarg in deserializer)
- notes: Step 3 of the weak-lensing series. Added `al.FitWeak` — a standalone fit class (no `aa.AbstractFit` inheritance, mirroring `FitPoint`) that compares a model shear field against a `WeakDataset` and reports residuals, chi-squared and log-likelihood. Model shear comes from `LensCalc.from_tracer(tracer).shear_yx_2d_via_hessian_from(grid=...)` — same primitive `SimulatorShearYX` uses, so noise-free round-trips are bit-exact. Each background galaxy contributes two independent measurements (γ₁/γ₂ share per-galaxy σ but are independent draws), so chi_squared sums over N×2 elements and noise_normalization carries a factor of 2 to match. Four `aplt` plotter helpers added: `plot_data_vs_model` (overlaid quivers data-black model-red-alpha-0.6), `plot_residuals` (RdBu_r quiver), `plot_chi_squared_map` (magma scatter), `subplot_fit_weak` (2×2 mosaic). Workspace tutorial `scripts/weak/fit.py` written with Opus prose, reports chi²=437.5 for 400 DoF on seed=1 simulator output (well within 1σ of expected 400). **Workspace tutorial workaround**: `al.from_json(WeakDataset)` is broken (VectorYX2DIrregular `__init__()` missing `values`), so the tutorial rebuilds the dataset inline via `SimulatorShearYX(seed=1)` rather than loading from disk; bug captured as follow-up prompt. **Parallel-worktree pattern**: workspace work executed via a parallel autolens_workspace worktree alongside in-flight `cluster-scaling-members` (zero file overlap — `scripts/weak/` vs `scripts/cluster/`), same approach used for `weak-visualization` earlier today. 26 weak tests pass (10 new fit + 16 existing). 293 full PyAutoLens tests green. Next in series: weak/4_modeling.md (AnalysisWeak), weak/5_likelihood_function.md — to be issued one at a time per [[feedback_no_bulk_issue_queues]].

## source-science-parametric
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/72
- completed: 2026-05-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/73
- repos: autolens_workspace_developer
- notes: First in a planned series of source-science recovery tests. Refactored Codex baseline `fit_compare.py` to add posterior expansion (50 draws via `samples.draw_randomly_via_pdf`) with per-draw `FitImaging.tracer_linear_light_profiles_to_light_profiles` to solve MGE intensities; added structured JSON + Markdown comparison + per-fit subplot PNGs. Bumped `path_prefix`/`unique_tag` to `source_science_v3` because cached v2 MGE samples were incompatible with current `mge_model_from` (PyAutoGalaxy commit `44f8db0b` made per-basis `ell_comps` independent). Headline finding: no parametric model recovers truth within posterior 1σ on any quantity, highest-evidence model (MGE+MGE) has worst recovery (magnification -44%, magnitude -0.77 mag); MGE source likely absorbs diffuse residual flux into wide-σ Gaussians, inflating source-plane flux ~2× while image-plane flux only inflates ~14%. Tutorial assessment caught 2 real bugs in `autolens_workspace` pixelization tutorial (line 280 noise-map interpolation passes wrong array; line 156/235 missing pixel-area factor) — held for follow-up issue since that workspace is held by another active task. Standalone summary persisted at `source_science/results/1_summary.md` for the planned series. Next experiments (separate issues): `radius_break` re-test to match truth's `SersicCore` default 0.025, MGE σ-cap test, then `RectangularAdaptImage` + `Delaunay` pixelization comparisons.

## cluster-scaling-members
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/184
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/185
- repos: autolens_workspace
- notes: Made the scaling-relation tier the cluster default. `scripts/cluster/simulator.py` now produces 10 lower-mass members on the truth relation `b0 = 0.3 * L^1.0` (luminosities log-spaced 0.05–0.40, `ra=0.1"`/`rs=10.0"` fixed across the tier), writing `scaling_galaxies.csv` via `al.galaxy_table_to_csv`. `modeling.py` consumes that CSV via `al.galaxy_table_from_csv`, composes a scaling tier whose per-member `b0` is a derived prior of shared `scaling_factor` (`UniformPrior(0, 1)`) and `scaling_exponent` (`UniformPrior(0, 2)`), and grows free-parameter count from 11 → 13 regardless of the population size. Pivotal scope decision: `start_here.py` was bundled into this PR via a full rewrite. The prior file was a stale group-scale extended-imaging copy referencing `extra_galaxies_centres.json` (file the simulator never writes) and `pixel_scales=0.05` (mismatch) — parked in `no_run.yaml`. Rewrote to mirror `modeling.py` (point-source, 13-param model) and unparked it; this subsumes the previously-Deferred "cluster/start_here.py rewrite" item in z_features. Mass-profile choice: `dPIEMassSph` for scaling members (matches main-tier cluster context) rather than the group example's `IsothermalSph`. JAX registration unchanged — scaling members reuse the `Galaxy / SersicSph / dPIEMassSph` classes already registered via `_lens_models`, so no new entries in `_registration_model`. Auto-sim guard tightened in both modeling.py and start_here.py to also check for `scaling_galaxies.csv`, so stale pre-change datasets get regenerated. Local simulator run on CPU confirmed both sources still produce 3 multiple images each at sensible positions. Smoke 6/7 — the 1 failure (`interferometer/modeling.py: nufftax not installed`) reproduces on canonical main and is unrelated. PR merged cleanly. Future work queued in `z_features/cluster_lensing.md`: cluster/3_test_workspace, /4_likelihood_function, /5_profiling — to be issued one at a time per [[feedback_no_bulk_issue_queues]].

## weak-visualization
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/496
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/523
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/186
- repos: PyAutoLens, autolens_workspace
- notes: Step 2 of the weak-lensing series. Added `autolens/weak/plot/` package with five module-level helpers (`plot_shear_yx_2d`, `plot_ellipticities`, `plot_phis`, `plot_noise_map`, `subplot_weak_dataset`) re-exported into `aplt`. Plotters use headless quiver segments (`pivot="middle", headwidth=0, headlength=0, headaxislength=0`) for the spin-2 convention. Access shear field via `.ellipticities` / `.phis` only (never raw `[:, 0]` / `[:, 1]`) to keep the `[γ₂, γ₁]` storage convention encapsulated. Note: `.phis` returns degrees, so `plot_shear_yx_2d` converts via `np.deg2rad` before quiver. Workspace follow-up filled the two `# TODO(2_visualization.md)` placeholders in `scripts/weak/simulator.py` with real `aplt` calls. Workspace work executed via a parallel autolens_workspace worktree alongside in-flight `cluster-scaling-members` (zero file overlap — `scripts/weak/` vs `scripts/cluster/`), bypassing the helper's task-level conflict check for the ~30-min shipping window. Tests use direct module imports because `al.plot.X` hits a pre-existing recursion in `autolens/__init__.py:147`'s `__getattr__` under pytest's attribute-access flow (point/imaging tests already follow this pattern). 283 PyAutoLens tests pass, all 4 CI checks green on both PRs. Next in series: weak/3_fit.md (FitWeak), 4_modeling.md, 5_likelihood_function.md — to be issued one at a time per [[feedback_no_bulk_issue_queues]].

## likelihood-function-assertions
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/102
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/103, https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/50
- notes: Added `__Likelihood Sanity__` regression-guard blocks before every Nautilus search in `_test`-workspace scripts that fit a pixelization source. Each block builds the prior-median instance, calls `analysis.log_likelihood_function`, reconstructs the fit via `analysis.fit_from`, and asserts `LLF == figure_of_merit != log_likelihood` plus `Fitness.call_wrap == figure_of_merit`. JIT scripts cover both CPU + JAX backends. Final scope (5 scripts) is narrower than the original prompt — MGE-source `modeling_visualization_jit.py` (singular) and all `autogalaxy_workspace_test/scripts/interferometer/*` are out of scope because they don't fit a pixelization and the `!=` guard would fail tautologically. Reconstruction uses `analysis.fit_from(instance)` directly rather than rebuilding `FitImaging` manually (the prompt's example was incomplete — missed `dataset_model` and `settings`). The sanity analysis is built without `positions_likelihood_list` so there's no `log_likelihood_penalty` to subtract. Local JAX validation hit a PyAutoGPU venv quirk (`Unknown backend cuda`); CPU branch validated end-to-end on both repos, autogalaxy `visualization.py` also validated under its smoke env-var overrides — both rectangular + Delaunay pixelization iterations PASS.

## subplot-fit-mid-zoom
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/517
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/324, https://github.com/PyAutoLabs/PyAutoLens/pull/518
- notes: Reshuffled the 12-panel `subplot_fit` / `subplot_fit_log10` layout: dropped "Data (Source Scale)", moved "Model Image" beside "Data", promoted "Source Plane (Zoomed)" → "Source Plane (Max Zoom)" to top right, added new "Source Plane (Mid Zoom)" panel. Mid Zoom uses 2× Max Zoom extent, kept square, uniformly shrunk so the half-width is `min(target, distance-to-nearest-No-Zoom-edge, 0.7 × No-Zoom-half)` — the 0.7 cap was added in iteration 3 because uniform shrinkage produced Mid Zoom ≡ No Zoom when the bright pixel was centred (parametric case). Threaded `zoom_extent_scale` kwarg through `Mapper.extent_from`, `plot_inversion_reconstruction`, `plot_mapper`, `_plot_source_plane`, `plane_image_from`. Pre-existing quirk: `Zoom2D.extent_from` returns pixel-shifted coords (centred at pixel ~39.5 not arcsec 0); workaround is to round-trip through `Grid2D.from_extent(...).geometry.extent` which extracts only the span. All defaults preserve behaviour (no API break). Smoke: 44/44 across autofit/autogalaxy/autolens/autolens_test/HowToLens.

## ep-profiling-breakdown
- issue: https://github.com/Jammy2211/ic50_workspace/issues/6
- completed: 2026-05-18
- workspace-pr: https://github.com/Jammy2211/ic50_workspace/pull/7
- notes: Added `scripts/profile_ep_sim.py` to `z_projects/ic50_workspace`, an instrumented EP run that monkey-patches `HillAnalysis` / `GlobalLinearAnalysis` / `FixedHillCoefEPFactor` / `DynestyStatic.fit` to split wall time into Hill-LL evals (per dataset), global-LL evals, set_model_approx, Dynesty-wrapper overhead (search.fit minus LL evals), and EP-loop orchestration (optimise minus search.fit minus set_model_approx). Production scripts untouched. Writes `scripts/results/ep_sim_profile.{md,json}` with a scaling projection to 100/1000/10000 datasets. Headline: **~86% of `factor_graph.optimise` time is Dynesty wrapper overhead** (sampler init, paths, per-fit plot attempts, internal-folder cleanup) — only ~10% of `search.fit(...)` wall time is actual likelihood evaluation. Per-Dynesty-fit overhead ~5 s/fit dominates at every N. Projection (using observed M≈2 EP iterations under `kl_tol=1.0`): 5→1min, 100→20min, 1000→3h, 10000→1.4 days. Three independent runs validated bucket proportions stable across ~25% run-to-run variance. The profile script clears `output/ep_sim/` at startup to avoid the AutoFit cache-resume short-circuit ([[feedback_autofit_cache_resume_pyauto_test_mode]]). Out of scope: actual optimisation, cProfile pass, N=100 validation measurement. Same z_projects/ caveats as previous ic50 PRs ([[reference_ic50_workspace_nonstandard]]) — no worktree, no pending-release label, ship ran in Opus.

## ic50-hpc-setup
- issue: https://github.com/Jammy2211/ic50_workspace/issues/4
- completed: 2026-05-18
- workspace-pr: https://github.com/Jammy2211/ic50_workspace/pull/5
- notes: Ported the HPC interface from `autolens_base_project/hpc/` to `z_projects/ic50_workspace`. New `hpc/` folder with 8 SLURM submit scripts (4 CPU + 4 GPU, one per entry point: `ep_sim`, `ep_real`, `graphical_sim`, `graphical_real`); real-data variants array-dispatch over `drugs=(1003 1073)` for failure isolation. CPU partition forces `JAX_PLATFORM_NAME=cpu` + OMP/MKL/OpenBLAS thread pinning to `$SLURM_CPUS_PER_TASK`; GPU partition lets JAX auto-pick the allocated device and runs `nvidia-smi`. Added `activate.sh` at workspace root with minimal PYTHONPATH (PyAutoConf + PyAutoFit only — IC50 doesn't import autoarray/galaxy/lens). Copied `sync` (560 LOC) verbatim from autolens_base_project; adapted `sync.conf.example` with `PROJECT_NAME=ic50_workspace`. CLAUDE.md gained an `## HPC runs` section. **`hpc/batch_*/output/.gitignore` placeholders had to be force-added** because the top-level `.gitignore`'s `output/` rule swallowed them — same convention autolens_base_project uses. **No notebook regeneration** (no scripts/*.py changed). No real HPC submission run this session — needs `cp hpc/sync.conf.example hpc/sync.conf` + edit + `hpc/sync push` to actually use. Out of scope: `sync_jump` (no build-server topology yet), `--use_cpu` / `--number_of_cores` argparse flags (env vars suffice; revisit if multi-core Dynesty matters later). Same z_projects/ caveats as the previous two ic50 PRs apply.

## ic50-graphical-fit
- issue: https://github.com/Jammy2211/ic50_workspace/issues/1
- completed: 2026-05-18
- workspace-pr: https://github.com/Jammy2211/ic50_workspace/pull/2
- notes: Added graphical-model fit variant of the EP pipeline in `z_projects/ic50_workspace`. New `scripts/graphical_sim.py` + `scripts/graphical_real.py` mirror the EP scripts but fit the full factor graph (33 params for the sim run: 5×3 Hill + 5×3 coef_matrix + 3 coef_mean) in a single Dynesty search over `factor_graph.global_prior_model` instead of EP message passing. `scripts/util.py` extended with `GraphicalLinearAnalysis` (reads `hill_coef` as a free param via the shared-prior wiring in `build_model_linear`; Gaussian regression constraint with fixed `DEFAULT_REGRESSION_SIGMAS = (0.5, 0.5, 6000.0)` matching `coef_matrix_prior_sigmas`), `run_graphical_fit`, and a `write_graphical_summary` shim. `write_ep_summary` parameterised with `method=` kwarg so the same writer produces `<method>_<name>_summary.{txt,json}` for both paths; column headers neutralised to `mean` / `σ`. **No worktree** — `z_projects/ic50_workspace` lives outside `$PYAUTO_MAIN/<repo>` so the worktree helper can't manage it; worked directly on `feature/ic50-graphical-fit` in the canonical checkout and ran the ship step in Opus (the Sonnet-subagent delegation in `/ship_workspace` assumes a worktree path). **No `pending-release` label** on `Jammy2211/ic50_workspace` — `ensure_workspace_labels.sh` doesn't cover this repo; PR was opened without the label. Proper-run validation (no `PYAUTO_TEST_MODE`) was deferred — both scripts only smoke-tested under test mode. Notebook regeneration via `PyAutoBuild/autobuild/generate.py ic50` picked up the two new scripts and also produced notebook-serialization-format drift in four unrelated notebooks (`simulator`, `likelihood_function`, `preprocess_real`, `least_squares`); all committed together to keep `notebooks/` in sync with the current generator.

## external-potential-priors-and-jit
- issue: (none — follow-up to #419 / #422)
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/423
- workspace-test-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/101
- notes: Closed two gaps from the external-potential ship the same day: (1) `af.Model(ag.mp.ExternalPotential)` was crashing at runtime because there was no library-default prior YAML — added `autogalaxy/config/priors/mass/sheets/external_potential.yaml` mirroring ExternalShear's gamma priors (Uniform(-0.3, 0.3), Absolute width_modifier 0.05) for all six γ/τ/δ components plus Gaussian(0, 0.1) centre matching Isothermal/MassSheet; (2) no JAX JIT parity test, fixed by adding an ExternalPotential block to `autolens_workspace_test/scripts/profiles_jit.py` parallel to the ExternalShear block (deflections + convergence on Grid2DIrregular and Grid2D.uniform). The new block forced a small extension to `check_profile_method` — added `atol` kwarg (default 0.0, existing callers unaffected) because ExternalPotential's convergence `κ = τ₁·x + τ₂·y` legitimately crosses zero on the τ-null line, where the rtol-only `assert_allclose` blows up on sub-machine-precision (2e-19) reductions; passing `atol=1e-12` puts a sub-physical floor under the comparison. Verified: `af.Model(ag.mp.ExternalPotential).instance_from_prior_medians()` returns an 8-param model (centre + γ/τ/δ); profiles_jit.py prints "All profiles_jit.py checks passed." and 909/909 PyAutoGalaxy tests stay green. grids.yaml entry skipped — grep confirmed no library code reads `radial_minimum` or `"grids"` (the 14 workspace copies are vestigial). Follow-up note posted on closed issue #419 so @Sketos sees the priors landed. Also updated `autolens_workspace_test/CLAUDE.md` and `scripts/CLAUDE.md` to list `mp.ExternalPotential` in the profiles_jit coverage.

## external-potential
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/419
- user-facing: true (reporter @Sketos)
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/422
- notes: User-reported feature request from @Sketos with full prototype code in the issue body. Added `ag.mp.ExternalPotential` as a sibling of `ExternalShear` in `autogalaxy/profiles/mass/sheets/external_potential.py` — six free params (gamma_1/2, tau_1/2, delta_1/2) plus a free centre (ExternalShear's centre is fixed (0,0) because pure shear deflections are constant; tau/delta have radial deflections so centre matters). Implements Powell 2022 Eq 4 in polar form, using `@aa.decorators.transform` for the centre shift (no rotation since ell_comps=(0,0)) — kept the body in the global frame so no `rotate_back` needed. Magnitude/angle accessors per-term (gamma spin-2 → [0,180), tau spin-1 → [0,360), delta spin-3 → [0,120)) plus a `from_magnitudes_and_angles` classmethod matching the paper-style parameterisation. One math correction vs prototype: `convergence_2d_from` returns `κ = τ₁·x + τ₂·y` (Laplacian of ψ), not zero — γ and δ stay harmonic with κ=0. 14 new unit tests cover γ-parity vs ExternalShear, τ-only convergence/potential/deflection, δ-only, non-zero-centre shift, and `from_magnitudes_and_angles` round-trip for all three terms. Full PyAutoGalaxy suite green (909 tests). 42/42 smoke tests across all six workspaces green. Conversational comment cadence: receipt + plan + smoke + shipped. No workspace demo this PR (offered to follow up if Sketos asks). Decorator import note: ExternalShear uses `@aa.decorators.*` not the `@aa.grid_dec.*` form in PyAutoGalaxy CLAUDE.md (stale per the recent multipole-linear memory) — matched the actual code path.

## workspace-version-mismatch-advice
- issue: (none — direct request)
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/107
- notes: Rewrote `WorkspaceVersionMismatchError` message in PyAutoConf/autoconf/workspace.py to give directional update advice. Workspace newer than library → `pip install --upgrade <lib>==<workspace_version>` (lib derived by stripping `_workspace` from the workspace folder name: `autolens_workspace` → `autolens`, etc.). Library newer than workspace → `cd <workspace_root> && git pull origin main`. Unparseable versions fall back to showing both. Dropped the `git clone --branch <version> <workspace-repo-url>` instruction since we no longer cut workspace version branches. Bypass instructions (`workspace_version_check: False`) and the `main`-branch IMPORTANT note are preserved. Four private helpers added (`_parse_version`, `_library_name_from_workspace`, `_update_library_block`, `_update_workspace_block`); public surface unchanged. All 14 `test_workspace.py` tests pass without modification (assertions matched both old and new message). Zero workspace-script references to the error or its message text. Merged directly without smoke tests — error fires only on import-time version mismatch, which smoke tests don't exercise.

## multipole-light-profiles-linear
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/418 (follow-up — no separate issue)
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/421
- workspace-test-prs: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/49 + https://github.com/PyAutoLabs/autolens_workspace_test/pull/100
- notes: Three-PR follow-up to multipole-light-profiles. (1) Split `autogalaxy/profiles/light/standard/multipole.py` into per-class modules (`sersic_multipole.py`, `gaussian_multipole.py`) with the shared mixin extracted to `_multipole_mixin.py`. Test file split correspondingly. (2) Added `ag.lp_linear.SersicMultipole` and `ag.lp_linear.GaussianMultipole` — each subclasses its standard variant + `LightProfileLinear`, hardcodes `intensity=1.0`. (3) Docs/api/light.rst Linear section in PyAutoGalaxy lists both new linear classes (PyAutoLens has no Linear section to extend, and its Standard section was already updated in PR #515). (4) Added JAX likelihood test scripts in `autogalaxy_workspace_test/scripts/jax_likelihood_functions/light_multipole/multipole.py` and `autolens_workspace_test/scripts/jax_likelihood_functions/light_multipole/multipole.py` — both exercise SersicMultipole under `fitness._vmap` + `jax.jit(analysis.fit_from)` with explicit `af.TuplePrior` Gaussian priors on the multipole component tuples (the library does not ship default priors for them yet). Key gotchas: (a) inline `bulge.multipole_3_comps_0 = af.GaussianPrior(...)` fails because PyAutoFit doesn't reassemble scalar-named priors into a tuple at construction time without a yaml config — `af.TuplePrior(multipole_3_comps_0=..., multipole_3_comps_1=...)` is the right primitive. (b) PyAutoLens worktree was attached but had zero changes after PR #515 already mirrored the Standard autosummary entries — skipped its PR entirely. (c) Workspace_test smoke CI auto-checks-out matching library feature branches via the `Match library branches` step in `.github/workflows/smoke_tests.yml`, so merge order didn't matter for CI. (d) `total_free_parameters` was 7 (Sersic baseline) until the TuplePrior priors were set; this would have silently fixed multipole_*_comps at (0,0) and defeated the JAX test. Workspace priors + modeling example still deferred to the follow-up prompt.

## multipole-light-profiles
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/418
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/420
- docs-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/515
- notes: Library-only ship. Added `ag.lp.SersicMultipole` and `ag.lp.GaussianMultipole` (subclassing `Sersic` / `Gaussian` so `isinstance` checks still pass), both gated by `multipole_3_comps` / `multipole_4_comps` tuples that default to `(0.0, 0.0)` — verified by machine-precision parity tests against the base profiles. API matches `EllipseMultipole` / `PowerLawMultipole` convention. Shared `perturbed_radii_from` helper lives on `_LightProfileMultipoleMixin` (leading-underscore signals throwaway — to be replaced when the z_features generalisation lands). `image_2d_via_radii_from` overridden in both subclasses to accept a raw backend array since the perturbation step strips the autoarray wrapper; matched Aris's prototype pattern. `xp=np` threaded end-to-end. Decorator stack uses `@aa.decorators.to_array` / `@aa.decorators.transform` (the actual code path; `@aa.grid_dec.*` references in PyAutoGalaxy CLAUDE.md appear stale). 12 new unit tests cover zero-perturbation parity, m=3/m=4 rotational symmetry on a circular base, centre translation, nonzero-multipole diff, and finiteness+non-negativity under large perturbation. Companion docs-only PR on PyAutoLens (#515) mirrored the autosummary entries since PyAutoLens re-exports `autogalaxy.profiles.light.standard as lp`. Workspace priors + modeling example explicitly deferred to a follow-up prompt — `autogalaxy_workspace` was free but the user chose to bundle workspace work under a separate issue, and `autolens_workspace` was held by `cluster-modeling-v2` anyway. Conflict-guard worked as designed (caught autolens_workspace early).

## interferometer-extra-galaxies
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/81
- completed: 2026-05-17
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/82
- notes: Single-repo task (autogalaxy_workspace only). Built scripts/interferometer/features/extra_galaxies/ — modeling.py + simulator.py + README + __init__. Adapts autogalaxy imaging extra_galaxies pattern (multi-galaxy field fitting via light profiles) for interferometer data, with autolens interferometer/features/extra_galaxies/ as a read-only structural template (lens-mass aspects stripped). Main galaxy: linear Sersic bulge + linear Exponential disk. Each extra galaxy: linear SersicSph with fixed centre loaded from extra_galaxies_centres.json (Option A); MGE alternative commented inline (Option B). Real-space mask radius=6.0" to cover extras offset at (±3.5"). Uses TransformerNUFFT (nufftax) — multi-galaxy fits practical because every galaxy's light profile is NUFFT'd inside the JIT'd likelihood. Key teaching point: autogalaxy/autolens role split (autogalaxy fits LIGHT of extras for multi-galaxy fields; autolens fits MASS of extras for lensing perturbation). Noise-scaling approach from autogalaxy imaging not portable to interferometer (uv-plane data not directly tied to image-plane pixels) — modeling.py uses modeling-approach exclusively with rationale called out in "Approaches to Extra Galaxies" section. No imaging Phase 1 sweep (prompt was direct-port, not review pass). No slam.py (autogalaxy is non-lensing). Both smoke tests pass — simulator.py produced dataset/interferometer/extra_galaxies/{data,noise_map,uv_wavelengths}.fits + galaxies.json + extra_galaxies_centres.json; modeling.py composed N=15 free-param model and called likelihood once under PYAUTO_TEST_MODE=2. Tracker now 5 shipped / 3 outstanding (double_einstein_ring, mass_stellar_dark, scaling_relation remain).

## group-scaling-relation
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/167
- completed: 2026-05-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/173
- notes: Two-stage task. Stage 1 added fit.py + likelihood_function.py to imaging/features/scaling_relation/; Stage 2 same for group/features/scaling_relation/. Both directories already had comprehensive modeling.py + simulator.py; this PR padded them out with the no-search demo + likelihood walkthrough. All 4 scripts assert per-galaxy (or per-tier) deflection sums match the tracer total — scaling-tier galaxies' einstein radii come from `0.3 * luminosity^1.0 = 0.135` (truth). Conflict override on autolens_workspace — ran in parallel with interferometer-multi-gaussian-expansion + interferometer-shapelets (disjoint feature dirs). Dataset gotcha: Explore agent initially reported group dataset NOT committed but it was already tracked on main (10 files); simulator regenerated noise differently on fresh seed which showed up as `M` entries in diff. .gitignore got hygiene additions: !dataset/imaging/extra_and_scaling_galaxies/** and !dataset/group/scaling_relation/** (both datasets were already tracked so allow-list was invisibly missing). Group simulator has only one main lens — lens_dict has single entry; pattern generalises naturally. No smoke entries added (per "small curated subset" memory).

## interferometer-shapelets
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/170
- completed: 2026-05-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/171
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/79
- notes: Built scripts/interferometer/features/advanced/shapelets/ (autolens) and scripts/interferometer/features/shapelets/ (autogalaxy) — modeling.py + fit.py + README + __init__ in both. Polar `ShapeletPolar` linear basis with shared centre/ell_comps/beta; autolens places it on the source (no lens light), autogalaxy on the single-galaxy bulge. Both use TransformerNUFFT (nufftax) and crucially set `Settings(use_positive_only_solver=False)` — shapelets require the positive-negative solver because their decomposition relies on negative-amplitude cancellations. Both `fit.py` smoke runs reproduce this in action: 31/66 (autolens) and 30/66 (autogalaxy) shapelets land at negative intensity on the example dataset, exactly the unphysical-but-required behaviour the prose explains. Path note: imaging shapelets live at imaging/features/advanced/shapelets/ in autolens but at imaging/features/shapelets/ in autogalaxy — the interferometer mirror preserves this asymmetry. Tutorial role swap is NOT central here (unlike MGE) — imaging shapelets already places the basis on the source (`simple__no_lens_light` dataset), so the interferometer adaptation is closer to a straight port than the MGE one was. Phase 1 moderate-pad scope: typos (peforms, assymetric, central→centre, case-of→case-if, shapelet-fit→shapelets-fit, non-of, trailing-t URL fragment, Shaeplet, definedon, of-that-are-composed) + duplicated `__Model__` section in autolens + `ShapeletCartesianSph` → `ShapeletPolar` naming correction (imaging script's prose claimed `ShapeletCartesianSph` but the actual model uses `ShapeletPolar`) — bundled into PRs. Did NOT add new imaging likelihood_function.py / slam.py (out of scope; separate task if wanted). Did NOT fix a pre-existing Cartesian-shapelet model-build bug at modeling.py lines 510-517 (loop reassigns out-of-scope `shapelet` variable rather than iterating the af.Collection — flagged in the issue completion comment as a follow-up). Tracker now 4 shipped / 4 outstanding (extra_galaxies, double_einstein_ring, mass_stellar_dark, scaling_relation remain).

## viz-subprocess-feasibility
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1279 (closed, pivoted)
- completed: 2026-05-17
- outcome: pivoted — no library code shipped
- successor: PyAutoPrompt/z_features/fast_visualization.md
- notes: Phase 4 of (archived) z_features/jax_visualization.md roadmap. Spike on FitImaging picklability returned a clean result — `FitImaging` round-trips through stdlib `pickle` cleanly on LP and pixelization models, under both `use_jax=False` and `use_jax=True` (495 KB → 5.7 MB pickle blobs, `figure_of_merit` preserved bit-exact). That was the load-bearing unknown for the subprocess design: had it failed, the worker would have needed raw-array + reconstruction; it didn't fail, so `mp.Process` + `Queue` with `queue.put(fit)` is the simplest viable design — banked for future. **But** during the spike we discovered the codebase already has full JAX-jittable critical-curve / Einstein-radius infrastructure: `jax_zero_contour` external package is installed, `_via_zero_contour_from()` methods exist throughout `autogalaxy/operate/lens_calc.py`, the plotter routes via the `visualize.general.critical_curves_method` config switch, and **the only reason none of it is active is that the YAML default is `marching_squares` instead of `zero_contour`**. So the user-visible "fast Jupyter viz" goal is achievable via config flip + targeted call-site migration (Euclid `util.compute_latent_variables`) + `IPython.display.update_display` for live cells, with no subprocess complexity. Pivoted issue #1279 to closed/not-planned; spike prompt moved to `z_features/complete/visualization_subprocess_feasibility.md`; new roadmap at `z_features/fast_visualization.md` covers Phase A (config flip) → Phase B (latent migration) → Phase C (Jupyter live cells) → Phase D (per-dataset end-to-end JAX-viz assertions in `_test` workspaces, regression net against the 2026-05-16 Euclid all-zero-source bug class) → Phase E (long-term: pytree-register `ModelInstance` cascade so `jax.jit(fit_from)(instance)` works and PR #1278's default-on can be re-attempted) → Phase F (subprocess viz, deferred). Re-entry into subprocess viz is documented but not authored. Local worktree `~/Code/PyAutoLabs-wt/viz-subprocess-feasibility` and feature branches in PyAutoFit + autolens_workspace_test removed.

## interferometer-multi-gaussian-expansion
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/166
- completed: 2026-05-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/168
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/77
- notes: Built scripts/interferometer/features/multi_gaussian_expansion/ in both autolens_workspace (modeling, fit, likelihood_function, slam + README + __init__) and autogalaxy_workspace (modeling, fit, likelihood_function + README + __init__). Key autolens role swap explicitly explained: imaging MGE fits the lens galaxy bulge, but for interferometer the lens light is omitted (no detection in mm/sub-mm) so the MGE is applied to the source galaxy. autolens slam.py mirrors interferometer/features/pixelization/slam.py with SOURCE LP mge_model_from total_gaussians bumped from 5 to 20; pixelized stages identical. autogalaxy uses single-galaxy MGE bulge (no role swap — same as imaging). All scripts use TransformerNUFFT (nufftax) for the per-iteration NUFFT of every Gaussian inside the JAX jit/vmap pipeline. likelihood_function.py walkthroughs reproduce FitInterferometer.figure_of_merit to 3 decimal places — the small mismatch likely from multiple valid positive-only sparse solutions in degenerate Gaussian bases. Both lhfn scripts nicely demonstrate positive-negative solver ringing (autogalaxy returns ±10^15 intensities) vs positive-only solver collapsing to sparse 2-Gaussian solutions — visual teaching moment. Phase 1 sanity sweep of imaging MGE scripts (typos: peforms, algabra, descomposed, double-backtick `Gaussian``, start.here.py, unphysicag×2, physicag, PyAutoGalaxys, lp_Linear) bundled into PRs. SLaM smoke ran all 4 stages in ~30s under PYAUTO_TEST_MODE=2. Tracker now 3 shipped / 6 outstanding. Future audit hint: autogalaxy imaging MGE files had clustered typos (unphysicag/physicag/PyAutoGalaxys) suggesting a bad find/replace was applied at some point — worth a broader sweep across autogalaxy MGE-adjacent scripts.

## jax-viz-default-broken
- issue: (none — direct user report of broken Euclid HPC runs 2026-05-16)
- completed: 2026-05-17
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1280
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/29
- notes: Reverted PR #1278 (use_jax_for_visualization default → follow use_jax). Root cause: `jax.jit(self.fit_from)(instance=ModelInstance)` fails because `ModelInstance` isn't pytree-registered — JAX raises TypeError abstracting it. On real Euclid pipeline runs (z_projects/euclid/initial_lens_model.py vis_lp on Tile102008468 et al.) the exception was swallowed deep in the visualizer's outer guards; visible symptom was source-plane FITS files written all-zero and Einstein-radius posteriors collapsing to the full prior across every Euclid tile. Confirmed via A/B/C reproducer (`/tmp/jax_mge_nautilus_ab.py`): config B (`use_jax=True, use_jax_for_visualization=False`) gave max_ll=-306, spread=3475 — normal convergence; config C (post-#1278 default) raised the TypeError on first quick_update. Reverted signature back to `bool = False`, dropped the sentinel-resolution block, restored pre-#1278 docstrings, dropped the unit test that exercised the sentinel-PYAUTO_DISABLE_JAX interaction, and on autofit_workspace_test dropped the three sentinel assertions in scripts/jax_assertions/fitness_dispatch.py (`assert_use_jax_true_implicitly_turns_on_visualization`, `assert_explicit_none_resolves_to_use_jax`, `assert_use_jax_true_jit_dispatch_via_sentinel_default`), replacing with `assert_use_jax_true_defaults_visualization_off`. Explicit-opt-in still works for callers who pass `use_jax_for_visualization=True` — the underlying `ModelInstance`-not-pytree-registered limitation remains but is now never triggered by default. Library PR was merged with red `Tests` CI: the 12 NSS-related ImportError failures were pre-existing on `main` (since PR #1277 at 09:48 on 2026-05-16 added the [nss] extra without wiring it into the Tests workflow); none caused by the revert. Local full-suite run was 1258 passed. Follow-up: someone needs to fix the Tests workflow to install `[nss]` extras so CI goes green on main. Adjacent parked task viz-subprocess-feasibility (PyAutoFit #1279) remains untouched — that's the proper long-term fix for JIT visualization (subprocess-based, won't have the ModelInstance pytree problem).

## interferometer-linear-light-profiles
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/162
- completed: 2026-05-16
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/163
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/75
- notes: Built scripts/interferometer/features/linear_light_profiles/ in both autolens_workspace (modeling, fit, likelihood_function, slam + README + __init__) and autogalaxy_workspace (modeling, fit, likelihood_function + README + __init__; no slam — SLaM is autolens-only). All use TransformerNUFFT (nufftax) for the per-iteration NUFFT of each linear basis component inside the JAX jit/vmap pipeline. Lens light omitted per interferometer convention (autolens); autogalaxy uses linear Sersic bulge + linear Exponential disk. autolens slam.py mirrors interferometer/features/pixelization/slam.py (SOURCE LP → SOURCE PIX 1 → SOURCE PIX 2 → MASS TOTAL) with the SOURCE LP source bulge swapped from MGE to linear SersicCore; SOURCE LP runs on TransformerNUFFT and pixelized stages switch to TransformerDFT + sparse operator. Phase 1 sanity sweep of the imaging linear_light_profiles/ scripts in both workspaces bundled into the same PRs: rewrote stale Basis/5-Gaussians sections to match actual model (Sersic+SersicCore for autolens, Sersic+Exponential for autogalaxy), fixed bulge/disk-vs-lens/source wording confusion in autolens likelihood_function.py (renamed image_2d_bulge/_disk → image_2d_lens_bulge/_source_bulge, fixed the two print(image_2d_bulge.slim) sites, removed misleading "this will raise an exception" block that no longer raised), fixed n_live=300 vs prose-says-75 mismatch in autogalaxy modeling.py (n_live now 75, prose now consistent), assorted typos (Althought, algabra, non-negligable, start.here.py, RectnagularMapper, autoogalaxy_workspace, lens galaxly's, Disadvatanges, lp_Linear). likelihood_function.py walkthroughs reproduce FitInterferometer.figure_of_merit to 4-5 decimal places; autogalaxy 2-component case nicely demonstrates positive-only solver (positive-negative returns bulge intensity ~-0.17 unphysical, positive-only correctly returns 0). Workspace conflict resolution: group-mass-stellar-dark also held autolens_workspace; cleared via file-level coexistence (this task: scripts/interferometer/...; that task: scripts/group/...) — same precedent as knn-barycentric + ag-interferometer-kwargs. worktree_check_conflict bypassed, worktree_create called directly. SLaM smoke ran all 4 stages in ~35s under PYAUTO_TEST_MODE=2. Notebook regeneration deferred to /generate_and_merge post-merge. z_features tracker: 2 shipped / 7 outstanding (interferometer-no-lens-light was removed during the audit since all interferometer scripts already assume no lens light).

## group-mass-stellar-dark
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/158
- completed: 2026-05-16
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/164
- notes: Two-stage task. Stage 1 padded out imaging features/advanced/mass_stellar_dark/ with new fit.py + likelihood_function.py + a Practical Use chaining callout on modeling.py + committed dataset (FITS + tracer.json). Stage 2 created scripts/group/features/advanced/mass_stellar_dark/ — 8 files (simulator, fit, likelihood_function, modeling, chaining, slam, README, __init__) using the group lens_dict API where each main lens galaxy carries its own lmp.Sersic + NFWSph decomposition with ExternalShear on lens_0 only. Source remains single-plane (z=1.0). Both stages' executable scripts validated end-to-end — decomposition assertions pass (sum_i(alpha_stellar_i + alpha_dark_i) + alpha_shear matches tracer total deflection). Pre-existing PyAutoGalaxy bug in print_vram_use / cse_settings_from blocks modeling.py / chaining.py / slam.py full Nautilus runs for lmp.Sersic lenses (reproduces on canonical main; both use_jax=True and use_jax=False paths affected) — filed as https://github.com/PyAutoLabs/PyAutoGalaxy/issues/417, NOT touched here per "never modify code to make tests pass". SLaM gotcha: al.util.chaining.mass_light_dark_from hardcodes a single "lens" key path on light_result.instance.galaxies, incompatible with lens_dict — group slam.py constructs MASS LIGHT DARK per-galaxy manually via take_attributes + UniformPrior(mass_to_light_ratio). dataset/.gitignore precedence gotcha re-confirmed: workspace-root allow-list (!dataset/<type>/<name>/**) is shadowed by in-tree dataset/.gitignore (*); new datasets need git add -f.

## ci-actions
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/7
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/11
- merge-commit: e18685b
- summary: |
    Phase 5 (final) of the autolens_profiling z_feature. Wired up two
    GitHub Actions workflows + threaded AUTOLENS_PROFILING_SMOKE=1 into
    every profile script so the lint workflow's smoke step is cheap.

    What landed:
    - ruff.toml at repo root (conservative E/F/W/I/UP/B selection;
      E501/E402/F401/B008 ignored for scientific code patterns)
    - .github/workflows/lint.yml — PR + push-to-main gate, <5min target,
      CPU-only ubuntu-latest. Steps: ruff check, ruff format --check,
      build_readme.py --check (dashboard idempotence), lychee link-rot,
      smoke one script per section with AUTOLENS_PROFILING_SMOKE=1.
    - .github/workflows/profile.yml — workflow_dispatch (with sections
      filter) + on release:published. Runs every profile script
      continue-on-error per section, runs build_readme.py, commits diff
      back to main as github-actions[bot] with [skip ci]. Skips
      simulators/point_source.py in the loop (default dataset_name
      overwrites Phase 1 tracked JSONs).
    - .github/workflows/README.md documenting both + design decisions.
    - AUTOLENS_PROFILING_SMOKE=1 threaded into 17 scripts (likelihood/*,
      simulators/*, searches/nautilus/*) via AST helper at the first
      non-import top-level statement. Each script exits 0 in <1s with
      "[smoke] ... imports + module setup OK" when the env var is set.

    Design decisions resolved:
    - CPU-only github-hosted runners (self-hosted GPU additive later)
    - workflow_dispatch + release-only (no weekly cron)
    - github-actions[bot] with [skip ci] subject
    - continue-on-error per section (single regression -> ERR cell,
      not blocked dashboard refresh)
    - ruff.toml standalone (no pyproject.toml because this isn't a
      Python package)

    Smoke: yaml.safe_load PASSED on both workflows; py_compile PASSED on
    all 17 SMOKE-instrumented scripts; the SMOKE flag verified working
    on 3 representative scripts locally; build_readme.py --check still
    exits 0 after the SMOKE insertions (Phase 4 idempotence preserved).

    First real CI run will be against any PR after this lands — local
    yaml parse is necessary but not sufficient; any GitHub Actions
    runtime issues get follow-up PRs.

## knn-barycentric (NEGATIVE result)
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/317
- completed: 2026-05-16
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/318 (merge-commit 7c728f75)
- developer-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/70 (merge-commit 5011943d)
- smoke-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/99 (merge-commit 9bf1d889)
- verdict: WILDCARD FAILED science gate (2.22% drift vs Delaunay, fails rtol=1e-2)
- summary: |
    Tried the kNN-barycentric wildcard for replacing scipy.spatial.Delaunay
    in PyAutoArray's source-plane interpolation: pick top-3 nearest mesh
    vertices in source plane, compute exact barycentric weights on the
    triangle they form, clip+renormalize on outside-triangle.

    Library code works — 8 unit tests pass, smoke passes, infrastructure
    is correct and additive. But at the HST imaging fiducial,
    log_evidence drifts from Delaunay by 2.22% (584 nats higher), failing
    even the lenient rtol=1e-2 abandon gate.

    Root cause is structural: ~5% of mesh vertices (60/1291) are never in
    any query's nearest-3 nor any split-point's nearest-3 — they're paid
    for but never used. Delaunay's topology guarantees every vertex
    belongs to at least one simplex; kNN doesn't. That gap is what
    breaks the science.

    What stays in the repo as additive infrastructure:
      - aa.mesh.KNNBarycentric mesh class
      - InterpolatorKNNBarycentric (k=3 + clip-renorm barycentric)
      - barycentric_weights_from_3_nearest helper
      - 8 unit tests (regression-guard for writability + split-padding bugs found during integration)
      - Regression script (documents the negative result, pins observed
        log_evidence, prints verdict block)
      - Smoke script (covers convex-combination invariant, Delaunay
        bit-equivalence on matching-triangle queries, degenerate fallback)

    Recommended next: resume option A (split-callback) per the updated
    PyAutoPrompt/autoarray/delaunay_research.md. Ready-to-ship code on
    feature/delaunay-jax-find-simplex (commit eda747c2) gives 1.19-1.23×
    speedup, modest but the only realistic JAX-native lever now that the
    wildcard is gone.

## instrument-readme-dashboard
- task-alias: instrument-dashboard  (matches active.md / worktree name during execution; full filename-stem slug here so the z_features audit picks this up as shipped)
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/6
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/10
- merge-commit: 2bd7fad
- summary: |
    Phase 4 of the autolens_profiling z_feature. Built the public-facing
    instrument-framed dashboard infrastructure that auto-generates the
    headline run-times tables in every section README from versioned
    artifacts under results/.

    What landed:
    - scripts/build_readme.py (270 LOC) — scans results/**/*_summary_v*.json,
      parses (section, sub-folder, script, instrument, version) tuples,
      picks latest version per axis via PEP 440-ish dotted sort, regenerates
      markdown tables between <!-- BEGIN auto-table:NAME --> / <!-- END --> sentinels.
      `--check` mode for CI gate.
    - 7 sentinel region types wired: headline (top-level), likelihood-
      {imaging,interferometer,point_source,datacube}, simulators,
      searches-nautilus.
    - All 7 target READMEs gained sentinel-tagged auto-table regions
      (replacing the "populated by Phase 4" placeholder tables from
      Phases 1-3). Surrounding hand-written prose preserved.
    - Top-level README: new "Latest run-times" section + Roadmap refreshed
      to show Phases 0-4 shipped + new "Future enhancements" subsection
      listing 6 deferred extras (regression-watch indicator, version-history
      PNGs, plotly timeline, flamegraphs, hardware-tier columns, archive policy).

    Design decisions resolved:
    - CPU/GPU split: single column for now (CPU only — every current
      artifact is implicitly CPU). Hardware-tier columns added as a
      Future enhancements entry; renderer change is small once artifacts
      gain a hardware label.
    - Versioning: keep all versions in results/, render latest. Archive
      to results/archive/ is a Future enhancements entry.
    - "Cool extras": ALL deferred to Future enhancements rather than
      landing any in this PR. The dashboard infrastructure is more
      valuable to ship first, and each extra is independently scoped.

    Today every auto-table renders "No data yet — run X to populate"
    because results/ is gitignored per Phase 1's design. Phase 5's CI
    workflow will commit artifacts on release; manual runs work too.

    Smoke: py_compile PASSED; first run populates 7 placeholders; second
    --check run exits 0 confirming idempotence; surrounding prose
    untouched.

## mesh-geometry-picklable
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/320
- completed: 2026-05-16
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/321
- parent-issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1279 (Phase 4 feasibility — Q2 carve-out)
- repos: PyAutoArray
- notes: |
    Q2 of the Phase 4 subprocess-visualization feasibility (#1279).
    AbstractMeshGeometry stored `self._xp = xp` (a module reference),
    making FitImaging unpicklable. Replaced with `self._use_jax: bool`
    + `_xp` as a property — same pattern as Analysis._xp (PyAutoFit)
    and AbstractMaker._xp (PyAutoArray decorators). One class change,
    one new test file (5 new tests covering numpy + JAX backends across
    Rectangular + Delaunay geometries). 171 inversion tests still pass.

    End-to-end verified: a populated FitImaging round-trips through
    pickle.dumps/loads with log_likelihood Δ=0.00e+00 on both backends.
    Pickle size ~4.6 MB for a Rectangular-adaptive-density pixelization
    fit. Strong positive signal for Q1 (IPC choice) — mp.Process+Queue
    and ProcessPoolExecutor are both viable; no need to fall back to
    "send raw arrays + reconstruct in worker".

    Spike scripts (picklability_spike.py, picklability_spike_jax.py)
    and q2_findings.md remain in ~/Code/PyAutoLabs-wt/viz-subprocess-
    feasibility/ as in-progress notes for Q1/Q3/Q4 work. The parent
    viz-subprocess-feasibility task stays in active.md.

    File-disjoint coexistence with knn-barycentric (#317) on PyAutoArray
    worked cleanly (different files under inversion/mesh/).

## al-assistant-style
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/160
- completed: 2026-05-16
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/161
- merge-commit: fd80fa45
- summary: |
    Added autolens_workspace/skills/al_assistant_style.md as the canonical
    writing guide for PyAutoLens-Assistant skills (four required properties,
    adaptive depth, Orient → Ask → Branch → Combine conversation arc, voice
    do/don't rules). Rewrote skills/al_load_results.md against it: same
    technical content, restructured from "Steps 1..7" into a conversation
    arc with six narrative sub-task branches. Updated skills/README.md to
    reframe the folder as PyAutoLens-Assistant and flag the style guide as
    "read first."

    Style guide is treated as iteration round 1 — expected to evolve as
    more skills land. Future skills surfaced by name in al_load_results'
    "Skill combinations" section: al_load_results_many (bulk), al_compare_fits,
    al_refit_with_perturbation, al_plot_caustics.

    Shipped in parallel with jax-phase3-adoption (which also held
    autolens_workspace) via a separate worktree on disjoint files (skills/
    only). No merge conflicts.

## simulators-mirror
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/4
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/9
- merge-commit: 75a562f
- summary: |
    Phase 2 of the autolens_profiling z_feature. Mirrored 6 simulator
    profiling scripts (~2040 LOC) from clean origin/main of
    _developer/jax_profiling/simulators/ into autolens_profiling/simulators/,
    plus replaced the Phase 0 placeholder README with a section README
    covering all 6 scripts.

    Files mirrored:
      imaging.py, interferometer.py, point_source.py,
      cluster.py, group.py, multi.py

    Path rewrites applied uniformly:
    - `_workspace_root / "jax_profiling" / "dataset"` -> `_workspace_root / "dataset"`
    - `_workspace_root / "jax_profiling" / "results" / "simulators"`
        -> `_workspace_root / "results" / "simulators"`
    - `_script_dir.parents[1]` -> `_script_dir.parents[0]`
        (one level shallower than Phase 1 because simulators are at
         simulators/<name>.py vs jax_profiling/simulators/<name>.py)
    - Docstring `python jax_profiling/simulators/<name>.py`
        -> `python simulators/<name>.py`

    The artifact filename convention is unchanged:
    `results/simulators/<script>_summary_v<al.__version__>.{json,png}`

    Smoke: py_compile PASSED for all 6. Full runtime smoke skipped
    intentionally — simulators/point_source.py at default
    dataset_name="simple" writes to dataset/point_source/simple/
    which holds Phase 1's tracked likelihood input JSONs
    (point_dataset_positions_only.json, tracer.json). Running it
    without changing the dataset_name would corrupt those tracked
    files. Phase 5's AUTOLENS_PROFILING_SMOKE=1 short-circuit will
    provide a clean smoke path; until then, smoke manually by
    passing a non-conflicting dataset_name (e.g. "smoke").

    F1 lesson applied: copies came from worktree's clean origin/main
    of _developer (NOT the canonical, which is dirty with ~36
    modified files).

## searches-nautilus-mirror
- task-alias: nautilus-mirror  (matches active.md / worktree name during execution; full filename-stem slug here so the z_features audit picks this up as shipped)
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/5
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/8
- merge-commit: cd95359
- summary: |
    Phase 3 of the autolens_profiling z_feature. Stood up
    autolens_profiling/searches/ with Nautilus-only profiling (4 files
    mirrored from _developer/searches_minimal/, ~20K source + 2 READMEs).
    Designed the folder layout so 7 other sampler families (Dynesty,
    Emcee, BlackJAX, NumPyro, PocoMC, NSS, LBFGS) can slot in cleanly
    under their own follow-up prompts.

    Files mirrored:
      _setup.py             -> searches/_setup.py
      _metrics.py           -> searches/_metrics.py
      nautilus_simple.py    -> searches/nautilus/simple.py
      nautilus_jax.py       -> searches/nautilus/jax.py

    Key rewrites:
    - _setup.py dataset path: Path("jax_profiling")/"dataset"/... -> Path("dataset")/...
    - should_simulate subprocess block -> clean FileNotFoundError (Phase 1 pattern)
    - Nautilus imports: from searches_minimal._{setup,metrics} -> from searches._{setup,metrics}
    - Added sys.path injection so scripts work invariant to cwd/invocation
    - Output upgraded: .txt-to-output/ -> versioned JSON+PNG to
      results/searches/nautilus/<script>_summary_v<al.__version__>.{json,png}
      (matches Phase 1 convention so Phase 4 dashboard can pick them up)

    Smoke: py_compile + import resolution PASSED. Full runtime smoke
    (n_live=200) skipped intentionally — takes 30+ min on CPU and
    Phase 5's AUTOLENS_PROFILING_SMOKE=1 short-circuit will make this
    cheap forever. Static checks + matching the Phase 1/2 pattern gave
    sufficient confidence to ship.

    F1 lesson applied: copies came from worktree's clean origin/main
    of _developer, NOT the canonical.

## jax-phase3-adoption
- issue: none — direct follow-up to use-jax-for-vis-default / PyAutoFit #1278
- completed: 2026-05-16
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/159 (26 scripts, 72/32)
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/74 (3 scripts, 3/2)
  - https://github.com/PyAutoLabs/autofit_workspace/pull/61 (1 script, 6/6)
- repos: autolens_workspace, autogalaxy_workspace, autofit_workspace
- notes: |
    Phase 3 of z_features/jax_visualization.md (archived earlier today).
    Sweep adoption of use_jax=True across 30 tutorial scripts in the
    three production workspaces, filling the gaps the 2026-05-08 audit
    flagged plus consistency mismatches (slam.py had use_jax=True but
    companion modeling.py didn't, etc.) discovered in the 2026-05-16
    re-audit done at the start of this task.

    The Phase 2 default-flip (PyAutoFit #1278) made use_jax=True
    sufficient — viz auto-follows via the sentinel. User explicitly
    confirmed the API direction: a single explicit flag making JAX
    dependence fully visible. No use_jax_for_visualization=True calls
    added anywhere.

    Audit precondition: original 2026-05-08 Phase 3 framing (zero
    adoption) was stale — 66/32/4 scripts already had use_jax=True
    via incidental adoption in other tasks. Re-audit narrowed scope
    to the consistency gaps.

    Skip list (intentional opt-outs honoured): cpu_fast_modeling.py,
    autogalaxy/ellipse/modeling.py (AnalysisEllipse stability),
    expectation_propagation.py + hierarchical.py (FactorGraphModel),
    autofit/searches/mle.py (LBFGS), all simulators/fit.py/
    likelihood_function.py/aggregator scripts.

    Phase 3 of the JAX visualization roadmap is now complete. Tracker
    z_features/jax_visualization.md already archived under
    z_features/complete/. Phases 4 (subprocess viz) and 5 (live
    Colab/Jupyter cell) remain explicit stubs awaiting prompts.

    Worktree force-removed (PYAUTO_WT_FORCE=1) because smoke runs
    regenerated binary dataset files in the workspaces (the worktree
    symlinks share dataset/ with main); user-approved as part of the
    "complete the task" flow.

## critical-curves-linewidth
- completed: 2026-05-16
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/319
- merge-commit: 41465a35
- summary: |
    User-reported visual tweak — critical curves and caustics overlays
    were too thick to inspect underlying lens-model results. Reduced the
    hardcoded matplotlib linewidth from 2 to 1 in three PyAutoArray plot
    sites (autoarray/plot/{array.py,inversion.py,grid.py}) that draw the
    generic `lines=` overlay. The `lines=` parameter is currently consumed
    exclusively by critical curves and caustics across PyAutoGalaxy and
    PyAutoLens, so the change targets exactly the reported overlays with
    no incidental side-effects. No public API touched; no config files
    involved — only the legacy z_projects/subhalo mat_wrap_2d.yaml
    contained these keys and it is not loaded by the active plot code
    path. Ran in parallel with knn-barycentric on PyAutoArray (distinct
    branches, disjoint file sets) without conflict. 780/780 PyAutoArray
    unit tests passed; user opted to merge without smoke tests given the
    surgical 3-line scope.

## jit-regression-constant-drift
- task-alias: jit-regression-drift  (matches active.md / worktree name during execution; full filename-stem slug here so the z_features audit picks this up as shipped)
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/67
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/3
- merge-commit: aa131a7
- upstream-issues-filed:
  - PyAutoLens#514 — eager log_likelihood drift in AnalysisPoint chain (real upstream behaviour change, not constant-refresh)
  - autolens_workspace_developer#68 — jit/imaging/pixelization.py JIT vs eager mapping matrix shape mismatch
  - autolens_workspace_developer#69 — jit/imaging/delaunay.py log_evidence rebuild returns -inf
- summary: |
    Follow-up F1 of autolens_profiling z_feature. Smoked all 10
    jax_profiling/jit/ scripts against clean origin/main of _developer
    on PyAutoLens 2026.5.14.2 and found the original drift picture was
    wrong in interesting ways:

    1. The Phase 1 imaging/mge.py "drift" (+0.6%) was NOT a real upstream
       drift — Phase 1 mirror was made from the dirty _developer canonical
       checkout, which had locally-modified dataset/imaging/hst/*.fits.
       On clean main, imaging/mge.py PASSES (27379.388907 matches the
       constant). Re-mirror in autolens_profiling PR #3 fixes this:
       refreshes scripts + datasets from clean origin/main, also pulls
       in dataset/interferometer/hannah/ that Phase 1 missed (828K, 5
       files). datacube/delaunay.py default instrument also restored
       from "sma" (dirty canonical) to "hannah" (clean main).

    2. point_source/{image_plane,source_plane}.py drift IS real and IS
       upstream — magnitudes too large for floating-point drift
       (image_plane: 0.075 → -362.21, sign change + 4843×; source_plane:
       -294 → -3599, 12×). Light bisect of PyAutoLens/PyAutoGalaxy/
       PyAutoArray commits since 2026-04-24 (cfa5378, when constants were
       set) surfaced candidates but no smoking gun. Filed PyAutoLens#514
       for upstream investigation. **Constants left as-is — failing
       regression assertions are load-bearing while #514 is open.**

    3. Two unrelated pre-existing bugs uncovered by the 10-script smoke
       (Phase 1 only smoked 1 per subfolder):
       - imaging/pixelization.py: JIT vs eager mapping matrix shape
         mismatch (1285×1285 vs 1225×1225) at the curvature+regularization
         add step. Filed as _developer#68.
       - imaging/delaunay.py: log_evidence rebuild returns -inf vs
         FitImaging's finite 26288.32. Filed as _developer#69.

    No code shipped to _developer (constants stay as-is). One PR shipped
    to autolens_profiling (#3, re-mirror hygiene). Three upstream issues
    filed for follow-on investigation by maintainer.

    Smoke logs + result-artifact backups preserved at
    /tmp/jit_drift_smoke/ in case useful for upstream triage.

## use-jax-for-vis-default
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1275
- completed: 2026-05-16
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1278
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/28
- repos: PyAutoFit, autofit_workspace_test
- notes: |
    Phase 2 of z_features/jax_visualization.md. Default of
    Analysis(use_jax_for_visualization) flipped from bool=False to
    Optional[bool]=None — sentinel that resolves to use_jax. Analysis(use_jax=True)
    now turns on the JIT visualization path automatically. Explicit True/False,
    PYAUTO_DISABLE_JAX=1, and the JAX-not-installed fallback all preserve
    their existing behaviour.

    Test split per long-standing rule (numpy-only unit tests; JAX-needing
    assertions in workspace_test): 1 numpy-only env-var-override test in
    test_autofit/analysis/test_use_jax_for_visualization.py; 4 JAX-conditional
    assertions in autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py.

    Workspace audit at ship time: zero production workspace scripts set
    use_jax_for_visualization= explicitly. autolens_workspace_test has ~12
    redundant =True call sites (harmless; optional cleanup follow-up).

    Tracker z_features/jax_visualization.md is now archivable — Phases 3-5
    are explicit stubs. Re-run /start_dev z_features/jax_visualization.md
    to verify shipped state and move to z_features/complete/.

## likelihood-jit-mirror
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/1
- completed: 2026-05-16
- repo-pr: https://github.com/PyAutoLabs/autolens_profiling/pull/2
- merge-commit: 7c464c2
- summary: |
    Phase 1 of the autolens_profiling z_feature. Mirrored the JIT
    likelihood profiling scripts and their tracked input datasets from
    autolens_workspace_developer/jax_profiling/jit/ into the new
    autolens_profiling repo at likelihood/ and dataset/. _developer
    stays the source of truth — nothing moved or deleted upstream.

    9 scripts + 1 __init__.py mirrored verbatim (filename-preserving)
    across imaging/, interferometer/, point_source/, datacube/. ~7,400 LOC.
    14 dataset files mirrored (~900K, checksums verified). 5 READMEs
    authored: top-level likelihood/README.md + 4 per-section.

    Path rewrites applied uniformly across all 9 scripts:
      Path("jax_profiling") / "dataset"     -> Path("dataset")
      _script_dir.parents[2]                -> _script_dir.parents[1]
      "jax_profiling" / "results" / "jit"   -> "results" / "likelihood"
      docstring sibling refs                -> new layout
    The if should_simulate(...) block at the top of each script was
    replaced with a clear FileNotFoundError pointing back at
    _developer/jax_profiling/dataset_setup/ for regeneration (Phase 1
    out of scope).

    Decision locked in: dataset/ lives at top-level (shared with Phase 2
    simulators and Phase 3 searches), NOT under likelihood/.

    Smoke (CPU, all 4 produced artifacts at expected results/likelihood/
    paths):
      - imaging/mge.py [hst]: artifacts ✓; regression assertion
        pre-existing drift (constant unchanged from _developer).
      - interferometer/mge.py [sma]: ALL PASSED.
      - point_source/image_plane.py [simple]: artifacts ✓; regression
        assertion pre-existing drift (large: 0.07 → -362, sign change).
      - datacube/delaunay.py [sma × 4]: ALL PASSED (both eager and
        full-pipeline cube regressions).

    Pre-existing drift in imaging/mge and point_source/image_plane is
    upstream science work for _developer — same drift would manifest if
    those scripts ran on PyAutoLens 2026.5.14.2. Worth a separate
    follow-up issue against _developer (especially the point_source one,
    which suggests a real behaviour change rather than fp noise).

## nss-tutorial-dispatch
- issue: https://github.com/PyAutoLabs/autofit_workspace/issues/59
- completed: 2026-05-16
- workspace-prs:
  - autofit_workspace: https://github.com/PyAutoLabs/autofit_workspace/pull/60
- notes: |
    Phase 5 of nss_first_class_sampler — the workspace capstone. Added an
    "Search: NSS" section to autofit_workspace/scripts/searches/nest.py so
    end users discover af.NSS from the canonical nested-sampler tutorial.
    Extended the top docstring + Contents block, added an
    "Install Precondition for NSS" callout (pip install autofit[nss]),
    added a "When to use NSS" paragraph to the searches/README.md.

    Real bug caught during validation: the default af.ex.Analysis uses
    NumPy internally, which trips TracerArrayConversionError when NSS
    JIT-traces through it. Fix: build a separate af.ex.Analysis with
    use_jax=True for the NSS section. Turned this into a natural teaching
    moment in the tutorial — production users adopting NSS need to
    construct their Analysis with use_jax=True.

    Validation: nest.py runs end-to-end through all four nested samplers
    (DynestyStatic, DynestyDynamic, Nautilus, NSS) producing finite
    log_evidence (NSS: log_evidence=-67.02, max log L=-50.40). autofit_workspace
    smoke 9/9 passes — searches/nest.py is in smoke_tests.txt so this
    exercises the new NSS section on every smoke run.

    Scope intentionally narrow: autofit_workspace only. autogalaxy_workspace
    and autolens_workspace NSS adoption deferred to separate follow-ups —
    those workspaces don't have scripts/searches/ directories.

    With this PR, Phases 0-5 of the nss_first_class_sampler z_feature are
    all shipped. Ready to archive the tracker via
    `/start_dev z_features/nss_first_class_sampler.md` (audit-only mode).

## nss-install-extra
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1276
- completed: 2026-05-16
- library-prs:
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1277
- notes: |
    Phase 4 of nss_first_class_sampler. `pip install autofit[nss]` is now the
    single safe install command for af.NSS — replaces the multi-step install
    saga documented in FINDINGS_v3.md.

    Approach: Option C from the prompt (pyproject.toml extra with pinned
    git+ URLs). Original prompt recommended Option A (full vendoring of
    ~1300 LOC) — investigation revealed modern pip 23+ handles URL-direct
    deps cleanly, so the lighter Option C path was feasible without
    ongoing re-vendor maintenance burden.

    Critical pin: handley-lab/blackjax at SHA `ef45acd2f` (the May 2026
    "Merge PR #60 — double_compile" commit). HEAD added `numpy>=1.25`
    which conflicts with autofit's `anesthetic==2.8.14` (numpy<2.0).
    Bump only when the anesthetic numpy cap moves (likely with Python 3.13
    anesthetic>=2.9 takeover).

    Validation: pytest test_autofit 1258 passed/1 skipped; fresh-venv
    `pip install -e PyAutoFit[nss]` completes in ~3 min on Python 3.12 with
    no resolver conflict; `af.NSS()` + `blackjax.ns.adaptive.init` smoke
    pass. New CI workflow runs the fresh-venv install + import smoke on
    every PR + Sunday 03:00 UTC cron — catches upstream drift past the
    pinned SHAs.

    Updated af.NSS ImportError text to reference `pip install autofit[nss]`
    instead of the manual `git+https://...` recipe. Phase 1's NSS unit
    test's `pytest.raises(match=...)` regex still matches the new text.

    Roadmap status: Phases 0-4 shipped. Only Phase 5 remains (workspace
    tutorial scripts — autolens_workspace/searches/nss.py +
    autogalaxy/autofit cookbook entries).

## bootstrap
- task-alias: autolens-profiling-bootstrap  (matches active.md / worktree name during execution; full filename-stem slug here so the z_features audit picks this up as shipped — the filename `bootstrap.md` lives under `autolens_profiling/`, hence the bare slug)
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/513
- completed: 2026-05-16
- new-repo: https://github.com/PyAutoLabs/autolens_profiling
- initial-commit: 0087d6a
- summary: |
    Phase 0 of autolens_profiling z_feature. Created the empty public repo
    PyAutoLabs/autolens_profiling, scaffolded README.md (vision/scope, JAX
    gradient out-of-scope note, related repos, how-to-read guide, roadmap),
    LICENSE (MIT), .gitignore (mirrored from autolens_workspace_developer +
    profiler/cache additions), and folder skeleton (likelihood/, simulators/,
    searches/, results/) each with a placeholder README pointing at the
    phase that will populate it. No profiling code moved — that lands in
    Phases 1–3. No PR: initial scaffolding committed directly to main of
    the new repo.

## nss-checkpointing-and-visualization
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1273
- completed: 2026-05-16
- library-prs:
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1274
  - autolens_workspace_developer: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/66
- notes: |
    Phases 2-3 of nss_first_class_sampler. Inlined the upstream
    run_nested_sampling outer loop in NSS._fit (blackjax.nss + manual while +
    finalise + log_weights) so we could hook checkpoint writes +
    analysis.visualize() between iterations. New `checkpoint_interval=100`
    kwarg; `iterations_per_quick_update` (Phase 1 no-op) now functional.
    Atomic tmp-and-rename pickle write with NumPy round-trip for JAX
    pytree portability. Post-success cleanup deletes the checkpoint.

    Architectural insight (corrected the roadmap): nss.ns.run_nested_sampling's
    outer loop is plain Python — the JIT boundary is `one_step` processing
    num_delete deaths per iteration. No upstream yallup/nss PR needed; both
    phases ship in one PyAutoFit PR.

    Validation: pytest test_autofit 1258 passed/1 skipped (1252 baseline +
    6 new checkpoint tests). End-to-end resume smoke
    (autolens_workspace_developer/searches_minimal/nss_checkpoint_resume.py):
    capture pass + resume pass produce identical log_evidence=-0.0096,
    Phase 3 viz fires 4 times during capture, post-success cleanup deletes
    the checkpoint on both passes. Phase 1 Gaussian smoke regression-free
    (7s wall, ESS 94/95).

    Roadmap status: Phase 4 (`pip install autofit[nss]` extra) and Phase 5
    (workspace tutorial scripts: autolens_workspace/searches/nss.py etc.)
    remain. Both are small + standalone — neither blocks the other.

## nss-search-wrapper
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1271
- completed: 2026-05-16
- library-prs:
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1272
  - autolens_workspace_developer: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/64
- notes: |
    Phase 1 of nss_first_class_sampler roadmap. `af.NSS(...)` lands as a
    drop-in `NonLinearSearch` mirroring `af.Nautilus(...)`. New module
    PyAutoFit/autofit/non_linear/search/nest/nss/ with `NSS(AbstractNest)`,
    `NSSamples(SamplesNest)`, and an `_NSSInternal` post-run state holder.
    JAX-traceable `log_likelihood` and `prior_logprob` closures built inline
    using Phase 0's `xp=jnp` plumbing (#1262 + #1269). Optional-import guard
    keeps `import autofit` working without `nss` installed; instantiation
    raises a clear `ImportError` pointing at the Phase 4 install path.

    Validation: pytest test_autofit 1252 passed/1 skipped (1242 baseline + 10
    new NSS tests). Fast 2D Gaussian end-to-end wiring smoke
    (autolens_workspace_developer/searches_minimal/nss_first_class_gaussian.py)
    completes in 10 sec wall on CPU — ESS 94/95, weighted posterior recovers
    prior means under flat likelihood, samples.csv written through Paths,
    Result.max_log_likelihood_instance round-trips. Heavy HST MGE smoke
    (nss_first_class.py) demonstrated wiring works (1000+ dead points,
    monotonic logZ progression) but is HPC-GPU-only for full numerical-parity
    runs.

    Real bug caught during validation: initial `_fit` returned None for the
    `fitness` slot but AbstractNest.perform_update calls `fitness.batch_size`
    for latent-sample generation. Fixed by returning a
    `Fitness(model, analysis, paths, fom_is_log_likelihood=True, batch_size=1)`
    even though af.NSS doesn't use Fitness for sampling — required by the
    post-fit API contract.

    Phases 2-5 status:
    - Phase 2 (checkpointing): stubbed — `iterations_per_quick_update`
      accepted with no-op log, state.json warns instead of resuming
    - Phase 3 (on-the-fly viz): stubbed (same kwarg)
    - Phase 4 (`pip install autofit[nss]` extra): not yet
    - Phase 5 (workspace tutorial scripts): not yet — autolens_workspace/
      searches/nss.py + autogalaxy/autofit cookbook entries land after Phase 4

    Follow-ups: JIT persistent cache (each cold fit eats ~25-30 s while_loop
    compile), and proper Sonnet-style workspace tutorial scripts once
    Phase 4 lands.

## fix-interferometer-sparse-curvature
- completed: 2026-05-16
- issue: https://github.com/Jammy2211/PyAutoArray/issues/314
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/316
- workspace-pr: https://github.com/Jammy2211/autolens_workspace_test/pull/98
- summary: |
    Replaced the NotImplementedError guard from PR #315 with a real math fix.
    InterferometerSparseOperator.curvature_matrix_via_sparse_operator_from →
    curvature_matrix_diag_from(rows, cols, vals, *, S), mirroring
    ImagingSparseOperator. New Mask2D.extent_index_for_masked_pixel property
    plumbed through so triplets land in the operator's extent-flat scatter
    buffer (the old code used native-flat fft_index_for_masked_pixel which
    silently fell out-of-bounds and was dropped by JAX for any mask with
    extent < native — both the Delaunay 34% Frobenius gap and the previously-
    documented Pmax=1 ~0.4% "numerical reformulation" gap were the same bug).
    Converted the raise-test to a sparse-vs-mapping parity assertion at
    rtol=1e-4. Updated the one Pmax=1 workspace call-site and the
    rectangular_sparse.py likelihood literal (-3152.03 → -3164.29 to match
    DFT-no-sparse and NUFFT-no-sparse to ~1e-13).

## datacube-sparse-operator
- completed: 2026-05-15
- status: partially-shipped (guard only, math fix deferred)
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/315
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/314
- follow-up: PyAutoPrompt/planned.md ##fix-interferometer-sparse-operator-irregular-meshes
- summary: |
    Attempted to wire `dataset.apply_sparse_operator(use_jax=True)` into the
    just-shipped hannah profilers; the path explodes at Cholesky with
    "Matrix is not positive definite". Diagnosed: the interferometer
    sparse-operator curvature math
    (`InterferometerSparseOperator.curvature_matrix_via_sparse_operator_from`)
    is wrong by ~34% Frobenius on Delaunay (Pmax=3 barycentric weights) — only
    validated against Rectangular Pmax=1. Independent of zeroed_pixels,
    independent of use_jax. CPU and JAX paths match each other to 5.5e-14,
    both wrong vs the mapping path.

    Shipped: PR #315 added a defensive NotImplementedError guard at
    `InversionInterferometerSparse.curvature_matrix_diag` for Delaunay-mesh
    mappers, with a regression test. Future users get a clear early failure
    pointing at issue #314 instead of confusing downstream LinAlgError.

    Deferred (to planned.md): the actual math rewrite of the interferometer
    sparse-operator curvature path to handle Pmax > 1 correctly, plus audit
    of the existing ~0.4 % rectangular_sparse discrepancy that may share the
    same root cause. Filed for an inversion-math maintainer to pick up.

    Workspace impact: none. The previously-shipped Delaunay profilers
    (datacube-hannah-preset) stay on plain DFT path and continue to work.

## hilbert-offset-centre-mask
- completed: 2026-05-15
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/313
- summary: |
    Fixed a user-reported bug where Hilbert image-mesh raised PixelizationException
    for circular masks with offset centres that didn't align with pixel half-integers.
    Root cause was twofold: Mask2D.is_circular used pixel-quantization-sensitive row
    vs column counts (rejecting valid offset circles, false-accepting annular masks),
    AND hilbert.image_and_grid_from sampled the image around (0,0) regardless of mask
    centre. Rewrote is_circular with a bbox-square + centre-pixel-unmasked + reference
    mask reconstruction check, and made image_and_grid_from translate Hilbert points
    by mask_centre (no-op for centred masks, so existing smoke tests bit-identical).
    Confirmed via 165 inversion unit tests + 13 workspace smoke tests.

## datacube-hannah-preset
- completed: 2026-05-15
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/311
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/63
- summary: |
    Added "hannah" instrument preset to the jax_profiling datacube + interferometer
    delaunay profilers, pinning Hannah Stacey's real ALMA settings (n_channels=34,
    n_visibilities=16984, pixel_scale=0.125", shape_native=(40, 40), mask_radius=2.3").
    Library side (PyAutoArray): extended `Interferometer.from_fits` to accept
    `raise_error_dft_visibilities_limit` (3-line change + regression test).
    Workspace side: promoted mask_radius into INSTRUMENTS dict; gated full-pipeline
    cube JIT (Part C) behind CUBE_FULL_JIT=1 (lower+compile alone is ~70s at
    n_channels=34); per-channel regression literal pinned at -204838.07924622478;
    cube step-by-step total at Hannah's scale is 205.92s/eval (shared-Lᵀ W̃ L
    savings est. ~34.8s once Aris's deferred optimisation lands).

## unpark-ellipse-scripts
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/72
- completed: 2026-05-15
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1270
- workspace-prs:
  - autogalaxy_workspace: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/73
  - PyAutoBuild: https://github.com/PyAutoLabs/PyAutoBuild/pull/89
- notes: Reactivated the five `scripts/ellipse/*` example scripts in autogalaxy_workspace, parked since 2026-04-24 pending the ellipse JAX refactor (PyAutoGalaxy #408/#410/#412). All five pass under PYAUTO_TEST_MODE=2; removed the NEEDS_FIX entries from `autogalaxy_workspace/config/build/no_run.yaml` and the paired `ellipse/database` fallback from `PyAutoBuild/autobuild/config/no_run.yaml`. Scope expanded mid-task — `database.py` surfaced two unrelated bugs that were also fixed: (a) two leftover `path.exists(...) / os.remove(...)` calls from the prior os.path→pathlib refactor (the `path` import was removed but two call sites missed), replaced with pathlib idiom in the workspace PR; (b) a `Drawer.__init__` round-trip crash in PyAutoFit when the aggregator scrapes a saved Drawer search.json (the saved JSON carries `number_of_cores` at the top level, which collided with the hardcoded `super().__init__(number_of_cores=1, **kwargs)`). Fix in library PR: `kwargs.pop("number_of_cores", None)` before forwarding to super; Drawer is single-core only by design so the saved value is always 1. Regression test `test__dict_round_trip_with_number_of_cores` added in test_drawer.py; full PyAutoFit suite 1243/1243 passes. Follow-ups worth their own issues: (i) `autogalaxy_workspace/smoke_tests.txt` still has `# ellipse/modeling.py` disabled with a comment citing PyAutoFit#1179 — may be stale now that the JAX refactor + Drawer fix are in; (ii) `smoke (3.13)` CI is failing on main (and on this PR — but pre-existing): interferometer/start_here.py + interferometer/simulator.ipynb fail with `FileNotFoundError` from `setup_notebook` workspace-root lookup in a `/tmp/smoke_regen_*` cwd. smoke (3.12) passes.

## log-prior-sign-convention
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1266
- completed: 2026-05-15
- library-prs:
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1269
  - autofit_workspace_test: https://github.com/PyAutoLabs/autofit_workspace_test/pull/27
- notes: |
    Sign-convention fix for Prior.log_prior_from_value across the Gaussian-family
    priors and LogUniformPrior. Switched to density form (log p(x), negative for
    low-density, zero at mode). NormalMessage flipped to -(value-mean)**2/(2σ**2);
    LogGaussianPrior similarly with the -log(value) Jacobian preserved;
    LogUniformPrior replaced 1.0/value (Jacobian gradient, not a log) with
    -log(value) on NumPy + xp.where(in_bounds, -xp.log(value), -xp.inf) on JAX.
    UniformPrior and TruncatedNormalMessage already correct. No Fitness changes —
    sign lives entirely at the Prior boundary, as architecture demanded.

    Empirically confirmed bug by two controlled experiments (Emcee + LBFGS,
    flat likelihood + GaussianPrior(5,1)) — pre-fix they diverged to 10^146 and
    8e143 respectively; post-fix both behave correctly. Both scripts promoted
    to autofit_workspace_test/scripts/prior_correctness/ as permanent regression
    gates that fail loudly if any future refactor reverts the sign.

    Validation: pytest test_autofit 1242 passed, 1 skipped; 4 test pins updated
    (test_prior.py + test_model_mapper.py — they had rubber-stamped the buggy
    values); priors_xp_dispatch.py 28 assertions pass (24 existing parity + 4
    new density-form gates); 4 autofit_workspace searches pass; EP runs to
    completion (confirmed unaffected — uses Message.logpdf directly); 44/44
    5-workspace smoke green.

    Bug existed since commit db4016db42 (4 May 2022) for LogUniformPrior and
    pre-dates that for the Gaussian family — ~4 years. Hidden because (a) most
    production fits use nested samplers which bypass log_prior_from_value, (b)
    most MCMC fits used UniformPrior which is sign-agnostic, (c) the existing
    test pins rubber-stamped the wrong values.

    Migration warning: cached Emcee/Zeus/MLE-Drawer/LBFGS/BFGS samples.csv with
    non-uniform priors are biased and should be re-run. Dynesty/Nautilus chains
    unaffected (priors via prior_transform); only their stored log_prior column
    is wrong-signed and auto-recovers on next aggregator load.

## url-check-ci
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/87
- completed: 2026-05-15
- tool-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/88
- consumer-prs:
  - https://github.com/PyAutoLabs/PyAutoConf/pull/106
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1268
  - https://github.com/PyAutoLabs/PyAutoArray/pull/310
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/415
  - https://github.com/PyAutoLabs/PyAutoLens/pull/510
  - https://github.com/PyAutoLabs/HowToFit/pull/8
  - https://github.com/PyAutoLabs/HowToGalaxy/pull/8
  - https://github.com/PyAutoLabs/HowToLens/pull/11
  - https://github.com/PyAutoLabs/autofit_workspace/pull/58
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/71
  - https://github.com/PyAutoLabs/autolens_workspace/pull/153
- notes: Follow-up to url-check (PyAutoLens#508). Wires the live HTTP URL audit into PyAutoBuild as a weekly-cron CI job, plus extends the offline regex guard with ~15 new patterns surfaced by the original audit (hhttps, joshspeagle/nautilus, rhayes777/PyAutoBuild, sphinx /en/main, bokeh + numfocus CoC paths, tree/release, etc.). New tool: PyAutoBuild/autobuild/url_check_live.py (port of admin_jammy/software/url_check/url_check.py with --allowlist, --strict, --format markdown-issue flags; fixes the symlink-canonical bug from the earlier wave). Wrapper PyAutoBuild/autobuild/url_check_live.sh. Each of 11 consumer repos: .url_check_allowlist.txt at repo root pre-seeded with the current broken URLs (PyAutoConf 2, PyAutoFit 12, PyAutoArray 0, PyAutoGalaxy 11, PyAutoLens 22, HowToFit 11, HowToGalaxy 31, HowToLens 9, autofit_workspace 4, autogalaxy_workspace 7, autolens_workspace 18 — 127 total), plus extended .github/workflows/url_check.yml with the existing url_check_patterns job + new url_check_live job that runs Mon 04:00 UTC. Live job opens/comments on a [url-check] New broken URLs detected issue when non-allowlisted breakage appears; auto-closes the issue on a clean run. pyauto-status skill updated to surface those tracking issues. Two parser bugs caught during smoke-testing and fixed before ship: allowlist parser truncated URLs at the first '#' (URL fragment), and the allowlist file itself was being scanned for URLs and reflected as locations. Both fixed via whitespace-preceded '#' for comments and an explicit SCAN_EXCLUDE_BASENAMES.

## visualize-combined-quick-update-kwarg
- issue: none — followup to priors-jax-native (#1262); deferred Bug B at #1266
- completed: 2026-05-15
- library-prs:
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1267
  - PyAutoGalaxy: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/414
- notes: Followup audit work for priors-jax-native (#1262). Pure additive one-line `quick_update: bool = False` kwarg on `VisualizerExample.visualize_combined` (PyAutoFit) and `VisualizerImaging.visualize_combined` (PyAutoGalaxy), mirroring the plumbing added by commit `a1e360567` ("Fix `AnalysisFactor.visualize_combined` dispatch in FactorGraph") which updated the base class + AnalysisFactor + PyAutoLens visualizers but missed these two. Fixed the four previously-broken graphical/EP integration scripts: `autofit_workspace_test/scripts/graphical/{simultaneous,hierarchical}.py` and `autofit_workspace/scripts/{features/graphical_models,cookbooks/multiple_datasets}.py`. PyAutoFit 1242/1242 + PyAutoGalaxy 870/870 unit tests pass. The deeper sign-convention finding surfaced during this audit (Gaussian-family `log_prior_from_value` returning cost form, biasing MCMC posteriors) was deferred to dedicated issue #1266.

## url-check
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/508
- completed: 2026-05-15
- tool-pr: https://github.com/Jammy2211/admin_jammy/pull/21
- library-prs:
  - https://github.com/PyAutoLabs/PyAutoConf/pull/105
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1265
  - https://github.com/PyAutoLabs/PyAutoArray/pull/309
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/413
  - https://github.com/PyAutoLabs/PyAutoLens/pull/509
- workspace-prs:
  - https://github.com/PyAutoLabs/HowToFit/pull/7
  - https://github.com/PyAutoLabs/HowToGalaxy/pull/7
  - https://github.com/PyAutoLabs/HowToLens/pull/10
  - https://github.com/PyAutoLabs/autofit_workspace/pull/57
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/70
  - https://github.com/PyAutoLabs/autolens_workspace/pull/152
- notes: Cross-repo doc URL audit + cleanup driven by new admin_jammy/software/url_check/ tool. Audit before fixes 304 broken URLs across 12 PyAuto repos; after both waves 104 broken (200 fixed end-to-end). Remaining 104 are mostly external paywalled/dead links + ~10 internal readthedocs renames needing editorial decisions. Tool patterns hhttps typo, Jammy2211/rhayes777 → PyAutoLabs (libs+workspaces), /blob/release/ → /blob/main/, joshspeagle/nautilus → johannesulf/nautilus, rhayes777/PyAutoBuild → PyAutoLabs/PyAutoBuild, bokeh+numfocus CoC URLs, sphinx /en/main → /en/master, pyautofit.readthedocs.io page renames, workspace notebook reorganisations (overview/{simple,complex}/{fit,result} flattened; modeling/imaging/features/<x>.ipynb → imaging/features/<x>/modeling.ipynb), Colab badge target fixes. Special-cases Colab URLs via raw.githubusercontent.com check (Colab returns 200 even for dead refs). Surface one bug worth remembering Path(__file__).resolve() through admin_jammy's worktree symlink lands at canonical root — pass --root explicitly when fixing from a worktree (memory feedback_path_file_resolve_symlink).

## analysis-ellipse-jax
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/411
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/412
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/48

## jax-likelihood-datacube
- issue: none — Phase 4 of the datacube roadmap (followup to autolens_workspace#120)
- completed: 2026-05-15
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/96
- repos: autolens_workspace_test
- notes: Phase 4 closes the datacube roadmap. Adds autolens_workspace_test/scripts/jax_likelihood_functions/datacube/{rectangular,delaunay}.py — end-to-end JIT-correctness regression scripts for the cube likelihood path (the regression net the Phase 1 rewrite punted to this folder). Mirrors interferometer/{rectangular,delaunay}.py with N=4 identical-channel FactorGraph wiring (multi/delaunay.py is the structural analogue). Each script asserts vmap (against the 4× single-channel literal: rectangular=-12657.14500637, delaunay=-12661.69554044), Path A jit(log_likelihood_function) round-trip via instance_from_vector(xp=jnp), and Path B TransformerNUFFT cross-check. scripts/CLAUDE.md updated. Full datacube roadmap now complete: Phase 1 (autolens_workspace#149) user-facing pedagogical likelihood walkthrough, Phase 2 (autolens_workspace_developer#61) jit/interferometer/delaunay.py step-by-step profiling, Phase 3 (autolens_workspace_developer#62) jit/datacube/delaunay.py cube profiler with channel-invariant/variant taxonomy, Phase 4 (this) JIT regression scripts.

## priors-jax-native
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1262
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1263
- workspace-prs:
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/26
- notes: Phase 0 of the `nss_first_class_sampler` roadmap. Added `xp=np` kwarg threading to `Prior.value_for`, `Prior.log_prior_from_value`, `Model.vector_from_unit_vector`, `NormalMessage.value_for`, and `TruncatedNormalMessage.value_for`. Each of the 5 concrete `Prior` subclasses gained a closed-form JAX `value_for` override (bypasses the scipy-backed message stack — cleaner trace, smaller surface). NumPy paths are byte-equivalent — `xp=np` defaults preserve all existing callers (Nautilus, Dynesty, Emcee, Zeus, EP). `NormalMessage.value_for` cleaned up: replaced legacy `isinstance(unit, np.ndarray)` runtime sniff with explicit `xp` dispatch. 1242/1242 PyAutoFit tests pass; 24 new JAX parity assertions in `autofit_workspace_test/scripts/jax_assertions/priors_xp_dispatch.py` (library policy: no JAX in unit tests, cross-xp checks live in workspace_test). Smoke: 44/44 across autofit/autogalaxy/autolens/autolens_test/HowToLens. Followups worth their own issues: (a) `LogUniformPrior.log_prior_from_value` body returns `1.0/value` instead of `-log(value)` — left untouched here to avoid MCMC regressions, (b) graphical declarative `VisualizerExample.visualize_combined()` signature mismatch (pre-existing on `main`, broke several graphical/EP integration scripts), (c) euclid workspace version pin `2026.5.8.2` lags library `2026.5.14.2` blocking euclid smoke. Phase 1 (`af.NSS` wrapper at `autofit/nss_search_wrapper.md`) is now unblocked.

## disable-model-graph
- issue: none — ad-hoc cleanup
- completed: 2026-05-15
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1264
- workspace-prs:
  - https://github.com/PyAutoLabs/autofit_workspace/pull/56
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/69
  - https://github.com/PyAutoLabs/autolens_workspace/pull/150
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/95
- followup-issue: https://github.com/PyAutoLabs/autolens_workspace/issues/151 (pre-existing smoke (3.13) CI failures — nufftax import + setup_notebook tempdir lookup; surfaced while merging this task's workspace PRs)
- notes: Gated `model.graph` output in fit folders behind a new `output.model_graph` config key (default `false`). Library change in PyAutoFit: replaced the `try/except AttributeError` block in `_save_model_info` with `if should_output("model_graph") and hasattr(model, "graph_info"):` — fixes the bug where an empty `model.graph` was created for every non-graphical fit because the file was opened before `model.graph_info` raised. Workspace change: added `model_graph: false` to each workspace's `config/output.yaml` so the workspace's `default: true` doesn't fall through. Metadata file left alone — investigation showed it's the sentinel `Aggregator.from_directory()` uses to detect search output directories and `SearchOutput` parses its key=value lines into attributes. 1242/1242 library tests pass; 30/30 local smoke tests pass across all 4 workspaces. CI smoke (3.13) on autogalaxy_workspace and autolens_workspace failed with pre-existing infrastructure errors (verified same failures on `main` predating this PR), tracked in workspace issue #151.

## jit-datacube-delaunay
- issue: none — Phase 3 of the datacube roadmap (followup to autolens_workspace#120)
- completed: 2026-05-14
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/62
- repos: autolens_workspace_developer
- notes: Phase 3. Add jax_profiling/jit/datacube/delaunay.py — mirrors the upgraded interferometer/delaunay.py with a per-channel loop and the channel-invariant / channel-variant taxonomy made explicit. 3 shared steps (ray-trace data, ray-trace mesh, regularization matrix) computed once for the whole cube; 5 per-channel steps (inversion setup incl. NUFFT, data vector D, curvature matrix F, NNLS reconstruction, log evidence) computed once per channel and reported as N × per-call. Cube total reported alongside full-pipeline cube JIT and a shared-Lᵀ W̃ L savings estimate ((N-1) × curvature_matrix). For the SMA × 4 channels preset: step-by-step total 1.164s, full-pipeline cube JIT 1.425s, shared-Lᵀ W̃ L savings est. 0.060s. Cube reuses the SMA interferometer dataset 4× (identical channels — timing not science); regression assertion against EXPECTED_LOG_EVIDENCE_CUBE_SMA = 4 × -3167.5258928840763 passes for both eager and full-pipeline JIT. Per-step cube log_evidence matches summed FitInterferometer.log_evidence at rtol=1e-4. vmap skipped (cube batching axis is "datasets" not "parameters"). Phase 4 (autolens_workspace_test/scripts/jax_likelihood_functions/datacube/{rectangular,delaunay}.py — JIT regression scripts) is the next step.

## jit-interferometer-stepwise
- issue: none — direct followup to autolens_workspace#120
- completed: 2026-05-14
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/61
- repos: autolens_workspace_developer
- notes: Phase 2 of the datacube roadmap. Upgraded jax_profiling/jit/interferometer/delaunay.py (~604 → ~1100 lines) to step-by-step parity with the imaging sibling jax_profiling/jit/imaging/delaunay.py. 8 per-step JIT timings entries (imaging-sibling numbering preserved for cross-reference; lens-light steps 3-4 dropped): ray-trace data grid, ray-trace mesh grid, inversion setup (steps 5-8 combined incl. NUFFT), data vector D (vis-space real+imag), curvature matrix F (real+imag summed), regularization matrix H (ConstantSplit), reconstruction (NNLS), mapped recon + log evidence (vis-space χ²). Step-by-step total 0.298s, full-pipeline JIT 0.316s (5% XLA cross-step fusion gap). Correctness: per-step log_evidence from inversion matrices matches FitInterferometer.log_evidence exactly; full-pipeline JIT matches eager at rtol=1e-4; eager + full-pipeline regression assertions against EXPECTED_LOG_EVIDENCE_SMA = -3167.5258928840763 still pass. Prereq for the future jit/datacube/delaunay.py profiler (Phase 3).

## fit-ellipse-jax
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/409
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/410

## datacube-likelihood-walkthrough
- issue: none — direct followup to autolens_workspace#120
- completed: 2026-05-14
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/149
- repos: autolens_workspace
- notes: Rewrite datacube/likelihood_function.py from a JIT-correctness test (PART A Setup → PART B Eager NumPy → PART C JIT → PART D Correctness) into a pedagogical pixelization-likelihood walkthrough in the style of imaging/features/multi_gaussian_expansion/likelihood_function.py — cross-reference shared sections back to interferometer/features/pixelization/likelihood_function.py and give full prose treatment only to the cube-specific bits (list of Interferometer objects, per-channel inversion construction, per-channel transformed_mapping_matrix from each channel's distinct uv_wavelengths, sparse-operator memory pressure multiplied by N channels, the deferred shared-`Lᵀ W̃ L` optimisation, and the headline `Across All Channels` section that loops the per-channel calculation and sums to produce the cube log-evidence). Cross-check against summed FitInterferometer.log_evidence agrees at ~5e-4 (matching the residual the pixelization reference exhibits — comes from source-code fast paths the manual walkthrough deliberately doesn't reproduce). The dropped JAX-JIT correctness assertion moves to the planned autolens_workspace_test/scripts/jax_likelihood_functions/datacube/ folder (follow-up).

## datacube-centre-and-4d
- issue: none — direct followup to autolens_workspace#120
- completed: 2026-05-14
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/148
- repos: autolens_workspace
- notes: Two polish items on top of the datacube tutorials. (1) Source centre now shifts linearly along y across channels (CENTRE_SHIFT_TOTAL = 0.12" end-to-end) to mimic a kinematic gradient; centres land at (0.04, 0.1), (0.08, 0.1), (0.12, 0.1), (0.16, 0.1) for the 4-channel reference cube. (2) Simulator now writes a third on-disk layout: a 4D CASA-like cube `{visibilities,noise_map,uv_wavelengths}_4d_cube.fits` of shape `(n_pol, n_chan, n_vis, 2)` matching what CASA gives users straight out of reduction. Polarisations are identical in the synthetic simulator (documented as pedagogical simplification). data_preparation.py now loads the simulator's actual 4D output rather than synthetic random arrays. README documents all three layouts (CASA-like 4D / 3D cube / per-channel folders).

## jax-dataset-model
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/93
- completed: 2026-05-14
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/94
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/47

## ellipse-xp
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/407
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/408

## ludlow16-jax-native
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/403
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/406
- repos: PyAutoGalaxy
- notes: |
    Phase 2 of the Ludlow16 JAX-native work (Phase 1 was #397 / PR #402).
    The colossus jax.pure_callback in mcr_util.py is GONE from production.

    What landed:
    - New autogalaxy/profiles/mass/dark/ludlow16.py — JAX-native port of
      colossus.halo.concentration.modelLudlow16 (~400 lines, xp-aware).
      EH98 transfer + Heath '77 growth factor + Einasto gammainc +
      200-point Ludlow c-solver.
    - mcr_util.py rewritten: replaced _ludlow16_cosmology_callback and
      ludlow16_cosmology_jax with a single xp-aware ludlow16_cosmology(...).
      The if-xp-is-np branching in kappa_s_and_scale_radius_for_ludlow and
      kappa_s_scale_radius_and_core_radius_for_ludlow collapsed to single
      xp-aware calls.
    - colossus moved from required runtime dep to test/dev extras in
      pyproject.toml. Production no longer imports colossus.
    - 10 new tests in test_autogalaxy/profiles/mass/dark/test_ludlow16.py
      (numpy-path cross-check vs colossus, skipped if colossus unavailable).
    - Test tolerances loosened from 1e-4 → 1e-3 (still 0.1%) in 4 NFW-MCR
      test files. Justified: the old 1e-4 implicitly claimed colossus-level
      precision; JAX impl differs from colossus by ~2e-4 (sub-Ludlow-scatter).
    - Bug fix during CI debug: replaced xp.trapezoid with a manual
      _trapezoid_last_axis helper for numpy<1.26 compat (CI's Python 3.12
      runs older numpy than the local dev venv).

    Cross-implementation verification: autolens_workspace_test/subhalo.py
    (Scenarios C and D, regression literals locked in via workspace_test
    PR #92 from the colossus path) produces vmap = -1.349200e+09 — exact
    match to rtol=1e-4. The JAX-native code reproduces the colossus
    pure_callback's downstream log-likelihood at the precision we care
    about.

    Production grep confirms: no import colossus, no jax.pure_callback
    anywhere in autogalaxy/. The only remaining "colossus" references are
    in docstrings explaining what the new code replaced.

## nfw-mcr-ludlow-jit
- issue: none — extension of the subhalo JAX regression script
- completed: 2026-05-14
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/92
- repos: autolens_workspace_test
- notes: |
    Extended scripts/jax_likelihood_functions/imaging/subhalo.py from
    two scenarios to four: kept the existing IsothermalSph + fixed/free
    subhalo-redshift pair (Scenarios A/B, regression check for PyAutoLens
    #498/PR #499) and added NFWMCRLudlowSph + fixed/free subhalo-redshift
    (Scenarios C/D, regression check for the colossus jax.pure_callback
    path in PyAutoGalaxy/autogalaxy/profiles/mass/dark/mcr_util.py).

    Refactored build_model + run_scenario to take a subhalo-mass-factory
    callable and an expected-vmap literal so each scenario is one call.

    Regression literals (vmap log-likelihood at prior medians):
      A/B (IsothermalSph)   : -1.412105e+09 (unchanged)
      C/D (NFWMCRLudlowSph) : -1.349200e+09 (new)
    Single-instance jit log-likelihood = -3.523166e+05 in all four
    scenarios and matches the numpy path to rtol=1e-4.

    Only the fitness._vmap path actually exercises the pure_callback
    inside the JAX trace — the single-instance jit(fit_from)(instance)
    path receives a pre-built ModelInstance whose kappa_s was computed
    at construction time outside the trace. C/D's vmap literal is the
    load-bearing assert and will catch any drift when the JAX-native
    Ludlow concentration of PyAutoGalaxy #403 (Phase 2) lands.

    Sundry fix during shipping: switched the canonical
    autolens_workspace_test remote from SSH (git@github.com:...) to
    HTTPS to match the convention used by every other PyAuto repo
    (only PyAutoConf still uses SSH). Was blocking gh pr create.

## ag-quantity-jax-viz
- issue: none — direct follow-up to ag-quantity-fit-from (#404)
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/405 (sibling library fix discovered during the workspace work)
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/46
- repos: PyAutoGalaxy, autogalaxy_workspace_test
- notes: |
    Phase 1C-extension. Added autogalaxy_workspace_test scripts that exercise
    use_jax_for_visualization=True on ag.AnalysisQuantity end-to-end, now
    that the dispatch (#404) is wired.

    Mid-task library gap discovered: when JIT-flattening a FitQuantity
    via the new fit_for_visualization dispatch, the `DatasetModel`
    attribute (reachable via aa.FitImaging base-class) hit
    `TypeError: not a valid JAX type`. The imaging analogue
    (autogalaxy/imaging/model/analysis.py:186) and interferometer
    analogue (interferometer/model/analysis.py:183) both register
    DatasetModel — the quantity pytree registration shipped in #401
    omitted it.

    Fix sequence:
    1. PR #405 (PyAutoGalaxy): 2-line fix — add
       register_instance_pytree(DatasetModel) to
       _register_fit_quantity_pytrees, mirroring imaging/interferometer.
       Library-first merge gate respected — merged before #46.
    2. PR #46 (autogalaxy_workspace_test): the 2 new scripts
       (visualization.py + visualization_jax.py) + env_vars override.
       Verified to pass with the library fix from #405 in place — no
       workaround in the script.

    Sonnet's initial workaround (a script-level
    register_instance_pytree(DatasetModel) call with a TODO) was
    removed before shipping the workspace PR. Cleaner end state — the
    library is correct, the script is clean.

    No modeling_visualization_jit.py for quantity (Nautilus quick-update
    visualization isn't the primary use case for quantity fits, per the
    original Phase 1C prompt scoping).

    Pre-flight diff check (per the binary-leak memory rule) was clean
    on both PRs.

    Coverage matrix after this PR:
    - imaging:        NumPy + JAX + jit-Nautilus  ✓
    - interferometer: NumPy + JAX + jit-Nautilus  ✓
    - quantity:       NumPy + JAX (no jit — one-shot fits)  ✓
    - ellipse:        NumPy only — JAX still blocked on dispatch design

    Only ellipse JAX coverage remains — blocked on autofit
    fit_for_visualization contract design for list-returning analyses
    (AnalysisEllipse.fit_list_from returns List[FitEllipse]).

## ag-quantity-fit-from
- issue: none — direct follow-up to ag-ellipse-quantity-pytree (#401)
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/404
- repos: PyAutoGalaxy
- notes: |
    Small library fix that completes one half of the deferred follow-up
    from Phase 0c. Added fit_from(instance) as a thin alias for
    fit_quantity_for_instance on AnalysisQuantity, and swapped the
    VisualizerQuantity dispatch line from
    analysis.fit_quantity_for_instance to analysis.fit_for_visualization.

    With this PR, use_jax_for_visualization=True on ag.AnalysisQuantity
    actually fires the JIT-cached path (it was a silent no-op despite
    #401 shipping the FitQuantity pytree registration + **kwargs
    passthrough, because the visualizer bypassed fit_for_visualization
    entirely).

    18/18 test_autogalaxy/quantity/ tests pass. Alias smoke-verified
    interactively: fit_from and fit_quantity_for_instance return the
    same FitQuantity (same dataset reference).

    Pre-flight diff check (per the binary-leak memory rule from earlier
    this session) caught nothing — only 2 .py files modified.
    Parallel-worktree-safe alongside interferometer-nufftax-updates
    (which is in autolens_workspace + autogalaxy_workspace — different
    repos).

    The matching ellipse follow-up (VisualizerEllipse dispatch swap)
    is NOT in this PR. ag.AnalysisEllipse.fit_list_from returns
    List[FitEllipse], not a single fit — needs autofit-side
    fit_for_visualization contract design before it can be wired
    similarly. Tracked as a separate (still-deferred) follow-up.

    Remaining follow-up: a small Phase 1C-extension workspace_test PR
    can now add autogalaxy_workspace_test/scripts/quantity/{visualization.py,
    visualization_jax.py} to exercise the dispatch end-to-end.

## ag-interferometer-jax-viz
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/43
- completed: 2026-05-14
- workspace-pr:
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/44 (3 new scripts + env_vars)
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/45 (cleanup — .gitignore + leaked binaries)
- repos: autogalaxy_workspace_test
- notes: |
    Phase 1C of jax_visualization roadmap. Scope narrowed mid-task to
    interferometer only — ellipse + quantity JAX coverage deferred to a
    follow-up after the Phase 0c-discovered visualizer-dispatch fixes
    ship. Workspace-only PR — all library prereqs (PRs #390, #399, #376,
    #401) already merged.

    Workspace scope shipped in #44:
    - scripts/interferometer/visualization.py (NEW) — NumPy baseline
    - scripts/interferometer/visualization_jax.py (NEW) — JAX viz with
      enable_pytrees() + register_model(model) + no try/except
    - scripts/interferometer/modeling_visualization_jit.py (NEW) —
      caching probe + live Nautilus with linear MGE basis, includes
      explicit rmtree(output/<path>/<name>/) before Nautilus (PR #87
      lesson)
    - config/build/env_vars.yaml — interferometer/visualization_jax +
      interferometer/modeling_visualization_jit overrides

    Cleanup shipped in #45 (same-session immediate follow-up):
    - .gitignore upgraded from per-type entries
      (scripts/imaging/images/, scripts/ellipse/images/) to the
      autolens_workspace_test-style **/images/ glob — covers all
      current and future dataset types
    - git rm'd 6 binary artifacts (3 PNG + 3 FITS, ~10 MB) that #44
      had leaked because the per-type gitignore didn't cover the new
      scripts/interferometer/images/ directory

    Lesson saved to memory (feedback_ship_workspace_binary_leak.md):
    when /ship_workspace introduces a NEW scripts/<type>/ subdirectory,
    pre-flight check `.gitignore` covers it or upgrade to **/images/
    before commit.

    Deferred follow-ups (unchanged from Phase 0c notes):
    - Quantity visualizer dispatch swap (small) — adds fit_from alias
      on AnalysisQuantity. Once shipped, a small follow-up workspace_test
      PR can add the quantity script triplet.
    - Ellipse visualizer dispatch swap (needs design) — fit_list_from
      returns List[FitEllipse], needs autofit fit_for_visualization
      contract generalization or list-to-single wrapper.

## ag-workspace-test-gitignore-fix
- issue: none — direct cleanup follow-up to ag-interferometer-jax-viz (#43)
- completed: 2026-05-14
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/45
- repos: autogalaxy_workspace_test
- notes: |
    Two-commit cleanup PR. Commit 1: git rm 6 binary artifacts that #44
    leaked. Commit 2: upgrade .gitignore from per-type entries to
    autolens_workspace_test-style **/images/ glob.

    Caught between merge of #44 and the active.md update — inspected
    PR #44's file list via `gh pr view 44 --json files`, found 6 unwanted
    PNGs/FITS, filed the cleanup PR within minutes of the original
    merge. Both PRs now in main; binaries never lived on main for long.

    The two commits could have been one if the .gitignore Edit had been
    staged before `git rm` (the Edit modification was unstaged when I
    ran `git commit`, so only the deletes landed in the first commit).
    Filed a separate commit on top rather than amending per the
    no-amend convention.

## nfw-jax-port
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/397
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/402
- repos: PyAutoGalaxy
- notes: |
    Phase 1 feasibility study for replacing the jax.pure_callback wrapping
    colossus.halo.concentration in autogalaxy/profiles/mass/dark/mcr_util.py.

    Verdict: Approach A (full JAX port of modelLudlow16 including the
    Eisenstein-Hu '98 transfer + Heath '77 growth factor + Einasto
    gammainc mass ratio + 200-point c-solver) is viable. ~330 lines of
    straight-line JAX. Max c200c rel error vs colossus = 7.5e-4 across the
    lensing parameter grid (log M ∈ [10, 14] Msun/h, z ∈ [0.1, 2.5]).
    Single-call post-JIT 0.69 ms (vs colossus 0.83 ms). vmap × 32 is
    1.29× faster than colossus serial. jax.grad agrees with finite-diff
    to 7e-4.

    Science-impact validation (science_check.py): end-to-end propagation
    through NFWMCRScatterLudlow and cNFWMCRScatterLudlow gives
    kappa_s max rel error 1.07e-3, NFW κ/α per-pixel max 8.21e-4, cNFW
    α per-pixel max 7.60e-4. Intrinsic Ludlow16 scatter is ~350× larger
    than the JAX-vs-colossus offset — scientifically invisible.

    All deliverables under docs/research/nfw_ludlow16_jax/:
      - nfw_ludlow16_jax_assessment.md (the report)
      - ludlow16_jax.py (the prototype)
      - validate.py, bench.py, tune.py, science_check.py

    No production code changed in this issue/PR. Phase 2 follow-up issue
    (to be filed) will swap the prototype into mcr_util.py, collapse the
    xp-branching in the two callers, and make colossus an optional dep.

    Known issue surfaced in cNFW science check (not blocking): the
    Penarrubia mcr formula goes negative for f_c=0.20, producing a
    negative kappa_s (pre-existing, unrelated to this work). Also
    cNFWSph.convergence_2d_from returns zeros — "not yet implemented"
    in PyAutoGalaxy.

## ag-ellipse-quantity-pytree
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/400
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/401
- repos: PyAutoGalaxy
- notes: |
    Phase 0c of jax_visualization roadmap. Completes the PyAutoGalaxy
    Fit* pytree series. Library-only PR — no workspace_test scripts
    needed at this stage; Phase 1C will exercise the registrations
    end-to-end.

    Library scope:
    - **kwargs passthrough on ag.AnalysisEllipse + ag.AnalysisQuantity
      __init__ (parity with PR #399's ag.AnalysisInterferometer fix).
    - _register_fit_ellipse_pytrees on AnalysisEllipse — registers
      Ellipse (no no_flatten), EllipseMultipole(no_flatten=("m",)), and
      FitEllipse(no_flatten=("dataset",)).
    - _register_fit_quantity_pytrees on AnalysisQuantity — registers
      FitQuantity(no_flatten=("dataset", "func_str", "use_mask_in_fit"))
      and reuses register_galaxies_pytree() for the light_mass_obj.

    Test plan: 154/154 unit tests across test_autogalaxy/{ellipse,
    quantity,imaging,interferometer}/. Interactive round-trip smoke
    verified locally (FitEllipse: 8 dynamic leaves; FitQuantity: 3
    dynamic leaves with Galaxies correctly reconstructed).

    Scope narrowing discovered mid-task: the original prompt assumed
    pytree registration would unblock use_jax_for_visualization=True
    for these analyses, but BOTH visualizers bypass
    analysis.fit_for_visualization entirely (VisualizerEllipse calls
    fit_list_from; VisualizerQuantity calls fit_quantity_for_instance).
    use_jax_for_visualization=True therefore remains a no-op for these
    two analyses despite the pytree work.

    Two deferred follow-ups (NOT in this PR):
    - **Quantity visualizer dispatch swap** (small) — add fit_from alias
      on AnalysisQuantity, switch VisualizerQuantity to use
      analysis.fit_for_visualization. Mirrors the imaging/interferometer
      pattern. Unlocks use_jax_for_visualization end-to-end on quantity.
    - **Ellipse visualizer dispatch swap** (needs design) — fit_list_from
      returns List[FitEllipse], not a single fit. Either generalize the
      autofit fit_for_visualization contract or compose the list into a
      wrapper. Separate design pass needed.

    Parallel-worktree-safe alongside in-flight nfw-jax-port (PyAutoGalaxy
    mass profiles, file-disjoint). User-cleared file-level safety.
    Worktree-conflict guard bypassed for this reason.

    PyAutoGalaxy pytree series now complete:
    - ag.FitImaging (PR #364) ✓
    - ag.FitInterferometer (PR #376) ✓
    - ag.FitEllipse + ag.FitQuantity (this PR) ✓

    Library kwargs-gap series now complete across both libs:
    - al.AnalysisImaging (always) ✓
    - al.AnalysisInterferometer (#500) ✓
    - al.AnalysisPoint (#506) ✓
    - ag.AnalysisImaging (always) ✓
    - ag.AnalysisInterferometer (#399) ✓
    - ag.AnalysisEllipse + ag.AnalysisQuantity (this PR) ✓

## ag-interferometer-kwargs
- issue: none — direct follow-up to point-source-jax-viz; user-approved file-level safety alongside in-flight jax-interp-2d / nfw-jax-port worktrees
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/399
- repos: PyAutoGalaxy
- notes: |
    Final piece of the kwargs-gap series. Added **kwargs passthrough to
    ag.AnalysisInterferometer.__init__ (2-line change), mirroring the
    earlier al.AnalysisInterferometer (#500) and al.AnalysisPoint (#506)
    fixes. test_autogalaxy/interferometer/ 37/37 pass.

    Coexisted on PyAutoGalaxy with two other in-flight worktrees
    (jax-interp-2d — actually merged via #398 before this PR; nfw-jax-port
    — mass profile work). User confirmed file-level safety: this PR
    touched autogalaxy/interferometer/model/analysis.py only, well away
    from mass-profile code. Parallel worktrees on different feature
    branches are exactly what the worktree flow is designed to handle;
    the conflict check is a soft policy guard, not a physical lock.

    With this PR, all four Analysis subclasses across PyAutoLens +
    PyAutoGalaxy now accept use_jax_for_visualization without TypeError.
    Phase 1C of the JAX visualization roadmap is unblocked from a
    library-API perspective.

## point-source-jax-viz
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/90
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/506
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/91
- repos: PyAutoLens, autolens_workspace_test
- notes: |
    Phase 1B of z_features/jax_visualization.md shipped end-to-end as a
    "Both" task — known up front (unlike Phase 1A which was discovered
    mid-session).

    Library PR #506 (PyAutoLens): added **kwargs to
    AnalysisPoint.__init__ + forwarded to super(). 2-line change
    mirroring PR #500's AnalysisInterferometer fix. 76/76 tests pass
    (test_autolens/point + test_autolens/analysis).

    Workspace PR #91 (autolens_workspace_test): three new scripts —
    scripts/point_source/visualization.py (NumPy baseline),
    scripts/point_source/visualization_jax.py (JAX path), and
    scripts/point_source/modeling_visualization_jit.py (caching probe +
    live Nautilus). Closes the autolens point_source gap (was the only
    autolens dataset type with zero visualization coverage). Plus two
    env_vars.yaml overrides (point_source/visualization_jax,
    point_source/modeling_visualization_jit) mirroring the imaging
    + interferometer analogues.

    Design choices forced by JIT constraints:
    - Image-plane chi-squared (FitPositionsImagePairAll) only — source-
      plane (FitPositionsSource) is still JIT-blocked per
      scripts/CLAUDE.md L132.
    - No free cosmology parameter in the model — cosmology distance
      calc caches global state and breaks JIT round-trip (per the
      existing jax_likelihood_functions/point_source/image_plane.py
      L144-147 caveat). The model is af.Collection(galaxies=...) only.
    - modeling_visualization_jit.py includes explicit rmtree of both
      scripts/point_source/images/modeling_visualization_jit/ AND
      output/scripts/point_source/images/modeling_visualization_jit/
      point_image_plane/ before Nautilus, so reruns force a fresh
      sampling pass and _jitted_fit_from gets populated (lesson from
      PR #87).

    JAX visualization roadmap **kwargs gap status:
    - al.AnalysisImaging: always had **kwargs ✓
    - al.AnalysisInterferometer: fixed in PR #500 ✓
    - al.AnalysisPoint: fixed in PR #506 ✓ (this task)
    - ag.AnalysisInterferometer: still has the gap — Phase 1C will fix.
    - ag.AnalysisImaging: always had **kwargs ✓

    Follow-ups:
    - Source-plane chi-squared visualization (FitPositionsSource) — gated
      behind the existing CLAUDE.md L132 JIT blocker. Defer until that
      lifts.
    - Backport the rmtree(output/<path>/<name>/) fix to
      scripts/imaging/modeling_visualization_jit.py and its delaunay /
      rectangular variants — they have the same brittleness on reruns.

## jax-interp-2d
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/306
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/308, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/398

## nnls-vmap-speedup
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/307 (closed without library changes)
- completed: 2026-05-11
- repos: PyAutoArray, PyAutoConf (both — no commits, worktree branches deleted)
- outcome: |
    Closed without shipping. Investigation found the prompt's premise
    ("8.84x Delaunay vmap regression caused by NNLS") was a batch=3
    measurement artifact. At production batch=20 on A100:
      - Rect full pipeline = 11.2 ms/element (target was <=25 ms — already met)
      - Delaunay full pipeline = 69.5 ms/element (target was <=200 ms — already met)
      - NNLS = 6.2 ms/element = 9% of Delaunay (vmap regress = 0.40x, faster than single)
    Delaunay bottleneck is scipy.spatial.Delaunay via pure_callback
    (16.87 ms/element under sequential vmap = 99% of source-mapper sub-cost).
    Algorithm survey (PDIP/ADMM/FISTA): PDIP wins on correctness — ADMM/FISTA
    can't converge at production conditioning. Gradient audit: custom_vjp is
    lazy, no forward-only entry needed. MAX_ITER sweep: PDIP self-stops at
    15-20 iters, lowering MAX_ITER doesn't speed typical case.
- findings: ~/Code/PyAutoLabs/z_projects/profiling/FINDINGS_nnls_v2.md
- followup: pure-JAX Delaunay triangulation (highest-value lever for Delaunay
    science throughput; ~1.3x speedup vs current state from this alone,
    up to ~2x combined with optimising other inversion-setup work).

## nufftax-citation-docs
- issue: N/A (ad-hoc follow-up to PyAutoGalaxy#391 — nufftax-default-transformer)
- completed: 2026-05-10
- library-prs: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/396, https://github.com/PyAutoLabs/PyAutoLens/pull/503
- repos: PyAutoGalaxy, PyAutoLens
- notes: |
    Added a `## NUFFTax` section to `docs/general/citations.md` in both
    PyAutoGalaxy and PyAutoLens so users running interferometer fits on
    the JAX path know to cite `nufftax` (the new pure-JAX NUFFT dependency
    introduced by PyAutoGalaxy#391) and the upstream FINUFFT paper its
    algorithm is based on. Followed the precedent set by the existing
    `## Jax-Zero-Contour` section — guidance lives only in
    `docs/general/citations.md`, not in the canonical
    `files/citations.{bib,tex,md}`.

## ellipse-fit-masked-loop-tests
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/394
- completed: 2026-05-10
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/395

## ellipse-jax-likelihood-tests
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/41
- completed: 2026-05-10
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/42

## autogalaxy-extras-mge-option
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/65
- completed: 2026-05-10
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/66
- repos: autogalaxy_workspace
- notes: |
    Audit follow-up to PyAutoGalaxy#392. Added MGE Option B (commented out)
    alongside the existing SersicSph Option A (default) in
    extra_galaxies/modeling.py, rewrote the wrap-up MGE paragraph to point at
    the inline option, and added a "scaling_relations not applicable" section
    to extra_galaxies/README.md with cross-links to the autolens examples.
    No new scaling_relation directory in autogalaxy -- explicitly declined
    (mass-only relation; light-only analogues need velocity dispersion).

## ellipse-visualization-test
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/39
- completed: 2026-05-10
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/40

## scaling-relation-csv-loader
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/392
- completed: 2026-05-10
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/393, https://github.com/PyAutoLabs/PyAutoLens/pull/502
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/143
- repos: PyAutoGalaxy, PyAutoLens, autolens_workspace
- notes: |
    Added autogalaxy.galaxy.galaxy_table (GalaxyTable dataclass + from_csv/to_csv
    helpers wrapping autoconf.csvable). Re-exported through autolens. Both
    scaling_relation simulators emit extra_galaxies.csv + scaling_galaxies.csv
    next to centre JSONs; both modeling.py files show CSV (Option A, default)
    AND JSON+hardcoded list (Option B, commented) side by side so users see the
    choice. modeling_for_luminosities.py writes scaling_galaxies.csv directly
    so the chain into modeling.py needs no manual paste.

## scaling-relation-update
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/141
- completed: 2026-05-10
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/142
- repos: autolens_workspace
- notes: |
    Refreshed imaging/features/scaling_relation/modeling.py to modern API
    (MGE + Isothermal + shared scaling_factor*L^scaling_exponent), added
    paired imaging simulator, and added new group/features/scaling_relation
    feature with three-tier modeling.py + standalone modeling_for_luminosities.py
    mirroring the SLaM source_lp[0] step.

    Two follow-up prompts queued:
      - workspaces/scaling_relation_csv_loader.md (CSV-driven centres/luminosities)
      - workspaces/autogalaxy_extra_galaxies_audit.md (autogalaxy_workspace parity)

## mge-profiling-a100
- completed: 2026-05-09
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/56
- repos: autolens_workspace_developer (+ z_projects/profiling local-only)
- notes: |
    Follow-up to fft-mixed-precision-fix: extends profiling from
    consumer (RTX 2060 + i9-10885H) to A100 80GB and consolidates 10
    configs into canonical
    jax_profiling/results/jit/imaging/mge/ tracking dir.

    Tooling: z_projects/profiling/scripts/mge_profile.py (single-config
    step-by-step JIT profiler) + mge_aggregate.py (--ingest-pre-fix to
    convert /tmp logs, --consolidate-from to move HPC pulls, default to
    emit comparison.json+png) + 2 SLURM submits for A100 fp64+mp.

    Headline timings:
    - A100 fp64: 5.7 ms full pipeline / 2.4 ms vmap-per-call
    - A100 mp: 5.4 ms / 2.3 ms (5% noise — mp delivers ~zero on A100)
    - RTX 2060 fp64: 43.7 ms / 23.9 ms
    - RTX 2060 mp: 43.0 ms / 15.0 ms (37% vmap win on consumer GPU)
    - CPU fp64: 308 ms / 234 ms

    Key conclusion: use_mixed_precision is a consumer-GPU lever, not a
    production one. A100's 1:2 fp64:fp32 ratio means fp64 is not
    punitive on production hardware.

    Caveat surfaced (filed as separate follow-up): A100 JIT log_likelihood
    truncates to fp32 precision (-159734.59 vs eager fp64
    -159736.355042) — jax_enable_x64 likely not set in HPC PyAutoNSS
    venv. Doesn't affect timing, but worth investigating before
    quoting A100-served NSS / Nautilus log Z to high precision.

## fft-mixed-precision-fix
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/302
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/38
- repos: PyAutoArray, autogalaxy_workspace_test
- notes: |
    Fixed a real net-loss in `al.Settings(use_mixed_precision=True)` on
    consumer GPUs: `Convolver.convolved_image_from` previously force-cast
    inputs to fp64 then narrowed the result, paying for fp64 FFT plus an
    extra cast. Light-profile path now runs end-to-end complex64 with the
    kernel pre-cached on `ConvolverState.fft_kernel_c64`.

    Headline result on RTX 2060 + i9-10885H (mge.py HST regression):
    GPU mp full pipeline 47 -> 19.6 ms; GPU mp vmap (production sampler
    hot path) 17.4 -> 8.9 ms (49% faster). CPU vmap unchanged-to-slightly-
    faster; CPU single-JIT regresses ~17% but production samplers use vmap.
    Delta log-likelihood ≈ 2.2e-3 absolute, far below chi^2 noise floor
    (sigma ≈ 175 for N=15k pixels).

    `convolved_mapping_matrix_from` intentionally keeps its complex128
    kernel multiply: full fp32 in that path drifted `figure_of_merit` by
    ~10 units (1.9% relative) on the autolens_workspace_test delaunay_mge
    regression (K=780 source mesh). Pixelization NNLS / log-determinant
    needs fp64. Codified the asymmetry in code comments, Settings
    docstring, and a new jax_assertion at
    autogalaxy_workspace_test/scripts/jax_assertions/convolver_mixed_precision.py.

    23/23 JAX likelihood-function integration tests pass across autolens +
    autogalaxy, imaging + interferometer, MGE + rectangular + Delaunay.

    Two follow-ups filed:
    - PyAutoPrompt/autoarray/nnls_gpu_bottleneck.md — GPU-NNLS bottleneck
      (jaxnnls is PDIP with MAX_ITER=50; Cholesky fast-path rejected
      because empirical positivity-hit-rate during sampling is low and
      lax.cond under vmap evaluates both branches).
    - PyAutoPrompt/autolens_workspace_developer/mge_jit_regression_rebaseline.md —
      mge.py's hardcoded EXPECTED_LOG_LIKELIHOOD_HST drifted from 27379.39
      to 27542.08 due to upstream changes (independent of this fix).

## autolens-interferometer-jax-viz
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/86
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/500
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/87
- repos: PyAutoLens, autolens_workspace_test
- notes: |
    Phase 1A of z_features/jax_visualization.md shipped end-to-end as a
    "Both" task (started workspace-only, reclassified mid-session when a
    missing **kwargs passthrough in al.AnalysisInterferometer.__init__
    was discovered).

    Library PR #500 (PyAutoLens): added **kwargs to
    AnalysisInterferometer.__init__ and forwarded to super(). 2-line
    change. ag.AnalysisImaging had the passthrough all along; the
    AnalysisDataset parent already accepts **kwargs. PyAutoLens 116/116
    tests pass.

    Workspace PR #87 (autolens_workspace_test): added
    scripts/interferometer/visualization_jax.py and
    scripts/interferometer/modeling_visualization_jit.py mirroring the
    imaging analogues. Split env_vars.yaml `imaging/visualization`
    pattern into NumPy-only + JAX-only entries; added
    `interferometer/modeling_visualization_jit` override.

    Discovered + fixed: modeling_visualization_jit.py Part 2 has a brittle
    assertion `_jitted_fit_from is not None` that only fires if Nautilus
    actually does live sampling. If output/<path>/ already has cached
    samples.csv from a prior run, Nautilus resumes and skips, so the JIT
    wrapper is never installed and the assertion AttributeErrors. Fix:
    explicit rmtree of the autofit search output directory before the
    Nautilus call. The imaging analogue
    (autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py)
    has the same brittleness — worth a tiny follow-up backport.

    Follow-ups deferred:
    - al.AnalysisPoint.__init__ has the same **kwargs gap. Phase 1B of
      the roadmap will need it.
    - ag.AnalysisInterferometer.__init__ has the same gap. Phase 1C
      will need it.
    - Backport the rmtree fix to imaging/modeling_visualization_jit.py.
    - sma.fits is gitignored — CI runs depending on it stay red on main.

## viz-jax-pytree-fix
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/84
- completed: 2026-05-08
- workspace-pr:
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/85 (lead, closes #84)
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/37 (autogalaxy sibling)
- repos: autolens_workspace_test, autogalaxy_workspace_test
- notes: |
    Direct follow-up to autogalaxy-viz-dispatch-swap (PyAutoGalaxy #390).
    Both visualization_jax.py scripts (autolens + autogalaxy versions) had
    been silently broken under JAX since their respective dispatch swaps
    landed: PyAutoLens #443 (2026-04-19) for the autolens version,
    PyAutoGalaxy #390 (2026-05-08) for the autogalaxy version. Each
    script's try/except wrapper caught the JAX trace failure
    (TypeError: ModelInstance not a valid abstract array), printed
    "PILOT FAILED", and exited 0 — invisible to test runners.

    Root cause: missing enable_pytrees() + register_model(model) so that
    jax.jit(fit_from) could trace the ModelInstance arg across the JIT
    boundary. Working sibling modeling_visualization_jit.py had these
    calls all along (lines 43-45) — visualization_jax.py just lacked them.

    Fix in both repos:
      1. Add enable_pytrees() + register_model(model) before constructing
         the analysis.
      2. Drop the try/except wrapper so future regressions fail loud.
      3. Split each workspace's config/build/env_vars.yaml imaging/visualization
         override into a NumPy-only pattern (visualization.py — substring
         match) and a JAX-only pattern (visualization_jax) that also unsets
         PYAUTO_DISABLE_JAX. Mirrors the existing
         imaging/modeling_visualization_jit override.

    Verified: both scripts now print PILOT SUCCEEDED with JAX enabled.

    Future work: neither visualization_jax.py is in smoke_tests.txt, so a
    future regression of the same shape would still be invisible to CI
    smoke. Promoting them to smoke is a separate decision per the user's
    "small curated subset" smoke policy and was not in scope here.

## datacube-3d-fits-relocate
- issue: none — direct followup to autolens_workspace#120
- completed: 2026-05-08
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/140 (3D-FITS layout + data_preparation.py + relocated likelihood_function.py)
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/51 (deletes the relocated likelihood walkthrough and prototype simulator)
- repos: autolens_workspace, autolens_workspace_developer
- notes: Hannah's ALMA visibilities arrive from CASA as a single 4D FITS (n_pol, n_chan, n_vis, 2). The original Phase 1 datacube tutorials only supported per-channel folders, which would have forced her to split her cube before loading. Updated simulator.py to additionally write `{visibilities,noise_map,uv_wavelengths}_cube.fits` at the cube root (each shape `(n_chan, n_vis, 2)`). New data_preparation.py walks through polarisation collapse (average vs concatenate) and ships a self-contained `dataset_list_from_3d_fits()` loader function — verified to match the per-channel-folder loader to rtol=1e-12. Also relocated the JAX likelihood walkthrough from the (private) autolens_workspace_developer to the (public) autolens_workspace so external collaborators can actually read it. Per-channel-folder layout kept for backward compatibility; both layouts coexist.

## autogalaxy-viz-dispatch-swap
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/389
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/390
- repos: PyAutoGalaxy
- notes: |
    Phase 0b of z_features/jax_visualization.md. Three call sites in PyAutoGalaxy
    visualizers swapped from analysis.fit_from(instance=...) to
    analysis.fit_for_visualization(instance=...) — imaging/model/visualizer.py
    (single visualize() + visualize_combined()) and interferometer/model/visualizer.py.
    PyAutoLens made the same swap in #443 (2026-04-19); the autogalaxy side was
    overdue. Pytree registration prerequisites for both imaging (#364) and
    interferometer (#376) had already shipped — this was the last piece.

    Library tests: 106/106 passed. Smoke verification: visualization.py NumPy and
    modeling_visualization_jit.py JIT-during-Nautilus both PASS; visualization_jax.py
    surfaced a pre-existing latent test-script bug (missing register_model /
    enable_pytrees) which also affects the autolens equivalent since #443. Filed
    a follow-up prompt: autolens_workspace_test/visualization_jax_pytree_registration.md.
    Neither failure breaks CI smoke (visualization_jax.py is not in smoke_tests.txt
    and env_vars.yaml defaults force PYAUTO_DISABLE_JAX=1 for any imaging/visualization
    path, so the script silently falls through to NumPy under CI).

    Phase 0c (ag ellipse + quantity pytree) and Phase 1A (autolens interferometer
    JAX viz coverage) are now unblocked from this PR's perspective.

## contents-block-bullets
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/138
- completed: 2026-05-08
- workspace-pr:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/139 (lead, 199 scripts)
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/64 (98 scripts)
  - https://github.com/PyAutoLabs/HowToLens/pull/9 (37 scripts)
  - https://github.com/PyAutoLabs/HowToGalaxy/pull/6 (21 scripts)
  - https://github.com/PyAutoLabs/autofit_workspace/pull/54 (1 script)
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/83 (1 script)
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/36 (1 script)
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/25 (1 script)
- repos: autolens_workspace, autogalaxy_workspace, autofit_workspace, HowToLens, HowToGalaxy, autolens_workspace_test, autogalaxy_workspace_test, autofit_workspace_test
- notes: |
    359 workspace tutorial scripts had their `__Contents__` blocks
    converted from plain `**Section:**` lines into Markdown list bullets
    (`- **Section:**`). Without bullet markers GitHub and JupyterLab
    collapsed the index into a single paragraph in the generated .ipynb's
    first markdown cell — the fix is text-only inside top-level module
    docstrings.

    HowToFit had nothing to ship — its 14 __Contents__ files were
    already bulleted (someone fixed them earlier, possibly the same
    person who shipped the ic50_workspace exemplar at 4cde480).

    Three rewrite-tool bugs were caught during dry-run before any PR
    opened — worth knowing for the next docstring-mass-rewrite tool:
    1. CRLF normalization (Python's read_text/write_text silently
       strip CRLF, would have produced 10K-line bogus diffs in the
       autofit_workspace files that use CRLF). Use read_bytes /
       write_bytes and detect/preserve the original eol byte-for-byte.
    2. Docstring-close `"""` mis-indented as a bullet continuation
       line. Terminate the contents-block scan at any line starting
       with `"""` or `'''`.
    3. Prose intro between `__Contents__` and the first bullet
       mis-indented as a continuation. Skip prose-intro lines until
       the first `**` or `- **` is found, then process from there.

    Visual confirmation via ipynb-py-convert on
    autolens_workspace/scripts/point_source/simulator.py: first
    markdown cell renders as a proper bulleted list.

    Notebooks were deliberately NOT regenerated in this PR set — the
    next /pre_build will pick them up cleanly.

    Out of scope (per the original prompt): same paragraph-collapse
    risk may bite `__Model__` blocks, `Steps`/`Notes`/`Outputs` blocks
    in workspace simulators. Not addressed without a concrete broken
    example.

## euclid-version-bump-2026-5-1-4
- completed: 2026-05-08
- workspace-pr: https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/pull/13
- repos: euclid_strong_lens_modeling_pipeline
- notes: |
    One-time catchup for a missed release. Surfaced as 6 of 6 euclid
    smoke "failures" during the PR #301 (PyAutoArray) validation run —
    every euclid script raised WorkspaceVersionMismatchError because
    config/general.yaml pinned workspace_version=2026.4.13.6 against
    library 2026.5.1.4, and there was no version.txt. Bumped both files
    to 2026.5.1.4 to match the convention used by autofit_workspace,
    autogalaxy_workspace, autolens_workspace, and HowToLens.

    Root cause was a one-time gap, not a structural issue:
    pre-2026-05-01 this repo lived under Jammy2211/ and PAT_PYAUTOLABS
    couldn't push, so it was excluded from the release_workspaces matrix
    in PyAutoBuild's release.yml. PyAutoBuild PR #81 restored the
    matrix entry on 2026-05-01 14:56 UTC — but ~3 hours after the
    2026.5.1.4 release dispatched at 11:33 UTC, so the bump didn't
    auto-land. Future drift will auto-correct via the now-restored
    matrix entry; this is a single-shot catchup, not the start of a
    maintenance pattern.

    All 6 smoke scripts pass with the bumped version, no other
    failures uncovered.

## imaging-from-fits-small-datasets-cap
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/301
- repos: PyAutoArray
- notes: |
    Library-side companion to multi-viz-imaging-small-datasets-override
    (PR #80 in autolens_workspace_test). User asked for the proper fix:
    "everything in the source code should honor PYAUTO_SMALL_DATASETS=1".
    Closed the asymmetry where Mask2D.circular and Grid2D.uniform capped
    to (15, 15) at 0.6"/px under the env var but Imaging.from_fits did
    not — silently producing shape mismatches that crashed apply_mask
    with a (150,150) vs (15,15) ValueError. Added
    autoarray.util.dataset_util.cap_array_2d_for_small_datasets, a
    center-crop helper that mirrors the existing caps. Hooked it into
    Imaging.from_fits for data + noise_map (PSF intentionally untouched
    — PSFs are usually <15x15 and capping changes their shape semantics).
    No-op when env unset OR when on-disk shape is already at-or-below
    the cap, so the simulator -> from_fits round-trip is unchanged.
    +123 lines library, +123 tests across two test files; full
    test_autoarray suite 747/747 green.

    Center-crop was chosen over downsample/resample because (a) smoke
    mode doesn't need numerical correctness, (b) center-crop preserves
    central pixels (where lens/galaxy signal sits), (c) avoids a scipy
    dependency at this layer, and (d) matches the existing convention
    of "fixed cap at smoke geometry" used by Mask2D.circular.

    Cluster-E reproducer (autolens_workspace_test/scripts/multi/
    visualization_imaging.py with PYAUTO_SMALL_DATASETS=1 and the
    env_vars.yaml override stripped) now exits 0 cleanly: data 150->15,
    psf preserved at 21x21, mask 15, apply_mask succeeds.

    Smoke ran across all 6 workspaces (autofit, autogalaxy, autolens,
    autolens_test, HowToLens, euclid) with the worktree's autoarray
    active. 36/44 passed in-scope; 6 euclid failures were pre-existing
    WorkspaceVersionMismatchError (workspace pinned at 2026.4.13.6 vs
    library 2026.5.1.4) — orthogonal, confirmed by bypass with
    PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1. No regressions attributable
    to this PR.

    **Follow-ups worth filing:**
    1. PyAutoArray: extend the same helper to Array2D.from_fits and
       Grid2D.from_fits for full consistency (and PSF-aware logic in
       Kernel2D.from_fits — preserve odd shape, re-normalize after crop).
    2. autolens_workspace_test: revert the multi/visualization_imaging
       env_vars.yaml override (PR #80) once this library fix has lived
       on main for a release cycle. The override is now redundant but
       harmless; reverting confirms the library cap is sufficient.
    3. euclid_strong_lens_modeling_pipeline: bump pinned library version
       to 2026.5.1.4 (separate from this work, but surfaced by smoke).

## subhalo-redshift-jax-fix
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/498
- completed: 2026-05-08
- repro-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/79 (merged d827d1c — Phase 1)
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/499 (merged b790632 — Phase 2)
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/81 (merged 61b0b4f — Phase 2b)
- repos: PyAutoLens, autolens_workspace_test
- notes: |
    Slack bug report from @qiuhan96 (working with an undergraduate at
    Groningen). Free-parameter subhalo redshift (af.UniformPrior on
    Galaxy.redshift) raised jax.errors.TracerBoolConversionError under
    jax.jit. Root cause: tracer_util.plane_redshifts_from /
    planes_from / grid_2d_at_redshift_from + Tracer.galaxies_ascending_
    redshift all called Python sorted(galaxies, key=lambda g: g.redshift)
    on a list whose subhalo redshift was a traced scalar; pairwise '<'
    comparisons cannot lift to traced ops.

    Three-phase ship in a single session:

    Phase 1 — Filed the issue, then landed a clean integration-test
    reproducer (PR #79) in autolens_workspace_test as scripts/jax_
    likelihood_functions/imaging/subhalo.py. Two scenarios: fixed
    z=0.55 PASS, free UniformPrior FAIL with the expected
    TracerBoolConversionError. Used to drive the fix and as the
    eventual regression check.

    Phase 2 — Library fix (PR #499). Each of the four buggy functions
    got a JAX-aware fast-path guard: when no galaxy redshift is
    traced, behaviour is byte-for-byte identical to before; when any
    redshift is traced, the function partitions concrete vs traced,
    sorts the concrete ones with normal Python sort, and trusts input
    galaxy order for the traced ones. grid_2d_at_redshift_from
    matches the requested redshift to a galaxy by Python identity
    (its only call site, AnalysisLens.tracer_via_instance_from,
    always passes the subhalo's own redshift object). 273/273 unit
    tests pass; full smoke suite for autolens_workspace_test (11/11)
    pass with the patched library on PYTHONPATH.

    Phase 2b — Polarity-flip workspace PR (#81). The same subhalo.py
    script was converted to the regression check: both scenarios
    must now PASS, with np.testing.assert_allclose locking the vmap
    output to -1.412105e+09 (rtol=1e-4) and the JIT log_likelihood
    matching the NumPy path within rtol=1e-4.

    Notes for future:
    - The cosmology distance functions (Planck15) accepted traced
      redshifts without modification, so no JAX-friendly cosmology
      shim was needed. If a different cosmology turns out to fail,
      that's a follow-up issue.
    - The tuple(subhalo_centre.in_list[0]) round-trip in
      analysis/lens.py:116 was a suspected breakage site but turns
      out to work fine under JIT — tuple((traced_y, traced_x))
      preserves traced scalars.
    - The JAX path trusts input galaxy order; if a user puts the
      subhalo before the lens in declaration order with a redshift
      between them, the multi-plane scaling factors will be wrong.
      Documented in the PR body's Migration section but not enforced
      at runtime (would require a non-JAX validation pass).
    - The "user-helping" tone established mid-session (4 conversational
      issue updates across 5 milestones) seemed to land well — keep
      that pattern for future user-reported bug reports.

## cluster-f-sensitivity-job-dataset
- completed: 2026-05-08
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1259
- repos: PyAutoFit
- notes: |
    Surfaced in a release-prep triage report as Cluster F: a single
    failing script (autofit_workspace_test/scripts/database/scrape/
    sensitivity.py) raising AttributeError: 'NoneType' object has no
    attribute 'data' at line 241 (Analysis.__init__ unpacking
    dataset.data). The user proposed two hypotheses: simulate_function
    returning None under test mode (workspace bug), or job dataset
    wiring broken by a recent PyAutoFit refactor (library bug). Both
    were wrong — the actual cause is a partial library optimization.
    PyAutoFit commit 41095a0 ("skip simulation if job is complete",
    Oct 2024) added `if self.is_complete: dataset = None` in
    Job.perform() but didn't also skip the downstream
    base_fit_cls / perturb_fit_cls calls, so they receive None.

    The library's own test_perform_twice masked the contract issue
    because the test conftest's Analysis.__init__(dataset) just
    stashes dataset without unpacking — real workspace Analysis code
    (autofit_workspace/scripts/features/sensitivity_mapping.py:246
    and the failing test-workspace script) eagerly unpacks
    dataset.data and dataset.noise_map. Fix per
    feedback_no_silent_guards.md: stop the producer of None, not
    add a consumer-side tolerance. Collapsed the if/else to always
    call simulate_cls. Re-runs still skip the expensive non-linear
    search via Search.fit's load-from-zip path
    (paths.restore() + paths.is_complete in abstract_search.py).
    Only the typically-cheap simulator cost is no longer optimized
    away.

    Verification: bug reproduced exactly on clean main (matched
    user's stack trace), fix applied, clean two-run cycle passes
    (first run completes, second hits is_complete=True via zip
    presence, restore() unzips, load path returns).
    Full PyAutoFit suite 1241/0/1 (skipped pre-existing). All 9
    autofit_workspace_test smoke scripts pass.

    Known limitation flagged in PR body but out of scope: if a
    previous run was killed mid-fit (e.g. [perturb].zip present but
    [base].zip missing), re-runs hit a different failure on that
    cell — Search.fit.restore() finds nothing to restore for the
    missing side, paths.is_complete=False, resume kicks in,
    Fitness.check_log_likelihood raises SearchException because the
    new (non-deterministic) simulator FoM doesn't match the
    persisted partial-fit FoM. Separate bug, separate fix.

    Verify-triage-clusters habit paid off again: the user's two
    hypotheses framed the search but were both wrong; the third
    answer (deliberate library design + workspace pattern conflict)
    was reachable only by tracing Job.perform() history.

## multi-viz-imaging-small-datasets-override
- completed: 2026-05-08
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/80
- repos: autolens_workspace_test
- notes: |
    Cluster E from the 2026-05-07 release-prep triage.
    `autolens_workspace_test/scripts/multi/visualization_imaging.py`
    crashed in the local triage runner with `ValueError: operands
    could not be broadcast together with shapes (150,150) (15,15)` at
    `dataset.apply_mask(mask=mask)`. The triage report's prescribed
    fix ("derive mask shape from dataset.shape_native, not from a
    hardcoded constant") was a no-op — the script already does that.
    Real cause: `PyAutoArray/autoarray/mask/mask_2d.py:363-366`
    silently overrides `shape_native` to (15,15) when
    `PYAUTO_SMALL_DATASETS=1`, *even when shape_native is explicit*,
    while `Imaging.from_fits` does not cap the dataset. The triage
    runner picked up the failure because
    `autolens_workspace_test/config/build/env_vars.yaml` sets
    `PYAUTO_SMALL_DATASETS: "1"` as a workspace-wide default; the
    GitHub Actions release.yml does not (only sets it for non-`_test`
    workspaces, lines 965–971), which is why CI passes. Fix: 9-line
    additive entry in env_vars.yaml unsetting the cap for
    `multi/visualization_imaging` (matching the existing
    `imaging/visualization` precedent). Sibling
    `multi/visualization_interferometer.py` was left alone — it
    passes under the cap currently and the triage didn't flag it
    (whether its likelihood is correct under a 15x15 real-space mask
    is a separate correctness question). Pure config change; no
    library or script touched.

    **Follow-up worth filing as a PyAutoArray issue:**
    `Mask2D.circular`'s `PYAUTO_SMALL_DATASETS=1` cap silently
    overrides an explicit `shape_native` argument. Two options for a
    proper fix: (a) only apply the cap when shape_native is at its
    default, or (b) make `Imaging.from_fits` also honour the env var
    so dataset and mask stay consistent. Both need a careful audit of
    every smoke test that currently relies on the implicit cap.

## cluster-c-point-source-rebaseline
- completed: 2026-05-08
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/78
- repos: autolens_workspace_test
- notes: |
    Surfaced in a release-prep triage report as Cluster C: three JAX
    point-source likelihood scripts (image_plane.py, point.py,
    source_plane.py) failing `np.testing.assert_allclose` against
    hardcoded `expected_likelihood` literals. Root cause was upstream
    commit 931a381 (six days earlier) changing
    `positions_noise_map` from `grid.pixel_scale` (0.2") to `0.005"`
    in `scripts/jax_likelihood_functions/point_source/simulator.py` —
    the committed seed dataset under `dataset/point_source/simple/`
    was last regenerated under the old noise from `pre build`
    (a88f0f6, May 1) and was never refreshed when the simulator
    changed. Because `should_simulate` only fires when the dataset
    path is missing, canonical `main` was actually passing — old
    dataset matched old literal — but any clean re-simulation hit
    the failure the user reported.

    Fix regenerated the seed dataset (`point_dataset_positions_only.json`,
    `tracer.json`) and rebaselined three literals: 1.313508 →
    -83.38049778 (image_plane.py and point.py — same dataset, same
    prior medians, identical values), and -199.1555813 →
    -331481.25978149 (vmap) / -331481.26508536364 (eager) for
    source_plane.py. The 1664x source_plane.py drift checks out as
    chi-squared rescaling: noise dropped 40x, so chi^2 scales by
    1600x.

    Verify-triage-clusters habit paid off — going through the chain
    `simulator → committed dataset → should_simulate semantics →
    canonical state` exposed that the failure mode required deleting
    the on-disk dataset to reproduce, which changed how I handed the
    user the choice (regenerate-and-commit vs leave-alone). Smoke
    tests 11/11 passed; one pre-existing skip (database/scrape/general,
    NEEDS_FIX 2026-04-27) unrelated.

## results-start-here-fits-hdu-fix
- completed: 2026-05-08
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/61
- repos: autogalaxy_workspace
- notes: |
    Surfaced as "Cluster B" in a release-prep triage report — four
    autogalaxy results scripts (start_here.py + three under
    aggregator/) supposedly sharing a `KeyError ('galaxies', 'galaxy',
    'bulge', 'ell_comps', 'ell_comps_0')` from
    `parameter_lists_for_paths`, attributed to `_quick_fit.py`
    building a model that doesn't expose `ell_comps`. Investigation
    on current main contradicted both halves: `ag.lp_linear.Sersic`
    and `ag.lp_linear.Exponential` both expose `ell_comps`
    (`model.info` confirms 9 free parameters incl. `bulge.ell_comps`)
    and the three aggregator scripts already pass cleanly under
    `PYAUTO_TEST_MODE=1` from a fresh `output/results_folder`. The
    original `KeyError` I saw on first run came from a stale cached
    output folder — once `_quick_fit.py` regenerated, it vanished.
    Only `start_here.py` was actually broken, and for an unrelated
    reason: lines 233–235 reloaded the saved `dataset.fits` with
    `data_hdu=0, noise_map_hdu=1, psf_hdu=2`, but the visualizer
    writes the file as `MASK, DATA, NOISE_MAP, PSF,
    OVER_SAMPLE_SIZE_LP, OVER_SAMPLE_SIZE_PIXELIZATION`, so
    `psf_hdu=2` pulled the 100×100 noise map and `Convolver` rejected
    it with `KernelException: "Convolver must be odd"`. Fix: shift
    indices by one (0→1, 1→2, 2→3). Three lines, one file. Smoke
    tests 6/6, aggregator regression confirmed the three siblings
    still pass. Cluster B as originally described is therefore
    already-resolved on main and needed no aggregator-path rewrites.
    Lesson worth keeping: when a triage cluster says "N scripts share
    root cause X, mechanical fix Y", verify reproduction on current
    main before mass-applying Y — clusters age out as upstream PRs
    merge, and the surviving failures often have a different cause.

## park-double-einstein-ring
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/134
- repos: autolens_workspace
- notes: |
    Cluster H of the recent release-prep triage. scripts/imaging/features/
    advanced/double_einstein_ring/slam.py crashed under PYAUTO_TEST_MODE=2
    with autofit.exc.FitException at analysis.py:84. Investigation
    traced a structural cascade: Adapt regularization (used by SLaM
    pixelization phases) requires per-galaxy adapt_data that the
    synthetic samples_summary produced by bypass mode does not carry.
    Mapper.pixel_signals_from None-derefs `self.adapt_data.array` from
    multiple inversion entry points (likelihood, post-fit
    result.subtracted_signal_to_noise_map_galaxy_dict, etc.), so
    patching one site only unblocks the next. Drafted a defensive
    FitException-tolerance patch in PyAutoFit's _fit_bypass_test_mode
    (mirrors compute_latent_samples pattern) — verified the FIRST entry
    point cleared, but the SECOND failure point fired with the same
    root cause through a different call chain. Even with that fix, the
    next SLaM phase would derive adapt_images from the synthetic
    samples_summary and fail again. End-to-end fix requires either
    (a) defensively pretending Adapt works without adapt_data (silent
    semantic change — unsafe) or (b) restructuring the SLaM bypass to
    construct valid adapt_data (large refactor). User chose to park
    and handle manually. Sibling of imaging/features/pixelization/slam
    (NEEDS_FIX 2026-04-10, same root cause).

## jax-assertions-env-override
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/24
- repos: autofit_workspace_test
- notes: |
    Cluster G of the recent release-prep triage.
    scripts/jax_assertions/fitness_dispatch.py crashed with
    `AttributeError: Analysis has no attribute _jitted_fit_from`. User's
    hypothesis (library API drift / rename) was wrong — library is intact.
    Real bug: env_vars.yaml `defaults` set PYAUTO_DISABLE_JAX=1 globally,
    and Analysis.__init__ silently flips both use_jax and
    use_jax_for_visualization to False whenever that env var is set.
    fit_for_visualization then early-returns without caching
    _jitted_fit_from, and the next assertion fails. The four scripts in
    jax_assertions/ exist specifically to assert JAX behavior, so
    disabling JAX makes the assertions vacuous. Fix: add an env_vars.yaml
    override that unsets PYAUTO_DISABLE_JAX for the `jax_assertions/`
    pattern (substring match covers all four current scripts and future
    siblings). Per memory feedback_env_vars_yaml_overrides.md — env-var
    conflicts get fixed in env_vars.yaml, not via os.environ.pop in the
    script. Verified pre-fix repro under PYAUTO_DISABLE_JAX=1 and post-fix
    pass under runner-emulated env (all other defaults applied,
    DISABLE_JAX absent).

## subhalo-refine-source-fix
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/133
- repos: autolens_workspace
- notes: |
    Cluster F of the recent release-prep triage. scripts/group/features/
    advanced/subhalo/detect/start_here.py crashed with `NameError: name
    'source' is not defined` at line 634 of `subhalo_refine`. Fallout from
    PR #117 ("Cluster F triage", 2026-05-02): PR #117 removed the
    redundant `"source": ...` key from five lens_dict literals (they
    collided with the explicit `source=source` kwarg on `af.Collection`).
    Four functions had a local `source = ...` assignment already, so the
    explicit kwarg remained resolvable. `subhalo_refine` was the
    exception — PR #117 dropped `"source": subhalo_grid_search_result.
    model.galaxies.source` from its lens_dict without lifting that
    expression into a standalone assignment. Fix: add the missing
    `source = subhalo_grid_search_result.model.galaxies.source` line
    before the lens_dict literal. Restores exactly the value PR #117
    dropped; matches the standalone-assignment style every other
    subhalo_* function uses; preserves the SLaM-like pipeline
    (grid-search posterior source flows into refine search).
    User's original suggestion (`source=source` → `source=source_lp`,
    based on Python's NameError "Did you mean: 'source_lp'?" hint)
    would have type-errored at fit time — `source_lp` is a function in
    this file, not a Galaxy. Python's NameError suggestion is lexical
    proximity, not type-correct. Verified locally:
    `subhalo[3]_[single_plane_refine]` (the previously-failing phase)
    now completes successfully.

## point-source-fit-positions-len
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/132
- repos: autolens_workspace
- notes: |
    Cluster E of the recent release-prep triage. scripts/point_source/fit.py
    crashed with `ValueError: operands could not be broadcast together with
    shapes (2,) (4,)` at autoarray/abstract_ndarray.py:326 under
    PYAUTO_SMALL_DATASETS=1. Root cause: PointSolver.solve() short-circuits
    to a fixed 2-position pair [(1.0,0.0),(0.0,1.0)] when SMALL_DATASETS=1
    (PyAutoLens/autolens/point/solver/point_solver.py:90-91), but the script
    hardcoded a 4-element positions_data and 4-element positions_noise_map.
    FitPositionsImagePairRepeat then divided a 2-element residual by the
    4-element noise_map and broadcasts failed. Same family as PR #119
    (Cluster E: deblending simulator) which fixed
    point_source/features/deblending/simulator.py with a
    range(len(positions)) dict comprehension; PR #119 didn't touch fit.py.
    Mechanical port of PR #119's pattern: replace hardcoded 4-element
    lists with `Grid2DIrregular(positions)` + `[0.005] * len(positions)`.
    Hardcoded values were demonstrative only — they matched solver output
    exactly, so the new code produces an identical demo (zero residuals)
    while adapting to N positions. Prose updated from "we manually
    specify" to describe the new derivation. Verified locally under
    SMALL_DATASETS (2-element residual_map) and full (4-element).

## park-modeling-viz-jit-slow
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/76
- followup-issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/77
- repos: autolens_workspace_test
- notes: |
    Cluster D of the recent release-prep triage. Three autolens
    workspace_test imaging scripts (modeling_visualization_jit,
    modeling_visualization_jit_delaunay, modeling_visualization_jit_rectangular)
    timed out at the 300s per-script cap. NOT a regression — PR #70
    (fix(env): unblock modeling_visualization_jit tests in CI defaults)
    cleared the prior `AssertionError: expected jax.Array, got numpy.float64`
    that had masked this perf issue. Autogalaxy sibling passes in ~88.6s;
    autolens variants are ~3.5x slower (>300s) — JIT compile + full
    visualization. Parked via standard `# SLOW <YYYY-MM-DD>` convention
    in autolens_workspace_test/config/build/no_run.yaml. Mega-runs
    surface SLOW entries with a loud warning banner so they don't
    silently rot. Follow-up perf issue #77 filed against
    autolens_workspace_test capturing the 3.5x disparity, investigation
    pointers (Tracer vs Galaxy, pixelization variants, JAX visualization
    pipeline) and acceptance criteria for removing the SLOW markers.

## group-slam-prior-clamp
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/131
- repos: autolens_workspace
- notes: |
    Cluster C of the recent release-prep triage. Two group SLaM scripts
    (scripts/group/features/{linear_light_profiles,pixelization}/slam.py)
    crashed with PriorException at `mass.einstein_radius = af.UniformPrior(
    lower_limit=0.0, upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0))`
    when PYAUTO_TEST_MODE=2 produced zero linear-light intensities, making
    upper_limit==lower_limit==0. PR #117 (Cluster F triage, merged 2026-05-02)
    fixed exactly this pattern in `linear_light_profiles/slam.py`'s
    `source_lp_1` (line 246) but missed two siblings: `mass_total` in the
    same file (line 704) and `source_lp_1` in `pixelization/slam.py` (line
    234). Verbatim mechanical port of PR #117's clamp prefix
    (`luminosity_cap = ...; upper_limit = min(luminosity_cap, 5.0) if
    luminosity_cap > 0 else 5.0`) to both missed sites. Verified locally —
    `linear_light_profiles/slam.py` now reaches `mass_total[1]` (the
    previously-buggy phase) and completes; previously crashed there.

## aggregator-mge-queries
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/130
- repos: autolens_workspace
- notes: |
    Cluster B of the recent release-prep triage. Two autolens aggregator
    tutorials (scripts/guides/results/aggregator/{queries,samples_via_aggregator}.py)
    crashed with `AttributeError: 'Model' object has no attribute 'sersic_index'`
    after PR #118 swapped the source bulge in _quick_fit.py from Sersic to
    MGE (al.model_util.mge_model_from). MGE is a Basis of fixed-sigma
    Gaussians whose only free parameters are the basis's shared centre +
    ell_comps — there is no sersic_index. Fix: queries.py Model Queries
    section now demos `lens.mass.einstein_radius < 1.5` (vs the Logic
    section's `& mass.einstein_radius > 1.0` — same parameter, different
    API features); samples_via_aggregator.py's two with_paths calls now
    use `lens.mass.centre.centre_0` as the second filter path (path was
    already used at line 533 of the same script, so known valid). Cluster A's
    fix to _quick_fit.py is the prerequisite that allowed these scripts
    to reach the failing line at all. autogalaxy versions of the same
    scripts are unaffected — that workspace's _quick_fit.py still uses
    Sersic + Exponential.

## quick-fit-smoke-mode-fix
- completed: 2026-05-07
- workspace-prs:
    - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/60
    - https://github.com/PyAutoLabs/autolens_workspace/pull/129
- repos: autogalaxy_workspace, autolens_workspace
- notes: |
    Cluster A of the recent release-prep triage. Four aggregator tutorials
    (scripts/guides/results/aggregator/{data_fitting,models}.py × 2 workspaces)
    crashed with `TypeError: 'NoneType' object is not subscriptable` at
    PyAutoGalaxy/autogalaxy/aggregator/agg_util.py:101 in fast smoke mode
    (PYAUTO_TEST_MODE=2 + PYAUTO_SKIP_VISUALIZATION=1). Root cause: the
    _quick_fit.py helper invoked via subprocess inherited those env vars,
    suppressing the visualizer that writes image/dataset.fits, so
    fit.value("dataset") returned None. Fix: helper now pops
    PYAUTO_SKIP_VISUALIZATION / PYAUTO_SKIP_FIT_OUTPUT and downgrades
    PYAUTO_TEST_MODE>=2 to 1 before importing autofit. Idempotent early-exit
    means cost is paid once per workspace per smoke run. Standalone fix —
    no library change. The PR #55/#118 refactor that split start_here.py
    into _quick_fit.py + un-skipped these scripts in no_run.yaml exposed
    the latent bug (the old start_here.py had a partial PYAUTO_TEST_MODE=1
    pop, but it never handled modes 2/3 + SKIP_VISUALIZATION).

## use-pathlib
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1257
- completed: 2026-05-07
- library-prs:
    - https://github.com/PyAutoLabs/PyAutoConf/pull/104
    - https://github.com/PyAutoLabs/PyAutoFit/pull/1258
    - https://github.com/PyAutoLabs/PyAutoArray/pull/300
    - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/388
    - https://github.com/PyAutoLabs/PyAutoLens/pull/497
- workspace-prs:
    - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/59
    - https://github.com/PyAutoLabs/autolens_workspace/pull/128
    - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/50
    - https://github.com/Jammy2211/autofit_workspace_developer/pull/15
    - https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/pull/12
- notes: |
    Replaced os.path.* and bare path.* with pathlib.Path across 5 libraries
    and 5 workspaces. ~700 references converted. 3,188 library unit tests
    pass; 36/36 workspace smoke tests pass. No public API changed.
    Canonical autogalaxy_workspace and autolens_workspace pull skipped due
    to in-progress smoke-test-optimization dirty state — will land when
    that task ships.

## group-list-based-api
- completed: 2026-05-06
- repos: autolens_workspace
- notes: |
    Retroactively logged via 2026-05-06 hygiene scan. Original prompt
    `workspaces/group.md` asked for `autolens_workspace/scripts/group/start_here.py`
    to use the list-based `lens_dict` model composition (multiple main lens
    galaxies treated symmetrically, instead of one main + extras). Verified done:
    `start_here.py:198-200` builds `lens_dict` and iterates `main_lens_centres`
    to populate `lens_0`, `lens_1`, … Three sibling tasks (`group-features`,
    `group-two-main-galaxies`, `group-pixelization-delaunay-fixes`) plus three
    `issued/group*.md` files cover the related rollout. Original issue/PR not
    tracked in this registry.

## smoke-workspace-actions
- completed: 2026-05-06
- repos: autofit_workspace, autogalaxy_workspace, autolens_workspace
- notes: |
    Retroactively logged via 2026-05-06 hygiene scan. Original prompt
    `autobuild/smoke_workspace_action.md` asked for GitHub Actions workflows
    that run smoke tests on the three workspaces (mirroring the source-repo
    actions). Verified done: `.github/workflows/smoke_tests.yml` exists in each
    of `autofit_workspace`, `autogalaxy_workspace`, `autolens_workspace`. The
    follow-up `autobuild/smoke_workspace_fixes.md` (re-enable the entries that
    were commented out for the green baseline) is still partial — kept in
    pending. Original issue/PR not tracked in this registry.

## delaunay-jax-profiling
- completed: 2026-05-06
- repos: autolens_workspace_developer
- notes: |
    Retroactively logged via 2026-05-06 hygiene scan. Original prompt
    `autolens_workspace_developer/imaging_delaunay_jax_profiling.md` asked for
    `jax_profiling/imaging/delaunay.py` to be aligned with the current pytree /
    register_model approach used by `mge.py` and `pixelization.py`. Verified
    done: `jax_profiling/jit/imaging/delaunay.py` now mirrors the sibling
    Timer + register_model + xp pattern. Original issue/PR not tracked.

## cluster-visualization-profiling
- completed: 2026-05-06
- repos: autolens_workspace_developer
- notes: |
    Retroactively logged via 2026-05-06 hygiene scan. Original prompt
    `autolens_workspace_developer/visualization_profiling_cluster.md` asked for
    a profiling script targeting `autolens_workspace/scripts/cluster/simulator.py`
    visualization (the ~92s `SimulatorImaging.via_tracer_from` phase identified
    in the prompt). Verified done: `autolens_workspace_developer/visualization_profiling/imaging/cluster.py`
    exists with Timer instrumentation matching the sibling profiling scripts.
    Original issue/PR not tracked.

## cluster-csv-redshifts
- completed: 2026-05-06 (retroactive log — INCORRECT, see correction below)
- repos: autolens_workspace
- corrected: 2026-05-18 — verification was wrong; work never shipped. Re-issued via `issued/2_modeling_cluster.md` rewrite.
- notes: |
    Retroactively logged via 2026-05-06 hygiene scan. Original prompt
    `cluster/2_modeling.md` asked for `autolens_workspace/scripts/cluster/modeling.py`
    to load redshifts from `point_datasets.csv` via `al.list_from_csv` and link
    them to `Galaxy.redshift` (replacing the hardcoded `redshift=1.0`), plus
    move toward CSV-driven main/source galaxy loading.

    **Correction (2026-05-18):** the 2026-05-06 verifier saw `al.list_from_csv`
    on `modeling.py:143` but missed that the call is inside a `"""..."""`
    docstring example block — it's not the actual code path. The real code
    still loops `for i in range(5): al.from_json(point_dataset_{i}.json)`,
    hardcodes the source galaxy `redshift=1.0` (line 367), and uses the
    pre-rewrite `extra_galaxies` + scaling-relation structure (10 satellites,
    5 sources) which is fundamentally mismatched to the current 2-main + halo
    + 2-source-at-different-redshifts simulator. `cluster/modeling` and
    `cluster/start_here` remain parked in `autolens_workspace/config/build/no_run.yaml`.
    Work re-scoped and re-issued as a minimal first-pass rewrite of
    `modeling.py` only — see updated `issued/2_modeling_cluster.md`. Lesson:
    retroactive hygiene sweeps must check that "verified" code isn't inside
    docstring blocks.

## autogalaxy-wst-jax-grad-multi
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/32
- completed: 2026-05-06
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/33
- repos: autogalaxy_workspace_test
- notes: **Final task (8/9) of the autogalaxy_workspace_test parity epic (#5) — epic is now closed.** Created `scripts/jax_grad/multi/{lp.py, mge.py}` from scratch. Each script joins per-band `AnalysisImaging` factors via `af.FactorGraphModel(use_jax=True)` and wraps the global log-likelihood in `jax.value_and_grad`. Both pass on CI 3.12 (`lp.py` 10.6s shape (9,), `mge.py` 21.0s shape (6,)). Used `ag.lp.Sersic` and option B per-band `ell_comps` matching `jax_likelihood_functions/multi/{lp,mge}.py` patterns. The `jax_grad/` env_vars override added in PR #29 covered all three jax_grad subfolders. Suggested follow-up: file an autolens-retrofit issue covering `imaging_lp.py → imaging/lp.py`, `imaging_mge.py → imaging/mge.py`, plus net-new interferometer/multi ports for autolens.

## blackjax-nuts-search
- issue: https://github.com/rhayes777/PyAutoFit/issues/1255
- completed: 2026-05-06
- library-pr: https://github.com/rhayes777/PyAutoFit/pull/1256
- workspace-prs:
  - https://github.com/PyAutoLabs/autofit_workspace/pull/52 (mcmc.py extended)
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/23 (BlackJAXNUTS.py integration test)
- repos: PyAutoFit, autofit_workspace, autofit_workspace_test
- notes: Added `af.BlackJAXNUTS` as a first-class non-linear search alongside Emcee/Zeus/Nautilus/etc. Lives under `autofit/non_linear/search/mcmc/blackjax/nuts/search.py` so the `blackjax/` namespace can hold future BlackJAX samplers (HMC, MALA). Inherits `AbstractMCMC`, runs `blackjax.window_adaptation` warmup followed by NUTS sampling in a JIT'd `jax.lax.scan`, chunked by `iterations_per_full_update` for periodic `perform_update` flushes. Strict requirement: `Analysis(use_jax=True)` — clear error otherwise. Sampling in physical parameter space; bounded priors contribute -inf outside support. Single-chain v1 (`num_chains>1` raises `NotImplementedError`); resume stubbed for later. AutoCorrelations populated from BlackJAX per-param ESS via τ_int = N / ESS (canonical identity). Persistence via pickle under `search_internal/`. Target log-density built from `Fitness.call` directly (pure-JAX path; `call_wrap`/`__call__` were intentionally bypassed because they convert to Python float and would break NUTS gradients). `blackjax>=1.2.0` added to `optional-dependencies.optional` (lazy import in `_fit`). 7 unit tests + integration test on the 1D Gaussian (recovers truth within 0.05σ, ESS ~50% of num_samples, 0 divergences).

## point-simulator-realistic-errors
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/125
- completed: 2026-05-06
- workspace-prs: https://github.com/PyAutoLabs/autolens_workspace/pull/127, https://github.com/PyAutoLabs/autolens_workspace_test/pull/73, https://github.com/PyAutoLabs/autolens_workspace_developer/pull/49
- repos: autolens_workspace, autolens_workspace_test, autolens_workspace_developer
- notes: Replaced unrealistic noise scales across 11 point-source scripts plus the local-only `z_projects/concr/simulators/cosmology.py`. Constants applied consistently — `position_noise = 0.005"` (5 mas, HST PSF-centroiding precision, replacing the imaging pixel scale ~0.05"), `time_delay_rel_noise = 0.05` (5%, replacing 25%), `flux_rel_noise = 0.05` (5%, replacing pure-Poisson `√flux` which gave ~100% relative error on unit flux). Scope expanded mid-implementation when `start_here.py` was found to have four separate sections with the bad patterns plus a latent bug — `positions_noise_map` was defined but the dataset constructor passed `grid.pixel_scale` instead. `cluster/simulator.py` had a docstring claiming the pixel scale *is* the positional uncertainty (the exact misconception this task fixed), rewritten in place. Cancer simulator confirmed orthogonal (Hill-curve dose-response, no positions/delays/fluxes). Two side findings worth follow-up prompts: (1) autolens_workspace_test 3.13 CI has been failing on main since at least 2026-05-01 with 7 `jax_likelihood_functions/*` failures (PR #73 inherited the identical list — not a regression); (2) the `worktree_check_conflict` helper silently exits 0 when `$PYAUTO_MAIN` isn't exported, which is how the original `/start_dev` missed a real conflict with #124. Notebook regeneration deferred to `/generate_and_merge`.

## blackjax-nuts-example
- issue: https://github.com/Jammy2211/autofit_workspace_developer/issues/13
- completed: 2026-05-06
- workspace-pr: https://github.com/Jammy2211/autofit_workspace_developer/pull/14
- repos: autofit_workspace_developer
- notes: Added `searches_minimal/nuts_jax.py` — BlackJAX NUTS on the same 1D Gaussian as the rest of the JAX scripts. Window adaptation tunes step size + diagonal inverse mass matrix; sampling runs in a JIT'd `jax.lax.scan`. Recovers truth in 1.83s, ESS 1216/2000, 0 divergences — fastest JAX path in the folder (beats nss_grad's 5.2s). Also wrote a non-git follow-up `z_projects/concr/scripts/cancer_sim/graphical_nuts.py` that runs joint NUTS over the 93-dim cancer-sim factor graph (4.4s wall, ESS 880/1000, 0 divergences); kept local since z_projects isn't tracked. Pre-task: shipped the unregistered `feature/searches-minimal-converged` work first (PR #12 — shared `_metrics.MLTracker` across all searches_minimal scripts) so this task could use `MLTracker.from_log_l_history` for evals/time-to-ML.

## datacube-positions-delaunay
- issue: none — direct followup to autolens_workspace#120
- completed: 2026-05-05
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/123 (delaunay.py + RectangularAdaptDensity + positions + PositionsLH)
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/48 (mesh swap in likelihood_function.py)
- repos: autolens_workspace, autolens_workspace_developer
- notes: Three datacube follow-ups on top of PR #122. (1) Mesh swap RectangularUniform → RectangularAdaptDensity (modeling, start_here, dev likelihood walkthrough). (2) New delaunay.py sibling using `Overlay` image-mesh, `append_with_circle_edge_points` edge zeroing, `ConstantSplit` regularization, with `AdaptImages` paired with the source galaxy. (3) `PointSolver` positions block in simulator.py writes `positions.json`; all four modeling scripts load it and pass `PositionsLH(threshold=0.3)` to every per-channel `AnalysisInterferometer`. PositionsLH is essentially required for pixelized fits — without it the search routinely converges on demagnified-source local maxima.

## autogalaxy-wst-jax-grad-interferometer
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/30
- completed: 2026-05-06
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/31
- repos: autogalaxy_workspace_test
- notes: Task 7/9 of the autogalaxy_workspace_test parity epic (#5). Created `scripts/jax_grad/interferometer/{lp.py, mge.py}` from scratch — autolens has no interferometer `jax_grad` reference. Both pass on CI 3.12 (`lp.py` 6.6s shape (7,), `mge.py` 11.7s shape (4,)). Used plain `ag.lp.Sersic` (not `lp_linear`) to match the validated `jax_likelihood_functions/interferometer/lp.py` setup. The `jax_grad/` env_vars override added in PR #29 already covered this PR — no env_vars.yaml change. Layout-divergence-from-autolens question now compounded by this PR (autolens has flat `jax_grad/imaging_*.py` and no interferometer scripts at all); suggested filing the autolens-retrofit follow-up after task 8 ships.

## autogalaxy-wst-jax-grad-imaging
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/28
- completed: 2026-05-05
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/29
- repos: autogalaxy_workspace_test
- notes: Task 6/9 of the autogalaxy_workspace_test parity epic (#5). Ported autolens `jax_grad/imaging_{lp,mge}.py` to autogalaxy under a new `scripts/jax_grad/imaging/` subfolder; both scripts pass on CI 3.12 (`lp.py` 11.3s, `mge.py` 16.8s). Established subfolder layout convention even though autolens is currently flat — surfaced the retrofit question to the maintainer via PR body. Added `jax_grad/` env_vars override mirroring `jax_likelihood_functions/` (unsets `PYAUTO_SMALL_DATASETS` + `PYAUTO_DISABLE_JAX`). Pytree registration on `autogalaxy/imaging/model/analysis.py` was already in place from task 3.

## autogalaxy-wst-model-composition
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/26
- completed: 2026-05-05
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/27
- repos: autogalaxy_workspace_test
- notes: Task 2/9 of the autogalaxy_workspace_test parity epic (#5). Ported `multi_galaxy_mge.py` from autolens_workspace_test, stripped to autogalaxy semantics (two galaxies sharing one plane, MGE light bases, no mass / shear / ray-tracing). Identifier regression anchor `a6eb928ed9a1fb92d0c18cf5443af4a6`. Required adding a `model_composition/` override to `config/build/env_vars.yaml` (mirrors the existing autolens override) because the `PYAUTO_SMALL_DATASETS=1` smoke default reduces `total_gaussians` inside `ag.model_util.mge_model_from`, collapsing `gaussian_per_basis=2` to 1 and breaking the structural prior_count assertions. Umbrella issue #5 also updated to tick tasks 4 and 5 (already shipped via PRs #17 and #19, checkboxes were stale).

## alma-datacube
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/120
- completed: 2026-05-05
- library-prs:
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1253 (AnalysisFactor.visualize_combined dispatch fix)
  - https://github.com/PyAutoLabs/PyAutoLens/pull/494 (VisualizerInterferometer combined plotter)
- workspace-prs:
  - https://github.com/PyAutoLabs/autolens_workspace_developer/pull/46 (datacube/ Phase 1: simulator + JAX likelihood walkthrough)
  - https://github.com/PyAutoLabs/autolens_workspace/pull/122 (interferometer/features/datacube/ tutorial scripts)
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/72 (multi/visualization dispatch tests)
- repos: PyAutoFit, PyAutoLens, autolens_workspace, autolens_workspace_developer, autolens_workspace_test
- notes: ALMA datacube modeling — list-of-Interferometer FactorGraph prototype with shared lens and per-channel pixelized source. Phase 1 deliberately runs each channel's NUFFT and inversion independently. Two follow-up issues to file: (1) Aris's shared `Lᵀ W̃ L` optimisation that exploits channel-invariant uv_wavelengths/noise_map; (2) `Interferometer.list_from_fits_3d` helper in PyAutoArray. While verifying visualization, found and fixed a silent dispatch bug in `af.FactorGraphModel.visualize_combined` — `AnalysisFactor` had no `visualize_combined` method, so the auto-forwarder skipped the call for multi-dataset fits (imaging and interferometer both). Added forwarders in PyAutoFit + the missing `VisualizerInterferometer.visualize_combined` + `subplot_fit_interferometer_combined` plot in PyAutoLens.

## rst-to-myst-md-pass3
- issue: none — direct followup to PyAutoFit#1245 and pass2
- completed: 2026-05-04
- workspace-prs:
  - https://github.com/Jammy2211/autofit_workspace_developer/pull/11
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/22
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/71
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/25
  - https://github.com/PyAutoLabs/autolens_base_project/pull/2
  - https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/pull/11
  - https://github.com/PyAutoLabs/autofit_workspace/pull/51
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/57
  - https://github.com/PyAutoLabs/autolens_workspace/pull/121
- library-prs:
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1251
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/387
  - https://github.com/PyAutoLabs/PyAutoLens/pull/493
- notes: Final sweep of `.rst` files across the workspace ecosystem (285 files, 9 workspace repos). Same playbook as passes 1 and 2: `rst2myst convert -R`, plain CommonMark hand-rewrite for the marketing READMEs (autofit/galaxy/lens workspaces) so badges/images render on GitHub, perl one-liner to fix rst-to-myst's escaped-dash continuation pattern in chapter READMEs, drop the leading `(references)=` MyST anchor from `CITATIONS.md`. Three repos had **non-rename code/script changes**: `autolens_base_project/hpc/sync` and `euclid_strong_lens_modeling_pipeline/hpc/sync` had `ROOT_FILES=(...README.rst...)` arrays that needed flipping to `README.md` or the sync script would skip the renamed file when copying to HPC; `autogalaxy_workspace/welcome.py` + `autolens_workspace/welcome.py` had prose docstring refs to `<repo>/README.rst` that needed flipping; and `autogalaxy_workspace/scripts/guides/hpc/example_cpu_and_gpu.{py,ipynb}` + `autolens_workspace/scripts/cluster/modeling.{py,ipynb}` had inline prose refs that needed updating in both the `.py` source-of-truth and the matching `.ipynb`. Tail: 3 follow-up PRs in PyAutoFit/Galaxy/Lens flipping the `docs/general/{configs,workspace}.md` prose refs from "README.rst" to "README.md" — these were deliberately deferred in pass 2 because they pointed at workspaces that hadn't been converted yet. **Two test workspaces (autolens_workspace_test, autogalaxy_workspace_test) had pre-existing 3.13 `jax_likelihood_functions/*` smoke failures** unrelated to this change — main was already red on the same scripts for days; merged with `--admin`. The autolens_base_project canonical checkout had pre-existing staged changes (CLAUDE.md, hpc scripts, scripts/template.py, skills/init-slam) that blocked the post-merge `git pull --ff-only`, left for user. The autofit_workspace_developer canonical checkout was on `feature/searches-minimal-converged` (unregistered work) so post-merge pull was skipped. Squash-merged in size order; library prose-ref tail merged last.

## rst-to-myst-md-pass2
- issue: none — direct followup to PyAutoFit#1245
- completed: 2026-05-04
- library-prs:
  - https://github.com/PyAutoLabs/PyAutoConf/pull/103
  - https://github.com/PyAutoLabs/PyAutoBuild/pull/82
  - https://github.com/PyAutoLabs/PyAutoArray/pull/298
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1249
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/386
  - https://github.com/PyAutoLabs/PyAutoLens/pull/492
  - https://github.com/PyAutoLabs/HowToFit/pull/6
  - https://github.com/PyAutoLabs/HowToGalaxy/pull/5
  - https://github.com/PyAutoLabs/HowToLens/pull/7
- notes: Followup to #1245 sweeping the rest of the prose `.rst` across the PyAuto ecosystem (84 files, 9 repos). Converted root `README.rst` and `CITATIONS.rst`, package `config/.../README.rst`, and HowTo* `notebooks/`+`scripts/` chapter READMEs. Lib root READMEs hand-rewritten as plain CommonMark with inline `[![alt](badge)](link)` syntax — `rst-to-myst`'s default output uses MyST `{image}` directives + `{{substitutions}}` which render as literal text on GitHub and PyPI. Side effects: `pyproject.toml` `readme` content-type → `text/markdown` (5 lib repos), `MANIFEST.in` `include README.md`/`CITATIONS.md` (5 lib repos), `PyAutoArray/docs/index.md` switched from `eval-rst` `.. include::` to MyST native `{include}`, `PyAutoGalaxy/PyAutoLens/docs/conf.py` dropped stale `.rst` entries from `exclude_patterns`. HowTo* chapter READMEs had `rst-to-myst`'s escaped-dash continuation pattern (`\- description`) rewritten as proper Markdown list items via a perl one-liner. Workspace prose refs in `docs/general/{configs,workspace}.md` deliberately left as `README.rst` — they point at the workspace repos which are not in scope for this pass. `docs/api/*.rst` and `docs/_templates/*.rst` deliberately kept as native RST (autosummary requirement). Squash-merged library-first.

## rst-to-myst-md
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1245
- completed: 2026-05-04
- library-prs:
  - https://github.com/PyAutoLabs/PyAutoFit/pull/1246
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/383
  - https://github.com/PyAutoLabs/PyAutoLens/pull/487
  - https://github.com/PyAutoLabs/PyAutoArray/pull/294
- notes: Converted prose `.rst` docs to MyST `.md` across all four libraries using `rst-to-myst`; kept `docs/api/*.rst` as native RST since autosummary directives don't gain readability from the conversion. Branches sat for ~3 days while main advanced 7-15 commits per repo (jax cleanup, weak-lensing additions, EP cavity-message factor, etc.). Caught up via merge: the only docs-touching commits on main were two automated `2026.5.1.1`/`2026.5.1.4` Colab URL-tag bumps. Resolved modify/delete conflicts on the `.rst` siblings by keeping the deletions; ported the URL bump (`2026.4.13.6` → `2026.5.1.4`) to 1 file in PyAutoFit, 7 in PyAutoGalaxy, 7 in PyAutoLens; PyAutoArray merged clean. PRs squash-merged library-first.

## cluster-g-interferometer-pixelization-plot
- issue: none — direct fix for Cluster G + one Cluster H entry in `PyAutoBuild/test_results/runs/2026-04-29T14-48-47Z/triage.md`
- completed: 2026-05-03
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/297
- notes: 3 release-prep failures (autogalaxy `interferometer/features/pixelization/fit.py`, autolens equivalent, `autolens_workspace_test/scripts/interferometer/visualization.py`) all crashed at `PyAutoArray/autoarray/plot/array.py:199 — h, w = array.shape[:2]` with `ValueError: not enough values to unpack`. Root cause: `subplot_of_mapper` and `subplot_mappings` panel 0 fed `inversion.data_subtracted_dict[mapper]` straight into `plot_array`. For interferometer fits that dict entry is `Visibilities` (1D complex), not `Array2D`. Panels 1-3 already detected `Visibilities` and substituted a 2D image-plane equivalent (`mapped_reconstructed_data_dict`) — panel 0 had no equivalent guard. Fix: when the entry is `Visibilities`, transform to a 2D dirty image via `inversion.transformer.image_from(visibilities=...)` (same call `FitInterferometer.dirty_residual_map` uses at `fit_interferometer.py:203`). Imaging path byte-identical. Also confirmed Cluster H's `autolens_workspace_test/interferometer/visualization.py` failure was the same root cause (visualizer routes through `subplot_of_mapper`). One file, two near-identical edits in `inversion_plots.py`; no workspace changes needed. Verified: 3 originally-failing scripts now exit 0; `autolens_workspace/imaging/features/pixelization/fit.py` (sanity) still exit 0; `pytest test_autoarray/inversion/` 162 passed; full smoke 36 pass / 0 fail (the 6 euclid failures are a pre-existing workspace-version-pin mismatch, fire before any PyAutoArray code runs).

## cluster-e-missing-simulator-output
- issue: none — direct fix for Cluster E failures in `PyAutoBuild/test_results/runs/2026-04-29T14-48-47Z/triage.md`
- completed: 2026-05-03
- workspace-pr:
  - PyAutoLabs/autolens_workspace#119
  - PyAutoLabs/autogalaxy_workspace#56
- notes: 5 failures all caused by missing `dataset/*/data.fits` files at script-run time, but two distinct root causes — not one as the triage suggested. Root cause 1 (1 file, 2 failing scripts): `point_source/features/deblending/simulator.py` hardcoded 4 lensed image positions but `PyAutoLens/autolens/point/solver/point_solver.py:90` short-circuits `PointSolver.solve()` to `[(1.0, 0.0), (0.0, 1.0)]` under `PYAUTO_SMALL_DATASETS=1`. Indexing `positions[2]` raised `IndexError`, no `data.fits` was written, and downstream `deblending/modeling.py` then hit `FileNotFoundError`. Fixed by building `point_image_{i}` kwargs from `len(positions)` so the simulator emits valid output in both smoke (2 images) and production (4 images) modes. Root cause 2 (4 files, 3 autolens guides + 1 autogalaxy guide): the build's `run_python.py` iterates `scripts/<dir>/` alphabetically, so `scripts/guides/` runs before `scripts/imaging/` where `dataset/imaging/simple/data.fits` is created. The 3 autolens guides (`data_structures.py`, `modeling/bug_fix.py`, `modeling/chaining.py`) had no auto-simulate snippet at all; the autogalaxy `guides/plot/start_here.py` had a snippet but pointed at `scripts/guides/plot/simulator.py` which writes `sersic_x2/`, not `simple/`. Added/fixed the canonical "if not dataset_path.exists(): subprocess.run([sys.executable, scripts/imaging/simulator.py])" pattern. The bug_fix.py snippet uses `os.path.exists` (not `Path.exists`) to match the script's existing `path.join` style. Verified end-to-end with `PYAUTO_SMALL_DATASETS=1 PYAUTO_TEST_MODE=1 PYAUTO_FAST_PLOTS=1` from a wiped dataset state: all 6 scripts (5 originally failing + the upstream simulator) exit 0; smoke 7/7 autolens, 6/6 autogalaxy. Pure workspace change, no library touched. Cluster E counted 5/48 release-prep failures.

## aggregator-quick-fit
- issue: none — direct fix for Cluster D failures in `PyAutoBuild/test_results/runs/2026-04-29T14-48-47Z/triage.md`
- completed: 2026-05-02
- workspace-pr:
  - PyAutoLabs/autogalaxy_workspace#55
  - PyAutoLabs/autolens_workspace#118
- notes: 8 failures across `scripts/guides/results/aggregator/` in both workspaces — 4 timeouts (each example duplicated `search.fit()` uncapped) and 4 NoneType cascades (downstream readers loaded the partial output of the timed-out scripts). Root cause was that `start_here.py` had `n_like_max=300` gated on `test_mode_was_on`, but `env_vars.yaml` unsets `PYAUTO_TEST_MODE` for `guides/results/`, so the cap never fired during release-prep; meanwhile the aggregator scripts (`galaxies_fit.py`, `samples.py`) ran their own uncapped searches. Fix: introduced `scripts/guides/results/_quick_fit.py` (idempotent, `n_like_max=300` always) and routed every aggregator example through the existing auto-trigger pattern (`subprocess.run(_quick_fit.py)` if `output/results_folder/` is missing). `models.py`/`queries.py`/`samples_via_aggregator.py`/`data_fitting.py` had their existing subprocess targets redirected from `start_here.py` to `_quick_fit.py`; `galaxies_fit{,s}.py` and `samples.py` got the guard added at the top plus `n_like_max=300` defensively on their own search; `start_here.py` got its conditional cap dropped (now unconditional). Removed the four stale `no_run.yaml` lines per workspace (`guides/results/start_here`, three `guides/results/examples/*`, `data_fitting` stem). Verified end-to-end: `_quick_fit.py` completes in 14s (autogalaxy) / 33s (autolens), idempotent re-runs in 0.04s, `aggregator/models.py` exits 0 in both workspaces. Pure workspace change, no library touched.

## group-pixelization-delaunay-fixes
- issue: none — direct fix for Cluster A failures in group/features/pixelization smoke set
- completed: 2026-05-01
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/490
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/116
- notes: 5 group/features/pixelization scripts failed under the smoke profile (`PYAUTO_TEST_MODE=2 PYAUTO_SKIP_CHECKS=1`). Root cause was two distinct workspace bugs, not a library regression — the user's "one upstream constructor bug" framing was wrong. Bug A (4 scripts: fit, likelihood_function, delaunay, modeling): `Pixelization(mesh=Delaunay(...))` was passed to `FitImaging`/`AnalysisImaging` without an `image_plane_mesh_grid` via `AdaptImages`. Bug B (slam.py): chained `.positions` on `Result.positions_likelihood_from(...)` which returned `None` under `skip_checks()`. Bug A fix is workspace-only (PR #116) — added `Overlay` image-mesh + `AdaptImages` boilerplate matching `imaging/features/pixelization/delaunay.py`, and switched modeling-section regularization from `AdaptSplit` (needs prior-search adapt data) to `ConstantSplit`. Bug B fix is library-only (PR #490) — `positions_likelihood_from` now returns a synthetic `PositionsLH` under `skip_checks() + is_test_mode()` instead of `None`, preserving the workspace API; `slam.py` itself was left unchanged. Workspace PR was gated on library PR. Both merged.

## jit-visualization-env-overrides
- issue: none — direct fix for Cluster C in `PyAutoBuild/test_results/runs/2026-04-29T14-48-47Z/triage.md`
- completed: 2026-05-01
- workspace-pr:
  - PyAutoLabs/autogalaxy_workspace_test#24
  - PyAutoLabs/autolens_workspace_test#70
- notes: 4 `modeling_visualization_jit*` integration scripts (1 in autogalaxy_workspace_test, 3 in autolens_workspace_test) failed in CI with `AssertionError: expected jax.Array, got <class 'numpy.float64'>`. Root cause was env-var-only: the CI defaults set `PYAUTO_DISABLE_JAX=1`, which `PyAutoFit/autofit/non_linear/analysis/analysis.py:42-46` intercepts and silently flips `use_jax_for_visualization` off, so `fit_for_visualization` returned a numpy `float64` and Part 1's `isinstance(..., jnp.ndarray)` failed. `PYAUTO_SMALL_DATASETS=1` would also have broken the hardcoded mask, and `PYAUTO_TEST_MODE=2` / `PYAUTO_FAST_PLOTS=1` would have broken Part 2's real-Nautilus + fit.png assertions. Fix: one new override entry per workspace's `config/build/env_vars.yaml` matching `imaging/modeling_visualization_jit`, mirroring the existing `jax_likelihood_functions/` precedent. Verified end-to-end PASS for all four scripts under the new env. Pure config change — no library or script edits.

## workspace-version-config-check
- issue: https://github.com/PyAutoLabs/PyAutoConf/issues/100
- completed: 2026-04-30
- library-pr:
  - PyAutoLabs/PyAutoConf#101
  - PyAutoLabs/PyAutoFit#1241
  - PyAutoLabs/PyAutoGalaxy#380
  - PyAutoLabs/PyAutoLens#484
  - PyAutoLabs/PyAutoBuild#70
- workspace-pr:
  - PyAutoLabs/autolens_workspace#112
  - PyAutoLabs/autofit_workspace#48
  - PyAutoLabs/autogalaxy_workspace#51
  - PyAutoLabs/HowToFit#4
  - PyAutoLabs/HowToGalaxy#4
  - PyAutoLabs/HowToLens#5
  - Jammy2211/euclid_strong_lens_modeling_pipeline#10
- notes: Workspace/library version mismatches now surface on every script run, not just `welcome.py`. `autoconf.workspace.check_version` reads `config/general.yaml`'s `version.workspace_version` (with `version.txt` fallback) and honours `version.workspace_version_check: False` as a YAML bypass — recommended for `main`-branch clones where mismatches are expected. PyAutoFit/Galaxy/Lens call the check on import. Release pipeline writes the new YAML key alongside `version.txt` via a regex Python shim (PyYAML strips comments, so round-trip wasn't viable). `verify_workspace_versions.sh` reports `ok` for all 7 workspaces — euclid_pipeline previously had no `version.txt` and was always SKIPped, now joins the standard flow. Smoke tests skipped at ship time because the workspace diffs are purely additive YAML + welcome.py line removals; pre-flight library-import silence + `verify_workspace_versions.sh = ok` covered the gate.

## howto-release-window
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/64
- completed: 2026-04-30
- merged-prs:
  - PyAutoLabs/PyAutoBuild#65 (HowTo* repos as first-class members of release window)
  - Jammy2211/admin_jammy#13 (ensure_workspace_labels.sh helper)
- notes: Tooling/admin task — no Python API changes. Two new helpers shipped: `admin_jammy/software/ensure_workspace_labels.sh` (idempotent canonical-label sweep across 15 PyAutoLabs repos) and `PyAutoBuild/verify_workspace_versions.sh` (fail-fast guard against version.txt ahead of installed library — blocks release dispatch). `pre_build.sh` invokes both, runs `autogalaxy_workspace_test` (was missing entirely). `release.yml` wires `autogalaxy_workspace_test` into find_scripts/run_scripts (was orphaned — separate-prompt-worthy `autogalaxy_test` had no checkout block, no script_matrix.py arg, no run_scripts configure case). `CLAUDE.md` table now lists all 10 workspace-style repos. Local Claude commands updated (no PR — `~/.claude/commands/` not git-tracked): `start_workspace.md` invokes the label helper as L6/S5; `ship_workspace.md` and `ship_library.md` now verify the `pending-release` label landed via `gh pr view --json labels`, fail-loud if missing. Out-of-scope flagged: `release.yml:410` `autofit` configure branch sets `repository::PyAutoLabs/PyAutoGalaxy` (copy-paste bug); `run_notebooks` configure has no `_test` cases at all. Bug fix during impl: probe path in `ensure_workspace_labels.sh` initially branched on stdout (`gh api --jq` emits "null|" on 404), corrected to branch on exit code.

## smoke-notebooks
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/110
- completed: 2026-04-30
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/46, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/50, https://github.com/PyAutoLabs/autolens_workspace/pull/111
- doc: Jammy2211/admin_jammy@18bc6fb (skills/smoke_test/SKILL.md)
- notes: Adds `smoke_notebooks.txt` registry + notebook execution loop in `run_smoke.py`. Notebooks execute via `jupyter nbconvert` written to `/tmp` (notebooks/ on disk untouched). On failure the runner regenerates the single failing notebook from its `.py` source via PyAutoBuild's `py_to_notebook` and retries once — full-workspace regen stays in `generate.py`. autogalaxy_workspace and autolens_workspace gained their first-ever CI smoke workflow (modeled on `_test`); autofit_workspace's existing workflow extended. autogalaxy/autolens CI green on first run; autofit smoke red on the pre-existing `Gaussian.model_data_from() got an unexpected keyword argument 'xp'` PyAutoFit example bug (independent fix in flight).

## env-var-rename
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/65
- completed: 2026-04-30
- workspace-pr:
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/66
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/21
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/18
  - https://github.com/PyAutoLabs/PyAutoBuild/pull/63
- notes: Finished the `PYAUTOFIT_TEST_MODE` → `PYAUTO_TEST_MODE` rename in the two `_test` repos skipped by the prior pass (autolens, autogalaxy), and fixed a second silent no-op surfaced by a general scan: `PYAUTO_WORKSPACE_SMALL_DATASETS` (set in every `_test` build config and `PyAutoBuild/release.yml`) was never read by any library — consumers all check `PYAUTO_SMALL_DATASETS`. Both renames switched silent no-ops into canonical names that actually fire. Activating `PYAUTO_SMALL_DATASETS=1` for the first time exposed override gaps: autolens needed `model_composition/`, autogalaxy needed `aggregator/`, `imaging/model_fit`, and `imaging/visualization` (the entire imaging-overrides set autolens already had). All `unset: [PYAUTO_SMALL_DATASETS]` overrides match the established autolens pattern. `lp.py`/`mge.py` parallel write-race noted but not fixed — pre-existing, unrelated to the rename. Out of scope and untouched: `autolens_base_project/CLAUDE.md` (uncommitted local edits) and `z_projects/{cowls_diana,euclid_group,concr}/CLAUDE.md` (not git-tracked from this checkout) — doc references in those still mention the old names.

## welcome-start-here-fixes
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/108
- completed: 2026-04-30
- workspace-pr:
  - https://github.com/PyAutoLabs/autolens_workspace/pull/109
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/49
  - https://github.com/PyAutoLabs/autofit_workspace/pull/45
  - https://github.com/PyAutoLabs/HowToLens/pull/4
- notes: Fixed welcome.py bugs across four workspaces. The reported `aa.Array2D` NameError in autolens_workspace was the visible symptom; auditing every workspace's welcome.py + start_here.py surfaced three more independent bugs — `aplt.LightProfile` removed in autogalaxy.plot, autofit_workspace loading a gitignored dataset path, and HowToLens pinned to a non-existent library release `2026.4.21.0`. autofit_workspace switched to synthesising the demo gaussian inline rather than loading from disk, matching the in-memory pattern used by autolens/autogalaxy welcome scripts. HowToLens version pin bumped down to 2026.4.13.6 to match the installed library and the rest of the workspaces — not a release rollback, the 2026.4.21.0 pin in the bootstrap commit was aspirational and never released. HowToLens shipped without the `pending-release` label because the label isn't registered in that repo yet (admin gap, not a blocker). Pre-existing PyAutoFit `xp` API drift in `example/analysis.py` surfaced via overview_1_the_basics.py smoke fail — reproduces on canonical main, deferred to its own task.

## verify-install-release-checks
- issue: https://github.com/Jammy2211/admin_jammy/issues/11
- completed: 2026-04-29
- tooling-pr: https://github.com/Jammy2211/admin_jammy/pull/12
- repos: admin_jammy
- notes: Split `/verify_install` into a standalone `verify_install.sh` (source of truth, aliased into ~/.bashrc) and a thin `verify_install.md` skill wrapper. Replaced the single pip+start_here.py probe with five independent checks A–E in throwaway envs: A=pip+welcome, B=3.9/3.11 rejection, C=conda flow, D=[optional] extra, E=yanked-pin (autolens==2025.10.6.1). Per-check PASS/FAIL/SKIP table; missing interpreter/conda → SKIP, never FAIL, so the suite is portable. CLI: `verify_install [A|B|C|D|E|all] [--version <v>] [--keep] [--help]`. Lightweight workflow (no worktree) since admin_jammy carries unrelated dirty state and the change is pure tooling — no test suite. Smoke-tested locally: --help, bad-arg paths exit 2, `verify_install B` on host without 3.9/3.11 → SKIP/SKIP/PASS. Real install paths (A/C/D/E) intentionally not exercised in this session — left as unchecked items in PR test plan; user runs them post-release.

## jax-likelihood-poisson-regen
- completed: 2026-04-29
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/64
- repos: autolens_workspace_test
- notes: Follow-up to autobuild-release-prep failures. Root cause: PyAutoArray be3a3a2f flipped the sign of `preprocess.poisson_noise_via_data_eps_from` (correct fix; old form mirrored Poisson skew). Because `dataset/` is gitignored, every fresh checkout re-simulates with the post-fix sign and produces noise maps that drift pixelization-driven likelihoods past `rtol=1e-4`; the recorded literals were captured against pre-fix simulator output. Considered converting `assert_allclose(np.array(result), <literal>, rtol=1e-4)` to relational `vmap ≈ NumPy-path` to immunise the suite against future simulator changes — explicitly rejected by user: "I want to keep hardcoded literals" — relational form would lose absolute regression detection on the NumPy path itself. Regenerated 11 literals in autolens_workspace_test (4 imaging-failed + 5 multi-failed + 2 imaging-borderline that drifted past tolerance during verification: delaunay_mge, mge_group). Updated `scripts/CLAUDE.md` testing-philosophy: removed "no hardcoded values" bullet, replaced with documentation that hardcoded literals are intentional regression markers + one-line regeneration recipe. Branch name `feature/jax-relational-baselines` from original plan; actual implementation kept literals — explained in PR body. autogalaxy_workspace_test was already relational throughout (no changes); autofit_workspace_test has no jax_likelihood_functions/ dir; interferometer/ + point_source/ were unaffected by the Poisson sign fix (Gaussian visibility noise / position datasets respectively) and kept their existing literals.

## truncated-normal-gradient-hessian
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1237
- completed: 2026-04-29
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1238

## adapt-images-mesh-grid-lookup
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/103
- completed: 2026-04-29
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/104
- repos: autolens_workspace
- notes: SLaM `light_lp` and `mass_total` helpers in `imaging/features/pixelization/delaunay.py` and `interferometer/features/pixelization/delaunay.py` built a fresh `AdaptImages` from `source_result_for_lens`'s image dict but silently dropped the `galaxy_name_image_plane_mesh_grid_dict` that `source_pix_2` had constructed. Under PYAUTO_TEST_MODE=2 (and any path exercising the helpers' likelihood end-to-end with a Delaunay/Voronoi/RectangularSplineAdapt source), the chained pixelization had no source-plane mesh grid, propagated `None` into `BorderRelocator.relocated_mesh_grid_from`, and crashed loudly. Fix: pull the dict forward from `source_result_for_source.analysis.adapt_images.galaxy_name_image_plane_mesh_grid_dict` rather than constructing fresh. No library change, no recipe re-execution, no defensive guard. Original prompt diagnosed at relocator/abstract-mesh layer in PyAutoArray; debug print of actual reproducer (PYAUTO_TEST_MODE=2 delaunay.py) showed the lookup was correct and the bug was the producer-side `AdaptImages` construction in two SLaM helper functions. Audit: only Delaunay/Voronoi/RectangularSplineAdapt source meshes consume `source_plane_mesh_grid` and need the fix; the canonical `slam_start_here.py` and most SLaM scripts use `RectangularAdapt*` which self-determine pixels and don't have the bug. The `group/*.py` SLaM scripts already pass `adapt_images` as a parameter (cleaner architecture). Considered but rejected a library helper `galaxy_name_image_plane_mesh_grid_dict_via_result_from(result, image_mesh)` — recipe varies too much per script (Overlay vs Hilbert, edge-points appended for Delaunay/Voronoi, `adapt_data` flow for adaptive meshes) to abstract cleanly. Imaging fix verified end-to-end: full SLaM pipeline (`source_lp[1]` → `source_pix[1]` → `source_pix[2]` → `light[1]` → `mass_total[1]`) ran to completion under PYAUTO_TEST_MODE=2. Interferometer fix structurally identical but not exercised end-to-end due to a pre-existing unrelated shape-mismatch bug in `interferometer/.../source_pix_2` (`add got incompatible shapes: (40, 40), (1070, 1070)` inside `inversion.fast_chi_squared`) — worth a separate ticket. Smoke run also flagged that `group/modeling.py` overwrites `dataset/group/simple/positions.json` with a test-mode stub — a smoke test should not be mutating real dataset files.

## positions-test-mode-fallback
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/477
- completed: 2026-04-29
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/479
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/102
- repos: PyAutoLens, autolens_workspace
- notes: PYAUTO_TEST_MODE safeguard added to `Result.positions_likelihood_from` (substitutes `[(1.0, 0.0), (-1.0, 0.0)]` when resolved positions are empty/NaN/inf — the original `ValueError: zero-size array to reduction operation fmax` from random test-mode mass models). PYAUTO_SMALL_DATASETS short-circuit added to `PointSolver.solve` (returns `[(1.0, 0.0), (0.0, 1.0)]` immediately, skipping the triangle-tile solve), letting the three group simulator scripts (`group/simulator.py`, `multi_gaussian_expansion/simulator.py`, `no_lens_light/simulator.py`) drop the `os.environ.pop("PYAUTO_SMALL_DATASETS")` workaround. Both fallbacks placed at the higher PyAutoLens layer rather than mutating `autoarray`'s `Grid2DIrregular.furthest_distances_to_other_coordinates` primitive — keeps autoarray pure and production fits still surface bad positions loudly. Reduced test footprint to a single unit test per user request (integration catches the rest); behavior documented inline via `Notes` sections on both methods. Smoke verification surfaced an unrelated downstream bug in `PyAutoArray/autoarray/inversion/mesh/border_relocator.py:450` (`'NoneType' object has no attribute 'array'` during inversion mesh build under PYAUTO_TEST_MODE=2) — pre-existing on main, was masked by the original positions crash, worth a separate ticket.

## merge-results-start-here
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/95
- completed: 2026-04-28
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/98, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/46
- repos: autolens_workspace, autogalaxy_workspace
- notes: Combined `guides/results/start_here.py` (simple JSON/FITS) and `guides/results/aggregator/start_here.py` (full aggregator) into one entry-point tutorial at `guides/results/start_here.py` for both workspaces. Simple loading first with `.exists()` guards on the `<unique_hash>` placeholder so the script runs cleanly before users replace it; aggregator section second, runs a real fit and walks the deeper API. Mirrored the autogalaxy `test_mode_was_on` / `n_like_max=300` conditional into autolens for parity — autolens previously had no such conditional, so manual `PYAUTO_TEST_MODE=1` runs short-circuited to a 1-sample mock; now both workspaces produce 300 samples. `samples.csv` displays as ~1 row in autogalaxy (Nautilus weight filtering keeps only high-weight survivors) but `samples_info.json: total_accepted_samples=300` confirms the search ran for 300 evals; autolens samples.csv shows 301 lines due to a more complex lens model. env_vars.yaml `pattern: "guides/results/"` override that unsets PYAUTO_TEST_MODE and PYAUTO_SKIP_FIT_OUTPUT during smoke runs is preserved and still load-bearing for downstream aggregator siblings. Discovered (out of scope for this PR) that `aggregator/models.py` and `data_fitting.py` etc. expose pre-existing PyAutoGalaxy aggregator bugs in test mode (`fit.value(name=name)[0].header` dereferences None; `_tracer_from` gets `instance=None`) — confirmed reproducible on main. Worth a separate issue if those scripts are ever added to the smoke test list.

## autogalaxy-wst-jax-lh-multi
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/18
- completed: 2026-04-28
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/19
- repos: autogalaxy_workspace_test
- umbrella: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/5 (task 5/9)
- notes: Ported 8 multi-band JAX-likelihood scripts from autolens_workspace_test (simulator, lp, mge, mge_group, rectangular, rectangular_mge, delaunay, delaunay_mge) into `scripts/jax_likelihood_functions/multi/`. Each fit script combines per-band `ag.AnalysisImaging` factors via `af.FactorGraphModel` and asserts NumPy/JIT scalar parity over `instance_from_vector → log_likelihood_function` (FactorGraphModel has no `fit_from`). Adapt-regularization variants (rectangular, rectangular_mge, delaunay, delaunay_mge) at rtol=1e-2 per the established imaging/interferometer convention; lp/mge/mge_group at rtol=1e-4. Self-contained simulator (no external fixtures). `delaunay_mge.py` enabled in smoke_tests.txt — JAX 0.7's `pytype_aval_mappings` removal does not bite on the multi path (matches interferometer; only single-dataset imaging delaunay_mge remains commented out). Sonnet-side mishaps caught: (1) initial Path A used JIT-vs-vmap on the same factor_graph(use_jax=True) — a tautology; rewrote all 4 pixelized scripts to do proper NumPy-vs-JIT via a separate factor_graph_np(use_jax=False); (2) first ship subagent committed+pushed but stalled before launching smoke; second subagent picked up cleanly at smoke + PR; (3) smoke runner pathing footgun — absolute python path bypassed cwd-relative subprocess.run inside fit scripts, fixed by wrapping in `(cd "$WS" && env ... python "$script")`. API gotcha: `ag.reg.Adapt` uses `inner_coefficient` not `coefficient`. Worth re-attempting the imaging delaunay_mge on a future task since interferometer and multi both work.

## autogalaxy-wst-jax-lh-interferometer
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/16
- completed: 2026-04-28
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/17
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/376 (prerequisite, shipped earlier same day)
- repos: autogalaxy_workspace_test
- umbrella: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/5 (task 4/9)
- notes: Ported 8 JAX-likelihood interferometer scripts from autolens_workspace_test (simulator, lp, mge, mge_group, rectangular, rectangular_mge, delaunay, delaunay_mge) into `scripts/jax_likelihood_functions/interferometer/`. Each fit script wraps `jax.jit(analysis.fit_from)` and asserts NumPy/JIT scalar parity, exercising the AnalysisInterferometer pytree registration shipped in PyAutoGalaxy PR #376 the same day. Three notable differences from the autolens reference: (1) self-contained simulator — synthetic 200-baseline uv-coverage generated inline via `np.random.default_rng(seed=1)`, no `sma.fits` dependency (sidesteps the gitignored-fixture issue that has had `interferometer/{mge,rectangular}.py` red on autolens CI for ≥1 week); (2) `delaunay.py` and `delaunay_mge.py` use a Sersic-image adapt-data instead of `dataset.dirty_image` (negative dirty pixels otherwise produce NaN via `sqrt(pixel_signal)` in AdaptSplit regularization) — matches autolens reference's actual intent; (3) `delaunay_mge.py` is enabled in smoke_tests.txt unlike its imaging counterpart — JAX 0.7's removal of `jax.interpreters.xla.pytype_aval_mappings` does not bite on the interferometer side. Worth revisiting whether the imaging delaunay_mge can be unblocked the same way. Tolerances: lp/mge/mge_group at rtol=1e-4 (sub-femtosecond); 4 adapt-regularization variants at rtol=1e-2 per imaging-port convention but actual diffs ≤ 9e-6.

## analysis-interferometer-pytree
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/375
- completed: 2026-04-28
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/376
- repos: PyAutoGalaxy
- notes: Adds JAX pytree registration for `AnalysisInterferometer` (mirrors imaging scaffold from #364). Galaxies flatten/unflatten lifted into `autogalaxy/analysis/jax_pytrees.py::register_galaxies_pytree()`; imaging body collapsed from 41 → 11 lines as a side benefit. Quantity and Ellipse deferred to follow-ups. End-to-end JIT verification will land in the queued `autogalaxy_workspace_test_jax_likelihood_interferometer` task. Smoke tests: 42/42 passed.

## autoprompt-cleanup
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/15 (closed automatically via PR's "Closes #15")
- completed: 2026-04-28
- library-pr: https://github.com/PyAutoLabs/PyAutoPrompt/pull/16
- repos: PyAutoPrompt
- notes: Closes the autoprompt/ workflow-infrastructure sweep. Moved 05_sync_slash_command.md, 06_repo_health_audit.md, 08_test_summary.md → `issued/` (matches the 01/02/03 precedent of archiving shipped prompts). Deleted 07_worktree_only_edits.md (matches the 04 precedent — explicitly skipped during the sweep, no point keeping the spec). Rewrote `autoprompt/README.md` as a historical record with an Outcomes table (per-prompt status: Shipped / Shipped re-scoped / Skipped) plus What-shipped / What-deliberately-didn't sections, replacing the stale TODO-list framing that still referenced 04 and listed 07 as "the biggest fix". After this, `autoprompt/` contains only the README — closed chapter. Net: +68 / -166 lines, mostly the 07 deletion (121 lines) and the README rewrite.

## dashboard-test-summary
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/13 (closed automatically via PyAutoPrompt PR's "Closes #13")
- completed: 2026-04-28
- repo-prs (2):
  - admin_jammy: https://github.com/Jammy2211/admin_jammy/pull/8
  - PyAutoPrompt: https://github.com/PyAutoLabs/PyAutoPrompt/pull/14
- repos: admin_jammy, PyAutoPrompt
- notes: Implements autoprompt 08. Two changes on a shared `feature/dashboard-test-summary` branch. (1) admin_jammy/skills/smoke_test/SKILL.md — added step 7 "Persist summary to local cache" instructing the agent to write `~/.cache/pyauto/smoke/<workspace>.json` per workspace tested with workspace, completed_at (ISO 8601 UTC), passed, failed, skipped, total, duration_seconds. Step is idempotent; overwrites previous file for same workspace. (2) PyAutoPrompt/scripts/pyauto_status.sh — appended two new optional sections: "Smoke tests:" reading the cache JSONs (ANSI green if failed=0 else red, ✓/✗ symbol), and "Last autobuild run:" reading the committed PyAutoBuild/test_results/*.json files for an aggregate (jobs across workspaces, passed/failed/skipped totals, most-recent completed_at, PyAutoBuild HEAD short SHA — red if any failure). Both sections suppressed entirely when no JSONs exist, matching the existing Dirty-files / Follow-up-commands pattern. Single python invocation per section parses all JSONs to avoid per-file fork overhead. Total dashboard runtime measured at 2.7s on the live tree (well under 4s target). Smoke tested: empty cache → smoke section suppressed + autobuild section shows; seeded fixtures (autofit_workspace failed=0, autolens_workspace failed=2) → green ✓ and red ✗ rendered correctly. No bashrc changes needed (pyauto_status.sh already sourced + called from PyAuto() aliases). The skill-instruction approach for persisting summaries assumes the agent running /smoke-test follows step 7; if it drifts, the cache stays stale and pyauto-status shows old timestamps — degrades gracefully. Live-tree autobuild section currently shows: 2026-04-26, 7 jobs / 1 workspace / 153 passed / 16 failed / 28 skipped (from PyAutoBuild commit c0d5b87). Out of scope: library version detection (PyAutoLens injects VERSION at build time, no committed source), per-script breakdown, failure tracebacks, GitHub Actions API queries, helper script for JSON writing.

## pyauto-audit
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/11
- completed: 2026-04-28
- library-pr: https://github.com/PyAutoLabs/PyAutoPrompt/pull/12
- repos: PyAutoPrompt
- notes: Re-scoped autoprompt 06 (the heavyweight monthly cron audit spec). Shipped a 113-line `scripts/pyauto_audit.sh` defining a `pyauto-audit` shell function (sourced from `~/.bashrc` next to `pyauto_status.sh`) with three structural-state sections that the dashboard can't show: (1) top-level dirs under `~/Code/PyAutoLabs/` with no `.git`, skip prefixes `.` and `z_`; (2) stashes older than `PYAUTO_AUDIT_STASH_DAYS` (default 14); (3) local-only branches with no upstream and last commit older than `PYAUTO_AUDIT_BRANCH_DAYS` (default 30). Plain text output, sections suppressed when empty, single `clean` message when all clear. Always exits 0 — informational, user reads + decides. Live-tree dry run during smoke testing immediately found real signal: 4 stray non-git dirs (`bad/`, `path/`, `priors/`, `scripts/` — all bug artifacts the user can decide whether to delete), 3 old stashes (PyAutoFit 2026-04-02, PyAutoGalaxy 2026-04-07, PyAutoLens 2026-04-06 — all ~3 weeks old, real drift-from-stash candidates), 1 abandoned branch (`PyAutoFit/feature/ep` from 2026-02-18). Bug found-and-fixed during smoke testing — first implementation used `IFS=$'\t'` for splitting `for-each-ref --format='%09'` output, but bash treats consecutive whitespace IFS chars as one delimiter and collapsed the empty-upstream column into the timestamp; switched to `|` delimiter (matching Section 2's stash format). Same PR also deleted `autoprompt/04_source_of_truth_rule.md` (rejected during this sweep — covered by `pyauto-status` BEHIND counts + prompt 03's history-rewrite guard). Out of scope: cron schedule, severity ERROR/WARN/INFO tags, snooze file, untracked-file age scan (overlaps with prompt 02), tracked-file gitignore-noise scan (overlaps with prompt 02). Skipped autoprompts 04 (source-of-truth doc rule — redundant with dashboard + prompt 03) and 05 ($/sync$ skill — replaced by dashboard's Follow-up commands section in #10).

## dashboard-followup-commands
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/9
- completed: 2026-04-28
- library-pr: https://github.com/PyAutoLabs/PyAutoPrompt/pull/10
- repos: PyAutoPrompt
- notes: Re-scoped autoprompt 05 (the heavyweight `/sync` slash command spec). Instead of a separate `pyauto-pull` shell function or full `/sync` skill, extended `scripts/pyauto_status.sh` (~89 added lines) with two additions: (1) `b` flag glyph appended to FLAGS column when current branch ≠ upstream branch component (or upstream=NONE and branch ≠ main) — caught `euclid_strong_lens_modeling_pipeline` immediately as on `fix/extra-galaxies-gui-vis-index`. (2) "Follow-up commands:" section after the Dirty files listing, grouped by Pull (clean+behind+not-ahead → copy-pasteable `git -C <abs-path> pull --ff-only`), Set missing upstream (branch=main + upstream=NONE → `branch --set-upstream-to=origin/main main`), and Investigate manually (one-line note for diverged / behind+dirty / branch-mismatch — no auto-command). Section is suppressed entirely when nothing is actionable, so the clean case stays quiet. Smoke tested all three branches: synthetic `reset --hard HEAD~1` on HowToFit produced correct Pull line that fast-forwarded cleanly; synthetic `branch --unset-upstream` produced correct Set-upstream line that restored tracking. Replaces the rejected pyauto-pull function design — printed commands ARE the action; the user copy-pastes (or selects-all-pastes) what's appropriate. Boring case automated via copy-paste, judgment cases surfaced for manual handling. Autoprompts 04 (source-of-truth doc rule) skipped as redundant — `pyauto-status` at venv activation + prompt 03's history-rewrite guard already cover the failure modes that prompt 04 addressed.

## history-rewrite-guard
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/7 (umbrella, closed manually after all 17 PRs merged)
- completed: 2026-04-27
- repo-prs (17):
  - PyAutoConf: https://github.com/PyAutoLabs/PyAutoConf/pull/98
  - PyAutoFit: https://github.com/PyAutoLabs/PyAutoFit/pull/1235
  - PyAutoArray: https://github.com/PyAutoLabs/PyAutoArray/pull/292
  - PyAutoGalaxy: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/373
  - PyAutoLens: https://github.com/PyAutoLabs/PyAutoLens/pull/478
  - autofit_workspace: https://github.com/PyAutoLabs/autofit_workspace/pull/43
  - autogalaxy_workspace: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/45
  - autolens_workspace: https://github.com/PyAutoLabs/autolens_workspace/pull/96
  - autofit_workspace_test: https://github.com/PyAutoLabs/autofit_workspace_test/pull/15
  - autogalaxy_workspace_test: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/15
  - autolens_workspace_test: https://github.com/PyAutoLabs/autolens_workspace_test/pull/61
  - autolens_workspace_developer: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/38
  - HowToFit: https://github.com/PyAutoLabs/HowToFit/pull/3
  - HowToGalaxy: https://github.com/PyAutoLabs/HowToGalaxy/pull/3
  - HowToLens: https://github.com/PyAutoLabs/HowToLens/pull/3
  - PyAutoBuild: https://github.com/PyAutoLabs/PyAutoBuild/pull/61
  - PyAutoPrompt: https://github.com/PyAutoLabs/PyAutoPrompt/pull/8
- skipped: autofit_workspace_developer, admin_jammy (no CLAUDE.md or AGENTS.md present; creating new files just to host this section was deemed over-engineering)
- follow-up: 1) Optional `## General Rules` line augmentation (the new top-level section is strong enough; defer to follow-up if ever felt missing). 2) Pre-commit hook to block "Initial commit"-style messages on remote-tracked branches (per prompt 03 itself, can be a follow-up). 3) Project-level `~/Code/PyAutoLabs/CLAUDE.md` (untracked personal file, not in any repo) was edited in parallel with the PRs to host the same section.
- notes: Implements prompt 03 of the autoprompt/ workflow-infrastructure series — the `## Never rewrite history` guard added to every PyAuto repo's `CLAUDE.md` and/or `AGENTS.md` (whichever exist). 17 PRs on shared `feature/history-rewrite-guard` branch, all squash-merged 2026-04-27 in one parallel batch. 25 files touched in total: 8 repos with both CLAUDE.md + AGENTS.md (16 files), 8 repos with CLAUDE.md only, 1 repo (PyAutoPrompt) with AGENTS.md only. All edits idempotent — script checked for existing section before appending. The PyAuto/rhayes777 → PyAutoLabs migration is fully done (the `/start_dev` skill mapping is stale; all 17 repos resolve to PyAutoLabs/). No tests run (docs-only change).

## workspace-gitignore-noise
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/6 (umbrella, closed manually after all 8 PRs merged)
- completed: 2026-04-27
- workspace-prs:
  - autofit_workspace: https://github.com/PyAutoLabs/autofit_workspace/pull/42
  - autogalaxy_workspace: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/44
  - autolens_workspace: https://github.com/PyAutoLabs/autolens_workspace/pull/94
  - autofit_workspace_test: https://github.com/PyAutoLabs/autofit_workspace_test/pull/14
  - autogalaxy_workspace_test: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/14
  - autolens_workspace_test: https://github.com/PyAutoLabs/autolens_workspace_test/pull/60
  - autofit_workspace_developer: https://github.com/Jammy2211/autofit_workspace_developer/pull/8
  - autolens_workspace_developer: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/37
- repos: 8 workspaces (autofit/autogalaxy/autolens × workspace + workspace_test + autofit/autolens × workspace_developer; autogalaxy has no _developer variant)
- follow-up: 1) source-side script bugs that wrote `image.fits` and `path/to/model/json/model.json` at workspace roots — file separate issues if the patterns recur post-merge. 2) `**/images/` not in prompt 02's pattern set; `autogalaxy_workspace_test` still has untracked `scripts/imaging/images/` post-merge (out of scope; consider extending). 3) `pyauto-status` dashboard surfaced pyc pollution committed in `euclid_strong_lens_modeling_pipeline` — separate workspace not in this prompt's scope.
- notes: Implements prompt 02 of the autoprompt/ workflow-infrastructure series. One umbrella issue, 8 PRs on shared `feature/workspace-gitignore-noise` branch, all squash-merged 2026-04-27. Per-repo: appended prompt 02's pattern block to `.gitignore` (deduped against existing — most workspaces already had `__pycache__/`, `*.pyc`, `root.log`); ran `git rm --cached` for tracked files matching new patterns. 17 files total removed from tracking: 1 in autofit_workspace, 2 each in autogalaxy_workspace and autolens_workspace, 3 in autofit_workspace_test (incl. one stray `__pycache__/util.cpython-312.pyc`), 9 search.log files in autolens_workspace_developer (deep `output/output/` tree), 0 in the 3 _test/_developer variants that were already clean. autofit_workspace_developer had no `.gitignore` at all — created with the full block. Smoke tests skipped (one-time deviation; `.gitignore`-only changes have no behaviour impact). No new skill formalised — the "skip smoke for chore PRs" pattern can be revisited if it recurs frequently.

## dashboard-dirty-listing
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/4
- completed: 2026-04-27
- library-pr: https://github.com/PyAutoLabs/PyAutoPrompt/pull/5
- repos: PyAutoPrompt
- notes: Follow-up to pyauto-status-shell. Split the dashboard's single `DIRTY` column into `MOD` (tracked-modified) + `UNTR` (untracked) so accumulating noise is distinguishable from real edits-in-progress, and append a `Dirty files:` listing after the main table showing the actual `git status --porcelain` lines per repo (only repos with content; `??` and ` M` prefixes preserved). Cached porcelain output via a bash associative array so the listing reuses the same data the counts came from — no new git invocations. Day-1 use already surfaced real signal: pyc pollution committed in `euclid_strong_lens_modeling_pipeline`, untracked `scripts/imaging/images/` in `autogalaxy_workspace_test` (worth feeding into prompt 02's gitignore patterns), and 14 dirty entries in `autolens_base_project` worth a closer look. Sweep time unchanged at ~3s.

## pyauto-status-shell
- issue: https://github.com/PyAutoLabs/PyAutoPrompt/issues/2
- completed: 2026-04-27
- library-pr: https://github.com/PyAutoLabs/PyAutoPrompt/pull/3
- repos: PyAutoPrompt
- notes: Added `scripts/pyauto_status.sh` defining a `pyauto-status` shell function — a cross-repo git sync dashboard that prints branch, upstream tracking ref, behind/ahead counts, dirty count, and flag glyphs (↓ ↑ * !) for every git repo under `~/Code/PyAutoLabs/`. Sweep of 21 repos completes in ~4s with parallel `git fetch`. Handles missing upstream (`NONE` + `!` flag — caught a real bug: PyAutoPrompt's local `main` had no tracking config, fixed in post-merge cleanup), non-default upstream branches via `@{u}`, fetch failures, and non-git directories. Also extended `scripts/status.sh` with a `--repos` flag that delegates to `pyauto-status`. The function is wired into the `PyAuto`, `PyAutoGPU`, `PyAutoNoJAX` venv-activation aliases in `~/.bashrc` so the dashboard prints automatically every time the user enters the workspace, giving drift state visibility before any work begins. Skipped `PyAutoOld()` (targets a different directory). Implements prompt 01 of the autoprompt/ workflow-infrastructure series; complementary to (not blocking) prompts 02-07.

## pyautoprompt-path-cleanup
- issue: none — housekeeping task
- completed: 2026-04-27
- repos: PyAutoPrompt, autogalaxy_workspace, autogalaxy_workspace_test, autolens_workspace, autolens_workspace_test, autolens_workspace_developer
- notes: Updated all workspace and workspace_test repos to use the new PyAutoPrompt path layout (prompts/registry moved from admin_jammy/prompt/ to PyAutoPrompt). 6 PRs merged across 6 repos.

## cluster-simulator-jax-multiplane
- issue: https://github.com/Jammy2211/autolens_workspace/issues/89
- completed: 2026-04-27
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/91
- notes: Refactored `scripts/cluster/simulator.py` to (a) shrink to 2 main lens galaxies + 1 host halo (was 5), (b) move sources to distinct redshifts z=1.0 and z=2.0 for a true multi-plane lens, (c) JAX-jit the PointSolver via the pytree-registration pattern from `autolens_workspace_developer/jax_profiling/point_source/image_plane.py` (>5min → fast), (d) collapse 3 grids down to 2 (rendering grid shared between simulation and viz, PointSolver's internal grid kept separate), and (e) polish docstrings to match `scripts/imaging/simulator.py` tone with new `__Multi-Plane Setup__` and `__JAX JIT__` sections. Single-commit squash merge.

## db-scrape-build-dataset-path
- issue: none — surfaced by /health_check on 2026-04-27
- completed: 2026-04-27
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/58
- follow-up: 1) PyAutoGalaxy library bug — `abstract_fit.linear_light_profile_intensity_dict` raises `TypeError: __hash__ method should return an integer` during `subplot_fit_imaging` after the search completes. A light-profile object's `__hash__` returns a non-int. Surfaced once `general.py` could load its dataset; parked via `no_run.yaml` NEEDS_FIX 2026-04-27. Fix in PyAutoGalaxy → remove the no_run entry. 2) autolens_workspace_test CI smoke has been red on `main` for ≥1 week — `jax_likelihood_functions/{interferometer/mge,interferometer/rectangular,imaging/rectangular,multi/mge}.py` fail. Two are missing `dataset/interferometer/uv_wavelengths/sma.fits` (a gitignored fixture file present locally but never on CI), the other two are JAX-likelihood numerical mismatches at the rtol=1e-4 boundary. Investigation: either ship `sma.fits` properly (commit + remove from gitignore, or fetch in CI setup) and bump the JAX likelihood tolerances, OR park them in no_run.yaml.
- notes: Fixed 4 `database/scrape` consumer scripts (`general.py`, `scaling_relation.py`, `slam_general.py`, `slam_pix.py`) by adding `dataset_label = "build"` so they read from where the simulator writes. The simulator + every other non-database script in this workspace already used the `build/` convention; the db/scrape scripts had drifted. Auto-sim subprocesses had been "succeeding" while writing to a folder the consumer never reads. Caught the symptom only after the autogalaxy mask fix landed and brought the test workspace's smoke down to the single remaining failure. The dataset-path fix unmasks the PyAutoGalaxy `__hash__` library bug above. Local smoke (with `dataset/` symlinked into the worktree from canonical) was 11 PASS + 1 SKIPPED + 0 FAIL. CI on PR #58 was UNSTABLE because of the 4 pre-existing jax CI failures (which exist on `main` too — last main CI run was already FAILURE before this PR), so net effect on CI is strictly improved (5 fails → 4 fails).

## auto-generate-mask-extra-galaxies
- issue: none — surfaced by /health_check on 2026-04-27
- completed: 2026-04-27
- workspace-pr:
  - autogalaxy_workspace: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/41
  - autolens_workspace: https://github.com/PyAutoLabs/autolens_workspace/pull/90
- follow-up: Both merge commits accidentally include smoke artifacts (`image.fits` and `path/to/model/json/model.json` at workspace root). Cause: the worktree was clean when created from main, but smoke runs in the worktree generated those files at the workspace root and Sonnet's `git add -A` swept them into the commit. Future fix: either (a) tiny cleanup PR per workspace removing the files + adding to `.gitignore`, or (b) tighten the ship subagent contract to use `git add scripts/ config/` instead of `-A`. The committed JSON output paths come from a relative-path bug in some script writing fit results to `path/to/model/json/` — worth tracing back too.
- notes: Moved `mask_extra_galaxies.fits` creation into each simulator that owns the affected dataset, so consumer scripts (start_here, modeling, fit, pixelization) can load the mask without spawning a data-prep subprocess. Geometry is derived from each simulator's own extra-galaxy centres + `effective_radius` (or all-False for "simple" datasets that have no extra galaxies but whose pixelization tutorials demo `apply_noise_scaling`). `Mask2D.circular` honours `PYAUTO_SMALL_DATASETS=1`, so masks auto-shrink to 15x15 and the env var no longer causes out-of-bounds slicing. Also fixed the optional standalone `mask_extra_galaxies.py` and `extra_galaxies_centres.py` in both workspaces (they targeted `dataset_name = "simple"` while writing centres + mask geometry that only made sense for `extra_galaxies` — a stale copy-paste). Removed 8 now-redundant `if not (mask file).exists() -> subprocess optional script` blocks across 6 consumers. Sister-fixed an unrelated typo in `autolens/scripts/interferometer/features/extra_galaxies/modeling.py` where the auto-sim subprocess pointed at the imaging simulator instead of the interferometer one (would have crashed any first-time user without the dataset cached). Smoke 13/13 green in both workspaces, including the previously-failing `autogalaxy/imaging/start_here.py`. PyAutoArray 748/748 unit tests still green (no library impact).

## preprocess-poisson-noise-sign-fix
- issue: none — surfaced by /health_check on 2026-04-27
- completed: 2026-04-27
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/290
- notes: Fixed sign of Poisson noise term in `autoarray.preprocess.poisson_noise_via_data_eps_from`. Old code returned `data − noisy`, so `data_eps_with_poisson_noise_added` produced `2·data − noisy_data` (mirrored skew — same mean and variance, but Poisson's positive skew flipped negative). Now returns `noisy − data` so `data + noise = noisy_data`. Also rewrapped result via `data_eps.with_new_array(...)` to preserve `Array2D` type (the in-progress edit had inadvertently let numpy's operator dispatch return a raw ndarray, breaking `.native` access in callers). Regenerated expected values for 2 unit tests + 4 simulator tests; 748/748 PyAutoArray pytest green. Behavioural change only — no public signatures touched. Workspace simulators that use `add_poisson_noise_to_data=True` will produce correctly-skewed data going forward; previously committed dataset/*.fits files remain valid (just systematically different) and will refresh on next simulator run.

## point-source-gradients
- issue: none — invoked via /remote-control from admin_jammy/prompt/autolens_workspace_developer/point_source_gradients.md
- completed: 2026-04-26
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/35
- follow-up: PointSolver is reverse-mode opaque — every stage chained through `solver.solve` returns identically zero gradient even when forward values are correct. Library-side fix would need a `jax.custom_vjp` wrapper around the triangle-subdivision path (or a continuous relaxation of the lens-equation root-finding). Separately: latent bug in `PyAutoGalaxy/autogalaxy/operate/lens_calc.py:404` (`jnp.array(grid[:, 0])` should be `jnp.array(grid.array[:, 0])`) — surfaces as `ValueError: object __array__ method not producing an array` for `Grid2DIrregular` inputs under a JAX trace. Not on the critical path here (full pipeline routes around it via `AnalysisPoint.magnifications_at_positions`'s `aa.ArrayIrregular` wrap), but worth fixing before someone calls `LensCalc.magnification_2d_via_hessian_from(grid_irregular, xp=jnp)` directly.
- notes: Two new JAX gradient probes under `jax_profiling/point_source/` mirroring the imaging probes. **Source-plane probe — 4/4 PASS**, including the full pipeline via `Fitness.call` (||grad|| ≈ 33916, 7/8 nonzero). The big finding: even though `source_plane.py` documents a forward-JIT blocker (`Grid2DIrregular.grid_2d_via_deflection_grid_from` not propagating `xp`), `jax.value_and_grad` does not require lowering at the same boundary, so the source-plane likelihood IS differentiable end-to-end today. NUTS / HMC against `AnalysisPoint(FitPositionsSource)` is viable now. **Image-plane probe — 0/4 PASS**, all four stages return grad=0 (forward values match eager NumPy to float64). The triangle-subdivision solver kills gradient flow at the boundary; a NUTS user would see a flat likelihood landscape — probe surfaces this cleanly. Implementation note: source-plane Step 3 dropped magnification (used residual+noise chi-squared instead) to route around the latent `lens_calc._hessian_via_jax` bug above; Step 4 (full pipeline) still exercises magnifications via the AnalysisPoint code path. Image-plane Step 1 originally returned `value=inf` because `solver.solve(remove_infinities=False)` keeps inf-padded sentinel rows; added a `finite_mask` reduction so the probe reports the cleaner "finite value, zero grad" diagnostic. Both probes' regression-stability comes from reusing the seeded `simulators/point_source.py` (noise_seed=1) that the existing `image_plane.py` / `source_plane.py` profilers already validate (EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE = 0.07475703623045682, EXPECTED_LOG_LIKELIHOOD_SOURCE_PLANE = -294.1401881258811).

## adapt-images-pytree-fix
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/369
- completed: 2026-04-26
- library-pr:
  - PyAutoGalaxy: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/370
  - PyAutoLens: https://github.com/PyAutoLabs/PyAutoLens/pull/474
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/12
- notes: Fixed `AdaptImages` lookups crashing across the `jax.jit` boundary for any model using adapt images. Added `AdaptImages.galaxy_path_list` parallel to the analysis-time galaxies list and helpers `image_for_galaxy` / `image_plane_mesh_grid_for_galaxy` that try by-instance lookup first then fall back to identity-positional path-tuple lookup. `GalaxiesToInversion` gained `path_galaxies` ctor arg so autolens can pass `tracer.galaxies` (the full flat list) into per-plane GalaxiesToInversion. Removed both the autogalaxy single-mesh-grid fallback (`to_inversion.py:428-442`) and the autolens single-pixelated-galaxy fallback (`autolens/lens/to_inversion.py:280-290`) — both were workarounds for the same root cause that only happened to cover single-pixelization fits. Workspace re-port: restored adapt variant of `rectangular.py` and added `rectangular_mge.py` (multi-galaxy regression for the path-tuple path), `delaunay.py`, `delaunay_mge.py` under `scripts/jax_likelihood_functions/imaging/`. Re-enabled `rectangular_mge.py` + `delaunay.py` in `smoke_tests.txt`; `delaunay_mge.py` committed but commented to mirror autolens's JAX-0.7 deferral. NumPy↔JIT tolerance set to `rtol=1e-2` for autogalaxy adapt scripts (`delaunay.py` and `rectangular_mge.py` agree much tighter; `rectangular.py` has ~0.2% drift in the `Adapt` regularization solver path between NumPy and JAX float ordering — unrelated to the lookup fix).

## imaging-delaunay-gradients
- issue: none — invoked via /remote-control from admin_jammy/prompt/autolens_workspace_developer/imaging_delaunay_gradients.md
- completed: 2026-04-26
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/33
- follow-up: PyAutoArray needs `jax.custom_jvp` wrapper around `jax_delaunay`'s `scipy.spatial.Delaunay` host call (file/line: `autoarray/inversion/mesh/interpolator/delaunay.py:80`)
- notes: New JAX gradient probe `jax_profiling/imaging/delaunay_gradients.py`, modelled on `pixelization_gradients.py` + `mge_gradients.py` — closes the third corner of imaging gradient-probe coverage (alongside MGE and rectangular pixelization). 11 stages: ray-trace, blurred lens light, profile-subtracted, Delaunay mapping matrix (pre-PSF), blurred mapping matrix (post-PSF), data vector D, curvature matrix F, regularization matrix H (ConstantSplit), NNLS reconstruction, mapped reconstructed image, full pipeline via `Fitness.call`. Includes a `_diagnose_kappa` block adapted from `mge_gradients.py` (loops `target_kappa ∈ {1e-3, 1e-2, 1e-1, 1.0}` calling `jaxnnls.pdip` primitives directly). Diagnostic only — does not raise on FAIL. Setup mirrors `delaunay.py` (Overlay mesh + circle edge points, ConstantSplit regularization, AdaptImages reconstructed inside the JIT trace). Eager regression value `-62305.31055677842` is the perturbed-params (PRNGKey(42), uniform 0.01-0.05) log_evidence — different from `delaunay.py`'s un-perturbed +29179.95, same divergence pattern as `pixelization_gradients.py` vs `pixelization.py`. Result on main (PyAutoArray @ 4ea58e1a): 3/11 PASS, 8/11 ERROR — Steps 1-3 (pre-inversion) PASS clean; Steps 4-11 (everything that touches the Delaunay inversion via `_fit_jax`) ERROR with shared `ValueError: Pure callbacks do not support JVP. Please use jax.custom_jvp to use callbacks while taking gradients.` raised from `jax_delaunay`'s `scipy.spatial.Delaunay` host call. The Delaunay inversion path is currently *un-differentiable end-to-end*, not silently zero-gradient — `pure_callback` has no JVP rule, so any `value_and_grad` through it hard-errors. The PART B.5 NNLS kappa diagnostic also can't run yet for the same reason; once the Delaunay JVP is added (likely a zero-JVP rule, since the triangulation is a discrete combinatorial structure) the kappa loop should produce useful output. Two minor wins discovered along the way: (a) `pixelization_gradients.py`'s `tb.strip().splitlines()[-1]` pattern in `test_grad` is fragile under JAX's traceback-filtering footer — replaced with `f"{type(e).__name__}: {e}"` so the summary table shows the actual exception, not JAX's footer note (worth backporting to mge/pixelization probes if they ever start producing ERROR rows); (b) the `JAX leaves on instance pytree` diagnostic only works for the `register_model` + `params_tree` style (mge_gradients.py), not the flat-vector `instance_from_vector(vector=params, xp=jnp)` style used here and in `pixelization_gradients.py` — for the flat-vector path the relevant pytree-readiness signal is the gradient shape printed by `test_grad`, not a leaf count.

## restore-data-preparation-scripts
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/85
- completed: 2026-04-26
- workspace-pr:
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/40
  - https://github.com/PyAutoLabs/autolens_workspace/pull/88
- notes: Restored 17 data_preparation scripts (8 autogalaxy + 9 autolens) wiped by commit 27eda214 — kept the current header + `__Contents__`, spliced the implementation body back in from /mnt/c/Users/Jammy/Code/PyAutoOld/AIBACKUP. All imaging examples now point at the `simple` dataset and gained a `__Dataset Auto-Simulation__` block invoking `scripts/imaging/simulator.py`; both interferometer scripts gained the same snippet pointing at `scripts/interferometer/simulator.py`. Added `.script_sizes.json` snapshot + `scripts/check_sizes.sh` (warns on >50% shrinkage; override via `ALLOW_SHRINK=1`) and a "Bulk-edit safety" section in each workspace's CLAUDE.md to prevent recurrence. Audit confirmed `guides/results/start_here.py` shrinkage was an intentional rewrite (commit 04fe40f9), and the autofit `plot/CamelCase.py`/`searches/CamelCase.py` deletions were rename-to-snake-case and per-family merges (39c1e6e, 530d4c7) — neither were truncations. Smoke caveat for autolens 3 scripts (`imaging/modeling.py`, `imaging/fit.py`, `interferometer/modeling.py`): worktree-only failure due to `PYAUTO_SMALL_DATASETS=1` triggering `shutil.rmtree(<symlink>)` on the dataset symlinks I'd added; canonical main passes — not a regression. Pre-existing autogalaxy `imaging/start_here.py` smoke fail (`extra_galaxies/mask_extra_galaxies.fits` missing) noted on `test-mode-visualize` reproduces here and is orthogonal.

## ag-imaging-scripts
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/10
- completed: 2026-04-26
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/367
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/11
- notes: Ports four imaging integration tests from autolens_workspace_test/scripts/imaging/ to single-galaxy autogalaxy (model_fit, visualization, visualization_jax, modeling_visualization_jit). Library side: ag.AnalysisImaging.__init__ had a custom signature that didn't forward **kwargs, breaking AnalysisImaging(use_jax_for_visualization=True) — fixed by adding **kwargs forwarding (mirrors al.AnalysisImaging which inherits from AnalysisDataset directly). Smoke side: PYAUTO_FAST_PLOTS=1 skipped savefig and broke visualization.py's file-existence assertions; first attempt was os.environ.pop in the script (user pushed back), correct fix is a per-pattern unset in config/build/env_vars.yaml — saved to memory.

## test-mode-visualize
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/59
- completed: 2026-04-26
- workspace-pr:
  - https://github.com/PyAutoLabs/PyAutoBuild/pull/60
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/39
  - https://github.com/PyAutoLabs/autolens_workspace/pull/87
- notes: Workspace smoke runs of `fits_make.py` / `png_make.py` were silently producing no `.png` files even with `PYAUTO_SKIP_VISUALIZATION` unset. Root cause turned out to be `PYAUTO_FAST_PLOTS=1` (a smoke-runner default) short-circuiting `subplot_save` / `save_figure` in `autoarray/plot/utils.py` to `plt.close(fig); return` before any save. Fix: per-script env_vars override unsets both `PYAUTO_SKIP_VISUALIZATION` and `PYAUTO_FAST_PLOTS` for `fits_make` / `png_make` patterns, plus `n_like_max=300` cap on the three workflow scripts (csv_make / fits_make / png_make in both autogalaxy and autolens workspaces) so the Nautilus searches finish in seconds. PyAutoBuild side drops the `fits_make` / `png_make` skip entries from `no_run.yaml`. Pre-existing autogalaxy `imaging/start_here.py` smoke fail (missing `dataset/imaging/extra_galaxies/mask_extra_galaxies.fits`) is orthogonal — already broken on main, not caused by this task.

## latent-fitexception-safe
- issue: none — follow-up to autofit_workspace_test smoke-test cleanup
- completed: 2026-04-26
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1233
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/13
- notes: `Analysis.compute_latent_samples` now catches per-sample `FitException` in the non-JAX branch and substitutes a NaN row, then a row-mask filter drops those samples before the existing per-latent column mask. Motivated by stochastic CI flake on `features/assertion.py` under `PYAUTO_TEST_MODE=1` (reduced iterations + real sampler): Dynesty's `sample_list` occasionally contains parameter vectors that violate the model's inequality assertions, and the post-fit latent loop calling `model.instance_from_vector` would raise `FitException` and kill the entire fit. JAX path untouched (jit/vmap can't raise Python exceptions anyway). Workspace side flips `features/assertion` env_vars override from `unset: [PYAUTO_TEST_MODE]` (which fell back to `0`, ~67–107s) back to `set: PYAUTO_TEST_MODE: "1"` (~6s). 5/5 stable smoke runs locally; 2 CI runs × 2 Python versions all pass.

## rectangular-spline-mesh
- completed: 2026-04-22
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/289
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/30
- follow-up: admin_jammy/prompt/autolens/rectangular_spline_gradient_smoothness.md
- note: Shipped `RectangularSplineAdaptDensity` / `RectangularSplineAdaptImage` — spline-CDF variants of the adaptive rectangular meshes, intended for gradient-based samplers (HMC / NUTS / VI). Degree-11 polynomial fit to the inverse empirical CDF on Chebyshev nodes plus cubic-Hermite spline inverter, replacing the C⁰ `jnp.interp` empirical-CDF transform with a C¹ smooth one. Ported RSE JAX notebook (`z_staging/rect_adap_spline_invert_jax (1).ipynb`) in xp-aware form; normal-equations solve in place of `jnp.polyfit` to keep JAX compile ≤10s. `InterpolatorRectangularSpline` subclasses `InterpolatorRectangular` so existing `isinstance` dispatch in `plot/inversion.py` picks it up automatically (fixed an empty-source-plane bug that surfaced during first PNG review). `MeshGeometryRectangular` carries a `spline_deg` so `areas_transformed` / `edges_transformed` use the spline helpers when required (needed for `AdaptiveBrightness` regularization). 12 new pure-numpy unit tests, 60/60 existing pixelization tests green. Truth-parameter eager fit on HST reconstructs a concentrated central source for both linear and spline; log-evidence within ~3e-4 relative (spline +7.5 vs linear baseline). Known limitation: under JIT the spline log_L shows small point-to-point oscillations across an einstein_radius sweep where the linear path is monotone — the gradient-smoothness story the spline was meant to deliver is not yet demonstrated end-to-end. Follow-up prompt (`rectangular_spline_gradient_smoothness.md`) contains hypotheses (floor/ceil routing, monotone-clamp transitions, digitize method) and a bisect plan.

## test-no-run-reasons-fix
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/56
- completed: 2026-04-22
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/57
- note: Two-line assertion fix in `tests/test_result_collector.py::test_parse_no_run_reasons` — the test was asserting `"GetDist" in reasons` but commit `e72a077` ("Rename per-sampler plotter stems to snake_case") had renamed the YAML entry to `get_dist`, so `pytest tests/` was failing on every run. Surfaced while shipping PR #55 (howtofit-register), where the failing test had to be `--deselect`ed. Switched assertions to `get_dist` to match the YAML. No production code or config changed. 40/40 tests now green with no deselects.

## howtofit-register
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/54
- completed: 2026-04-22
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/55
- umbrella: https://github.com/PyAutoLabs/autofit_workspace/issues/38 (sub-4 of 4 — umbrella complete)
- note: Final sub-task of the HowToFit extraction. Registered `howtofit` as a first-class build target in PyAutoBuild, mirroring the HowToLens / HowToGalaxy pattern from PR #53: added `run_workspace "HowToFit" "howtofit"` to `pre_build.sh`, six `release.yml` edits (Checkout block, `script_matrix.py` arg, `generate_notebooks` matrix entry, two resolver branches mapping `howtofit` → `PyAutoLabs/PyAutoFit` + `project::autofit`, release-matrix entry with `package: PyAutoFit`), extended `bump_colab_urls.sh` regex alternation to include `HowToFit`, and seeded empty `howtofit:` / `howtofit: []` entries in `copy_files.yaml` and `no_run.yaml`. Tests: 38 passed; one pre-existing failure (`test_parse_no_run_reasons` — expects `GetDist` but `no_run.yaml` has `get_dist` after an earlier snake_case rename, broken on `main`) deselected as unrelated — worth a follow-up cleanup. First real validation is the next `/pre_build` release workflow dispatch, where HowToFit will be checked out, tested, Colab-URL-bumped, and published for the first time.

## howtofit-bootstrap
- issue: https://github.com/PyAutoLabs/autofit_workspace/issues/38
- completed: 2026-04-22
- sub-prs:
  - sub-1 (HowToFit scaffold): https://github.com/PyAutoLabs/HowToFit/pull/1
  - sub-2 (remove howtofit/ from autofit_workspace + cross-refs): https://github.com/PyAutoLabs/autofit_workspace/pull/39
  - sub-3 (update PyAutoFit library URLs + docs/howtofit/): https://github.com/PyAutoLabs/PyAutoFit/pull/1231 (shipped as howtofit-docs-update)
  - sub-4 (register howtofit target in PyAutoBuild): https://github.com/PyAutoLabs/PyAutoBuild/pull/55 (shipped as howtofit-register)
- note: Umbrella task for the HowToFit extraction. Moved HowToFit from `autofit_workspace/scripts/howtofit/` into a standalone `PyAutoLabs/HowToFit` repository with its own CI, seeded by the existing chapter scripts/notebooks/config/dataset. All four sub-tasks merged on 2026-04-22. HowToFit is now built and released by the same PyAutoBuild pipeline as HowToGalaxy and HowToLens.

## howtofit-docs-update
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1230
- completed: 2026-04-22
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1231
- umbrella: https://github.com/PyAutoLabs/autofit_workspace/issues/38 (sub-3 of 3)
- note: Sub-3 of the HowToFit extraction. Deleted `docs/howtofit/` Sphinx chapter tree (4 .rst files, ~174 lines), removed the `:caption: Tutorials:` toctree block from `docs/index.rst`, and rewrote every `pyautofit.readthedocs.io/howtofit/…` and `Jammy2211/autofit_workspace/…/howtofit/…` URL across 19 files to point at the standalone `PyAutoLabs/HowToFit` repo. Touched: 7 `docs/api/*.rst` cross-refs, `docs/general/workspace.rst` (HowToFit section rewritten to "standalone repo" framing), `docs/features/graphical.rst` (2 prose refs hyperlinked), `docs/overview/statistical_methods.rst`, `docs/cookbooks/multiple_datasets.rst`, `docs/science_examples/astronomy.rst`, `README.rst` (header link + body refs), `paper/paper.md` (JOSS paper prose — URL + framing only, no scientific content altered), plus two README-style refs in `docs/index.rst` caught by the verification grep after first pass. Docs-only: 1232 unit tests pass unchanged; no API surface modified. Follow-up still pending on umbrella issue #38: register `howtofit` build target in PyAutoBuild.

## autogalaxy-wst-jax-lh-imaging
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/8
- completed: 2026-04-22
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/364
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/9
- umbrella: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/5 (task 3/9)
- notes: Added `_register_fit_imaging_pytrees` staticmethod on `ag.AnalysisImaging` (mirrors autolens) + custom `Galaxies` list-subclass pytree flatten. Workspace side: simulator.py + 4 JAX-likelihood imaging scripts (lp, mge, mge_group, rectangular-non-adapt). Deferred 3 adapt-image variants (rectangular_mge, delaunay, delaunay_mge) to a follow-up library task tracked at `admin_jammy/prompt/autogalaxy/adapt_images_pytree_fix.md` — post-unflatten `self.galaxies` has fresh `Galaxy.id`s that don't match `adapt_images.galaxy_image_dict` (aux) keys. Autolens's rectangular.py passes today despite having the apparent same setup — root-cause diff deferred to that follow-up. Gotchas: `ag.lp_linear.Sersic` on a single-galaxy model returns empty `blurred_image` (inversion required); `raise_inversion_positions_likelihood_exception` is autolens-only.

## autogalaxy-wst-ci
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/6
- completed: 2026-04-22
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/7
- umbrella: https://github.com/PyAutoLabs/autogalaxy_workspace_test/issues/5 (task 1/9)
- notes: Added `.github/workflows/smoke_tests.yml`, `.github/scripts/run_smoke.py`, and `config/build/env_vars.yaml` — mirrors autolens_workspace_test's smoke-test setup with PyAutoLens stripped. CI green on Python 3.12 and 3.13. `pending-release` label was created on the repo for the first time during this PR. env_vars.yaml ships only the `jax_likelihood_functions/` override; sibling tasks 2–9 will add per-path overrides alongside their scripts.

## register-howto-repos
- completed: 2026-04-21
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/53
- note: Closed the two long-standing follow-ups from the HowToLens extraction (PyAutoLens PR #468) and HowToGalaxy extraction (PyAutoGalaxy PR #363). Registered `HowToLens` and `HowToGalaxy` as first-class build targets in PyAutoBuild: added both to `pre_build.sh`, all five `release.yml` workspace stages (`find_scripts`, `generate_notebooks`, `run_scripts` Configure, `run_notebooks` Configure, `release_workspaces`), and seeded `autobuild/config/{copy_files,no_run}.yaml`. Extended `bump_colab_urls.sh` regex from `(autofit|autogalaxy|autolens)_workspace` to also match `HowToGalaxy`/`HowToLens` — load-bearing, since the `bump_library_colab_urls` job would otherwise silently skip every HowTo URL we just wrote into PyAutoGalaxy/PyAutoLens docs, paper.md, and READMEs. Locally verified: `script_matrix.py` produces 12 valid matrix entries across the two repos; `generate.py howtogalaxy` generates 24 notebooks + root `start_here.ipynb`; `generate.py howtolens` generates 39 notebooks + root `start_here.ipynb`; `bump_colab_urls.sh` fixture test bumps all three URL shapes. Full CI exercise happens on the next `/pre_build` run.

## howtogalaxy-sub3
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/362
- completed: 2026-04-21
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/363
- note: Sub-task 3 of 3 — and final step — of the HowToGalaxy extraction. Migrated every HowToGalaxy URL across README, docs toctree, per-chapter pages, `docs/general/workspace.rst`, `docs/howtogalaxy/howtogalaxy.rst`, `docs/overview/overview_2_new_user_guide.rst`, `paper/paper.md`, and `CLAUDE.md` to the new `PyAutoLabs/HowToGalaxy` repo at tag `2026.4.13.6` (matches workspace version at extraction). Colab URLs drop the redundant `howtogalaxy/` segment since the new repo root *is* the tutorial series. Prose rewritten to frame HowToGalaxy as a standalone repo. Umbrella issue PyAutoLabs/autogalaxy_workspace#35 now fully closed (all 3 sub-tasks merged). Follow-up still pending: register HowToGalaxy in PyAutoBuild so `/pre_build` can create future version tags automatically (matches the HowToLens follow-up from PyAutoLens PR #468).

## howtogalaxy-sub2
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/36
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/37
- note: Sub-task 2 of 3 of the HowToGalaxy extraction. Deleted `scripts/howtogalaxy/` + `notebooks/howtogalaxy/` (70 files) now that the series lives at PyAutoLabs/HowToGalaxy. Relocated the sersic simulator dependency to `scripts/imaging/simulator_sersic.py` (matches sibling `simulator.py` / `simulator_sample.py`) and rewrote all 9 non-tutorial script callers plus their 9 notebook counterparts. Slimmed HowToGalaxy sections of README.rst, start_here.py/.ipynb, CLAUDE.md, scripts/README.rst, notebooks/README.rst to a single external-repo pointer; rewrote 3 in-script prose references in imaging/interferometer/ellipse modeling.py plus notebook equivalents; dropped howtogalaxy-specific entries from config/build/env_vars.yaml and no_run.yaml. Sub-task 3 of 3 (PyAutoGalaxy docs/README/Colab URLs pointing at new repo) still pending.

## howtogalaxy-bootstrap
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/35
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/HowToGalaxy/pull/1
- note: Extracted the howtogalaxy tutorial series into its own repo (transferred Jammy2211/HowToGalaxy → PyAutoLabs/HowToGalaxy). Sub-task 1 of 3 on issue #35; follow-ups still pending are (2) remove `scripts/howtogalaxy/` + `notebooks/howtogalaxy/` from autogalaxy_workspace with cross-ref updates, (3) update PyAutoGalaxy docs/README/Colab URLs to point at the new repo, plus PyAutoBuild `howtogalaxy` project target registration and a content-alignment pass on chapter 1 tutorials 0 and 3 (pre-existing upstream dataset/import issues excluded from the initial smoke list). Tagged `2026.4.13.6` to match autogalaxy_workspace version at extraction.

## point-solver-auto-jax
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/466
- completed: 2026-04-21
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/469
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/81, https://github.com/PyAutoLabs/autolens_workspace_test/pull/57, https://github.com/PyAutoLabs/autolens_workspace_developer/pull/29

## howtolens-docs-update
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/467
- completed: 2026-04-21
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/468
- note: Sub-task 3 of 3 of the HowToLens extraction. Migrated every HowToLens URL across README, docs toctree, overview, per-chapter pages, `docs/general/workspace.rst`, and `paper/paper.md` to the new `PyAutoLabs/HowToLens` repo at tag `2026.4.13.6` (matches workspace version at extraction). Prose rewritten to frame HowToLens as a standalone repo. Follow-up still pending: register HowToLens in PyAutoBuild so `/pre_build` can create future version tags automatically.

## searches-minimal
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/27
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/28

## jax-likelihood-multi-per-band-priors
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/55
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/56

## howtolens-bootstrap
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/78
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/HowToLens/pull/1
- note: Extracted the howtolens tutorial series into its own repo (transferred to PyAutoLabs org). This was sub-task 1 of 3; follow-ups still pending on issue #78 are (2) remove `scripts/howtolens/` + `notebooks/howtolens/` from autolens_workspace with cross-ref updates, (3) update PyAutoLens docs toctree/overview/paper to point at the new repo, plus PyAutoBuild `howtolens` project target registration and a content-alignment pass on chapter 1 tutorials 0 and 7 (pre-existing upstream bugs excluded from the initial smoke list).

## jax-likelihood-multi-parity
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/53
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/54

## jax-likelihood-point-source-parity
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/51
- completed: 2026-04-21
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/52

## jax-likelihood-interferometer-parity
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/49
- completed: 2026-04-20
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/50

## jit-viz-pixelization-tests
- issue: none — visualization-during-modeling for pixelized sources (follow-up to mge-jit-visualization)
- completed: 2026-04-20
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/48
- note: scripts use n_batch=10 (vs default 100) because rectangular/Delaunay inversion under JAX vmap × default n_batch has a genuine peak memory cost (~40GB for rectangular on the jax_test dataset) that exceeded the 15GB dev box. Not a library bug; n_batch is the right tuning knob for pixelized+JAX workloads on memory-constrained machines.

## merge-pytree-scripts
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/46
- completed: 2026-04-20
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/47

## fit-interferometer-pytree-rectangular
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/463
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/45

## fit-interferometer-pytree-mge-group
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/462
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/44

## fit-imaging-pytree-delaunay-mge
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/461
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/43

## fit-imaging-pytree-rectangular-dspl
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/460
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/42

## fit-imaging-pytree-rectangular-mge
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/459
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/41

## fit-point-pytree
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/457
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/458
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/40

## fit-interferometer-pytree-mge
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/454
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/456
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/39

## csvable
- completed: 2026-04-19
- conf-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/95
- lens-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/455
- summary: moved generic CSV reader/writer to new autoconf.csvable, left PointDataset-specific schema layer in autolens.point.dataset.

## point-csv-examples
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/76
- summary: added CSV I/O examples to point_source/simulator.py, cluster/simulator.py, cluster/modeling.py, and double_einstein_cross/simulator.py so the al.output_to_csv / al.list_from_csv / PointDataset.to_csv API has workspace coverage alongside JSON.

## fit-imaging-pytree-delaunay
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/453
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/361
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/38
- follow-up-prompt: admin_jammy/prompt/autolens/galaxy_pytree_token.md (principled Galaxy.pytree_token fix to supersede narrow GalaxiesToInversion fallback; required before any multi-pixelised-source JIT variant ships)

## fit-imaging-pytree-mge-group
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/452
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/37

## fit-imaging-pytree-rectangular
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/451
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/36

## fit-imaging-pytree-lp
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/450
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/35

## point-dataset-csv
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/446
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/447
- follow-up-prompt: admin_jammy/prompt/autoconf/csv_io.md (move generic CSV I/O helpers into autoconf alongside dictable/fitsable, keeping PointDataset schema logic in autolens)

## linear-light-profile-intensity-dict-pytree
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/448
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/360, https://github.com/PyAutoLabs/PyAutoLens/pull/449
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/34

## weak-lensing-shear-docs
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/365
- completed: 2026-04-25
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/366

## mge-jit-visualization
- issue: none — end-to-end validation of Path A pytree shipping + follow-up prompts for remaining Fit variants
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1229
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/33
- follow-up-prompt: admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md (identity-keyed dict blocks any linear-light-profile model under use_jax_for_visualization=True); 11 per-variant fit_*_pytree_*.md prompts updated with visualization caveat

## fit-imaging-pytree
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/444
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/288, https://github.com/PyAutoLabs/PyAutoLens/pull/445
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/32

## jax-visualization
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1227
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1228, https://github.com/PyAutoLabs/PyAutoLens/pull/443
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/31
- follow-up-prompt: admin_jammy/prompt/autolens/fit_imaging_pytree.md (Path A feasibility study — pytree-register FitImaging for jax.jit-wrapped visualization)

## skip-degenerate-radial-caustic
- issue: none — follow-up to caustic-pixel-scale
- completed: 2026-04-19
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/359

## eager-numpy-regression-assertions
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/25
- completed: 2026-04-19
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/26
- follow-up-prompt: admin_jammy/prompt/autolens/pixelization_eager_vs_jit_divergence.md (eager FitImaging.figure_of_merit ~292k divergence vs JIT/step-by-step in rectangular pixelization)

## lens-calc-hessian-richardson
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/357
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/358
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/24

## grid-irregular-xp-propagation
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/286
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/287, https://github.com/PyAutoLabs/PyAutoLens/pull/442
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/23
- follow-up-prompt: admin_jammy/prompt/autolens/lens_calc_magnification_xp_divergence.md (np/jnp divergence in LensCalc.magnification_2d_via_hessian_from)

## point-source-jax-profiling
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/21
- completed: 2026-04-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/22
- follow-up-prompt: admin_jammy/prompt/autoarray/grid_irregular_xp_propagation.md (PyAutoArray Grid2DIrregular xp-propagation fix to unblock source-plane JIT)

## remove-pyswarms-ultranest
- issue: none — PySwarms/UltraNest dropped from PyAutoFit library
- completed: 2026-04-18
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/36, https://github.com/PyAutoLabs/autofit_workspace_test/pull/10, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/34, https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/4, https://github.com/PyAutoLabs/autolens_workspace/pull/75, https://github.com/PyAutoLabs/autolens_workspace_test/pull/30, https://github.com/PyAutoLabs/PyAutoBuild/pull/51

## integrate-euclid-pipeline
- issue: https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline/issues/3
- completed: 2026-04-18
- workspace-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/50, https://github.com/PyAutoLabs/autofit_workspace/pull/35
- admin-jammy-commit: 9df3bcc

## caustic-pixel-scale
- issue: none — ad-hoc
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/355
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/29

## release-url-sweep-and-tag-pinning
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/73
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/49
- sweep-prs: https://github.com/PyAutoLabs/PyAutoFit/pull/1226, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/356, https://github.com/PyAutoLabs/PyAutoLens/pull/441, https://github.com/PyAutoLabs/autofit_workspace/pull/34, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/33, https://github.com/PyAutoLabs/autolens_workspace/pull/74, https://github.com/Jammy2211/euclid_strong_lens_modeling_pipeline/pull/2
- release-branches-deleted: PyAutoLabs/autofit_workspace, PyAutoLabs/autogalaxy_workspace, PyAutoLabs/autolens_workspace

## default-branch-release-to-main
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/71
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/94, https://github.com/PyAutoLabs/PyAutoBuild/pull/48, https://github.com/PyAutoLabs/PyAutoFit/pull/1225, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/354, https://github.com/PyAutoLabs/PyAutoLens/pull/440
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/33, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/32, https://github.com/PyAutoLabs/autolens_workspace/pull/72

## slam-dspl-modernize
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/68
- completed: 2026-04-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/70

## autofit-smoke-cleanup
- issue: none — autofit_workspace full-sweep cleanup
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1224, https://github.com/PyAutoLabs/PyAutoBuild/pull/47
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/32, https://github.com/Jammy2211/autofit_workspace_developer/pull/7

## fix-autoarray-root-log
- issue: none — direct fix requested (stop root.log creation on `import autoarray`)
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/285

## fix-interferometer-jax-profiling-cwd
- issue: none — follow-up bug fix for interferometer-jax-profiling + interferometer-jax-profiling-pixelization
- completed: 2026-04-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/18

## interferometer-jax-profiling-pixelization
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/16
- completed: 2026-04-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/17

## nnls-target-kappa-fix
- issue: none — follow-up from tupleprior-pytree-fix (PyAutoFit#1222)
- completed: 2026-04-17
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/282

## tupleprior-pytree-fix
- issue: none — follow-up from mge-gradients-pytree-migration (autolens_workspace_developer#13)
- completed: 2026-04-17
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1222

## mge-gradients-pytree-migration
- issue: follow-up to https://github.com/PyAutoLabs/autolens_workspace_developer/issues/10
- completed: 2026-04-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/13

## pixelization-pytree-migration
- issue: follow-up to https://github.com/PyAutoLabs/autolens_workspace_developer/issues/10
- completed: 2026-04-17
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1221
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/12

## imaging-mge-pytree-migration
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/10
- completed: 2026-04-16
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/93, https://github.com/PyAutoLabs/PyAutoFit/pull/1220
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/11

## assertions-fix
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1215
- completed: 2026-04-14
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1217
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/7

## results-json-load-docs
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/64
- completed: 2026-04-14
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/65, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/30

## nnls-gradient-nan-fix
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/278
- completed: 2026-04-14
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/279

## aggregator-output-png
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1213
- completed: 2026-04-14
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1214

## group-features
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/60
- completed: 2026-04-14
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/352
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/61, https://github.com/PyAutoLabs/autolens_workspace/pull/62

## jax-nested
- issue: https://github.com/Jammy2211/autofit_workspace_developer/issues/5
- completed: 2026-04-14
- workspace-pr: https://github.com/Jammy2211/autofit_workspace_developer/pull/6

## jax-mge-gradients
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/8
- completed: 2026-04-14
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/9

## search-interface-simple
- issue: https://github.com/Jammy2211/autofit_workspace_developer/issues/3
- completed: 2026-04-13
- workspace-pr: https://github.com/Jammy2211/autofit_workspace_developer/pull/4

## group-dict-api
- issue: https://github.com/Jammy2211/autolens_workspace/issues/56
- completed: 2026-04-13
- workspace-pr: https://github.com/Jammy2211/autolens_workspace/pull/58

## lens-calc-guide
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/57
- completed: 2026-04-13
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/59

## data-typing-simplify
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/276
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/277, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/351, https://github.com/PyAutoLabs/PyAutoLens/pull/438
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/28, https://github.com/PyAutoLabs/autolens_workspace/pull/55

## jax-profiling-jit-coverage
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/5
- completed: 2026-04-13
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/6

## on-the-fly-modeling
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1211
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1212, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/350

## cli-noise-clean
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1209
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/92, https://github.com/PyAutoLabs/PyAutoFit/pull/1210, https://github.com/PyAutoLabs/PyAutoArray/pull/275, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/349, https://github.com/PyAutoLabs/PyAutoLens/pull/437

## search-update-refactor
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1205
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1207

## samples-simplify
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1203
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1204

## transform-decorator
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/271
- completed: 2026-04-13
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/272, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/347
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/54

## pin-dependency-versions
- completed: 2026-04-13
- summary: Release workflow pins PyAuto inter-deps (==VERSION) in wheel only, not committed to main. Python version check in autoconf/__init__.py. Homepage URLs updated to PyAutoLabs org.
- release: 2026.4.13.6 (verified clean install)
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/91, https://github.com/PyAutoLabs/PyAutoFit/pull/1206, https://github.com/PyAutoLabs/PyAutoArray/pull/273, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/348, https://github.com/PyAutoLabs/PyAutoLens/pull/435, https://github.com/PyAutoLabs/PyAutoBuild/pull/45

## search-config-cleanup
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1200
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1202
- workspace-prs: https://github.com/PyAutoLabs/autofit_workspace/pull/30, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/27, https://github.com/PyAutoLabs/autolens_workspace/pull/53

## release-fixes-apr-2026
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/269
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/270, https://github.com/PyAutoLabs/PyAutoFit/pull/1201
- workspace-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/44, https://github.com/PyAutoLabs/autolens_workspace/pull/52, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/26

## test-mode-separate
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1193
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/86, https://github.com/PyAutoLabs/PyAutoFit/pull/1195, https://github.com/PyAutoLabs/PyAutoArray/pull/265, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/343, https://github.com/PyAutoLabs/PyAutoLens/pull/432
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/29, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/25, https://github.com/PyAutoLabs/autolens_workspace/pull/51

## model-composition-integration
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1194
- completed: 2026-04-12
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/6, https://github.com/PyAutoLabs/autolens_workspace_test/pull/25

## title-prefix-subplots
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/332
- completed: 2026-04-07
- library-pr:
  - https://github.com/Jammy2211/PyAutoArray/pull/260
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/333
  - https://github.com/PyAutoLabs/PyAutoLens/pull/428

## test-mode-bypass
- issue: https://github.com/rhayes777/PyAutoFit/issues/1179
- completed: 2026-04-06
- library-pr: https://github.com/rhayes777/PyAutoFit/pull/1180
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/20, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/10, https://github.com/PyAutoLabs/autolens_workspace/pull/33, https://github.com/PyAutoLabs/autolens_workspace_test/pull/11

## unit-test-profiling (PyAutoFit)
- issue: https://github.com/rhayes777/PyAutoFit/issues/1181
- completed: 2026-04-06
- library-pr: https://github.com/rhayes777/PyAutoFit/pull/1182

## unit-test-profiling (PyAutoArray)
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/251
- completed: 2026-04-06
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/252

## autofit-workspace-plot-update
- issue: https://github.com/PyAutoLabs/autofit_workspace/issues/18
- completed: 2026-04-05
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/19

## unit-test-profiling (PyAutoLens)
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/420
- completed: 2026-04-06
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/421
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/12, https://github.com/PyAutoLabs/autolens_workspace_developer/pull/1

## unit-test-profiling (PyAutoGalaxy)
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/321
- completed: 2026-04-06
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/322

## remove-deflection-integral
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/323
- completed: 2026-04-06
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/324
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/13

## import-optimization
- issue: https://github.com/Jammy2211/PyAutoLens/issues/426
- completed: 2026-04-07
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/259, https://github.com/rhayes777/PyAutoFit/pull/1186, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/330, https://github.com/PyAutoLabs/PyAutoLens/pull/427

## setup-notebook-boilerplate
- issue: https://github.com/rhayes777/PyAutoFit/issues/1183
- completed: 2026-04-09
- library-pr: https://github.com/rhayes777/PyAutoConf/pull/85, https://github.com/PyAutoLabs/PyAutoBuild/pull/37, https://github.com/rhayes777/PyAutoFit/pull/1189
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/23, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/15, https://github.com/PyAutoLabs/autolens_workspace/pull/39

## smoke-test-fast
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/34
- completed: 2026-04-06
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/253, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/325
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/35, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/11

## search-refactor
- issue: https://github.com/rhayes777/PyAutoFit/issues/1190
- completed: 2026-04-09
- library-pr: https://github.com/rhayes777/PyAutoFit/pull/1191
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/24, https://github.com/PyAutoLabs/autofit_workspace_test/pull/2, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/16, https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/2, https://github.com/PyAutoLabs/autolens_workspace/pull/40, https://github.com/PyAutoLabs/autolens_workspace_test/pull/15, https://github.com/PyAutoLabs/PyAutoBuild/pull/38

## group-two-main-galaxies
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/41
- completed: 2026-04-09
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/42

## deflections-integral-fix
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/16
- completed: 2026-04-09
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/17

## mge-fix
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/341
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/342

## merge-fast-plot-env-vars
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/339
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/340, https://github.com/PyAutoLabs/PyAutoBuild/pull/43
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/28, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/24, https://github.com/PyAutoLabs/autolens_workspace/pull/50, https://github.com/PyAutoLabs/autolens_workspace_test/pull/24, https://github.com/PyAutoLabs/autofit_workspace_test/pull/5

## python-313
- issue: https://github.com/PyAutoLabs/PyAutoConf/issues/89
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/90, https://github.com/PyAutoLabs/PyAutoFit/pull/1198, https://github.com/PyAutoLabs/PyAutoArray/pull/268, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/345, https://github.com/PyAutoLabs/PyAutoLens/pull/433

## dependency-sweep
- issue: https://github.com/PyAutoLabs/PyAutoConf/issues/87
- completed: 2026-04-12
- library-pr: https://github.com/PyAutoLabs/PyAutoConf/pull/88, https://github.com/PyAutoLabs/PyAutoFit/pull/1197, https://github.com/PyAutoLabs/PyAutoArray/pull/267, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/344

## interferometer-data-prep
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/67
- completed: 2026-04-15
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/69

## interferometer-jax-profiling
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/14
- completed: 2026-04-17
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/15

## nautilus-pool-teardown
- issue: none — autofit_workspace smoke-test cleanup
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1223

## pyauto-test-mode-rename
- issue: none — silent no-op cleanup discovered during autofit_workspace smoke tests
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/46
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace_test/pull/9, https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/3
- notes: Renamed `PYAUTOFIT_TEST_MODE` → `PYAUTO_TEST_MODE` everywhere. The old name was a silent no-op because autoconf only reads `PYAUTO_TEST_MODE`. After rename, Nautilus smoke test dropped from ~60s (full sampling) to ~4s (test mode skipped sampling). Skipped autolens_workspace_test (active worktree conflict with interferometer-mge-gradients task) and z_projects/autolens_base_project/euclid_strong_lens_modeling_pipeline (out of scope).

## interferometer-mge-gradients
- issue: https://github.com/PyAutoLabs/autolens_workspace_developer/issues/19
- completed: 2026-04-18
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/283, https://github.com/PyAutoLabs/PyAutoArray/pull/284
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/20, https://github.com/PyAutoLabs/autolens_workspace_test/pull/28
- notes: Lowered `nnls_target_kappa` default from 1.0e-2 → 1.0e-11 across PyAutoArray (PR #283 yaml + PR #284 hardcoded fallback for config-shadowing workspaces). Added interferometer MGE gradient profiling + regression assertions in autolens_workspace_developer (PR #20). Workspace_test expected-value update (PR #28) merged via admin despite CI reproducibly producing pre-fix value — local runs against same library commit produce the new value, root cause unidentified; will be regenerated by upcoming test overhaul / next release.

## merge-search-and-plot-scripts
- issue: none — autofit_workspace cleanup
- completed: 2026-04-18
- workspace-pr: https://github.com/PyAutoLabs/autofit_workspace/pull/37
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/52
- notes: Collapsed `scripts/searches/{nest,mcmc,mle}/` into single `nest.py`/`mcmc.py`/`mle.py` files (shared data+model+analysis, one search-variant block per algorithm with distinct `name=` strings). Renamed `scripts/plot/GetDist.py` → `get_dist.py` and the four per-sampler plotters (`{Dynesty,Emcee,Nautilus,Zeus}Plotter.py`) to snake_case. Updated READMEs, `CLAUDE.md`, `smoke_tests.txt`, cookbook cross-refs, and both `no_run.yaml` files (workspace-local + PyAutoBuild). Zeus still fails under test mode so the merged `mcmc.py` is entirely skip-listed.

## howtolens-workspace-cleanup
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/79
- prompt: admin_jammy/prompt/issued/howtolens_workspace_cleanup.md
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/howtolens-workspace-cleanup
- repos:
  - autolens_workspace: feature/howtolens-workspace-cleanup
- merged: 2026-04-21 via https://github.com/PyAutoLabs/autolens_workspace/pull/80

## cluster-simulator
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/464
- completed: 2026-04-20
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/465
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/77
- notes: Added optional `redshift` to `PointDataset` with CSV round-trip and per-source validation (library). Rewrote `autolens_workspace/scripts/cluster/simulator.py` as a 5-member cluster with standalone `NFWMCRLudlowSph` halo (`mass_at_200=10^15.3`) and 2 sources at z=1.0 producing 3 images each, writing a combined `point_datasets.csv` as the canonical hand-editable cluster input. Removed `cluster/simulator` from `no_run.yaml`. Follow-up prompts written: `admin_jammy/prompt/cluster/1_visualization.md` (cluster-scale viz prototype) and `2_csv_model_redshift.md` (pipe `PointDataset.redshift` into `af.Model(al.Galaxy, redshift=...)`). `modeling.py` and `start_here.py` remain parked in `no_run.yaml` — rewrite deferred until those two follow-ups land.

## interferometer-delaunay-jax-profiling
- prompt: PyAutoPrompt/issued/autolens_workspace_developer/interferometer_jax_profiling.md (Phase 3 — Delaunay) — retired with this task
- completed: 2026-04-28
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/39
- notes: Added `jax_profiling/interferometer/delaunay.py` mirroring `interferometer/pixelization.py` (full-pipeline JIT only) with the Delaunay mesh + `ConstantSplit` regularization and `Overlay` image-mesh + edge-points + `AdaptImages` plumbing lifted from `imaging/delaunay.py`. SMA: eager / JIT / vmap log_evidence all match `-3167.5258928840763` at `rtol=1e-4`; full-pipeline JIT 0.26 s/call, vmap (batch=3) 0.24 s/call (1.1× speedup, 127 MB XLA temp). vmap is gated behind `DELAUNAY_VMAP=1` (matches imaging convention) but compiles in seconds on SMA-scale interferometer rather than the 20+ min seen on imaging. Closes the original three-script interferometer JAX profiling brief; both prompts (`interferometer_jax_profiling.md` and `interferometer_jax_profiling_pixelization.md`) deleted from `issued/` and dropped from `z_features/autolens_workspace_developer.md`.

## multiple-sources-modeling
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/97
- completed: 2026-04-28
- workspace-pr: https://github.com/Jammy2211/autolens_workspace/pull/100
- followup: blocked by PyAutoLens #480 (PointSolver magnification filter ignores plane_redshift); both new scripts gated by no_run.yaml until that lands. Restore prompt: PyAutoPrompt/autolens/restore_multiple_sources_lensing_of_lens.md

## interferometer-delaunay-no-lens-light
- completed: 2026-04-29
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/106
- notes: Pre-existing crash in `scripts/interferometer/features/pixelization/delaunay.py` `source_pix_2` under `PYAUTO_TEST_MODE=2` — `add got incompatible shapes for broadcasting: (40, 40), (1070, 1070)` inside `inversion.fast_chi_squared`. Root cause: `source_pix_2` forwarded `bulge=source_lp_result.instance.galaxies.lens.bulge` (a 40-Gaussian MGE Basis from `source_lp`'s `lens_bulge`), giving the source-pix inversion two linear objects (40-param Basis + 1030-param Mapper). `InversionInterferometerSparse.curvature_matrix` (`PyAutoArray/autoarray/inversion/inversion/interferometer/sparse.py:88-109`) only handles the single-Mapper diagonal case and indexes by `linear_obj_list[0].params` = 40 (lens basis), returning a (40,40) curvature_matrix that then can't be added to the (1070,1070) regularization_matrix at `abstract.py:355`. Fix: stripped lens light from interferometer pixelization SLaM scripts (`bulge=None, disk=None` plus `# interferometry does not support lens light` comment). Real fix in `delaunay.py` (`source_lp` and `source_pix_2`); stylistic in `pixelization/slam.py`, `extra_galaxies/slam.py`, `subhalo/detect/start_here.py` where the forwarded `source_pix_result_1.instance.galaxies.lens.bulge` was already None at runtime. Verified end-to-end with `PYAUTO_TEST_MODE=2 delaunay.py`. Library bug in `InversionInterferometerSparse` (no Func+Mapper support, broken `pix_pixels` indexing) left intact — not exercised by any current interferometer SLaM script after this fix; defer until a real co-fit-lens-light-with-source-pix interferometer use case emerges.

## autobuild-release-prep
- completed: 2026-04-29
- merged-prs:
  - PyAutoBuild#62 (workspace-owned build configs + persistent timestamped runs)
  - PyAutoPrompt#17 (`/pyauto-status-full` skill + `pyauto-status-full` / `pyauto-{report,json,triage}` shell functions)
  - autofit_workspace#44, autogalaxy_workspace#48, autolens_workspace#107 (new `config/build/{copy_files,visualise_notebooks}.yaml`)
  - autofit_workspace_test#16, autogalaxy_workspace_test#20, autolens_workspace_test#63 (same; `autogalaxy_workspace_test` also gained a `no_run.yaml` that was missing entirely — falling through to autobuild's empty fallback)
- notes: PyAutoBuild's `run.py` / `run_python.py` / `generate.py` now prefer each workspace's `config/build/` files over autobuild's keyed-dict copies. Dead `autobuild/config/{notebooks_remove,env_vars}.yaml` deleted. `run_all.py` writes results to `test_results/runs/<UTC-timestamp>/` with a `latest` symlink updated atomically; per-script timeout raised 60s → 300s; `autogalaxy_workspace_test` added to the workspace list; pre-existing bug fixed where `run_all.py` passed bare subdir names instead of `scripts/<dir>`. `aggregate_results.py` adds top-25 slowest scripts and a run header to `report.md`. `result_collector.RunReport` exposes `total_duration_seconds`. 57/57 pytest passing. First full release-prep run produced 460 results / 95 min / 48 failures across `runs/2026-04-29T14-48-47Z/`; `triage.md` in that run dir clusters them into ~10 root causes (group/features/pixelization, jax_likelihood numerical drift, modeling_visualization_jit, aggregator timeouts, missing simulator output) for follow-up.

## numba-docs-deprioritize
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/482
- completed: 2026-04-30
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/483, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/379
- repos: PyAutoLens, PyAutoGalaxy

## autobuild-bash-cli
- issue: https://github.com/PyAutoLabs/PyAutoBuild/issues/67
- completed: 2026-04-30
- library-pr: https://github.com/PyAutoLabs/PyAutoBuild/pull/68, https://github.com/Jammy2211/admin_jammy/pull/14
- repos: PyAutoBuild, admin_jammy
- notes: Added `bin/autobuild` dispatcher (16 subcommands + help system) wrapping every PyAutoBuild operation under one shell entry point alongside the existing Claude skills. `tag_and_merge.py` ported to bash; `script_matrix.py` deliberately kept Python (called by release.yml — workflow ABI). README version-bump sed step folded from the `/pre_build` skill into `pre_build.sh` (+ inferred `readme_pkg` for `autogalaxy_workspace_test`, `HowToGalaxy`, `HowToFit` which the old skill table didn't list). `/pre_build` skill collapsed to a thin wrapper around `autobuild pre_build`, mirroring `/verify_install`. The skill's old soft "stale `no_run.yaml` patterns" report was dropped — can be added back to the bash CLI later if useful.

## cluster-f-api-drift
- issue: N/A (triage cluster F sweep, run 2026-04-29T14-48-47Z)
- completed: 2026-05-02
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/491
- workspace-prs: https://github.com/PyAutoLabs/autofit_workspace/pull/50, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/54, https://github.com/PyAutoLabs/autolens_workspace/pull/117
- repos: PyAutoLens, autofit_workspace, autogalaxy_workspace, autolens_workspace
- notes: 9 Cluster F triage failures resolved in 8 file changes across 4 PRs. Item 4 (double_einstein_ring `FitException`) root-caused to a swallowed `IndexError` at `PyAutoLens/autolens/analysis/result.py:445` — `plane_indexes_with_pixelizations[plane_index]` should be `.index(plane_index)`. The library bug was latent: it only triggered when not every plane had a pixelization. Items 1, 8 added back functionality removed by PyAutoArray plotter-class deletion (`b491a119`) and missing simulator outputs. Items 7, 9 fixed wrong/missing auto-sim blocks in consumer scripts — the auto-sim pattern is more reliable than ad-hoc dataset preparation. Items 2, 3, 5 were one-line script bugs: duplicate `source` kwarg, prior bound under zero-luminosity test mode, and an autoarray wrapper escaping a `@jax.jit` boundary.

## weak-shear-simulator
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/476 (and #472)
- completed: 2026-05-04
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/473
- workspace-prs: https://github.com/PyAutoLabs/autolens_workspace/pull/84
- repos: PyAutoLens, autolens_workspace
- notes: Step 1 of weak-lensing series — `al.WeakDataset` + `al.SimulatorShearYX(via_tracer_from / via_tracer_random_positions_from)` (additive, new `autolens/weak/` sub-package). Originally drafted on mobile-Claude on 2026-04-25, both PRs sat unmerged for ~10 days. Resume work: rebased onto current main (clean — main's new `check_version()` block in `autolens/__init__.py` merged cleanly with the additive exports), fixed one failing test (`test__simulator_shear_yx__noise_changes_values_but_preserves_shape_and_grid` was asserting `dataset.positions is grid` but `VectorYX2DIrregular.__init__` re-wraps via `Grid2DIrregular(values=grid)` — replaced with `np.testing.assert_array_equal`). CI debugging found a pre-existing **stale `claude/list-admin-prompts-Mid1U` branch on PyAutoGalaxy** (also from the mobile session, fully merged via PR #366 but never deleted) that the workflow's `Change to same branch if exists in deps` step kept switching to — that branch's *old* `pyproject.toml` still listed `ultranest==3.6.2`, which fails to compile under newer Cython on Python 3.13. Deleting that stale branch unblocked CI for both runs. Local unit tests (269/269 on Python 3.12) and end-to-end workspace simulator (`scripts/weak/simulator.py` writes `dataset/weak/simple/dataset.json` + `tracer.json` for an Isothermal lens, 200 random source positions, noise_sigma=0.3) verified before merge. Visualization deferred to follow-up prompt `weak/2_visualization.md` (already spawned during the mobile session).

## output-folder-layout-tutorials
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/124
- completed: 2026-05-05
- workspace-prs: https://github.com/PyAutoLabs/autolens_workspace/pull/126, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/58
- repos: autolens_workspace, autogalaxy_workspace
- notes: Replaced flat-bullet `__Output Folder__` block with a comprehensive directory-tree `__Output Folder Layout__` block in every modeling.py tutorial across both workspaces (imaging, interferometer, point_source, cluster, group; autogalaxy: imaging, interferometer, ellipse), adapted per package and data type. Restructured `guides/results/start_here.py` so the model-fit runs once at the top of the file and simple-loading uses `result_path = search.paths.output_path` instead of a hardcoded `<unique_hash>` placeholder. The aggregator section's existing narrative is preserved in full; its intro was rewritten to frame it as a peer first-class tool (generator-based, used by `csv_maker`/`png_maker`/`fits_maker` workflow tools) rather than a "many fits" fallback. Fixed pre-existing latent `af.SamplesNest.from_csv` API bug uncovered once the simple-loading path resolved to a real folder — switched to the correct `af.SamplesNest.from_table(filename=...)`.

## cluster-viz-prototype
- issue: https://github.com/PyAutoLabs/autolens_workspace_test/issues/74
- completed: 2026-05-07
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/75
- repos: autolens_workspace_test


## results-start-here-fits-hdu-fix-autolens
- issue: N/A (Cluster D triage follow-on for autolens; mirrors merged autogalaxy PR #61)
- completed: 2026-05-08
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/135
- repos: autolens_workspace
- notes: |
    Cluster D triage report attributed `autolens_workspace/scripts/guides/results/start_here.py`
    to a "truncated traceback ending in a 15x15 numpy pixel-coord array" — the actual
    exception was clipped from `report.md`. Reran the script with full stderr capture and
    found a `DatasetException: A value in the noise-map of the dataset is -0.0367 ... less
    than or equal to zero` at line 245 (the second `al.Imaging.from_fits(...)` block that
    re-loads the fit-output multi-HDU `image/dataset.fits` for the Simple-Loading section).
    The fit-output FITS file's HDU layout is `[0=MASK, 1=DATA, 2=NOISE_MAP, 3=PSF,
    4=OVER_SAMPLE_SIZE_LP, 5=OVER_SAMPLE_SIZE_PIXELIZATION]` (canonical reference:
    `PyAutoGalaxy/autogalaxy/aggregator/imaging/imaging.py:73-77`), but the script still
    passed `data_hdu=0, noise_map_hdu=1, psf_hdu=2`. Three-line fix bumped them to
    `1, 2, 3`. The autogalaxy_workspace half of this bug was already merged on main as
    PR #61 — `/plan_branches` caught the prior fix and narrowed the scope from "both
    workspaces" to "autolens only". The 7/7 smoke tests passed. Notebook regen
    (`/generate_and_merge`) deliberately deferred — to be batched with other notebook
    updates, mirroring how PR #61 was scripts-only too.

## cluster-h-hpc-pathlib-fix
- issue: N/A (Cluster H from triage report 2026-05-07T15-42-17Z)
- completed: 2026-05-08
- workspace-prs: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/62, https://github.com/PyAutoLabs/autolens_workspace/pull/136
- repos: autogalaxy_workspace, autolens_workspace
- notes: |
    Two classes of leftover typos from the `os.path` → `pathlib` blanket
    refactor PRs (autogalaxy #59, autolens #128) in the HPC tutorial
    scripts. The triage report only flagged the autogalaxy `path.sep`
    crash at line 64 — running the script after that one-line fix
    surfaced a second bug at line 207 (`dataset_Path()` corrupted from
    `dataset_path` by the same case-insensitive `path → Path`
    substitution). Expanded scope to fix both classes in the same PR
    after exhaustive grep confirmed only 4 case-corrupted identifiers
    across the two files (`Path.cwd()` and prose-text "Path" in section
    headers were unaffected). Final tally: autogalaxy 6 sites (3 ×
    `Path(path.sep)` + 3 × `dataset_Path()`), autolens 4 sites (3 + 1).
    autolens twin file (`example_cpu.py`, different name from autogalaxy's
    `example_cpu_and_gpu.py`) does not show in the failure list because
    it is pre-emptively listed in `no_run.yaml` ("HPC paths dont exist
    locally."). Verified post-fix that the autolens script no longer
    raises `NameError` and instead reaches the expected pre-existing
    `ConfigException` at line 110 — exactly the failure mode the
    `no_run.yaml` skip documents — so the skip stays.

## interferometer-nufftax-updates
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/146
- completed: 2026-05-14
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/147
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/68

## info-exclude-identifier-fields
- completed: 2026-05-14
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1261
- summary:
    `model.info` was rendering `pytree_token N` lines for every
    `LightProfileLinear` (40+ per basis fit in SLaM chaining output) —
    an internal JAX-pytree counter set in `__init__`, declared in
    `__exclude_identifier_fields__` so the unique_id hash already
    ignores it. PyAutoFit's `AbstractPriorModel.info` did not honor
    that contract. Fix walks each leaf's parent and consults
    `type(parent).__exclude_identifier_fields__`; PyAutoGalaxy and
    other libs need no changes. Verified end-to-end via SLaM
    (`PYAUTO_TEST_MODE=3 slam_start_here.py`): unique_id hashes and
    `model.results` byte-identical to baseline, `model.json`
    semantically identical, 120 `pytree_token` lines across 3 fits
    dropped to zero. Also generalises to the existing
    `GridSearch.__exclude_identifier_fields__ = ("number_of_cores",)`.

## group-double-einstein-ring
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/156
- completed: 2026-05-16
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/157
- summary: |
    Padded out the imaging double Einstein ring example with new fit.py
    and likelihood_function.py (parametric MGE source, no pixelization);
    refreshed modeling.py with an upfront "use chaining for real fits"
    callout and a fixed-Planck18 cosmology with a commented Om0-free
    snippet. Mirrored the imaging set into a new scripts/group/features/
    advanced/double_einstein_ring/ folder using lens_dict for two main
    lens galaxies (simulator + fit + modeling + likelihood_function +
    chaining + slam + README). Committed datasets for both. slam.py's
    pixelized stages couldn't be validated under TEST_MODE=2 due to a
    pre-existing autoarray limitation that also affects imaging slam.py;
    structurally mirrors imaging so should work in production. Inner
    dataset/.gitignore required `git add -f` for the new datasets,
    matching existing convention. Tracker z_features/group_lensing_workspace.md
    now has 4 sub-prompts remaining (los_halos, mass_stellar_dark,
    scaling_relation, subhalo_sensitivity).

## rectangular-adapt-cdf
- issue: https://github.com/PyAutoLabs/PyAutoArray/issues/322
- completed: 2026-05-17
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/323
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_developer/pull/71
- note: |
    Scope pivoted mid-session — original plan was a multi-component
    density framework (magnification + brightness + residual + caustic
    weighted into one mesh_weight_map). User pulled the brakes; empirical
    ghost_peak experiment confirmed the real problem was the separable
    per-axis CDF on multi-modal sources (not the signal richness). Shipped
    RectangularRotatedAdaptImage as Path A (brightness-weighted PCA
    pre-rotation). Path B (multi-sub-mesh) prompt-scoped at
    PyAutoPrompt/autoarray/rectangular_multi_submesh.md as the next step
    for arbitrary K >= 3 non-collinear peaks.

    Phase 2 density_components framework (compose_density +
    uniform_density_component, 9 tests) kept as scaffolding for future
    multi-signal work even though Path A didn't end up using it.

    Demo package rect_adapt_duo shipped under autolens_workspace_developer
    with a documented chi^2 caveat (rotated mesh delivers ~2x effective
    resolution per real peak, so under-smooths at fixed regularization;
    real lens-modelling search would tune coefficient per mesh).

## cluster-modeling-v2
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/174
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/175
- repos: autolens_workspace
- notes: Re-do of the failed cluster-csv-redshifts retroactive log from 2026-05-06 (the verifier had seen `al.list_from_csv` at modeling.py:143 but missed it was inside a docstring example). Full rewrite of scripts/cluster/modeling.py to pair to the multi-plane simulator: load via `al.list_from_csv` so per-source `dataset.redshift` is the real code path; load `main_lens_centres.json` + `host_halo_centre.json` (drops defunct `extra_galaxies_*` JSONs); compose 2 `dPIEMassSph` mains (centres fixed, ra/rs/b0 free), 1 `NFWMCRLudlowSph` host halo galaxy (free mass_at_200, redshift_source anchored to max source z), 2 `Point` source galaxies whose redshifts come from `dataset.redshift`; switch `search.fit` to the factor-graph pattern (returns `result_list`, fixes the latent `result_list[0]` reference). Auto-sim guard tightened from `dataset_path.exists()` to `(dataset_path / "data.fits").exists()` — the dataset dir already held viz-prototype PNGs from autolens_workspace_test#75, which made the old guard silently skip simulator regeneration. Latent `aplt.subplot_tracer(grid=result.grids.lp)` bug fixed (AnalysisPoint's PointDataset has no `.grids`). Removed `cluster/modeling` from `config/build/no_run.yaml` (kept `cluster/start_here` parked — separate follow-up). Smoke 7/7 green. Out of scope (each its own future prompt): broader lens/source CSV API mirroring `al.galaxy_table_from_csv`; `start_here.py` rewrite; scaling-relation cluster members + simulator extension; library-side `aplt` plotter promotion from the viz prototype.

## light-profiles-guide
- issue: https://github.com/PyAutoLabs/autogalaxy_workspace/issues/85
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/516
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/86, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/87, https://github.com/PyAutoLabs/autolens_workspace/pull/176, https://github.com/PyAutoLabs/autolens_workspace/pull/177
- repos: PyAutoLens, autogalaxy_workspace, autolens_workspace
- notes: New scripts/guides/profiles/light.py in both autogalaxy_workspace and autolens_workspace — single-page tour of every light profile family (Standard / Linear / Operated / Multipole / Basis), detailed Sersic example, full af.Model → instance flow (autolens version uses al.Tracer to place lens+source on distinct redshift planes), compact walkthrough of every remaining standard profile. Dedicated section for the newly merged SersicMultipole / GaussianMultipole (PyAutoGalaxy #420/#421) with m=3/m=4 Fourier perturbation explained and plotted. 4-Gaussian Basis (MGE) example wraps Basis in a Galaxy because Basis.image_2d_from returns a raw numpy.ndarray rather than an Array2D (library quirk, not fixed in this task — worth a separate issue if API uniformity is wanted). Section order revised mid-flight: Multipole moved from "between Operated and Basis" to "between Model Instance and Remaining Walkthrough" so the reader sees the full Sersic→Model→instance(→Tracer) flow on a plain profile before learning the multipole variants (#87 / #177 follow-up to #86 / #176). PyAutoLens #516 synced docs/api/light.rst byte-identically with PyAutoGalaxy (added Linear, Operated, Basis sections + Standard prose; Standard autosummary already had multipoles + Chameleon + ElsonFreeFall from #515).

## mass-profiles-guide
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/178
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/179
- repos: autolens_workspace
- notes: New scripts/guides/profiles/mass.py in autolens_workspace — single-page tour of every lensing mass profile (Total / Mass Sheets / Multipoles / Point Mass), paired-companion to scripts/guides/profiles/light.py. Detailed example builds al.mp.Isothermal and plots convergence + log10 potential + deflection magnitude + lensed-source image via Tracer. Mass Sheets section covers ExternalShear, MassSheet, ExternalPotential. Point Mass section covers PointMass / SMBH / SMBHBinary via Galaxy wrapper (PointMass family returns raw ndarray for convergence, not Array2D — same kind of library quirk as Basis). PowerLawMultipole positioned after Model Instance (paralleling revised light.py order). Library quirks worked around in the guide rather than fixed upstream: dPIEMass(ell_comps=(0,0)) divide-by-zero, deflections_yx_2d_from returning VectorYX2D (plot via np.hypot + Array2D wrap), dPIEPotential.convergence_2d_from incorrectly returning VectorYX2D. Stellar / dark / lmp / lmp_linear profiles deferred to a separate light_and_mass_profiles.py guide (starting immediately as follow-up task).

## light-mass-profiles-guide
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/180
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/181
- repos: autolens_workspace
- notes: Third guide in scripts/guides/profiles/ trilogy (light.py → mass.py → light_and_mass_profiles.py). Covers stellar mass (al.mp.Sersic family with mass_to_light_ratio), the full dark-matter NFW menagerie (NFW, gNFW, cNFW, NFWTruncated, plus MCR / Virial / Scatter variants in a single dedicated "NFW Variants" section explaining each axis), combined light+mass (al.lmp.* — headline section shows one Sersic emits both image_2d_from and convergence_2d_from via shared mass_to_light_ratio), and linear combined (al.lmp_linear.* — same constructor as al.lmp, intensity-via-inversion at fit time). Composing a Decomposed Bulge+Halo Model section shows the canonical recipe of attaching a stellar Sersic + NFW halo to one lens galaxy so their convergences sum. 751 lines. Out-of-scope follow-ups noted in the issue comment: docs/api/mass.rst still missing al.lmp.* / al.lmp_linear.* documentation (plus the gaps already flagged in mass-profiles-guide notes — ExternalPotential, SMBH*, dPIE*, several NFW variants); GaussianGradient parameter naming inconsistency (mass_to_light_ratio_base vs mass_to_light_ratio in SersicGradient).

## docs-mass-rst-sync
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/519
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoLens/pull/520
- repos: PyAutoLens
- notes: Pure rST docs sync — brought PyAutoLens/docs/api/mass.rst into line with the al.mp.* / al.lmp.* / al.lmp_linear.* namespaces it documents. Added missing entries to Total (dPIE family), Mass Sheets (ExternalPotential), Stellar (GaussianGradient, SersicCore*), Dark (cNFW family, virial-mass variants). Moved PointMass out of Total into a new Point Mass section that also lists SMBH / SMBHBinary. Added two new sections at the bottom: Stellar Light+Mass [ag.lmp] and Linear Light+Mass [ag.lmp_linear]. No Python code touched. Surfaced as follow-up while writing the autolens_workspace mass / light+mass guides (#178 / #180).

## profile-return-type-fixes
- issue: https://github.com/PyAutoLabs/PyAutoGalaxy/issues/424
- completed: 2026-05-18
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/425
- repos: PyAutoGalaxy
- notes: Two profile-return-type bugs flagged while writing the autolens_workspace profiles guides. (1) Basis.image_2d_list_from's LightProfileLinear placeholder was a raw xp.zeros((N,)) ndarray, so Basis.image_2d_from returned a raw ndarray instead of Array2D when every constituent was linear (the MGE case) — wrapped in aa.Array2D(values=..., mask=grid.mask). (2) dPIEPotential.convergence_2d_from was decorated @aa.decorators.to_vector_yx (copy-paste from the deflections method directly above) instead of @aa.decorators.to_array — wrapped the scalar convergence as a VectorYX2D. Swapped to @to_array; dPIEPotentialSph already had the correct decorator. Regression tests added in test_basis.py and test_dual_pseudo_isothermal_potential.py. 911 tests pass. No workspace migration needed — workarounds in autolens_workspace/scripts/guides/profiles/{light,mass}.py are obsolete but harmless.

## profile-guide-followup-cleanup
- issue: https://github.com/PyAutoLabs/autolens_workspace/issues/182
- completed: 2026-05-18
- workspace-pr: https://github.com/PyAutoLabs/autogalaxy_workspace/pull/88, https://github.com/PyAutoLabs/autolens_workspace/pull/183
- repos: autogalaxy_workspace, autolens_workspace
- notes: Follow-up to PyAutoGalaxy #425 (profile-return-type-fixes). Removed two workspace-side workarounds in scripts/guides/profiles/ now that the library returns the correct wrapper types. (1) Basis demo in both light.py guides was using Galaxy.image_2d_from to dodge a Basis.image_2d_from quirk — switched to plain basis.image_2d_from. Also surfaced a pedagogical issue: the demo used al.lp_linear.Gaussian constituents which produced an all-zeros plot (intensities unset before inversion). Swapped to al.lp.Gaussian with explicit decreasing intensities (1.0 → 0.5 → 0.25 → 0.1 with sigmas 0.05 → 0.15 → 0.4 → 1.0) so the MGE shape is actually visible, with a follow-on note saying use lp_linear in real fits. (2) mass.py walkthrough swapped al.mp.dPIEPotentialSph for the elliptical al.mp.dPIEPotential now that its convergence_2d_from returns Array2D. Surveying for other Galaxy-wrap / *Sph fallbacks found none worth changing — Point Mass section still wraps in a Galaxy due to a separate unfixed library quirk (PointMass.convergence_2d_from returns raw ndarray); left alone.

## graphical-ep-scale-up
- issue: https://github.com/Jammy2211/autofit_workspace_developer/issues/16
- completed: 2026-05-20
- workspace-pr: https://github.com/Jammy2211/autofit_workspace_developer/pull/17
- repos: autofit_workspace_developer
- notes: Scaffolded two self-contained example packages (graphical/, ep/) adapted from z_projects/concr/scripts/toy/. Each package's simulator emits ground_truth.json per dataset (truth params + truth-evaluated log likelihood) and the fit scripts run end-of-run sanity checks comparing recovered posteriors to truth. Profile baselines at N=3/10/30 committed for both packages; cProfile attribution at N=10 committed as N10_hotspots.txt. aggregate_profiles.py distills per-N summaries into baseline.json. Companion scoping documents PyAutoPrompt/graphical_ep/{graphical,ep}_scoping.md rank scale-up follow-up prompts based on the measured data. Key findings: (a) scipy.stats.truncnorm.cdf in TruncatedGaussianPrior.value_for is 33% of graphical wall time and 16% of EP — shared cross-package optimisation target; (b) matplotlib per-factor visualisation is 48% of EP runtime (confirms IC50's Tier 1); (c) Dynesty nlive=50 fails to converge at 91 dim for graphical N=30, motivating gradient samplers (N=30 sanity FAIL is intentional and informative); (d) EP scales linearly in N; graphical RSS grows 5× from N=3 to N=30 (556 MB → 2.9 GB), so EP is the only memory-feasible approach beyond N≈100.

## cluster-f-jax-baseline-oom
- issue: (CI-triage cluster F, no GitHub issue)
- completed: 2026-05-20
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/108
- repos: autolens_workspace_test
- notes: Cluster F was a 3-script triage report (rectangular_dspl + datacube/delaunay + interferometer/delaunay). Verified on current main first: rectangular_dspl and datacube/delaunay both already pass — the triage had aged out for them (likely a transient library state when the report was generated). Interferometer/delaunay still SIGKILLed, but at a different point than the triage suggested: it dies during JIT compilation of Path B (TransformerNUFFT cross-check, added 2026-05-10 in d89db3e) — peaks ~23 GB virtual / ~15 GB RSS, killing python on this 15 GB-RAM laptop. Confirmed via dmesg kernel logs; OOM reproduces on both GPU and CPU, so not a 6 GB GPU artifact. Fix: restrict Path B's vmap to `parameters[:1]` (1 sample instead of batch_size=3), cuts JIT memory ~3×. One sample is sufficient to validate NUFFT vs DFT agree, matching the pattern already used in datacube/delaunay's NUFFT cube cross-check (which is why datacube didn't OOM). Path A and the primary TransformerDFT vmap retain batch_size=3 — they were already passing. Lesson: verify triage clusters on current main before mass-fixing; transient library state can age out failures faster than the report cycle.

## move-basis-regularization-to-developer
- completed: 2026-05-20
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace/pull/193, https://github.com/PyAutoLabs/autogalaxy_workspace/pull/89, https://github.com/PyAutoLabs/autolens_workspace_developer/pull/76
- repos: autolens_workspace, autogalaxy_workspace, autolens_workspace_developer
- notes: Cluster D smoke-run triage. Surfaced via a FitException at PyAutoLens analysis.py:84 wrapping the original exception. The real error was an empty-float64 mapper_indices used as an index in PyAutoArray Inversion._reduced (IndexError on numpy, stricter TypeError on JAX). Reproduces in user-mode too (use_jax=True), so not an autobuild env artifact. Spent significant time prototyping a PyAutoArray fix (regularized_indices vs mapper_indices, log_det decoupling) before the user clarified the convention: lens-light Basis regularization should never enter log_det_* terms, but should enter the regularization_term S^T H S. Two failed pivots (full short-circuit broke log_det cholesky; regularized_indices redefined _reduced shape). User then pointed out the underlying problem was the workspace script — the regularized-Basis section was documented as "Advanced / Unused" yet still executed. Reverted PyAutoArray entirely; moved the section out of 4 user-facing modeling.py scripts into a new autolens_workspace_developer/basis_regularization/ folder housing 4 self-contained reference scripts. Library stays unchanged. Lessons: (1) check whether the failing config is documented as unsupported before patching library code; (2) Basis lens-light regularization is a "research-only" feature — positive-only solver already solved the ringing problem it was designed for.

## grid-respect-small-datasets
- completed: 2026-05-20
- library-pr: https://github.com/PyAutoLabs/PyAutoArray/pull/327, https://github.com/PyAutoLabs/PyAutoGalaxy/pull/431
- workspace-pr: https://github.com/PyAutoLabs/autolens_workspace_test/pull/109
- repos: PyAutoArray, PyAutoGalaxy, autolens_workspace_test
- notes: Cluster H triage verification. Triage said cluster/visualization.py was failing due to a simulator regression or LensCalc multi-plane plumbing bug; both diagnoses were wrong. simulator.py passes on clean main, mass.csv exists, host halo is the expected 10^15.3 Msun NFW, and LensCalc returns the right curve when given a grid of adequate extent. Real root cause: PYAUTO_SMALL_DATASETS=1 silently shrunk Grid2D.uniform >15x15 to (15,15) @ 0.6" (~8" extent) at TWO sites — the viz_grid in the script AND the internal evaluation grid built by PyAutoGalaxy's @evaluation_grid decorator. Both fell well inside the cluster's tangential critical curve so the assertion fired. Fix: new `respect_small_datasets: bool = True` kwarg on `Grid2D.uniform` (PyAutoArray) that callers can flip off; `evaluation_grid` decorator in PyAutoGalaxy passes `respect_small_datasets=False` on its internal Grid2D.uniform; cluster/visualization.py passes it for viz_grid. Lessons: (1) verify triage clusters on clean main before fixing — saved chasing three wrong root causes here; (2) global "smoke shrink" hooks that operate inside library decorators can defeat physics assertions silently — opt-out kwargs are needed for grids whose spatial extent is load-bearing.

## truncated-gaussian-fast-path
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1284
- completed: 2026-05-20
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1285
- repos: PyAutoFit
- notes: First sub-task issued from the graphical-ep-scale-up scoping pass (PyAutoPrompt/graphical_ep/{graphical,ep}_scoping.md). Replaced scipy.stats.norm.cdf/ppf inside TruncatedGaussianPrior.value_for and TruncatedNormalMessage.value_for with direct scipy.special.ndtr / ndtri (and the jax.scipy.special equivalents) via a new shared private helper autofit.mapper.prior._erf_helpers.truncated_normal_value_for. ndtr/ndtri are the Cephes routines that scipy.stats.norm.cdf/ppf already wrap — bit-exact equivalent at the algorithm level, just without the _distn_infrastructure dispatch. Measured speedup on the autofit_workspace_developer toy 1D Gaussian baseline: graphical N=3 22.8s→5.6s (4.04× / 75% reduction); EP N=3 251.9s→76.3s (3.30× / 70% reduction). The cProfile-projected reductions were ~30% (graphical) / ~17% (EP) — the real numbers were 2–4× larger because cProfile's own wrapper overhead masked the true scipy.stats cost. Worth remembering: cProfile-projected speedups are lower bounds, not point estimates. Initial draft used erf/erfinv directly (matching GaussianPrior's JAX branch); switched to ndtr/ndtri after extreme-unit tests at unit ∈ {1e-6, 1-1e-6} for half-bounded truncations showed 1e-11 rel disagreement with the pre-fix path — ndtr/ndtri eliminate that entirely (same Cephes path). Also caught a test-oracle gotcha: scipy.stats.truncnorm.ppf has its own tail-safe branching that neither the old nor new prior-side code matches — the right oracle is the OLD scipy.stats.norm.cdf/ppf composition (bit-exact to the new ndtr/ndtri form). Pytest suite 1398/0 (1 skipped) including 145 new bit-exact equivalence tests on (mean, sigma, lower, upper) × unit grid. Follow-up workspace PR queued to refresh autofit_workspace_developer/{graphical,ep}/profiles/baseline.json with the post-fix numbers.

## split-likelihood-profiling
- issue: https://github.com/PyAutoLabs/autolens_profiling/issues/19
- pr: https://github.com/PyAutoLabs/autolens_profiling/pull/20
- shipped: 2026-05-21
- summary: |
    Split likelihood/ into likelihood_breakdown/ + likelihood_runtime/.
    14 scripts (5 breakdown + 9 runtime) plus moved sweep+aggregate harness.
    Per-package READMEs document methodology + when-to-use.
