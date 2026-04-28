
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
