# Active Tasks

## kaplinghat-sidm-cored-nfw
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/564
- session: codex --resume <session-id>
- status: pyautogalaxy-pr-created, pyautolens-local-commit-pending-push
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/471
- worktree: ~/Code/PyAutoLabs-wt/kaplinghat-sidm-cored-nfw
- suggested-branch: feature/kaplinghat-sidm-cored-nfw
- classification: library
- affected-repos:
  - PyAutoGalaxy
  - PyAutoLens
- notes: Initial implementation phase scoped to PyAutoGalaxy because PyAutoLens was claimed by datacube-shared-state. User confirmed this PyAutoLens follow-up does not clash with datacube work, so Codex continued on the canonical PyAutoLens branch.
- repos:
  - PyAutoGalaxy: feature/kaplinghat-sidm-cored-nfw
  - PyAutoLens: feature/kaplinghat-sidm-cored-nfw
- local-commits:
  - PyAutoGalaxy: 8aa03972 feat: support vmapped Kaplinghat deflections (local follow-up branch feature/kaplinghat-sidm-cored-nfw-vmapped; original PR worktree is read-only in this sandbox)
  - PyAutoLens: 3fd6dd6d7 feat: support Kaplinghat halos in substructure arrays

## datacube-shared-state
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/565
- session: claude --resume <session-id>
- status: workspace-dev (Phase 5); library curvature+mapper on #344/#566 (pending-release)
- library-pr:
  - https://github.com/PyAutoLabs/PyAutoArray/pull/344
  - https://github.com/PyAutoLabs/PyAutoLens/pull/566
- worktree: ~/Code/PyAutoLabs-wt/datacube-shared-state
- epic: z_features/analysis_shared_state.md (sub-task B — lensing datacube consumer)
- remaining: Phase 5 — datacube scripts opt into shared_preloads; autolens_workspace_test fast-assert + JAX assertion; autolens_profiling re-measure
- repos:
  - PyAutoLens: feature/datacube-shared-state
  - PyAutoArray: feature/datacube-shared-state
  - autolens_workspace: feature/datacube-shared-state
  - autolens_workspace_test: feature/datacube-shared-state
  - autolens_profiling: feature/datacube-shared-state
