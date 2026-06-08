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
  - PyAutoLens: 3fd6dd6d7 feat: support Kaplinghat halos in substructure arrays

## datacube-shared-state
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/565
- session: claude --resume <session-id>
- status: library-shipped (curvature); slice-2b + workspace pending
- library-pr:
  - https://github.com/PyAutoLabs/PyAutoArray/pull/344
  - https://github.com/PyAutoLabs/PyAutoLens/pull/566
- worktree: ~/Code/PyAutoLabs-wt/datacube-shared-state
- epic: z_features/analysis_shared_state.md (sub-task B — lensing datacube consumer)
- remaining: slice-2b mapper-sharing (library); Phase 5 datacube scripts + autolens_workspace_test + autolens_profiling
- repos:
  - PyAutoLens: feature/datacube-shared-state
  - PyAutoArray: feature/datacube-shared-state
