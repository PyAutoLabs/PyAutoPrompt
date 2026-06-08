# Active Tasks

## kaplinghat-sidm-cored-nfw
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/564
- session: codex --resume <session-id>
- status: pyautogalaxy-pr-created, pyautolens-pending
- library-pr: https://github.com/PyAutoLabs/PyAutoGalaxy/pull/471
- worktree: ~/Code/PyAutoLabs-wt/kaplinghat-sidm-cored-nfw
- suggested-branch: feature/kaplinghat-sidm-cored-nfw
- classification: library
- affected-repos:
  - PyAutoGalaxy
- notes: Initial implementation phase scoped to PyAutoGalaxy because PyAutoLens is currently claimed by datacube-shared-state. PyAutoLens substructure batching/validation should follow after that claim clears if source changes are needed there.
- repos:
  - PyAutoGalaxy: feature/kaplinghat-sidm-cored-nfw

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
