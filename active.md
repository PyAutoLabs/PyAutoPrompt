# Active Tasks

## nss-optional-dependency-workspace
- issue: https://github.com/PyAutoLabs/autofit_workspace/issues/70
- session: codex --resume <session-id>
- status: workspace-dev
- worktree: ~/Code/PyAutoLabs-wt/nss-optional-dependency-workspace
- suggested-branch: feature/nss-optional-dependency-workspace
- classification: workspace
- affected-repos:
  - autofit_workspace
- repos:

## kaplinghat-sidm-cored-nfw
- issue: https://github.com/PyAutoLabs/PyAutoLens/issues/564
- session: codex --resume <session-id>
- status: library-and-workspace-prs-created
- library-pr:
  - https://github.com/PyAutoLabs/PyAutoGalaxy/pull/471
  - https://github.com/PyAutoLabs/PyAutoLens/pull/567
- workspace-pr:
  - https://github.com/PyAutoLabs/autolens_workspace_test/pull/139
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
  - autolens_workspace_test: feature/kaplinghat-sidm-cored-nfw
