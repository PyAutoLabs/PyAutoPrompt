# Active Tasks

## latent-jax-release-failures
- issue: https://github.com/PyAutoLabs/PyAutoFit/issues/1316
- session: codex --resume <session-id>
- status: library-and-workspace-prs-created, ci-green, merge-pending
- library-pr: https://github.com/PyAutoLabs/PyAutoFit/pull/1317
- workspace-prs:
  - https://github.com/PyAutoLabs/autofit_workspace_test/pull/33
  - https://github.com/PyAutoLabs/autogalaxy_workspace/pull/112
  - https://github.com/PyAutoLabs/autogalaxy_workspace_test/pull/65
- worktree: ~/Code/PyAutoLabs-wt/latent-jax-release-failures
- suggested-branch: feature/latent-jax-release-failures
- classification: both
- affected-repos:
  - PyAutoFit
  - autofit_workspace_test
  - autogalaxy_workspace
  - autogalaxy_workspace_test
- notes: Start library-first at `autofit_workspace_test/scripts/jax_assertions/fitness_dispatch.py`; PyAutoFit canonical checkout currently has unrelated local latent cleanup work on `feature/latent-old-api-cleanup`, so use the task worktree.
- repos:
  - PyAutoFit: feature/latent-jax-release-failures
  - autofit_workspace_test: feature/latent-jax-release-failures
  - autogalaxy_workspace: feature/latent-jax-release-failures
  - autogalaxy_workspace_test: feature/latent-jax-release-failures

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
