# Active Tasks

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

## latent-class-phase2
- session: claude (local)
- status: library-dev
- worktree: ~/Code/PyAutoLabs-wt/latent-class-phase2
- suggested-branch: feature/latent-class-phase2
- classification: library
- depends-on: PyAutoFit feature/latent-class-phase1 (PR #1315, unmerged — the worktree's PyAutoFit rides that branch so af.Latent is available)
- prompt: PyAutoPrompt/autofit/latent_class_redesign.md (Phase 2)
- notes: Phase 2 of the latent redesign — ship LensLatent/GalaxyLatent subclasses of af.Latent in autolens/autogalaxy and declare Analysis.Latent. Edits analysis/latent.py + imaging/model/analysis.py + config/latent.yaml; ZERO file overlap with kaplinghat-sidm-cored-nfw (mass/dark profiles), verified parallel-safe. Kept off main so a release stays clean.
- repos:
  - PyAutoFit: feature/latent-class-phase1 (dependency, no new commits here)
  - PyAutoGalaxy: feature/latent-class-phase2
  - PyAutoLens: feature/latent-class-phase2
