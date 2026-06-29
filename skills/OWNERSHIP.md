# PyAuto workflow skill ownership — audit + relocation

Originally an **ownership/location audit** (PR #26); this version records the
**completed relocation** done by the skill-redesign task (`autoprompt/skill_redesign.md`).
Each workflow skill now lives in the organ that owns its responsibility, command
names preserved, discovery wired up via `admin_jammy/skills/install.sh`.

## What moved (and why)

Ownership reconciled with the **actual** organism boundary docs
(`PyAutoBrain/AGENTS.md`, `PyAutoBuild/AGENTS.md`), not the aspirational mapping
in `skill_redesign.md`:

- **Mind** = intent + task-registry state.
- **Memory** = accumulated knowledge (consulted for context).
- **Brain** = reasoning + how-work-gets-done; **owns the development workflow**.
- **Heart** = health / validation / readiness gates.
- **Build (Hands)** = **release/packaging executor only** (tag / notebooks / PyPI
  via `release.yml`). It owns **no** workflow skills; `ship_*` is feature-dev work
  that only *calls* Build's release step at release time.

| Skill | New home | Recommended owner | Action taken |
|-------|----------|-------------------|--------------|
| `create_issue` | `PyAutoMind/skills/` | **PyAutoMind** — the issue+registry **primitive** (Brain's `/start_dev` delegates the issue write to it; runnable standalone) | kept in Mind |
| `start_dev` | `PyAutoBrain/skills/` | **PyAutoBrain** (classification/routing entry) | **moved → Brain** |
| `start_dev_for_user` | `PyAutoBrain/skills/` | **PyAutoBrain** (routing variant) | **moved → Brain** |
| `plan_branches` | `PyAutoBrain/skills/` | **PyAutoBrain** (planning) | **moved → Brain** |
| `start_library` | `PyAutoBrain/skills/` | **PyAutoBrain** (dev-cycle setup) | **moved → Brain** |
| `start_workspace` | `PyAutoBrain/skills/` | **PyAutoBrain** (dev-cycle setup) | **moved → Brain** |
| `ship_library` | `PyAutoBrain/skills/` | **PyAutoBrain** dev-workflow → Heart gate (Build only at release) | **moved → Brain** |
| `ship_workspace` | `PyAutoBrain/skills/` | **PyAutoBrain** dev-workflow → Heart gate (Build only at release) | **moved → Brain** |
| `register_and_iterate` | `PyAutoBrain/skills/` | **PyAutoBrain** (dev-workflow orchestration loop) | **moved → Brain** |
| `repo_cleanup` | `PyAutoBrain/skills/` | **PyAutoBrain** (between-tasks git hygiene; Heart observes, Brain decides + executes — natural home is a future Cleanup Agent) | **moved → Brain** (from admin_jammy) |
| `pyauto-status` | `PyAutoHeart/skills/` | **PyAutoHeart** (status/readiness view) | **moved → Heart** |
| `pyauto-status-full` | `PyAutoHeart/skills/` | **PyAutoHeart** (release-readiness dashboard) | **moved → Heart** |
| `worktree_status` | `PyAutoHeart/skills/` | **PyAutoHeart** (diagnostic) | **moved → Heart** |
| `profile_likelihood` | `autolens_profiling/skills/` | **`autolens_profiling`** (science profiling) | **moved → autolens_profiling** |
| `handoff` | — (removed) | — | **deleted** — the phone↔laptop park/resume dance is obsolete now PyAutoBrain runs uniformly across execution environments; `active.md` is the shared task state, so any environment resumes a task directly |

### Second wave — general admin_jammy skills

A follow-up pass re-homed the remaining admin_jammy skills so `admin_jammy/`
trends toward installer + tooling only:

- **→ PyAutoHeart** (read-only validation/readiness checks): `dep_audit`,
  `verify_install`, `review_release`, `audit_docs`, `cli_noise_clean`.
- **→ PyAutoBrain** (dev-workflow): `update_issue`, `repo_cleanup`.
- **→ PyAutoBuild** (release execution): `pre_build`. PyAutoBuild now has a
  `skills/` root for its **release-execution** skills only — it still owns no
  dev-workflow skills (`ship_*` live in Brain and only *call* its release step).
- **Removed:** `start-new-project` was a duplicate of the canonical science-project
  skill already owned by `autolens_assistant` (`skills/start-new-project.md`),
  so the admin_jammy fork was deleted rather than moved.

After this, `admin_jammy/skills/` hosts **no skills** — only the installer
(`install.sh`) and the line-count guard (`check_skill_line_counts.sh`).

(`*/agents/openai.yaml` and the `SKILL.md` ↔ `<name>.md` pairs are bundled Codex
agent configs / dispatcher+body pairs, not separate skills. Long-form detail was
factored into per-skill `reference.md` files and the shared
`PyAutoBrain/skills/WORKFLOW.md`, keeping every primary skill file under 200
lines.)

## Discovery

`admin_jammy/skills/install.sh` now scans these roots and symlinks skills into
`~/.claude/skills/` and commands into `~/.claude/commands/`:

- `admin_jammy/skills/` — general PyAuto tooling
- `PyAutoMind/skills/` — registry-coupled (`create_issue`)
- `PyAutoBrain/skills/` — development-workflow
- `PyAutoHeart/skills/` — status / readiness
- `autolens_profiling/skills/` — science profiling

**PyAutoBuild's `skills/` root holds release-execution skills only** (`pre_build`)
— it owns no *dev-workflow* skills; `ship_*` live in Brain and only call its
release step. Registry
references stay workspace-root-anchored (`PyAutoMind/active.md`,
`PyAutoMind/complete.md`, `source PyAutoMind/scripts/prompt_sync.sh`), which
resolve from any sibling repo, so the moved skills keep working unchanged.

## Redesign summary

The moves were paired with the redesign in `autoprompt/skill_redesign.md`:

- Skills are **thin entry points** that delegate to the organism — Brain reasons
  and routes (Feature/Build/Health agents), Heart gates ship via
  `pyauto-heart readiness`, Build executes release only, Mind holds state, Memory
  supplies context.
- `start_dev` routes through the **Feature Agent** (+ Memory); `ship_*` gates
  through the **Health Agent → Heart** before the dev workflow commits/pushes/PRs.
- "Remote / mobile mode" is gone, replaced by the general execution-environment
  model (`local-dev` / `web-github` / `ci-only` / `analysis-only`) in
  `PyAutoBrain/skills/WORKFLOW.md`. No phone↔laptop handoff concept remains.
- Command names and the start/ship lifecycle are preserved.

## Validation

- `bash admin_jammy/skills/install.sh` → every moved `/command` resolves.
- `find <each skills dir> -type l` → no stray symlinks in source.
- `bash admin_jammy/skills/check_skill_line_counts.sh` → all primary workflow
  skill files within 200 lines.
- Grep for stale `PyAutoMind/skills/` workflow paths and mobile/remote-mode terms
  → clean across the moved skills.
