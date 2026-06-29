# PyAuto workflow skill ownership — audit

Driven by [`autoprompt/skill_location.md`](../autoprompt/skill_location.md). This
is an **ownership and location audit only** — no skill behaviour was redesigned
and no command names changed (that is the separate
[`autoprompt/skill_redesign.md`](../autoprompt/skill_redesign.md) task).

## Headline finding

Every PyAuto development-workflow skill currently lives as a **real, Git-tracked
file under `PyAutoMind/skills/<name>/`**. In the auditable scope there are:

- **no symlinks** inside `PyAutoMind/skills/` (`find -type l` is empty);
- **no `PyAutoLabs/.agents/skills/` directory** — it does not exist in this
  organism layout;
- **no stale installed copies** of these skills in any sibling repo
  (`PyAutoBrain`, `PyAutoBuild`, `PyAutoHeart`, `PyAutoMemory` have **no
  `skills/` directory at all**);
- **no `admin_jammy/` checkout** in scope, so its generic skills
  (`audit_docs`, `dep_audit`, `repo_cleanup`) and — importantly — the **install
  script** were not inspectable from this session.

So the premise that skills "may live under `PyAutoLabs/.agents/skills` with
symlinks or installed copies" does **not** hold here: the canonical source for
all of them is unambiguously `PyAutoMind/skills/`, and it is clean.

### How the skills are surfaced (the discovery mechanism)

Per `PyAutoMind/README.md` (Bootstrap section), skills are made visible to
Claude/Codex by:

```
bash admin_jammy/skills/install.sh   # symlinks skills + commands
```

`install.sh` **auto-discovers skills from both `admin_jammy/skills/` and
`PyAutoMind/skills/`** and creates symlinks under `~/.claude/skills/` and
`~/.claude/commands/`. The symlinks therefore live in the user's home, not in
any repo, and are regenerated on demand — they are install state, not canonical
source.

**This is the blocker for physically relocating skills (see below):
`admin_jammy/skills/install.sh` is the only thing that makes a skill
discoverable, it only scans `admin_jammy/skills/` and `PyAutoMind/skills/`, and
`admin_jammy` is out of scope for this session — it cannot be edited to add
`PyAutoBrain/skills/`, `PyAutoBuild/skills/`, etc. as discovery roots.**

## Coupling to the registry

The workflow skills are tightly bound to PyAutoMind's registry. They reference
**workspace-root-relative** paths such as `PyAutoMind/active.md`,
`PyAutoMind/complete.md`, and `source PyAutoMind/scripts/prompt_sync.sh`. Because
those paths are anchored at the `~/Code/PyAutoLabs/` checkout root (not relative
to the skill's own repo), a skill file *could* be relocated to a sibling repo
without breaking those references — but it would still be undiscoverable until
`admin_jammy/skills/install.sh` learns the new root.

## Ownership table

Ownership rule (from the prompt): **Mind** = intent/task-registry · **Brain** =
reasoning/agent-orchestration · **Build** = execution/release/build · **Heart** =
health/validation/readiness · `admin_jammy` should not own canonical organism
workflow skills.

| Skill | Current source | Symlink? | Tracked? | Current owner | Recommended owner | Action taken |
|-------|----------------|----------|----------|---------------|-------------------|--------------|
| `create_issue` | `PyAutoMind/skills/create_issue/` | no | yes | PyAutoMind | **PyAutoMind** (correct) | none — keep |
| `handoff` | `PyAutoMind/skills/handoff/` | no | yes | PyAutoMind | **PyAutoMind** (registry/work-state) | none — keep |
| `register_and_iterate` | `PyAutoMind/skills/register_and_iterate/` | no | yes | PyAutoMind | **PyAutoMind** today; orchestration loop → PyAutoBrain at redesign | defer to redesign |
| `plan_branches` | `PyAutoMind/skills/plan_branches/` | no | yes | PyAutoMind | **PyAutoBrain** (planning/reasoning) | defer to redesign |
| `start_dev` | `PyAutoMind/skills/start_dev/` | no | yes | PyAutoMind | **PyAutoBrain** (classification/routing entry point) + PyAutoMind registry interface | defer to redesign |
| `start_dev_for_user` | `PyAutoMind/skills/start_dev_for_user/` | no | yes | PyAutoMind | **PyAutoBrain** (routing variant) | defer to redesign |
| `start_library` | `PyAutoMind/skills/start_library/` | no | yes | PyAutoMind | **PyAutoBuild** (worktrees/branches) + PyAutoMind registry interface | defer to redesign |
| `start_workspace` | `PyAutoMind/skills/start_workspace/` | no | yes | PyAutoMind | **PyAutoBuild** (worktrees/branches) | defer to redesign |
| `ship_library` | `PyAutoMind/skills/ship_library/` | no | yes | PyAutoMind | **PyAutoBuild** (commit/PR/release) + PyAutoHeart gate | defer to redesign |
| `ship_workspace` | `PyAutoMind/skills/ship_workspace/` | no | yes | PyAutoMind | **PyAutoBuild** (commit/PR/release) + PyAutoHeart gate | defer to redesign |
| `pyauto-status` | `PyAutoMind/skills/pyauto-status/` | no | yes | PyAutoMind | **PyAutoHeart** (status/readiness view; reads Mind registry) | defer to redesign |
| `pyauto-status-full` | `PyAutoMind/skills/pyauto-status-full/` | no | yes | PyAutoMind | **PyAutoHeart** | defer to redesign |
| `worktree_status` | `PyAutoMind/skills/worktree_status/` | no | yes | PyAutoMind | **PyAutoHeart** (diagnostic) / reads Mind registry | defer to redesign |
| `profile_likelihood` | `PyAutoMind/skills/profile_likelihood/` | no | yes | PyAutoMind | **`autolens_profiling`** — a science-profiling skill, not an organism workflow skill | defer (out of workflow scope) |

(`*/agents/openai.yaml` and the `SKILL.md` ↔ `<name>.md` pairs are bundled Codex
agent configs / dispatcher+body pairs, not separate skills.)

## Why no files were physically moved in this audit

The prompt allows moving "if a skill is in the wrong canonical repo", but doing
so safely is **not possible from this session**, and doing it unsafely would
break live workflows — which the prompt forbids:

1. **Discovery cannot be updated.** The only mechanism that surfaces a skill is
   `admin_jammy/skills/install.sh`, which scans only `admin_jammy/skills/` and
   `PyAutoMind/skills/`. `admin_jammy` is out of scope, so a skill relocated to
   `PyAutoBrain/skills/` or `PyAutoBuild/skills/` would silently stop being
   installed. Moving without updating the installer = a broken workflow.
2. **No target infrastructure exists yet.** `PyAutoBrain`, `PyAutoBuild`, and
   `PyAutoHeart` have no `skills/` directory and no convention for hosting
   these registry-coupled skills. Establishing that is part of the redesign,
   not of a location audit.
3. **The move is inseparable from the redesign.** `skill_redesign.md` is the
   task that turns these standalone skills into thin Brain/Build/Heart
   delegating wrappers. Relocating the source now — before that rewiring — would
   either duplicate work or strand half-migrated skills. The prompt explicitly
   says: *"Do not redesign the skills yet… Do not substantially rewrite
   behaviour… avoid breaking existing workflows."*

The correct, non-breaking outcome of *this* audit is therefore: **canonical
ownership is recorded (this file), recommended target owners are decided, and
the physical relocation is deferred to `skill_redesign.md`** with its
prerequisites made explicit.

## Prerequisites to unblock the physical move (for the redesign task)

1. Add discovery roots for `PyAutoBrain/skills/`, `PyAutoBuild/skills/`, and
   `PyAutoHeart/skills/` to `admin_jammy/skills/install.sh` (out of this
   session's scope).
2. Create a `skills/` convention in each target repo.
3. Move each skill per the **Recommended owner** column, preserving the
   `name:` / command names so `/start_dev`, `/ship_library`, etc. keep working.
4. Keep the `PyAutoMind/...`-anchored registry references (`active.md`,
   `complete.md`, `prompt_sync.sh`) — they remain valid from any sibling repo.
5. Re-run `install.sh` and verify every command still resolves.

## Validation performed

- `find PyAutoMind/skills -type l` → empty (no symlinks).
- `ls PyAuto{Brain,Build,Heart,Memory}/skills` → absent (no stale copies).
- `git ls-files PyAutoMind/skills/` → all 24 files tracked.
- Grep for skill-install / symlink machinery across in-scope repos → only
  `PyAutoMind/README.md` (documenting `admin_jammy/skills/install.sh`).
