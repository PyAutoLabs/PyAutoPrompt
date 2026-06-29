---
name: create-issue
description: Convert a prompt file from PyAutoMind/ into a GitHub issue with overview, plan, and starting AI prompt, then register and move it. The Mind issue+registry primitive that start_dev delegates to.
user-invocable: true
---

Turn a `PyAutoMind/` prompt into a tracked GitHub issue and register it.

A **PyAutoMind** primitive — it owns the mechanical **issue + registry write**:
assemble the issue body, create the issue, register the task in `active.md`, move
the prompt to `issued/`, and push Mind state. The *reasoning* (classify repos,
explore code, generate the plan) belongs to **PyAutoBrain**:

- **Called by Brain.** `/start_dev` (Brain) does the triage/planning via the
  Feature Agent and then **delegates the issue write to this skill**, passing the
  primary repo, title, plan and suggested branch. Brain does not re-implement
  issue creation.
- **Runnable standalone.** You can run `/create-issue` by itself when you just
  want an issue from a prompt without full triage or worktree routing — it does a
  *light* pass to fill any inputs the caller didn't supply. For deep
  classification + dev-environment setup, use `/start_dev` instead.

Organ boundary and the execution-environment model: PyAutoBrain `skills/WORKFLOW.md`.

## Usage

```
/create-issue <prompt-file-path>
```

Path relative to `PyAutoMind/`. Prompts live under `<work-type>/<target>/` (see
README "Prompt taxonomy"); pre-migration `<target>/<name>.md` paths still resolve.
Examples: `bug/autofit/factor_graph_instance_iteration.md`,
`feature/autoarray/oversampling.md`.

## Inputs

These come from the **caller** (Brain/`start_dev`) when delegated, or from a
**light standalone pass** here when run directly:

| Input | From caller | Standalone fallback |
|-------|-------------|---------------------|
| primary repo | Feature Agent classification | most-referenced `@RepoName` in the prompt; ask if ambiguous |
| title | caller | concise title (<70 chars), conventional prefix (`feat:`/`fix:`/`refactor:`/`docs:`/`perf:`) |
| plan (high + detailed) | caller | brief plan from a quick read of the prompt + referenced files |
| suggested branch | caller (`plan_branches`) | `feature/<short-desc>` kebab-case, <50 chars |

Repo → owner mapping: PyAutoConf/PyAutoFit → `rhayes777/`; everything else →
`Jammy2211/` (full table in `WORKFLOW.md`).

## Steps

### 0. Sync new prompt ideas (Mind)

```bash
source PyAutoMind/scripts/prompt_sync.sh
prompt_sync_new_prompts          # no-op if nothing untracked; else commits + pushes new ideas
```

### 1. Read the prompt

Read `PyAutoMind/<argument>`. If missing, report and list prompts in that folder.

### 2. Resolve the inputs

If the caller supplied repo/title/plan/branch, use them **verbatim**. Otherwise
do the light standalone pass from the Inputs table (don't run a full Brain
triage — that's `/start_dev`).

### 3. Assemble + create the issue

Build the body in this structure, then create it (present for review first):

```markdown
## Overview
<2-4 sentence summary of what this task is and why it matters>

## Plan
<high-level bullet plan — human readable, no code>

<details>
<summary>Detailed implementation plan</summary>

### Affected Repositories
- repo1 (primary)

### Branch Survey
| Repository | Current Branch | Dirty? |
|-----------|---------------|--------|
| ./RepoName | main | clean |

**Suggested branch:** `feature/<name>`

### Implementation Steps
1. <step with file paths>

### Key Files
- `path/to/file.py` — description
</details>

## Original Prompt
<details>
<summary>Click to expand starting prompt</summary>

<original prompt content copied verbatim>
</details>
```

```bash
gh issue create --repo <owner/repo> --title "<title>" --body "$(cat <<'ISSUE_EOF'
<body content>
ISSUE_EOF
)"
```

### 4. Register the task in active.md (Mind)

Add the task entry to `PyAutoMind/active.md` with the issue URL (schema in
README). **If the caller is handling registration itself** — e.g. `/start_dev`
routing a conflicted task to `planned.md` — skip this step and let it register.

### 5. Move the prompt to issued/

```bash
mv PyAutoMind/<path> PyAutoMind/issued/<filename>
```

Timestamp-suffix the filename if one already exists in `issued/`.

### 6. Push Mind

```bash
source PyAutoMind/scripts/prompt_sync.sh
prompt_sync_push "prompt: file issue for <task-name> (#<issue>)"
```

If step 0 already pushed, this carries only the active.md + `issued/` changes.

## Notes

- Always present the issue body for review before creating it.
- If `gh auth status` fails, tell the user to run `! gh auth login`.
- The detailed plan should be thorough enough that a fresh session could start
  from the issue alone.
