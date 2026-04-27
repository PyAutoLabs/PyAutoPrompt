# AGENTS.md

This file is for AI coding agents (Claude Code, Codex, Cursor, etc.) discovering
this repository.

## What this repo is

**PyAutoPrompt is the starting point of the PyAuto workflow.** Every task that
ends up as a PR in PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens,
or any of the `*_workspace*` repos begins as a markdown file here.

For the full workflow narrative, conventions, and registry schemas, read
[README.md](README.md). The summary below is just enough to operate.

## Layout (operational)

- **Prompts** — `<category>/<name>.md` (free-form markdown, one task per file).
  Categories: `autoarray/`, `autofit/`, `autogalaxy/`, `autolens/`, `autobuild/`,
  `cluster/`, `weak/`, `workspaces/`, `autolens_workspace_developer/`,
  `autoprompt/`, `z_vault/`.
- **Registry** — root-level markdown files: `active.md`, `complete.md`,
  `planned.md`, `queue.md`, `priority.md`, `ideas.md`. Mutate these only via the
  skills in `skills/` so commit messages stay consistent.
- **Skills** — `skills/<name>/` are Claude Code skills/commands tightly coupled
  to the registry. They source `scripts/prompt_sync.sh` for commit/push.
- **Scripts** — `scripts/status.sh` (inventory), `scripts/prompt_sync.sh`
  (commit/push helpers).

## Hard rules

1. **Never rewrite history on any branch with a remote.** No `git init` over an
   existing repo, no `git push --force` to `main`. The 2026-04-27 drift incident
   that motivated `autoprompt/03_history_rewrite_guard.md` is the reason.
2. **Pull before edit.** `git fetch && git status` first, every time. If behind
   `origin/main`, `git pull --ff-only` before touching anything. See
   `autoprompt/04_source_of_truth_rule.md`.
3. **One prompt = one task = one PR.** If a prompt outlines multiple
   loosely-related changes, split into separate prompt files before issuing.
4. **`tmp/` is scratch.** Never commit anything under it.

## When you are asked to add a new prompt

Write the file under the appropriate category. Don't touch `active.md` or
`issued/` directly — those are managed by `/start_dev` / `/create_issue`.

## When you are asked to start work on an existing prompt

Use `/start_dev <category>/<name>.md`. It will route to `/start_library` or
`/start_workspace` based on the repos referenced in the prompt body.

## When in doubt

Read [README.md](README.md). It is current as of the last commit on this branch.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
