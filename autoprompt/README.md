# autoprompt/

Workflow-infrastructure prompts drafted on 2026-04-27 after a forensic
sweep found local checkouts up to 101 commits behind origin. The eight
prompts addressed root causes structurally — visibility, gitignore noise,
history-rewrite guards, source-of-truth rules, sync tooling, repo
audits, worktree-only enforcement, and persistent test/build summaries.

This directory now contains only this README. The shipped prompts live
in [`../issued/`](../issued/) (per-prompt `<NN>_<slug>.md`); the rejected
ones are gone.

## Outcomes

| # | Prompt | Outcome | Implementation |
|---|---|---|---|
| 01 | `pyauto-status` shell function | Shipped | [issued/01_status_dashboard.md](../issued/01_status_dashboard.md) — `scripts/pyauto_status.sh` |
| 02 | `.gitignore` workspace artifacts | Shipped | [issued/02_gitignore_noise.md](../issued/02_gitignore_noise.md) — 8 workspace PRs |
| 03 | "Never rewrite history" rule | Shipped | [issued/03_history_rewrite_guard.md](../issued/03_history_rewrite_guard.md) — 17 repos × CLAUDE.md/AGENTS.md |
| 04 | "Pull before edit" rule | **Skipped** | Redundant with 01 (BEHIND counts) + 03 (history guard) |
| 05 | `/sync` slash command | Shipped (re-scoped) | [issued/05_sync_slash_command.md](../issued/05_sync_slash_command.md) — dashboard's "Follow-up commands:" section instead of a heavyweight skill |
| 06 | Monthly repo-health audit | Shipped (re-scoped) | [issued/06_repo_health_audit.md](../issued/06_repo_health_audit.md) — `scripts/pyauto_audit.sh` (on-demand `pyauto-audit`, no cron) |
| 07 | Worktree-only edits enforcement | **Skipped** | Doc-rule layer wasn't worth 17 PRs; shell hook + `/sync` reset depended on rejected pieces |
| 08 | Persistent test/build summary | Shipped | [issued/08_test_summary.md](../issued/08_test_summary.md) — dashboard "Smoke tests:" + "Last autobuild run:" sections |

## What the sweep actually fixed

- **Drift visibility.** `pyauto-status` runs on every venv activation and
  prints branch / upstream / behind / ahead / dirty counts plus the
  `b` flag for forgotten feature branches. Drift can no longer hide.
- **Generated noise.** Workspaces' `.gitignore` files were rationalised
  (prompt 02), so `git status` shows real divergence instead of pyc
  pollution and stray artifacts.
- **History-rewrite guard.** Every PyAuto repo's `CLAUDE.md` /
  `AGENTS.md` documents the forbidden ops (`rm -rf .git && git init`,
  `Initial commit` resets, force-push to main, etc.). Catches the
  specific pattern that caused ~40 redundant local commits.
- **Actionable dashboard.** Beyond the table, `pyauto-status` prints
  copy-pasteable follow-up commands grouped by category (Pull /
  Set missing upstream / Investigate manually) — drift recovery is
  now one paste away.
- **Structural audit.** `pyauto-audit` finds non-git directories under
  `~/Code/PyAutoLabs/`, stashes >14 days old, and abandoned local-only
  branches (>30 days). Run on demand.
- **Test + release status.** Dashboard now reads
  `~/.cache/pyauto/smoke/<workspace>.json` (written by the
  `/smoke-test` skill) and PyAutoBuild's committed `test_results/` to
  show colored smoke counts and the latest autobuild aggregate on
  every venv activation.

## What the sweep deliberately didn't do

- Doc rules in CLAUDE.md repeating things the dashboard already
  surfaces (prompt 04).
- A heavyweight `/sync` slash command with stash-handling and
  duplicate-commit detection (prompt 05's full spec — replaced by
  copy-pasteable commands in the dashboard).
- A monthly cron schedule for the audit (prompt 06's full spec — the
  audit is on-demand only).
- Read-only-mirror enforcement of canonical checkouts via shell hooks
  or `/sync` resets (prompt 07 entirely).

## Reading order

If you want the historical narrative, read the shipped prompts in
numeric order in [`../issued/`](../issued/) — each preserves its
2026-04-27 framing, including its own caveat about being drafted from
a stale repo state. Re-evaluate before reusing as a template; the
cheap habits (pull before edit, never rewrite history) bought most of
the win.
