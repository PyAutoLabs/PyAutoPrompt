# `/sync` slash command — one-shot multi-repo sync

> ⚠️ **Caveat — drafted from a stale repo state.** This prompt was drafted on 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits behind origin. The trigger looked like a structural workflow flaw, but later analysis showed the drift was largely driven by **stale local checkouts being edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt is the canonical source-of-truth and `skills/install.sh` auto-discovers across both repos, some of the recommendations below may be over-engineered for the day-to-day case. Re-evaluate whether each measure is still warranted — the cheap habits (pull before edit, never rewrite history) buy most of the win.

A reproducible version of what was done by hand on 2026-04-27 to pull all 12
repos in line with origin/main. Drift recovery should be a one-command,
one-minute operation so doing it weekly is cheap and drift never compounds.

## What to ship

A new skill `PyAutoPrompt/skills/sync/SKILL.md` (or `sync.md` as a flat command
file — match whichever convention the existing prompt-coupled skills use).

Behavior, deterministic per-repo:

```
for repo in <repos under ~/Code/PyAutoLabs/>:
    if no .git:                       skip with warning
    elif no upstream:                 warn, list (don't auto-set; surfaces real bugs)
    elif merge-base = NONE:            surface for confirmation BEFORE any reset
    elif behind > 0 and ahead = 0:     ff-pull (auto, after handling dirty)
    elif behind > 0 and ahead > 0:     audit local-ahead commits per origin/main:
                                         - merge commits → check PR state via gh
                                         - non-merge commits → grep origin/main for subject match
                                         If all duplicates, reset --hard with confirm.
                                         If any unique → present diff, ask user.
    elif ahead > 0 only:               warn, don't push (pushing is user-driven)
    elif clean:                        OK
```

Dirty-file handling:

- Untracked files byte-identical to upstream: silently delete (the "lost"
  fits/npz fixtures pattern that blocked pulls on 2026-04-27).
- Tracked-modified files whose diff is fully present in upstream: stash, pull,
  drop stash if redundant (the PyAutoJAX→PyAutoLabs rename pattern).
- Anything else: stash with named label, pull, surface for user.

Output: a single dashboard at the end identical to `pyauto-status` (see
`01_status_dashboard.md`) so the user sees clear before/after.

## Implementation notes

- The forensic audit on 2026-04-27 has a working sequence — start from that
  conversation transcript or the commands recorded in skill development logs.
- For the "are local-ahead commits all duplicated" check, the heuristic that
  worked: subject-match against `git log origin/main --oneline | grep -F
  "$subject"`, plus PR-merged check via `gh pr view <num> --json state`. False
  positives are rare in practice but always confirm before resetting.
- Workspaces with `merge-base = NONE` are the dangerous case — never auto-reset
  these; always show the user the local-ahead commit list and the comparison
  with origin's, exactly as the 2026-04-27 audit did.

## Acceptance

- `/sync` run on a clean tree (all repos at origin/main) prints the dashboard
  and exits in <30s with no changes.
- `/sync` run on a synthetically-staled tree (one repo `git reset --hard HEAD~5`)
  fast-forwards it and prints clean dashboard.
- `/sync` run on a tree with truly-divergent local commits (one repo with
  unique work) prompts before any destructive action and aborts cleanly on "no".
- `/sync` doesn't touch `admin_jammy` or `PyAutoPrompt` differently — same
  rules apply, but those repos see fewer noisy false positives because they're
  small.

## Dependencies

- `01_status_dashboard.md` (the dashboard format the skill prints)
- `pyauto-status` shell function or equivalent

## Out of scope

- Auto-pushing local-ahead commits. Push is always user-initiated.
- Sync of node_modules / pip installs after pulling. That's `06_repo_health_audit.md`.

## Files touched

- `PyAutoPrompt/skills/sync/SKILL.md` (new)
- `admin_jammy/skills/install.sh` — add `sync` to the SKILLS array (if it has
  a SKILL.md — it should)
