# Monthly repo-health audit

> ⚠️ **Caveat — drafted from a stale repo state.** This prompt was drafted on 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits behind origin. The trigger looked like a structural workflow flaw, but later analysis showed the drift was largely driven by **stale local checkouts being edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt is the canonical source-of-truth and `skills/install.sh` auto-discovers across both repos, some of the recommendations below may be over-engineered for the day-to-day case. Re-evaluate whether each measure is still warranted — the cheap habits (pull before edit, never rewrite history) buy most of the win.

`pyauto-status` (prompt 01) catches commit-count drift but doesn't catch
*structural* problems — repos on dead branches, repos with no remote configured,
directories that look like repos but aren't, stale stashes, generated noise that
slipped past `.gitignore`. These compound silently. The 2026-04-27 audit found:

- PyAutoFit checked out on `main_build` (a defunct branch), 91 commits behind.
  No automation noticed because everything still imported.
- `autolens_workspace_developer` had no `origin` remote at all despite having a
  GitHub repo — committing into a void for weeks.
- `autofit_workspace_developer` had no `.git` at all — a directory that looked
  like a git repo to humans but wasn't.

A monthly cron / scheduled run catches these classes structurally.

## What to ship

A skill (or a plain script) `PyAutoPrompt/scripts/audit.sh` that, for every
directory under `~/Code/PyAutoLabs/`, reports:

1. **Branch on a non-default ref.** `git branch --show-current` vs the
   `origin/HEAD` symbolic ref. PyAutoFit on `main_build` would have lit up here.
2. **No remote configured.** `git remote -v` empty. Caught autolens_workspace_developer.
3. **No `.git` directory.** Caught autofit_workspace_developer.
4. **Working-tree files older than 30 days that aren't tracked.** Most often
   generated junk (`output_path/`, `image.fits`, etc.) that survived because
   `.gitignore` doesn't cover it yet.
5. **Stash entries older than 14 days.** Drift-from-stash is a real failure mode
   (this run almost fell into one when an early stash pop conflict was left
   sitting). Old stashes either matter (recover) or don't (drop).
6. **Branches local-only (no upstream) that haven't been touched in 30 days.**
   Likely abandoned feature branches.
7. **Any tracked file matching the noise patterns from `02_gitignore_noise.md`.**
   Means the gitignore patches haven't shipped yet for that repo.

Output format: per-repo block with severity-tagged findings (`ERROR` / `WARN` /
`INFO`), one block per repo, exit code = number of ERRORs.

Schedule via the existing `/schedule` skill or a plain cron — monthly is enough,
findings are slow-changing.

## Acceptance

- Running on the current tree finds zero ERRORs (everything was cleaned up
  manually on 2026-04-27).
- Synthetically introducing each failure mode (e.g. `git checkout -b dead-branch`
  in PyAutoFit, `rm -rf .git/refs/remotes` in another) makes the audit flag it.
- Output is short enough to read in 30 seconds — don't dump every untracked file,
  summarize.

## Companion config: snooze list

A `PyAutoPrompt/scripts/audit_snooze.txt` for things known-acceptable
(e.g. PyAutoConf intentionally on `feature/speed-up-unit-tests`). Format:
`<repo>:<finding-type>:<rationale>`. Keep it short — if the snooze list grows
beyond ~10 lines, that's a smell.

## Out of scope

- Auto-fixing findings. Audit reports; user decides.
- Disk-usage / pack-size audits (separate concern).

## Files touched

- `PyAutoPrompt/scripts/audit.sh` (new)
- `PyAutoPrompt/scripts/audit_snooze.txt` (new, possibly empty initially)
- Optionally a `/schedule` entry to run it monthly
