# Workspace `.gitignore` cleanup

> ⚠️ **Caveat — drafted from a stale repo state.** This prompt was drafted on 2026-04-27 during a forensic sweep that found local checkouts up to 101 commits behind origin. The trigger looked like a structural workflow flaw, but later analysis showed the drift was largely driven by **stale local checkouts being edited without `git pull` first**, not by missing tooling. Now that PyAutoPrompt is the canonical source-of-truth and `skills/install.sh` auto-discovers across both repos, some of the recommendations below may be over-engineered for the day-to-day case. Re-evaluate whether each measure is still warranted — the cheap habits (pull before edit, never rewrite history) buy most of the win.

Generated artifacts are routinely polluting `git status` in the workspace repos,
to the point where the user (and agents) stop reading `git status` because
"of course it's dirty". On 2026-04-27 this caused 101-commit drift on
PyAutoGalaxy to go unnoticed for weeks: the workspaces showed 8+ "dirty" files
each, all generated, so the actual modified files (real PyAutoJAX→PyAutoLabs
renames) were buried.

Examples of the noise pattern, all observed in working trees:

- `image.fits` at workspace root (a script wrote a literal default filename)
- `path/`, `scripts/path/` (a script took `"path"` as a literal arg)
- `scripts/scripts/` (a script ran from inside `scripts/` with `cd scripts/`)
- `output_path/` (similar literal)
- `root.log` (logger output)
- `__pycache__/` directories everywhere

Some of these reflect actual bugs in scripts (passing `"path"` as a positional
arg). The bugs should be fixed at source where reasonable, but `.gitignore` is
the backstop that prevents the pollution from compounding.

## What to ship

Add to the `.gitignore` of every workspace repo (`autofit_workspace`,
`autogalaxy_workspace`, `autolens_workspace`, the `*_test` and `*_developer`
variants):

```gitignore
# Generated artifacts — never check in
image.fits
path/
scripts/path/
scripts/scripts/
output_path/
root.log
*.log

__pycache__/
*.pyc
.codex/
```

Where the literal noise comes from a real script bug (e.g. `image.fits` from a
plotter that defaulted to that filename), file an issue or fix the source while
you're there.

## Acceptance

- After pulling the updated `.gitignore` into a clean workspace and running the
  smoke tests, `git status` is clean.
- The dataset/* simulator outputs that workspaces actually want to track
  (`dataset/imaging/.../data.fits` etc.) remain tracked — only the literal-string
  accidents are ignored.

## How to handle existing tracked files matching the new patterns

`git rm --cached <file>` (don't use plain `git rm`, that deletes content) then
commit the `.gitignore` and the `--cached` removal in the same PR.

## Out of scope

- Source-side fixes for the scripts that write `path/` etc. — file follow-up
  issues but don't bundle the fixes here.
- Library `.gitignore` updates (PyAutoConf/PyAutoFit/PyAutoArray/PyAutoGalaxy/
  PyAutoLens already have reasonable `.gitignore`s; the noise problem is
  workspace-specific).

## Files touched

One PR per workspace repo:

- `autofit_workspace/.gitignore`
- `autogalaxy_workspace/.gitignore`
- `autolens_workspace/.gitignore`
- `autofit_workspace_test/.gitignore`
- `autolens_workspace_test/.gitignore`
- `autofit_workspace_developer/.gitignore`
- `autolens_workspace_developer/.gitignore`
