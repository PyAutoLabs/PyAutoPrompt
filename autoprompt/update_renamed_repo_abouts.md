# Update GitHub "About" descriptions for renamed organism repos

As part of the PyAuto **organism** renames, several repositories have been (or
are being) renamed, but their GitHub **About** descriptions still describe the
old identity. Do a single pass — on the **laptop**, where `gh` and a desktop
browser are available — to bring every renamed repo's description in line with
its new organism role, then apply them.

The repo "About" description is **not** part of the git tree, so it cannot be
changed by a normal commit/push. Set each one with the GitHub CLI from the
laptop:

```bash
gh repo edit PyAutoLabs/<repo> --description "<new description>"
```

(or via the repo's main page → the ⚙ gear next to "About" → Description).

## Repos to update

Organism mapping: Mind → Brain → Hands → Heart (+ Memory).

- **PyAutoMind** (renamed from PyAutoPrompt) —
  "The Mind of the PyAuto organism: ideas, intent, goals, priorities, and the
  prompt registry that starts every PyAuto task."
- **PyAutoHeart** (renamed from PyAutoPulse) —
  "The Heart of the PyAuto organism: health and release-readiness checking for
  the PyAuto ecosystem." (confirm it already reads this way; update if not.)
- **PyAutoBrain** (to be renamed from PyAutoAgent) —
  "The Brain of the PyAuto organism: reasoning, planning and routing of PyAuto
  development work."
- **PyAutoMemory** (to be renamed from PyAutoPaper) —
  "The Memory of the PyAuto organism: accumulated knowledge — papers, wikis and
  reference material."
- **PyAutoBuild** (the "Hands"; repo name unchanged) — confirm the description
  reflects its executor/build-and-release role; update only if it still reads as
  an older description.

## Notes

- **Skip any repo whose GitHub-side rename has not yet happened.** Set its
  About in the same pass that performs the rename, so the description never
  points at a name that doesn't exist yet (PyAutoBrain/PyAutoMemory are pending
  at the time of writing).
- Keep each description to one sentence, matching the tone of the other org
  repos.
- This is a **metadata-only chore**: no code changes, no PR. Run the `gh repo
  edit` commands (or use the About gear) and confirm each on the repo page.
