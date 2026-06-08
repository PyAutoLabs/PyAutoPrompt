---
name: ship-library
description: Ship PyAutoLabs source-library changes. Use when Codex needs to inspect diffs, draft commit messages and PR bodies with API-change summaries, run library tests, commit, push, create pending-release PRs, analyze downstream workspace impact, update GitHub issues, and move PyAutoPrompt task state.
---

# Ship Library

Use `ship_library.md` in this directory as the authoritative workflow body.

Follow the command file exactly, adapting Claude-specific references to Codex:

- `/ship_library` means use this skill.
- If the command file delegates mechanical execution to a Claude subagent, Codex
  should either use an available subagent tool with the same contract or perform
  the same mechanical steps directly while preserving the judgment/mechanical
  split in the user-facing workflow.
- Preserve the `## API Changes` PR-body contract because `/start_workspace`,
  release review, and downstream workspace migration depend on it.

Do not duplicate or reinterpret the workflow here. If the workflow changes, edit
`ship_library.md`.
