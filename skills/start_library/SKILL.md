---
name: start-library
description: Set up PyAutoLabs source-library development after start-dev has created or registered an issue. Use when Codex needs to create or resume task worktrees for PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, or PyAutoLens, register claimed repos in PyAutoPrompt, and prepare the session for source-code edits.
---

# Start Library

Use `start_library.md` in this directory as the authoritative workflow body.

Follow the command file exactly, adapting Claude-specific references to Codex:

- `/start_library` means use this skill.
- Slash-command references such as `/ship_library` refer to the matching Codex
  skill or shared command body.
- When the command file says to display a summary, report the same operational
  details to the user before editing inside the task worktree.

Do not duplicate or reinterpret the workflow here. If the workflow changes, edit
`start_library.md`.
