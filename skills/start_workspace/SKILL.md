---
name: start-workspace
description: Set up PyAutoLabs workspace or tutorial repository development. Use when Codex needs to attach or create workspace worktrees, inspect upstream library PR API changes, identify affected scripts, register workspace repos in PyAutoPrompt, and prepare workspace-only or library-follow-up work.
---

# Start Workspace

Use `start_workspace.md` in this directory as the authoritative workflow body.

Follow the command file exactly, adapting Claude-specific references to Codex:

- `/start_workspace` means use this skill.
- Slash-command references such as `/ship_workspace` and `/smoke_test` refer to
  the matching Codex skill or shared command body.
- Maintain the library-first rule: linked workspace work follows the upstream
  library PR and must use that PR's API-change summary.

Do not duplicate or reinterpret the workflow here. If the workflow changes, edit
`start_workspace.md`.
