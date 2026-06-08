---
name: start-dev
description: Default entry point for PyAutoLabs development tasks, even when the user does not explicitly say start_dev. Use for requests to implement, fix, add, change, refactor, migrate, optimize, document, test, ship, or open a PR for any PyAutoLabs library, workspace, tutorial, build/release, or project-doc task. Creates or reads a PyAutoPrompt markdown prompt, creates or audits the GitHub issue, classifies library versus workspace work, surveys affected repository branches, generates the implementation plan, and registers the task before edits begin. Do not use for pure questions, reviews, status checks, or explicit workflow opt-outs.
---

# Start Dev

Use `start_dev.md` in this directory as the authoritative workflow body.

Follow the command file exactly, adapting Claude-specific references to Codex:

- `/start_dev <prompt>` means use this skill with the same prompt path.
- If the user gives a development task without a prompt path, first create a
  concise prompt file in the appropriate `PyAutoPrompt/` category and include
  the original request verbatim, then use this skill with that prompt path.
- "Plan Mode" means present the plan and wait for explicit user approval before
  file edits when this workflow is being used.
- Slash-command references such as `/plan_branches`, `/start_library`, and
  `/start_workspace` refer to the matching Codex skill or shared command body.

Do not duplicate or reinterpret the workflow here. If the workflow changes, edit
`start_dev.md`.
