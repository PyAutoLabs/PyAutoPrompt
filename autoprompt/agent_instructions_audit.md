# Audit AGENTS.md / CLAUDE.md across the whole ecosystem

Every PyAuto repo has accumulated its own `CLAUDE.md` and/or `AGENTS.md` independently
over time, the same way the pre-2026-04-27 sweep found 17 repos drifting on git hygiene
(see `z_vault/README.md` and `issued/03_history_rewrite_guard.md` — the "Never rewrite
history" block was the fix for that). The instructions files themselves now have the
same kind of unaudited drift. Two concrete things found just from reading the files in
this session, which is exactly the sort of thing this audit should surface systematically
instead of by accident:

- **CLAUDE.md and AGENTS.md have silently diverged where both exist.** `@PyAutoFit/AGENTS.md`
  (68 lines) and `@PyAutoFit/CLAUDE.md` (95 lines) are not the same document with a
  different name — they're independently written and already inconsistent. Same pattern
  in PyAutoArray, PyAutoConf, PyAutoGalaxy, PyAutoLens. `@autolens_assistant/CLAUDE.md` is
  the counter-example: it's a 7-line stub that does `@AGENTS.md` (Claude Code's import
  syntax) so there is exactly one document to maintain. Worth checking whether that pattern
  generalizes to the library repos.
- **The "Never rewrite history" section is pasted verbatim (~20 lines, identical text)
  into at least 10 repos' CLAUDE.md** (PyAutoFit, PyAutoGalaxy, PyAutoLens, PyAutoArray,
  PyAutoConf, PyAutoBuild, PyAutoPrompt, autolens_workspace, autolens_workspace_test,
  HowToLens). That's deliberate redundancy for a safety-critical rule, not an accident —
  but it's also the largest chunk of every file, and any future wording fix means editing
  10+ files by hand unless something de-duplicates it.
- Beyond duplication, some files (`@PyAutoGalaxy/CLAUDE.md` at 287 lines, `@PyAutoBuild/CLAUDE.md`
  at 184 lines) read like runbooks/reference manuals rather than agent-orientation docs —
  worth checking whether some of that belongs in a Skill instead (loaded on demand) rather
  than in context on every single session regardless of task.

## What to ship

1. **Inventory.** For every repo in scope (see Files touched), record current
   `CLAUDE.md` / `AGENTS.md` line counts, whether both exist, and whether they're
   single-sourced (one imports the other) or independently maintained.
2. **Diff CLAUDE.md vs AGENTS.md** wherever both exist and report how far they've
   diverged — same content reworded, or genuinely different sections.
3. **Inspect other AI agent infra already in use**, repo by repo: `skills/` directories,
   `.claude/skills/`, slash commands, hooks (`session-start`, pre-commit, etc.), and
   any `bin/`-style dispatcher referenced from the instructions file (e.g.
   `@PyAutoBuild/bin/autobuild`, `@autolens_assistant/skills/`, `@PyAutoPrompt/skills/`).
   Note which repos already push detail out of CLAUDE.md into a skill/wiki layer (the
   `autolens_assistant` three-layer instructions/skills/wiki model is the most developed
   example in the ecosystem) versus which ones keep everything as static prose in one file.
4. **Read up on the `llms.txt` convention** (a proposed standard for a root-level
   `llms.txt`/`llms-full.txt` curated link index aimed at LLMs consuming a *public*
   project — distinct in purpose from `AGENTS.md`/`CLAUDE.md`, which are operating
   instructions for a coding agent working *inside* the repo, not a docs index for an
   external reader). Summarize what it actually specifies, and give an explicit
   recommendation on whether any PyAutoLabs repo would benefit from one — the library
   repos with Sphinx docs (PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoConf)
   are the plausible candidates; internal-only test/admin/prompt repos almost certainly
   aren't.
5. **Write up findings as a single report** (a new file in this repo is fine — e.g.
   `autoprompt/agent_instructions_findings.md` — or the eventual issue/PR body) covering,
   per repo: current state, the duplication/drift found, and a concrete recommendation
   (consolidate via import / leave as-is / push section X into a skill / adopt llms.txt /
   not applicable).
6. **Ship the mechanical, low-risk fixes directly** if the audit confirms they're safe:
   collapsing a divergent CLAUDE.md+AGENTS.md pair into the `autolens_assistant` import
   pattern is reversible and low-blast-radius. Don't silently drop content when collapsing
   — reconcile differences first and flag anything that looks like it was intentionally
   different between the two files.
7. **File everything else as separate, scoped follow-up prompts** rather than doing it
   here — e.g. "move PyAutoGalaxy's JAX xp-threading runbook section into a skill" is its
   own task with its own review, not a rider on an audit.

## Acceptance

- A findings report exists, covering every repo listed below, with line counts and a
  recommendation per repo.
- `llms.txt` is explained in the report in plain terms, with a clear yes/no/which-repos
  recommendation — not just a link.
- Any CLAUDE.md/AGENTS.md consolidation actually shipped preserves all safety-critical
  content (the "Never rewrite history" block is NOT shortened or removed anywhere) and
  reduces total line count without losing information that was only in one of the two files.
- Any non-mechanical opportunity identified (content → skill, adopt llms.txt, etc.) exists
  afterward as its own prompt file in the relevant category, not as code already written.

## Out of scope

- Don't touch the wording or placement of the "Never rewrite history" section — it's
  deliberately repeated per-repo for redundancy and discoverability, confirmed working as
  intended.
- Don't design a brand-new cross-repo CLAUDE.md template wholesale in this task. If the
  audit makes a strong case for one, raise it with the user as a decision before drafting it.
- Don't restructure any repo's `skills/` directory here — note opportunities only.
- Don't touch `admin_jammy` (out of this session's repo scope) beyond noting in the report
  that it also has CLAUDE.md/AGENTS.md worth auditing later.

## Files touched

Audit (read-only) across every repo with a `CLAUDE.md` and/or `AGENTS.md`:

- `@PyAutoConf`, `@PyAutoFit`, `@PyAutoArray`, `@PyAutoGalaxy`, `@PyAutoLens`
- `@autolens_workspace`, `@autolens_workspace_test`
- `@HowToLens`
- `@autolens_assistant` (CLAUDE.md + AGENTS.md)
- `@PyAutoBuild`, `@PyAutoPrompt`

Likely edits (only the mechanical, low-risk consolidations from step 6), one PR per repo:
CLAUDE.md and/or AGENTS.md in whichever repos the audit confirms are safe to collapse.
