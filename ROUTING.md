# Routing — prompt taxonomy → PyAutoBrain agents

PyAutoMind stores **intent**. PyAutoBrain reasons over that intent and routes it
to the right specialist agent. This file defines the contract between the two:
the first folder of every prompt path declares the *kind of work*, and PyAutoBrain
maps that to a reasoning agent.

> PyAutoMind organises intent by the kind of thinking required; PyAutoBrain uses
> that structure to choose the right reasoning agent.

## The map

Prompts live at `<work-type>/<target>/<name>.md`. The **work-type** (first folder)
determines the agent; the **target** (second folder) tells the agent which repo or
domain is affected.

| Work-type folder | Intent | PyAutoBrain agent |
|------------------|--------|-------------------|
| `feature/`       | new user-facing or scientific capabilities | feature planner |
| `bug/`           | incorrect behaviour, crashes, regressions | debugger |
| `refactor/`      | internal restructuring, no intended behaviour change | refactor architect |
| `docs/`          | documentation, tutorials, notebooks, examples | documentation agent |
| `test/`          | test coverage, smoke tests, validation scripts | test engineer |
| `release/`       | packaging, versions, deployment, release readiness | release engineer |
| `maintenance/`   | dependency updates, hygiene, cleanup, small technical debt | hygiene agent |
| `research/`      | exploratory scientific / algorithmic investigation before implementation | research analyst |
| `experiment/`    | prototypes, spikes, proof-of-concept work | prototype agent |
| `triage/`        | classification still unclear | (human triages, then re-homes) |

## Targets (second folder)

The second folder names the affected repo or domain, e.g. `autoarray`, `autofit`,
`autogalaxy`, `autolens`, `autolens_assistant`, `autolens_profiling`,
`autolens_workspace_developer`, `autobuild`, `pyautobrain`; the workspace bucket
`workspaces`; or a topic series kept together as a unit (`jax_substructure`,
`weak`, `cluster`, `priors`).

Within the libraries, work classifies as **library** vs **workspace** for the
`/start_library` ↔ `/start_workspace` split — but that is decided from the
`@RepoName` references in the prompt body, *not* from the folder. The folder is
for human + agent legibility and PyAutoBrain routing.

## Scope of this file

This repository **only defines the taxonomy and the metadata/documentation** that
PyAutoBrain consumes. The agents themselves are **not** implemented here — they
live in PyAutoBrain. Prompts that *implement*
those agents are ordinary `feature/pyautobrain/*.md` prompts.

## Not routed by work type

`issued/`, `z_features/`, `z_vault/`, `shelved/` are workflow-lifecycle folders;
`autoprompt/` holds meta prompts about this repo's own infrastructure. None of
these are work-type folders and PyAutoBrain does not route them.
