# PyAutoMind

**The Mind of the PyAuto organism — its ideas, intent, goals and priorities.**

The PyAuto ecosystem is evolving into a software organism, and this repository
is its **Mind**: the place where the organism's *ideas, intentions, goals,
priorities and future direction* are captured. Although it began life as a
prompt repository, it has grown into the home of everything the organism *wants
to become* — ideas, prompts, active and completed work, planning, priorities,
workflow state and project direction.

It is still the **starting point of the PyAuto workflow**. Every piece of work
in the ecosystem starts here as a markdown file describing an intent in **plain
English**. You write what you want — that's it. An AI agent (or a human) picks
it up and turns it into a tracked GitHub issue, a feature branch, and a merged
pull request.

No template to fill in, no special syntax. **If you can describe the change
in a GitHub issue, you can drive the workflow.**

The Mind decides *what* the organism wants to become; the Brain
([PyAutoBrain](https://github.com/PyAutoLabs/PyAutoBrain))
decides *how* to achieve it. See [The PyAuto organism](#the-pyauto-organism) below.

## The PyAuto organism

The PyAuto ecosystem is organised as a software **organism**, with each
repository playing the role of an organ:

```
  Mind   →   Brain   →   Hands   →   Heart
  ideas      reasoning   execution    health
  intent     & planning  & delivery   & readiness
  goals
  priorities                ↘     ↙
                            Memory
                       accumulated knowledge
```

| Organ | Repository | Role |
|-------|------------|------|
| **Mind** | **PyAutoMind** (this repo) | What the organism *wants to become*: ideas, intent, goals, priorities, future work. |
| **Brain** | [PyAutoBrain](https://github.com/PyAutoLabs/PyAutoBrain) | *How* to achieve those goals: reasoning, planning, routing work. |
| **Hands** | [PyAutoBuild](https://github.com/PyAutoLabs/PyAutoBuild) *(the "Hands")* | Execution and delivery: building, testing, releasing. |
| **Heart** | [PyAutoHeart](https://github.com/PyAutoLabs/PyAutoHeart) | Health and release-readiness: monitoring, checks, the "is it safe to ship?" gate. |
| **Memory** | [PyAutoMemory](https://github.com/PyAutoLabs/PyAutoMemory) | Accumulated knowledge: literature summaries, wikis, scientific and project knowledge. |

> All organ repositories now carry their organism names. PyAutoAgent was renamed
> to PyAutoBrain and PyAutoPaper to PyAutoMemory; GitHub redirects the old URLs.
> PyAutoBuild ("Hands") and PyAutoHeart keep their repository names.

**Why this repository is the Mind.** Work in PyAuto begins as *intent* — an
idea, a goal, a priority — long before it becomes code. This repository is where
that intent lives and is shaped: raw ideas in `ideas.md`, scoped intentions as
prompt files, the priorities that order them, and the workflow state that tracks
what the organism is currently pursuing. It does not reason about *how* to build
something (that is the Brain) or carry the work out (that is the Hands); it holds
the organism's **wants and direction**. Everything downstream — planning,
execution, health checks — flows from the intent captured here.

---

## What a prompt looks like

Here's a real prompt — the contents of `autoarray/psf_oversampling.md` — that
became a tracked task. This is the level and style of detail to aim for in
your own GitHub issue: free-form prose, with `@RepoName/path/to/file.py`
references so the tooling knows which repo and files to target. No boilerplate.

````markdown
A point spread function is used to blur images via 2d convolution.

This blurring occurs predominantly in the package @PyAutoArray/autoarray/operators/convolver.py.

The source code currently requires PSF blurring to occur at the same resolution (pixel scale) as the
image, meaning the PSF is always the same resolution as the image.

However, for modeling, convolution can be performed at a higher resolution than the image, which allows for more accurate
blurring and modeling of the image. This requires us to have an oversampled PSF, which is a PSF that has a higher
resolution than the image.

For modeling, where images are generated PSF blurring happens in @PyAutoGalaxy/autogalaxy/operate/image.py.

Modeling can always evaluate images using a hgiher resolition grid, blurring them with the PSF at high
resolution and then downsample to the observed image resolution. Oversampling is implemented in
@PyAutoArray/autoarray/operators/over_sampling.

Note that over sampling often uses an adaptive sub-szie, which means that 2D covnolution with a PSF is not
well defined. for now, we will assume adaptive over sampling is not used.

I want us to be able to append the Convolver class with a convolve_over_sample_size integer, which specifies the over sample size of the PSF.
This will allow us to perform convolution at a higher resolution than the image, which will improve the accuracy of the blurring and modeling of the image.
For example, if convolve_over_sample_size is 2, then the PSF will be oversampled by a factor of 2, meaning it will have a resolution that is 2 times higher than the image.

This, in turn, means out imaging object @PyAutoArray/autoarray/dataset/imaging/dataset.py will need to be
extended to include the convolve_over_sample_size_lp and convolve_over_sample_size_pixelization attributes, which will
specify the over sample size of the PSF for the lensing and pixelization operations, respectively.

class Imaging(AbstractDataset):
    def __init__(
        self,
        data: Array2D,
        noise_map: Optional[Array2D] = None,
        psf: Optional[Convolver] = None,
        psf_setup_state: bool = False,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
        use_normalized_psf: Optional[bool] = True,
        check_noise_map: bool = True,
        sparse_operator: Optional[ImagingSparseOperator] = None,
    ):


Also,read through the @PyAutoArray/autoarray/inversion/inversion/imaging package, and parents, to see
how PSF convolution enters this. I think we can get it to work in PyAutoArray/autoarray/inversion/inversion/imaging/mapping.py,
and will leave work in PyAutoArray/autoarray/inversion/inversion/imaging/sparse.py to future work.

This is a complex task, therefore I think we should extend @autolens_workspace_test/scripts/imaging/convolution.py
with a numerical test.

We should then build on this test in a separate test file using a simple over sampled PSF, to get a numerical
result we can test the source code against.

@autolens_workspace/scripts/imaging/simulator.py is a good example we can build on to show how to use
over sampled PSFs in a real simulation. We can extend this script to show how to use over sampled PSFs in a real simulation.

Come up with a plan to implement over sampled PSFs.
````

That prompt becomes a GitHub issue, gets routed to the affected repos
(`PyAutoArray`, `PyAutoGalaxy`, the autolens workspaces), and lands as PRs
against each. Typos, half-finished thoughts, and "I think we should…" are
fine — write naturally, the AI fills in the rest.

---

## How a prompt flows through the workflow

```
  idea               ── you write it in ideas.md
    │
    ▼
  draft prompt       ── you write a markdown file under <work-type>/<target>/<name>.md
    │
    ▼
  /start_dev         ── reads the prompt, audits the code, drafts an issue,
    │                   creates the GitHub issue, registers the task in
    │                   active.md, moves the prompt to issued/
    ▼
  active.md entry    ── the task is now tracked across machines and sessions
    │
    ▼
  /start_library     ── creates a worktree, branch, opens dev environment
    or                  (or workspace variant — chosen automatically)
  /start_workspace
    │
    ▼
  development        ── code, tests, run smoke tests, commit
    │
    ▼
  /ship_library      ── runs tests, opens PR, waits for merge
    or
  /ship_workspace
    │
    ▼
  PR merged          ── post-merge cleanup deletes the worktree, removes the
    │                   active.md entry, appends a summary to complete.md
    ▼
  done
```

The slash commands above are skills hosted across the organism (Brain, Heart) but
all read/write Mind's registry via workspace-root-anchored paths. One operates
over the registry without starting work:

- `/pyauto-status` — dashboard of `active.md`, `planned.md`, `complete.md`
  (a PyAutoHeart status view). Continuity across execution environments needs no
  special step — any environment reads `active.md` and resumes an in-flight task.

---

## Repository layout

```
PyAutoMind/
├── README.md                ← this file
├── .gitignore
│
├── active.md                ← tasks currently in progress (one ## section per task)
├── complete.md              ← finished tasks (most recent first)
├── ideas.md                 ← raw incubating ideas, no structure required
├── parked.md                ← started/scoped but not in flight (stashes, orphan worktrees, deferred)
├── planned.md               ← issued tasks blocked from starting (created on demand)
├── priority.md              ← hand-curated priority hints
├── queue.md                 ← processing queue for /register_and_iterate
│
│   PROMPTS — organised by WORK TYPE (first folder), then TARGET (second folder).
│   See "Prompt taxonomy" below and ROUTING.md.
├── feature/                 ← new user-facing or scientific capabilities
│   ├── autoarray/  autofit/  autogalaxy/  autolens/  autolens_assistant/  …
│   ├── workspaces/          ← any *_workspace repo
│   ├── pyautobrain/         ← prompts that implement PyAutoBrain agents
│   ├── jax_substructure/  weak/  cluster/   ← numbered topic series (kept together)
├── bug/                     ← incorrect behaviour, crashes, regressions
│   ├── autofit/  autogalaxy/  autolens/  autoarray/  priors/  …
├── refactor/                ← internal restructuring, no intended behaviour change
├── docs/                    ← documentation, tutorials, notebooks, examples
├── test/                    ← test coverage, smoke tests, validation scripts
├── release/                 ← packaging, versions, deployment, release readiness
├── maintenance/             ← dependency updates, hygiene, cleanup, small tech debt
├── research/                ← exploratory scientific / algorithmic investigation
├── experiment/              ← prototypes, spikes, proof-of-concept work
├── triage/                  ← classification still unclear; needs manual review
│
│   LIFECYCLE / META — not work-types; keep their own names.
├── issued/                  ← prompts that have been routed via /start_dev
│   └── autolens_workspace_developer/   ← per-target subdirs preserved
│
├── z_features/              ← multi-task epic trackers (one tracker → many sub-prompts)
│   └── complete/            ← archived trackers (all sub-prompts shipped)
│
├── z_vault/                 ← deferred prompts (z_ prefix sorts last in listings)
├── shelved/                 ← shelved prompts
│
├── autoprompt/              ← prompts about THIS repo's own infrastructure (meta)
│
├── scripts/
│   ├── status.sh            ← prompt inventory helper
│   └── prompt_sync.sh       ← commit/push helpers sourced by skills
│
└── skills/                  ← Mind-owned skills + the ownership audit
    ├── OWNERSHIP.md          ← where every workflow skill lives, and why
    └── create_issue/         ← convert a prompt into a tracked GitHub issue
```

`PyAutoMind/skills/` now holds **only** the Mind-owned `create_issue` skill (plus
`OWNERSHIP.md`). The development-workflow skills were re-homed to the organs that
own them — **PyAutoBrain** (`start_dev`, `start_dev_for_user`, `plan_branches`,
`start_library`, `start_workspace`, `ship_library`, `ship_workspace`,
`register_and_iterate`), **PyAutoHeart** (`pyauto-status`, `pyauto-status-full`,
`worktree_status`), and **autolens_profiling** (`profile_likelihood`). The
`handoff` skill was retired (PyAutoBrain runs uniformly across execution
environments — see `OWNERSHIP.md`). General PyAuto tooling (release prep,
dependency audits, smoke tests, lint sweeps) lives in `admin_jammy/skills/`.

`scripts/prompt_sync.sh` is sourced by skills that mutate registry files
(`active.md`, `complete.md`, etc.) to commit and push back to origin. It
replaces the previous `admin_jammy/software/admin_sync.sh` which operated on
`admin_jammy/prompt/`.

---

## Prompt taxonomy

PyAutoMind organises **intent by the kind of thinking required; PyAutoBrain uses
that structure to choose the right reasoning agent.**

Prompts live at `<work-type>/<target>/<name>.md`:

- The **first folder** answers *what kind of thinking or agent is needed?* — the
  work type.
- The **second folder** answers *what domain or repository is affected?* — the
  target repo (`autoarray`, `autofit`, `autogalaxy`, `autolens`,
  `autolens_assistant`, `pyautobrain`, …), a workspace bucket (`workspaces`), or
  a topic series (`jax_substructure`, `weak`, `cluster`, `priors`).

### Work types → PyAutoBrain agents

| Folder | Holds | Future PyAutoBrain agent |
|--------|-------|--------------------------|
| `feature/` | new user-facing or scientific capabilities | feature planner |
| `bug/` | incorrect behaviour, crashes, regressions | debugger |
| `refactor/` | internal restructuring, no intended behaviour change | refactor architect |
| `docs/` | documentation, tutorials, notebooks, examples | documentation agent |
| `test/` | test coverage, smoke tests, validation scripts | test engineer |
| `release/` | packaging, versions, deployment, release readiness | release engineer |
| `maintenance/` | dependency updates, hygiene, cleanup, small tech debt | hygiene agent |
| `research/` | exploratory scientific / algorithmic investigation | research analyst |
| `experiment/` | prototypes, spikes, proof-of-concept work | prototype agent |

`triage/` holds prompts whose classification is still unclear — file there with a
short note and re-home once the work type is obvious. The full mapping (and the
note that the agents themselves live in PyAutoBrain, not here) is in
[`ROUTING.md`](ROUTING.md).

### Good prompt paths

```
feature/autolens/potential_corrections.md
bug/autoarray/mask_edge_case.md
refactor/autofit/result_object_cleanup.md
docs/workspaces/pixelization_tutorial.md
research/autofit/sbi_design.md
experiment/autoarray/jax_sparse_mapping.md
```

### Not work-types

`issued/`, `z_features/`, `z_vault/`, `shelved/` are **workflow lifecycle**
folders, and `autoprompt/` holds **meta** prompts about this repo's own
infrastructure. They keep their own names and are not routed by work type.

### Migration note

The repository previously used the target repo as the first folder
(`autoarray/foo.md`). Those prompts have moved to `<work-type>/autoarray/foo.md`.
Routing always keyed off the `@RepoName` references in a prompt's body, not its
folder, so the skills accept both old and new paths during the transition — but
new prompts should use the work-type layout.

---

## Conventions

### Naming

- Prompt filenames are lowercase `kebab_or_snake_case.md`.
- Numbered series use a leading number: `0_docs.md`, `1_simulator.md`. Skipping a
  number (e.g. `feature/weak/2_*.md` not present) is fine — it usually means a
  step was consolidated or deferred.
- **First folder = work type** (`feature/`, `bug/`, …); **second folder = target**
  repo or domain (lowercased, no `Py` prefix): `feature/autoarray/`,
  `bug/autofit/`, `refactor/autogalaxy/`. Workspace prompts go under
  `<work-type>/workspaces/` regardless of which workspace. See "Prompt taxonomy".

### Prompt file format

Free-form markdown. Strong conventions:

- Reference repos and files with `@RepoName/path/to/file.py` (e.g.
  `@PyAutoFit/autofit/non_linear/search.py`). `/start_dev` parses these to
  identify the primary target repo.
- One prompt = one task = one PR (ideally). If a prompt outlines several
  loosely-related changes, split before issuing.
- No frontmatter required. Title in the first line is helpful but optional.
- **Optional metadata header.** A prompt may carry a light, human-writable header
  near the top so both people and PyAutoBrain can see its type/target at a glance.
  This is a convention, not a schema — never required, no YAML frontmatter:

  ```markdown
  # Short task title

  Type: feature
  Target: PyAutoLens
  Repos:
  - PyAutoLens
  - autolens_workspace

  Status: draft
  ```

  When present, `Type:` should match the work-type folder. The goal is light
  structure, not bureaucracy — prompts stay free-form prose.

### `active.md` schema

Each task is an H2 section:

```markdown
## <task-name-kebab-case>
- issue: https://github.com/<owner>/<repo>/issues/<n>
- session: claude --resume <session-id>           # optional
- status: <library-dev | workspace-dev | ready-to-ship | …>
- location: <cli-in-progress | ready-for-mobile | …>   # optional, used by /handoff
- worktree: ~/Code/PyAutoLabs-wt/<task-name>
- repos:
  - <RepoName>: feature/<branch-name>
- summary: |
    Free-form summary of progress and next steps.
```

### `complete.md` schema

```markdown
## <task-name>
- issue: https://github.com/<owner>/<repo>/issues/<n>
- completed: YYYY-MM-DD
- library-pr: <url> [, <url>]
- workspace-pr: <url> [, <url>]
- notes: |
    Long-form description of what landed, gotchas, follow-ups.
```

### `z_features/` (multi-task epics)

`z_features/` holds **umbrella trackers** for multi-task epics — single
markdown files listing a sequence of sub-prompts that ship as their own PRs
under `autofit/`, `autogalaxy/`, etc. The tracker itself never becomes an
issue; only its sub-prompts do.

```
z_features/
├── latent_refactor.md            ← tracker (lists sub-prompt links)
├── ellipse_fitting_jax.md
├── ...
└── complete/                     ← archived trackers (all sub-prompts shipped)
```

Use this pattern when a single ask decomposes into 5+ dependent sub-tasks.
`/start_dev z_features/<tracker>.md` runs in **audit-only mode** — it reports
which sub-prompts are not-yet-issued / in-flight / shipped, and offers to
move the tracker to `z_features/complete/` once everything has landed.

**Naming convention for clean audit:** the audit derives task-name
candidates from each sub-prompt's `issued/` filename stem with `_`→`-`. For
the audit to auto-match against `complete.md` headings, **the task slug in
`active.md` / `complete.md` must equal the issued filename's stem after
that transform**.

| Issued filename | Task slug that matches | Task slug that does NOT match |
|---|---|---|
| `issued/latent_module_autogalaxy.md` | `latent-module-autogalaxy` ✓ | `latent-autogalaxy-module` ✗ |
| `issued/latent_smoke_test.md` | `latent-smoke-test` ✓ | `smoke-test-latent` ✗ |
| `issued/latent_variables_tutorial_expand_autofit.md` | `latent-variables-tutorial-expand-autofit` ✓ | `latent-tutorial-autofit` ✗ |

The third row is the trap — if `/start_dev` renames the prompt on move
(e.g. appends a repo suffix for disambiguation) and `active.md`'s task slug
diverges from the issued stem, the audit will report the sub-prompt as
"in flight" forever and never archive the tracker. The cure is to either:

- Pick the task slug at `/start_dev` time to match the eventual issued
  filename stem, or
- Manually archive the tracker (`mv z_features/<name>.md z_features/complete/`
  + `prompt_sync_push`) when you know it's all shipped.

---

## Tracking and inspection

### Quick inventory

```bash
bash scripts/status.sh
```

Prints counts per category, lists the active and recently-completed tasks, and
flags anything in `z_vault/` that's been sitting for a while.

### From inside Claude Code

- `/pyauto-status` — dashboard of registry state (active, planned, recent complete; PyAutoHeart)
- `/start_dev <work-type>/<target>/<name>.md` — read a prompt and route it (PyAutoBrain)
- `/worktree_status` — cross-references registry with task worktrees (PyAutoHeart)

---

## How this repo integrates with the rest

The PyAuto workflow has three repos with distinct roles:

| Repo | Purpose |
|------|---------|
| **PyAutoMind** (this repo) | The Mind: ideas, intent, goals, priorities, the prompt registry and prompt-coupled skills. The starting point. |
| **admin_jammy** | Personal admin notes (`euclid.md`, `grants.md`, …) and general PyAuto tooling (`software/worktree.sh`, `software/admin_sync.sh`, generic skills like `audit_docs`, `dep_audit`, `repo_cleanup`). |
| **PyAutoPaper** | Personal paper-management repo: source PDFs (gitignored) plus topical LLM wikis (`lensing_wiki/`, `smbh_wiki/`, `cti_wiki/`, `methods_wiki/`, `galaxies_wiki/`) and a reading queue (`reading-queue.md`, moved from `admin_jammy/papers.md`). |
| **`PyAuto*` libraries and `*_workspace*` repos** | Where the actual code work happens. Each task gets a feature branch + worktree under `~/Code/PyAutoLabs-wt/<task-name>/`. |

Helper scripts that this repo's skills source:

- `admin_jammy/software/worktree.sh` — task worktree management (create, remove, conflict check).
- `admin_jammy/software/admin_sync.sh` — admin_jammy/PyAutoMind sync helpers.

These intentionally live in `admin_jammy/software/` because they're general
multi-repo tooling, not prompt-specific. The skills that need them source by
absolute path.

---

## Bootstrap on a new machine

```bash
cd ~/Code/PyAutoLabs
git clone git@github.com:PyAutoLabs/PyAutoMind.git    # the Mind (this repo)
git clone git@github.com:PyAutoLabs/PyAutoBrain.git   # dev-workflow skills
git clone git@github.com:PyAutoLabs/PyAutoHeart.git   # status / readiness skills
git clone git@github.com:Jammy2211/admin_jammy.git    # installer + general tooling
bash admin_jammy/skills/install.sh                     # symlinks skills + commands
```

> **The local checkout directory must be named `PyAutoMind`.** The skills and
> scripts reference `PyAutoMind/...` paths directly — e.g.
> `source PyAutoMind/scripts/prompt_sync.sh` and `git -C PyAutoMind …` — so a
> differently-named directory breaks those commands.

`install.sh` auto-discovers skills from every present discovery root
(`admin_jammy/skills/`, `PyAutoMind/skills/`, `PyAutoBrain/skills/`,
`PyAutoHeart/skills/`, `autolens_profiling/skills/`) and creates symlinks under
`~/.claude/skills/` and `~/.claude/commands/`. Roots that aren't checked out are
skipped. Re-run any time after pulling new skills from any of those repos.
