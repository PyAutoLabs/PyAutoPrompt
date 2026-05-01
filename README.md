# PyAutoPrompt

**The starting point of the PyAuto workflow.**

Every piece of work in the PyAuto ecosystem starts as a markdown file
describing a task in **plain English**. You write what you want — that's it.
An AI agent (or a human) picks it up and turns it into a tracked GitHub issue,
a feature branch, and a merged pull request.

No template to fill in, no special syntax. **If you can describe the change
in a GitHub issue, you can drive the workflow.**

## What a prompt looks like

Here's a real prompt — the contents of `autoarray/psf_oversampling.md` — that
became a tracked task. This is the level and style of detail to aim for in
your own GitHub issue: free-form prose, with `@RepoName/path/to/file.py`
references so the tooling knows which repo and files to target. No boilerplate.

````markdown
A point spreadh function is used to blur images via 2d convolution.

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
  draft prompt       ── you write a markdown file under <category>/<name>.md
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

Two slash commands operate over the prompt registry without starting work:

- `/pyauto-status` — dashboard of `active.md`, `planned.md`, `complete.md`
- `/handoff` — park a task on this machine and resume on another (mobile, laptop, server)

---

## Repository layout

```
PyAutoPrompt/
├── README.md                ← this file
├── .gitignore
│
├── active.md                ← tasks currently in progress (one ## section per task)
├── complete.md              ← finished tasks (most recent first)
├── ideas.md                 ← raw incubating ideas, no structure required
├── planned.md               ← issued tasks blocked from starting (created on demand)
├── priority.md              ← hand-curated priority hints
├── queue.md                 ← processing queue for /register_and_iterate
│
├── autoarray/               ← prompts targeting PyAutoArray
├── autofit/                 ← prompts targeting PyAutoFit
├── autogalaxy/              ← prompts targeting PyAutoGalaxy
├── autolens/                ← prompts targeting PyAutoLens
├── autolens_workspace_developer/   ← prompts targeting the dev workspace
│
├── autobuild/                   ← prompts targeting build/release infrastructure (PyAutoBuild)
├── workspaces/              ← prompts targeting any *_workspace repo
│
├── cluster/                 ← cluster-lensing prompt series (numbered)
├── weak/                    ← weak-lensing prompt series (numbered)
│
├── issued/                  ← prompts that have been routed via /start_dev
│   └── autolens_workspace_developer/   ← per-target subdirs preserved
│
├── z_vault/                 ← deferred prompts (z_ prefix sorts last in listings)
│
├── autoprompt/              ← prompts about THIS repo's own infrastructure
│
├── scripts/
│   ├── status.sh            ← prompt inventory helper
│   └── prompt_sync.sh       ← commit/push helpers sourced by skills
│
└── skills/                  ← Claude Code skills tightly coupled to the prompt registry
    ├── start_dev/
    ├── start_library/
    ├── start_workspace/
    ├── ship_library/
    ├── ship_workspace/
    ├── pyauto-status/
    ├── register_and_iterate/
    ├── handoff/
    ├── worktree_status/
    ├── create_issue/
    └── plan_branches/
```

The `skills/` here hold **only the skills that read or write `active.md` / prompt
files**. General PyAuto tooling (release prep, dependency audits, smoke tests,
lint sweeps) lives in `admin_jammy/skills/`.

`scripts/prompt_sync.sh` is sourced by skills that mutate registry files
(`active.md`, `complete.md`, etc.) to commit and push back to origin. It
replaces the previous `admin_jammy/software/admin_sync.sh` which operated on
`admin_jammy/prompt/`.

---

## Conventions

### Naming

- Prompt filenames are lowercase `kebab_or_snake_case.md`.
- Numbered series use a leading number: `0_docs.md`, `1_simulator.md`. Skipping a
  number (e.g. `weak/2_*.md` not present) is fine — it usually means a step was
  consolidated or deferred.
- Category dirs match the target repo name (lowercased, no `Py` prefix):
  `autoarray/`, `autofit/`, `autogalaxy/`, `autolens/`. Workspace prompts go
  under `workspaces/` regardless of which workspace.

### Prompt file format

Free-form markdown. Strong conventions:

- Reference repos and files with `@RepoName/path/to/file.py` (e.g.
  `@PyAutoFit/autofit/non_linear/search.py`). `/start_dev` parses these to
  identify the primary target repo.
- One prompt = one task = one PR (ideally). If a prompt outlines several
  loosely-related changes, split before issuing.
- No frontmatter required. Title in the first line is helpful but optional.

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

---

## Tracking and inspection

### Quick inventory

```bash
bash scripts/status.sh
```

Prints counts per category, lists the active and recently-completed tasks, and
flags anything in `z_vault/` that's been sitting for a while.

### From inside Claude Code

- `/pyauto-status` — dashboard of registry state (active, planned, recent complete)
- `/start_dev <category>/<name>.md` — read a prompt and route it
- `/handoff park` / `/handoff resume` — cross-machine task transitions
- `/worktree_status` — cross-references registry with task worktrees

---

## How this repo integrates with the rest

The PyAuto workflow has three repos with distinct roles:

| Repo | Purpose |
|------|---------|
| **PyAutoPrompt** (this repo) | Prompts, registry, prompt-coupled skills. The starting point. |
| **admin_jammy** | Personal admin notes (`euclid.md`, `papers.md`, `grants.md`, …) and general PyAuto tooling (`software/worktree.sh`, `software/admin_sync.sh`, generic skills like `audit_docs`, `dep_audit`, `repo_cleanup`). |
| **`PyAuto*` libraries and `*_workspace*` repos** | Where the actual code work happens. Each task gets a feature branch + worktree under `~/Code/PyAutoLabs-wt/<task-name>/`. |

Helper scripts that this repo's skills source:

- `admin_jammy/software/worktree.sh` — task worktree management (create, remove, conflict check).
- `admin_jammy/software/admin_sync.sh` — admin_jammy/PyAutoPrompt sync helpers.

These intentionally live in `admin_jammy/software/` because they're general
multi-repo tooling, not prompt-specific. The skills that need them source by
absolute path.

---

## Bootstrap on a new machine

```bash
cd ~/Code/PyAutoLabs
git clone git@github.com:PyAutoLabs/PyAutoPrompt.git
git clone git@github.com:Jammy2211/admin_jammy.git    # if not already present
bash admin_jammy/skills/install.sh                     # symlinks skills + commands
```

`install.sh` auto-discovers skills from both `admin_jammy/skills/` and
`PyAutoPrompt/skills/` and creates symlinks under `~/.claude/skills/` and
`~/.claude/commands/`. Re-run any time after pulling new skills from either repo.
