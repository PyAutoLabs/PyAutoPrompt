# PyAuto Status Full: Latest Release-Prep Run Dashboard

Render the most recent PyAutoBuild full-run report as a release-readiness dashboard. Use this to inspect timing and failures from the last `python autobuild/run_all.py`. Read-only.

## Usage

```
/pyauto-status-full
```

This skill is the deeper sibling of `/pyauto-status`:

- `/pyauto-status` — what's in flight right now (planned/active/recently-completed tasks, branches, dirty repos).
- `/pyauto-status-full` — what the last full release-prep run produced (per-workspace pass/fail/timing, slowest scripts, failure tracebacks).

## Steps

### 1. Locate the latest run

PyAutoBuild stores every full run under `~/Code/PyAutoLabs/PyAutoBuild/test_results/runs/<UTC-timestamp>/` and updates a `latest` symlink on success.

```bash
LATEST=~/Code/PyAutoLabs/PyAutoBuild/test_results/latest
```

If the symlink does not exist, no full run has completed yet. Print:

```
No full release-prep run on disk.

To produce one, from PyAutoBuild root:
  source ../activate.sh
  python autobuild/run_all.py
```

…and exit. Do not invent data.

### 2. Read the aggregated report

```bash
cat "$LATEST/report.json"
```

The JSON is produced by `aggregate_results.aggregate()` and contains:

- `ready` — boolean release-readiness gate
- `run_label`, `run_path`, `total_duration_seconds`
- `summary` — totals by status (passed / failed / skipped / timeout)
- `per_project` + `per_project_duration_seconds` — per-workspace counts and total wall-clock
- `slowest` — top-25 slowest scripts (any status), with `duration_seconds`, `status`, `project`
- `failures` — every failed/timed-out script with `classification`, `error_message`, `traceback`
- `failure_pr_correlations` — which failures coincide with recently-merged PRs
- `slow_skips`, `needs_fix_skips` — surfaced from each workspace's `config/build/no_run.yaml`
- `runs` — per-job manifest (no per-result records, just metadata)

### 3. Render the dashboard

Print a concise Markdown dashboard that includes:

**Header**

```
Run:    <run_label>
Path:   <run_path>
Status: READY | NOT READY  (failures: N, timeouts: M)
Total:  <total_duration_seconds>s
```

**Per-workspace table** — columns: Workspace, Passed, Failed, Skipped, Timeout, Duration. Sort alphabetically.

**Failures (if any)** — group by `classification`. For each failure, show:

- Script path
- One-line error message
- PR correlations from `failure_pr_correlations[<file>]` if present
- Last 10 lines of `traceback` in a fenced block

**Slowest scripts (top 25)** — table of Script | Project | Status | Duration | Share. The `Share` column is `duration / total_duration_seconds * 100`.

**Slow-skip / needs-fix banners** — render `report["slow_skips"]` and `report["needs_fix_skips"]` verbatim using the same table format the markdown report uses (workspace, pattern, marked-date, age, reason).

**Footer**

```
Open full markdown report:  <run_path>/report.md
Open run JSON:              <run_path>/report.json
```

### 4. Drift since previous run (optional)

If the parent `runs/` directory contains at least two run subdirs, locate the second-most-recent one (sort by name descending, skip `latest`). Read its `report.json` and produce a "Drift since previous run" section showing:

- Per-workspace passed-count delta (`+N`, `-N`, `0`)
- Any test that was passed previously and is now failed/timeout (regression)
- Any test that was failed/timeout previously and is now passed (recovery)
- Top-5 timing regressions: scripts whose duration grew by ≥ 50 % between runs

If only one run exists, omit this section silently.

### 5. Notes

- Read-only — never modifies PyAutoBuild output, never deletes runs, never touches the `latest` symlink.
- No GitHub posting (matches `/pyauto-status` convention). The user can decide to share findings manually.
- For quick browsing, the markdown report at `<run_path>/report.md` already contains the same data — this skill is the conversational summary layer on top.

## Remote / mobile mode

This skill reads local artefacts produced by an autobuild run on the laptop. There is no mobile equivalent — full runs only happen on the dev machine.

If `~/Code/PyAutoLabs/PyAutoBuild/test_results/latest` does not exist (e.g. the user is on a fresh machine or has never run `run_all.py`), print the same "no run on disk" message from step 1 and exit.
