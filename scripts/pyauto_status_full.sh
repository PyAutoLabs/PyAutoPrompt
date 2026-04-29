#!/usr/bin/env bash
# pyauto_status_full.sh — release-prep run dashboard.
#
# Defines a shell function `pyauto-status-full` that reads the latest
# PyAutoBuild full release-prep run (the one symlinked from
# PyAutoBuild/test_results/latest/) and prints a dashboard:
#
#   - Run timestamp + path + ready/not-ready verdict + total duration
#   - Per-workspace pass / fail / skipped / timeout / duration table
#   - Failure counts grouped by classification
#   - Top-25 slowest scripts (any status) — surfaces timing regressions
#     before they cross the timeout threshold
#   - Slow-skip / needs-fix banner counts
#   - Pointer to triage.md if present (free-form analytical clustering)
#
# Usage:
#   source ~/Code/PyAutoLabs/PyAutoPrompt/scripts/pyauto_status_full.sh
#   pyauto-status-full
#
# Override the run path (e.g. to inspect a specific historical run)
# by passing it as the first argument:
#   pyauto-status-full ~/Code/PyAutoLabs/PyAutoBuild/test_results/runs/2026-04-29T14-48-47Z
#
# Note: this shell function shares its name with the /pyauto-status-full
# slash command (PyAutoPrompt/skills/pyauto-status-full/) but lives in a
# different namespace. The slash command is the conversational layer; this
# function is the same data, printed straight to stdout, no Claude needed.

PYAUTO_STATUS_FULL_DEFAULT="${PYAUTO_STATUS_FULL_DEFAULT:-$HOME/Code/PyAutoLabs/PyAutoBuild/test_results/latest}"

pyauto-status-full() {
  local run_dir="${1:-$PYAUTO_STATUS_FULL_DEFAULT}"

  if [[ ! -e "$run_dir" ]]; then
    cat >&2 <<EOF
pyauto-status-full: no run found at $run_dir

To produce one, from PyAutoBuild root:
  source ../activate.sh
  python autobuild/run_all.py
EOF
    return 1
  fi

  # Resolve symlink so the printed path is the actual run dir.
  run_dir="$(readlink -f "$run_dir")"

  local report_json="$run_dir/report.json"
  if [[ ! -f "$report_json" ]]; then
    echo "pyauto-status-full: $report_json missing — run incomplete?" >&2
    return 1
  fi

  python3 - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
with open(run_dir / "report.json") as f:
    r = json.load(f)

ready    = r.get("ready")
total    = float(r.get("total_duration_seconds", 0.0) or 0.0)
summary  = r.get("summary", {}) or {}
n_pass   = summary.get("passed", 0)
n_fail   = summary.get("failed", 0)
n_skip   = summary.get("skipped", 0)
n_to     = summary.get("timeout", 0)

GREEN = "\033[32m"
RED   = "\033[31m"
YEL   = "\033[33m"
DIM   = "\033[2m"
RST   = "\033[0m"

verdict = f"{GREEN}READY{RST}" if ready else f"{RED}NOT READY{RST}"
print(f"{'=' * 76}")
print(f" PyAuto Status Full")
print(f"{'=' * 76}")
print(f"Run:    {r.get('run_label','')}")
print(f"Path:   {run_dir}")
print(f"Status: {verdict}  (passed: {n_pass}, failed: {n_fail}, skipped: {n_skip}, timeout: {n_to})")
print(f"Total:  {total:.1f}s ({total/60:.1f} min)")
print()

# Per-workspace
print("Per-workspace")
print("-" * 76)
print(f"{'Workspace':<22} {'Passed':>6} {'Failed':>6} {'Skipped':>7} {'Timeout':>7} {'Duration':>10}")
pp  = r.get("per_project", {}) or {}
ppd = r.get("per_project_duration_seconds", {}) or {}
for proj in sorted(pp.keys()):
    c = pp[proj]
    f = c.get("failed", 0)
    t = c.get("timeout", 0)
    color = GREEN if (f == 0 and t == 0) else RED
    print(
        f"{color}{proj:<22}{RST} "
        f"{c.get('passed',0):>6} {f:>6} "
        f"{c.get('skipped',0):>7} {t:>7} "
        f"{ppd.get(proj,0):>9.1f}s"
    )
print()

# Failures by classification
failures = r.get("failures", []) or []
if failures:
    by_class = {}
    for fr in failures:
        cls = fr.get("classification", "unknown")
        by_class.setdefault(cls, []).append(fr)
    labels = {
        "source_code_bug":  "Source code bugs",
        "workspace_issue":  "Workspace issues",
        "workspace_data":   "Missing data files",
        "environment":      "Environment issues",
        "timeout":          "Timeouts",
        "known_numerical":  "Known numerical",
        "unknown":          "Unclassified",
    }
    print(f"Failures by classification ({len(failures)} total)")
    print("-" * 76)
    for cls in sorted(by_class.keys(), key=lambda c: -len(by_class[c])):
        items = by_class[cls]
        print(f"  {labels.get(cls, cls):<22} {len(items)}")
    print()

# Slowest 25
slowest = r.get("slowest", []) or []
if slowest:
    print(f"Slowest {len(slowest)} scripts")
    print("-" * 76)
    print(f"{'Duration':>9} {'Status':<8} {'Project':<16} Script")
    for s in slowest:
        proj  = s.get("project", "")
        stat  = s.get("status", "")
        fil   = s.get("file", "")
        # Trim absolute paths to last 3 segments for readability.
        short = "/".join(fil.split("/")[-3:])
        dur   = float(s.get("duration_seconds", 0.0) or 0.0)
        color = RED if stat in ("failed", "timeout") else (YEL if dur > 180 else "")
        print(f"{color}{dur:>8.1f}s {stat:<8} {proj:<16} {short}{RST}")
    print()

# Parked scripts banners
slow_skips = r.get("slow_skips") or []
nf_skips   = r.get("needs_fix_skips") or []
if slow_skips or nf_skips:
    print("Parked scripts (workspace no_run.yaml banners)")
    print("-" * 76)
    if slow_skips:
        print(f"  SLOW skips:      {len(slow_skips)}  (need performance fix)")
    if nf_skips:
        print(f"  NEEDS_FIX skips: {len(nf_skips)}  (parked broken)")
    print()

# Pointers
print("Pointers")
print("-" * 76)
print(f"  Markdown report:  {run_dir}/report.md       {DIM}(pyauto-report){RST}")
print(f"  Run JSON:         {run_dir}/report.json     {DIM}(pyauto-json){RST}")
triage = run_dir / "triage.md"
if triage.exists():
    print(f"  {GREEN}Triage notes:     {triage}{RST}  {DIM}(pyauto-triage){RST}")
PY
}

# _pyauto_run_file <subpath> [run-dir-arg] — resolve a file inside the latest
# (or supplied) run directory. Used by the pyauto-{report,json,triage} viewers.
_pyauto_run_file() {
  local subpath="$1"
  local run_dir="${2:-$PYAUTO_STATUS_FULL_DEFAULT}"

  if [[ ! -e "$run_dir" ]]; then
    echo "pyauto: no run found at $run_dir" >&2
    return 1
  fi
  run_dir="$(readlink -f "$run_dir")"

  local target="$run_dir/$subpath"
  if [[ ! -f "$target" ]]; then
    echo "pyauto: $target missing" >&2
    return 1
  fi
  printf '%s' "$target"
}

# pyauto-report [run-dir] — view report.md in the pager.
pyauto-report() {
  local f
  f="$(_pyauto_run_file report.md "$1")" || return 1
  "${PAGER:-less}" "$f"
}

# pyauto-json [run-dir] — view report.json. Uses jq for color + paging when
# available, falls back to plain cat otherwise.
pyauto-json() {
  local f
  f="$(_pyauto_run_file report.json "$1")" || return 1
  if command -v jq >/dev/null 2>&1; then
    jq -C . "$f" | "${PAGER:-less}" -R
  else
    "${PAGER:-less}" "$f"
  fi
}

# pyauto-triage [run-dir] — view triage.md in the pager.
pyauto-triage() {
  local f
  f="$(_pyauto_run_file triage.md "$1")" || return 1
  "${PAGER:-less}" "$f"
}
