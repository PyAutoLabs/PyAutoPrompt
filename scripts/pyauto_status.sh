#!/usr/bin/env bash
# pyauto_status.sh — cross-repo sync status dashboard.
#
# Defines a shell function `pyauto-status` that prints, for every git repo
# under ~/Code/PyAutoLabs/, the branch, upstream tracking ref, behind/ahead
# counts vs @{u}, dirty file count, and a flag column. Designed to run in
# under 10 seconds — fetches are parallelised one background job per repo.
#
# Usage:
#   source ~/Code/PyAutoLabs/PyAutoPrompt/scripts/pyauto_status.sh
#   pyauto-status
#
# Override the repo root (e.g. for testing) via PYAUTO_STATUS_ROOT.
#
# Flag glyphs (FLAGS column):
#   ↓  behind upstream
#   ↑  ahead of upstream
#   *  dirty (modified or untracked files)
#   !  no upstream / fetch failed
#   b  current branch ≠ upstream branch (forgotten feature branch)
#
# After the main table, four optional sections may follow:
#   - "Dirty files:"        — per-repo `git status --porcelain` for any repo
#                             with mod or untr > 0.
#   - "Follow-up commands:" — copy-pasteable git invocations grouped by
#                             category (pull / set-upstream / investigate).
#                             Suppressed entirely when nothing is actionable.
#   - "Smoke tests:"        — per-workspace counts from
#                             ~/.cache/pyauto/smoke/*.json (written by the
#                             /smoke-test skill). Green when failed=0, red
#                             otherwise. Suppressed when no JSONs exist.
#   - "Last autobuild run:" — aggregate from
#                             ~/Code/PyAutoLabs/PyAutoBuild/test_results/*.json
#                             (committed by the autobuild release pipeline).
#                             Suppressed when no JSONs exist.
#
# Note: this shell function shares its name with the /pyauto-status slash
# command (PyAutoPrompt/skills/pyauto-status/) but lives in a different
# namespace. The slash command shows workflow registry status (planned /
# active / complete tasks); this function shows git sync state.

PYAUTO_STATUS_ROOT="${PYAUTO_STATUS_ROOT:-$HOME/Code/PyAutoLabs}"

pyauto-status() {
  local root="$PYAUTO_STATUS_ROOT"
  if [[ ! -d "$root" ]]; then
    echo "pyauto-status: $root does not exist" >&2
    return 1
  fi

  # Discover repos. `.git` is a directory in normal checkouts and a file in
  # linked worktrees, so accept both. mindepth/maxdepth 2 limits us to the
  # immediate children of $root.
  local repos=()
  while IFS= read -r dir; do
    repos+=("$dir")
  done < <(
    find "$root" -mindepth 2 -maxdepth 2 \
      \( -name .git -type d -o -name .git -type f \) \
      -printf '%h\n' 2>/dev/null | sort
  )

  if [[ ${#repos[@]} -eq 0 ]]; then
    echo "pyauto-status: no git repos found under $root"
    return 0
  fi

  # Parallel fetch. One background job per repo; sentinel files mark fetch
  # failures so the dashboard can flag stale rows with `!` instead of
  # silently returning misleading counts.
  local fetch_status_dir
  fetch_status_dir="$(mktemp -d)"
  trap 'rm -rf "$fetch_status_dir"' RETURN

  # Run inside a subshell with monitor mode disabled so the interactive
  # shell's job-control notifications (`[N] PID` / `[N] Done ...`) do not
  # leak into the dashboard output when this function is sourced.
  local repo
  (
    set +m
    for repo in "${repos[@]}"; do
      (
        if ! git -C "$repo" fetch --quiet origin 2>/dev/null; then
          touch "$fetch_status_dir/$(basename "$repo")"
        fi
      ) &
    done
    wait
  )

  # Header.
  local fmt='%-32s %-30s %-36s %6s %5s %4s %4s  %s\n'
  printf "$fmt" REPO BRANCH UPSTREAM BEHIND AHEAD MOD UNTR FLAGS
  printf "$fmt" "--------------------------------" \
    "------------------------------" \
    "------------------------------------" \
    "------" "-----" "----" "----" "-----"

  # Per-repo row. Porcelain is cached so the dirty-files listing below can
  # reuse it without a second `git status` per repo. Action arrays collect
  # actionable follow-ups for the "Follow-up commands:" section printed at
  # the end.
  declare -A repo_porcelain
  local actions_pull=() actions_set_upstream=() actions_manual=()
  local name branch upstream upstream_branch behind ahead mod untr flags counts porcelain branch_mismatch b_int a_int
  for repo in "${repos[@]}"; do
    name="$(basename "$repo")"

    branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    [[ "$branch" == "HEAD" ]] && branch="(detached)"
    [[ -z "$branch" ]] && branch="?"

    upstream="$(git -C "$repo" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true)"
    flags=""

    if [[ -z "$upstream" ]]; then
      upstream="NONE"
      upstream_branch=""
      behind="?"
      ahead="?"
      flags+="!"
    else
      # Strip the remote prefix (e.g. "origin/main" → "main"). Branch names
      # may contain slashes (e.g. "feature/foo"), so #*/ is the right
      # operator — it removes only up to the first slash.
      upstream_branch="${upstream#*/}"
      counts="$(git -C "$repo" rev-list --left-right --count "$upstream"...HEAD 2>/dev/null || true)"
      if [[ -n "$counts" ]]; then
        behind="${counts%%[[:space:]]*}"
        ahead="${counts##*[[:space:]]}"
      else
        behind="?"
        ahead="?"
      fi
      [[ -e "$fetch_status_dir/$name" ]] && flags+="!"
    fi

    porcelain="$(git -C "$repo" status --porcelain 2>/dev/null || true)"
    repo_porcelain["$name"]="$porcelain"

    if [[ -z "$porcelain" ]]; then
      mod=0
      untr=0
    else
      untr="$(printf '%s\n' "$porcelain" | grep -c '^??' || true)"
      mod="$(printf '%s\n' "$porcelain" | grep -cv '^??' || true)"
    fi

    # Branch-mismatch detection. With no upstream, the heuristic is
    # "expected to be on main"; with an upstream, compare to its branch
    # component. Detached HEAD never matches.
    branch_mismatch=false
    if [[ "$upstream" == "NONE" ]]; then
      [[ "$branch" != "main" ]] && branch_mismatch=true
    elif [[ "$branch" != "$upstream_branch" ]]; then
      branch_mismatch=true
    fi

    [[ "$behind" =~ ^[0-9]+$ ]] && (( behind > 0 )) && flags+="↓"
    [[ "$ahead"  =~ ^[0-9]+$ ]] && (( ahead  > 0 )) && flags+="↑"
    (( mod + untr > 0 )) && flags+="*"
    [[ "$branch_mismatch" == "true" ]] && flags+="b"

    printf "$fmt" "$name" "$branch" "$upstream" "$behind" "$ahead" "$mod" "$untr" "$flags"

    # Categorise actionable follow-ups. Only the boring case (clean, behind,
    # not ahead) becomes an auto-runnable command; everything else is
    # surfaced for manual handling.
    b_int=0; a_int=0
    [[ "$behind" =~ ^[0-9]+$ ]] && b_int="$behind"
    [[ "$ahead"  =~ ^[0-9]+$ ]] && a_int="$ahead"
    if [[ "$upstream" == "NONE" ]]; then
      if [[ "$branch" == "main" ]]; then
        actions_set_upstream+=("$repo")
      else
        actions_manual+=("$name — branch=$branch, upstream=NONE; switch to main or set upstream")
      fi
    elif (( b_int > 0 && a_int == 0 )); then
      if (( mod + untr == 0 )); then
        actions_pull+=("$repo")
      else
        actions_manual+=("$name — behind=$b_int, dirty (mod=$mod untr=$untr); stash + pull manually")
      fi
    elif (( b_int > 0 && a_int > 0 )); then
      actions_manual+=("$name — diverged: ahead=$a_int, behind=$b_int; investigate")
    elif [[ "$branch_mismatch" == "true" ]]; then
      actions_manual+=("$name — on branch $branch (upstream $upstream_branch); switch to $upstream_branch if not a worktree")
    fi
  done

  # Per-repo dirty-file listing. Only repos with non-empty porcelain are
  # shown — keeps the output empty when everything is clean. The `??` and
  # ` M` etc. prefixes from porcelain are preserved so users can tell
  # untracked from modified at a glance.
  local printed_header=false
  for repo in "${repos[@]}"; do
    name="$(basename "$repo")"
    porcelain="${repo_porcelain[$name]}"
    [[ -z "$porcelain" ]] && continue
    if [[ "$printed_header" == "false" ]]; then
      echo ""
      echo "Dirty files:"
      printed_header=true
    fi
    echo "  $name:"
    printf '%s\n' "$porcelain" | sed 's/^/    /'
  done

  # Follow-up commands. Suppressed entirely when nothing is actionable so
  # the clean case stays quiet. The `git -C <abs-path>` form means each
  # printed line is independently copy-pasteable.
  local total=$(( ${#actions_pull[@]} + ${#actions_set_upstream[@]} + ${#actions_manual[@]} ))
  if (( total > 0 )); then
    echo ""
    echo "Follow-up commands:"
    if (( ${#actions_pull[@]} > 0 )); then
      echo "  # Pull (clean, behind, not ahead):"
      local r
      for r in "${actions_pull[@]}"; do
        echo "    git -C $r pull --ff-only"
      done
    fi
    if (( ${#actions_set_upstream[@]} > 0 )); then
      echo "  # Set missing upstream (branch=main, upstream=NONE):"
      local r
      for r in "${actions_set_upstream[@]}"; do
        echo "    git -C $r branch --set-upstream-to=origin/main main"
      done
    fi
    if (( ${#actions_manual[@]} > 0 )); then
      echo "  # Investigate manually:"
      local line
      for line in "${actions_manual[@]}"; do
        echo "    $line"
      done
    fi
  fi

  # Smoke tests. Reads per-workspace JSON written by the /smoke-test skill
  # (admin_jammy/skills/smoke_test/SKILL.md step 7). One python invocation
  # parses all files; bash formats with ANSI color (green if failed=0).
  local smoke_dir="$HOME/.cache/pyauto/smoke"
  if [[ -d "$smoke_dir" ]] && compgen -G "$smoke_dir/*.json" > /dev/null; then
    echo ""
    echo "Smoke tests:"
    local ws ts passed failed skipped total dur color symbol
    while IFS='|' read -r ws ts passed failed skipped total dur; do
      [[ -z "$ws" ]] && continue
      if [[ "$failed" == "0" ]]; then
        color='\033[32m'; symbol='✓'
      else
        color='\033[31m'; symbol='✗'
      fi
      printf "  ${color}%-32s %3s passed   %3s failed   %3s skipped   (%s)  %s\033[0m\n" \
        "$ws" "$passed" "$failed" "$skipped" "${ts:0:10}" "$symbol"
    done < <(python3 -c '
import json, os, glob
for f in sorted(glob.glob(os.path.expanduser("~/.cache/pyauto/smoke/*.json"))):
    try:
        d = json.load(open(f))
        print("|".join(str(d.get(k, "")) for k in
            ["workspace", "completed_at", "passed", "failed", "skipped", "total", "duration_seconds"]))
    except Exception:
        pass
' 2>/dev/null)
  fi

  # Last autobuild run. Reads aggregate from PyAutoBuild/test_results/*.json
  # (committed by the autobuild release pipeline). Counts only — failure
  # detail lives in the per-job JSON / the GitHub Actions run.
  local pab_dir="$HOME/Code/PyAutoLabs/PyAutoBuild/test_results"
  if [[ -d "$pab_dir" ]] && compgen -G "$pab_dir/*.json" > /dev/null; then
    local pab_summary
    pab_summary=$(python3 -c '
import json, os, glob
total_p = total_f = total_s = num = 0
projects = set()
latest = ""
for f in sorted(glob.glob(os.path.expanduser("~/Code/PyAutoLabs/PyAutoBuild/test_results/*.json"))):
    try:
        d = json.load(open(f))
        s = d.get("summary", {})
        total_p += s.get("passed", 0)
        total_f += s.get("failed", 0)
        total_s += s.get("skipped", 0)
        num += 1
        projects.add(d.get("project", "?"))
        ct = d.get("completed_at", "")
        if ct > latest:
            latest = ct
    except Exception:
        pass
print(f"{latest[:10]}|{num}|{len(projects)}|{total_p}|{total_f}|{total_s}")
' 2>/dev/null)
    if [[ -n "$pab_summary" ]]; then
      local pab_date njobs nproj pab_p pab_f pab_s sha
      IFS='|' read -r pab_date njobs nproj pab_p pab_f pab_s <<< "$pab_summary"
      sha=$(git -C "$HOME/Code/PyAutoLabs/PyAutoBuild" rev-parse --short HEAD 2>/dev/null)
      echo ""
      printf "Last autobuild run: %s (PyAutoBuild commit %s)\n" "$pab_date" "${sha:-?}"
      local color='\033[32m'
      [[ "$pab_f" != "0" ]] && color='\033[31m'
      printf "  ${color}%s jobs across %s workspaces: %s passed, %s failed, %s skipped\033[0m\n" \
        "$njobs" "$nproj" "$pab_p" "$pab_f" "$pab_s"
    fi
  fi
}
