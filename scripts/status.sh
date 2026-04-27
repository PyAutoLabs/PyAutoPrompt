#!/usr/bin/env bash
# PyAutoPrompt registry status.
#
# Prints prompt inventory grouped by category, plus the current state of
# active.md / planned.md / complete.md.
#
# Usage:
#   bash PyAutoPrompt/scripts/status.sh [--full | --repos]
#
# Without args: counts + active task list + last 5 completed.
# With --full:  also lists every prompt under every category.
# With --repos: delegate to pyauto-status (cross-repo git sync dashboard).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = "--repos" ]; then
  # shellcheck source=./pyauto_status.sh
  source "$ROOT/scripts/pyauto_status.sh"
  pyauto-status
  exit 0
fi

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
dim()  { printf "\033[2m%s\033[0m\n" "$1"; }

# ---------- Counts per category ----------

bold "== Prompt inventory =="
printf "%-35s %s\n" "category" "count"
printf "%-35s %s\n" "----------------------------------" "-----"

for dir in autoarray autofit autogalaxy autolens autolens_workspace_developer \
           autobuild cluster weak workspaces autoprompt z_vault issued; do
  if [ -d "$ROOT/$dir" ]; then
    count=$(find "$ROOT/$dir" -type f -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    printf "%-35s %s\n" "$dir/" "$count"
  fi
done

echo ""

# ---------- Registry state files ----------

count_h2() {
  if [ -f "$1" ]; then
    awk '/^## /{n++} END{print n+0}' "$1"
  else
    echo 0
  fi
}
active_count=$(count_h2 "$ROOT/active.md")
planned_count=$(count_h2 "$ROOT/planned.md")
complete_count=$(count_h2 "$ROOT/complete.md")

bold "== Registry =="
printf "active.md     %s task(s) in flight\n"  "$active_count"
printf "planned.md    %s task(s) queued\n"     "$planned_count"
printf "complete.md   %s task(s) recorded\n"   "$complete_count"

echo ""

# ---------- Active task list ----------

if [ "$active_count" -gt 0 ]; then
  bold "== Active tasks =="
  awk '
    /^## / { name=$0; sub(/^## /, "", name); print " - " name; in_task=1; next }
    in_task && /^- (issue|status|location|worktree):/ {
      sub(/^- /, "    ")
      print
    }
    /^### / || /^---/ { in_task=0 }
  ' "$ROOT/active.md"
  echo ""
fi

# ---------- Recently completed (last 5) ----------

if [ "$complete_count" -gt 0 ]; then
  bold "== Recently completed (last 5) =="
  awk '
    /^## / {
      if (count >= 5) exit
      name=$0; sub(/^## /, "", name)
      printf " - %s\n", name
      count++
    }
  ' "$ROOT/complete.md"
  echo ""
fi

# ---------- Full mode ----------

if [ "${1:-}" = "--full" ]; then
  bold "== Full prompt list =="
  for dir in autoarray autofit autogalaxy autolens autolens_workspace_developer \
             autobuild cluster weak workspaces autoprompt z_vault issued; do
    [ -d "$ROOT/$dir" ] || continue
    files=$(find "$ROOT/$dir" -type f -name "*.md" 2>/dev/null | sort)
    [ -z "$files" ] && continue
    bold "$dir/"
    while IFS= read -r f; do
      rel="${f#$ROOT/}"
      printf "  %s\n" "$rel"
    done <<< "$files"
    echo ""
  done
fi

dim "Run 'bash $ROOT/scripts/status.sh --full' for the full prompt list."
