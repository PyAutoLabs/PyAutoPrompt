---
name: plan-branches
description: After a plan is approved, survey all affected repositories for current branch state and suggest a unified working branch name to avoid overlap with other work or agents.
---

After a plan has been created and approved, run this skill to survey all repositories that the plan will touch, report their current branch state, and propose a single working branch name for the entire task.

## Steps

1. **Identify affected repositories**

   Review the approved plan and list every repository that will be modified. Only include repos that the plan actually touches — do not list all repos from `settings.json` if they are not relevant.

2. **Report current branch state for each affected repo**

   For each affected repository, run:
   ```bash
   git -C <repo_path> branch --show-current
   git -C <repo_path> status --short
   ```

   Display a table like:

   ```
   Repository              | Current Branch | Dirty?
   ------------------------|----------------|-------
   ./PyAutoFit             | main           | clean
   ./PyAutoArray           | feature/xyz    | 2 modified
   ```

   If a repo is on a non-main/non-master branch or has uncommitted changes, flag it with a warning — this may indicate another task or agent is active there.

3. **Check for active Claude agents, recent branches, and worktree claims**

   Source the worktree helper and list anything already claimed by another task:

   ```bash
   source admin_jammy/software/worktree.sh
   worktree_list_claimed
   ```

   This prints one line per `(task, repo, branch, worktree_path)` quadruple currently registered in `active.md`. For each affected repo the new plan wants to touch, check whether a different task already claims it via a `worktree:` field. If so, flag it as a **hard conflict** — the new task cannot start until the other one ships.

   Then, for each affected repo, also run:
   ```bash
   git -C <repo_path> branch --sort=-committerdate | head -5
   ```

   Show the 5 most recent branches per repo so the user can spot ongoing work that pre-dates the worktree flow. A feature branch on the main checkout that is **not** referenced by any `worktree:` claim is unregistered work, not a conflict — surface it as a warning, not a block.

4. **Suggest a unified branch name**

   Propose a single branch name to be used across all affected repos. Use the format:
   ```
   feature/<short-description-of-plan>
   ```

   The name should be:
   - Descriptive of the task from the plan
   - Lowercase, kebab-case
   - Short (under 50 chars)

5. **Present summary for approval**

   Display the full summary:
   - List of affected repos
   - Current branch and dirty state for each
   - Recent branches for each
   - Any warnings about potential overlap
   - The suggested branch name

   Then ask the user:
   - "Does this branch name work, or would you like a different one?"
   - "Are any of the flagged repos a concern? Should we wait or coordinate?"

   **Do not proceed with any work until the user confirms the branch name and acknowledges any overlap warnings.**

6. **On resume: verify branches and worktree match the plan**

   When resuming work on an existing plan (e.g. a new conversation continuing previous work, or returning after a break), run this verification step **before making any edits**:

   a. Read the plan or `active.md` entry to find the agreed branch name, the worktree root path (`worktree:` field), and the list of affected repositories.

   b. **Verify the worktree root exists on disk:**
   ```bash
   test -d "$WT_ROOT" && echo "present" || echo "MISSING"
   test -f "$WT_ROOT/activate.sh" && echo "activate.sh ok" || echo "activate.sh MISSING"
   ```

   If the worktree root has been deleted but `active.md` still lists it, stop and ask the user whether to:
   - Re-create the worktree via `worktree_create <task-name> <repos...>` (resumes the task), or
   - Abandon the task and remove its entry from `active.md`.

   c. For each affected repo listed in the task, run `git -C "$WT_ROOT/<repo>" branch --show-current` and compare against the expected branch. Display the comparison table:

   ```
   Repository              | Expected Branch       | Actual Branch         | Status
   ------------------------|-----------------------|-----------------------|--------
   $WT_ROOT/PyAutoFit      | feature/my-task       | feature/my-task       | OK
   $WT_ROOT/PyAutoArray    | feature/my-task       | main                  | MISMATCH
   ```

   d. If **all repos match**: confirm and continue work. Remind the user to `source "$WT_ROOT/activate.sh"` if they're in a fresh shell.

   e. If **any repo is on an unexpected branch**:
      - Flag each mismatch with a warning
      - Ask the user whether to:
        - Switch the mismatched worktree to the expected branch (`git -C "$WT_ROOT/<repo>" checkout <expected>`)
        - Continue on the current branch (with acknowledgement)
        - Abort and investigate
      - **Do not proceed with any edits until the user responds.**

## Remote / mobile mode

**Environment detection:**
- If `~/Code/PyAutoLabs` exists and contains repo subdirectories → **laptop mode** (use all steps above)
- Otherwise → **mobile mode** (follow this section)

**Mobile behavior:**
1. Skip worktree-related checks entirely (`worktree_list_claimed`, `worktree_check_conflict`)
2. Check branch state via GitHub API instead of local git:
   ```bash
   gh api repos/<owner>/<repo>/branches --jq '.[].name' | head -10
   gh api repos/<owner>/<repo>/branches/<branch> --jq '.name' 2>/dev/null
   ```
3. Use the same GitHub org mapping as other skills:
   - rhayes777: PyAutoConf, PyAutoFit
   - Jammy2211: PyAutoArray, PyAutoGalaxy, PyAutoLens, all workspaces
4. Suggest branch name and present summary as normal
5. On resume verification: check branches via GitHub API instead of local `git -C` commands
