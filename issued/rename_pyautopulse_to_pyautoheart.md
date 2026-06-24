# Rename PyAutoPulse to PyAutoHeart

Original request:

```text
We are renaming the repository/project concept from PyAutoPulse to PyAutoHeart.

Context:
- This repo is currently PyAutoLabs/PyAutoPulse.
- It owns the health/readiness/validation layer of the PyAuto ecosystem: reusable workflows, tests, readiness gates, workspace validation, URL hygiene, generated-artifact/noise classification, and related documentation.
- The rename is part of a broader PyAuto organism metaphor:
  - PyAutoMind: ideas, prompts, intent, priorities
  - PyAutoBrain: planning, reasoning, orchestration
  - PyAutoHands: execution, build/release/deployment
  - PyAutoHeart: health, vitality, quality checks, readiness, continuous monitoring

Task:
Perform a careful repository-wide rename from PyAutoPulse to PyAutoHeart.

Requirements:
1. Replace user-facing names:
   - PyAutoPulse -> PyAutoHeart
   - pyautopulse -> pyautoheart
   - pulse -> heart only where it refers to the project identity, commands, docs, paths, package names, workflow labels, or ecosystem role.
   - Do NOT blindly replace every generic word “pulse” if it is being used metaphorically or historically in a way that would become awkward. Prefer careful semantic replacements.

2. Update documentation:
   - README files
   - docs
   - architecture notes
   - Build/Pulse/Agent boundary docs should become Build/Heart/Brain or Hands/Heart/Brain if those names already exist in the repo context.
   - Explain PyAutoHeart as the health/readiness/vital-signs layer of the PyAuto organism.

3. Update code/package metadata:
   - pyproject.toml / setup.cfg / setup.py if present
   - package/module names if present
   - import paths
   - CLI entry points
   - scripts
   - tests
   - workflow names
   - badges
   - URLs where appropriate

4. Update GitHub Actions / reusable workflow references:
   - Rename workflow display names from Pulse to Heart.
   - Preserve backwards compatibility if external repos may still call reusable workflows by existing filenames.
   - Do not rename workflow files if that would break external `uses: PyAutoLabs/PyAutoPulse/.github/workflows/...` references, unless you also add compatibility wrappers or document the required downstream migration.
   - Prefer keeping stable workflow filenames initially and only changing displayed names / docs, unless the repo clearly has no external consumers.

5. Preserve compatibility:
   - If there is an importable Python package called `pyautopulse`, either:
     a) add a compatibility shim from `pyautopulse` to `pyautoheart`, or
     b) explicitly document why no shim is needed.
   - If there are CLI commands containing `pulse`, consider adding aliases during transition.
   - Avoid breaking existing automation unless unavoidable.

6. Search thoroughly:
   - Use ripgrep for:
     - PyAutoPulse
     - pyautopulse
     - AutoPulse
     - pulse
     - Pulse
     - PULSE
   - Review each occurrence manually.
   - Also check hidden/config files: `.github`, `.claude`, scripts, docs, tests.

7. Validation:
   - Run formatting/linting if configured.
   - Run the test suite if present.
   - Run any local smoke/readiness script documented in the repo.
   - Report anything that cannot be run locally.

Deliverables:
- One PR implementing the rename.
- PR title: `Rename PyAutoPulse to PyAutoHeart`
- PR description should include:
  - summary of renamed docs/code/workflows
  - compatibility notes
  - validation commands run
  - any follow-up tasks needed, especially downstream repo references still pointing at PyAutoPulse.

Important:
This is a branding / architecture rename, not a behavioural rewrite. Keep functionality unchanged unless a small compatibility shim is needed.
```
