Migrate the Health Agent into PyAutoBrain.

## Context

The Health Agent (the first PyAutoBrain specialist agent) was implemented under
the `feature/pyautobrain/health.md` task, but PyAutoBrain was not available in
the implementing environment. It was therefore staged in **PyAutoHeart** at
`health_agent/`:

- `health_agent.md`     — the agent definition (role, invoke Heart, reasoning,
                          GREEN/YELLOW/RED output schema, gate semantics).
- `capabilities.yaml`   — machine-readable manifest of every Heart capability
                          (the abstract-provider self-description the agent reads).
- `capabilities.md`     — human-readable audit of Heart's health surface.
- `pyautobuild_boundary_audit.md` — confirms no health logic drifted into Build.
- `README.md`           — index + the "canonical home is PyAutoBrain" note.

## Task

Once PyAutoBrain exists / is in scope:

1. Move `health_agent.md` (and `README.md`) into PyAutoBrain as the canonical
   Health Agent / first specialist agent, registered however PyAutoBrain
   registers its agents (skill, slash command, or agent definition).
2. **Keep `capabilities.yaml` in PyAutoHeart** — it is Heart self-describing its
   capabilities and belongs there. The Brain agent should read it from Heart (or
   from `pyauto-heart` output), not vendor a copy.
3. Update cross-references: the Heart `health_agent/README.md` note and this
   task; leave a pointer in Heart back to the Brain location.
4. Preserve the architectural boundary: Brain reasons, Heart checks, Hands
   executes. The agent must still treat Heart as an abstract provider and adopt
   `pyauto-heart readiness` as the authoritative verdict.

Keep each `.md` agent/skill file under 200 lines.
