This is Phase 0 of the JAX user introduction series ([z_features/jax_user_intro.md](../z_features/jax_user_intro.md)).

Goal: deeply audit how PyAutoArray's xp interface, PyAutoFit's `Analysis(use_jax=True)`
flag, and PyAutoLens / PyAutoGalaxy's manual `jax.jit + pytree register` pattern hang
together, *from a new user's perspective*. Output a single reference design doc at
@admin_jammy/notes/jax_interface.md that all later phases cite.

The audit is intentionally read-only on libraries — no PyAutoArray / PyAutoLens /
PyAutoGalaxy code changes in this phase. If the audit surfaces specific library
follow-ups, file them as separate prompts under `autoarray/` or as new entries in
the tracker. The single file that lands in this phase is `admin_jammy/notes/jax_interface.md`.

__Audit scope__

For each layer below, document: what the user sees, what the contract is, where it
leaks abstraction, and what could be cleaner. Be specific — file paths, function
signatures, worked-through examples.

1. **The xp parameter pattern**
   - @PyAutoArray/autoarray/structures/decorators/abstract.py — how `xp` flows in,
     what gets wrapped, what doesn't.
   - The "autoarray types are not JAX pytrees → cannot be returned from jax.jit"
     rule (see @PyAutoArray/CLAUDE.md "JAX Support"). Where the `if xp is np:`
     guard pattern fires (grep `if xp is np` across @PyAutoArray and @PyAutoGalaxy
     — especially @PyAutoGalaxy/autogalaxy/operate/lens_calc.py).
   - User implication: what shape do they receive back? autoarray wrapper or raw
     `jnp.ndarray`? When does each happen?

2. **`Analysis(use_jax=True)`**
   - What the flag actually flips. Where it lands inside the fitness path. What
     `_vmap` vs `_jit` dispatch looks like. Reference @PyAutoFit/autofit/non_linear/analysis/fitness.py
     and the Analysis classes in @PyAutoLens / @PyAutoGalaxy.
   - Per [[feedback_jax_validation_vmap_not_jit]], `jax.jit(fn)(concrete_instance)`
     hides un-threaded xp sites; the honest validation path is `fitness._vmap(...)`.
     The user-facing story needs to be clear about which path runs in production.

3. **Simulator status quo**
   - @PyAutoArray/autoarray/dataset/imaging/simulator.py and the PyAutoLens /
     PyAutoGalaxy subclasses do *not* currently accept `use_jax=True`. Confirm and
     describe what they do today.
   - @autolens_workspace/scripts/cluster/simulator.py shows the manual pattern:
     pytree registration + `jax.jit` + raw jnp arrays in / out. Read this in full
     as the working example of the contract a `Simulator.use_jax=True` would
     formalise. Note the `_registration_model` dance — that's the part Phase 2
     would need to absorb into the library.

4. **Manual `jax.jit` pattern (user-authored)**
   - The user's vision: every workspace example always shows the user writing
     `@jax.jit` themselves around a library function, then calling the jitted
     form (not the library doing it implicitly).
   - Audit @autolens_workspace/scripts/cluster/simulator.py and the relevant
     `autolens_workspace_test/scripts/jax_likelihood_functions/...` scripts for
     the canonical shape this pattern takes. Are there inconsistencies users
     would trip over?
   - Per [[feedback_jax_closure_cache_busts]], fresh closures per call bust the
     JIT cache. Does the user-facing API surface this trap, or hide it?

5. **Workspace surface**
   - Where today does a user encounter JAX in @autolens_workspace and
     @autogalaxy_workspace? Grep for `use_jax`, `jax.jit`, `jnp`, `xp=`,
     `from autoconf import jax_wrapper`. Map current coverage vs. the target
     coverage (start_here + 6 dataset types × 4 script types + 4 guides × 2
     workspaces).

__Judgement to produce__

Address each of these questions in the output doc, with a recommendation:

a. Is `xp=np / xp=jnp` the right interface for users to *encounter*, or is
   `use_jax=True` enough? Are there places where xp leaks into user code that
   shouldn't?
b. Should Simulator gain `use_jax=True`? If yes, what's the exact user contract —
   does the user pass jnp arrays in? Does the simulator return jnp or numpy data?
   Should the pytree registration be automatic (Phase 2's problem to solve) or
   remain manual?
c. Is the "user writes `jax.jit` themselves" rule the right one? Or should
   `AnalysisImaging` internally jit on first call? What's lost in either choice?
d. Where does the autoarray-not-pytree limitation hurt user experience? Is it
   worth surfacing in user docs (so users aren't surprised), or fixing
   structurally (a library task for a future phase)?

__Output__

Single markdown file at `admin_jammy/notes/jax_interface.md`. Suggested sections:

- **Survey** — the current interface, layer by layer (1–5 above), with code excerpts.
- **Judgement** — pros, cons, sharp edges, with questions (a)–(d) answered.
- **Design** — the user-facing contract Phases 1–5 should document and Phase 2
  should implement.
- **Open questions** — anything you couldn't decide alone; flag for user discussion.

Keep it tight — this is a reference, not a paper. ~10–30 pages of markdown is the
right scale. PyAutoPaper concept-page formatting is fine but not mandatory.

__Out of scope__

- Library code changes (those land in Phase 2 if the design calls for them).
- Gradients (explicitly deferred by the user; future series).
- Workspace doc edits (those are Phases 1, 3, 4, 5).

__References__

- @PyAutoArray/CLAUDE.md "JAX Support" section — the canonical statement of the
  autoarray-not-pytree rule and the `if xp is np:` guard pattern.
- @autolens_workspace/scripts/cluster/simulator.py — the canonical working example
  of the full manual JAX pattern (registration + jit + raw jnp).
- @autolens_workspace/start_here.py:33-50, 240-275 — the current user JAX
  touchpoint, which Phase 1 will expand.
- [autofit/on_the_fly_docs.md](on_the_fly_docs.md) — sibling prompt on workspace
  doc updates (cite from later phases, not absorbed here).
