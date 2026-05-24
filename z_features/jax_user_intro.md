JAX user introduction series — make JAX a first-class concept in autolens_workspace
and autogalaxy_workspace, so users discover it through start_here.py and per-dataset-type
scripts, understand the xp / jnp interface, the "use_jax=True in → JAX arrays out"
contract, and how to write their own `jax.jit` around library calls. Gradients are
explicitly out of scope for this series (future pass).

__Scope anchor (added 2026-05-24 from Phase 0)__

User-facing design targets the **core API** — `imaging`, `interferometer`, `point_source`
dataset types and the workspace guides (`data_structures`, `galaxies`, `tracer`,
`lens_calc`). `cluster`, `group`, `multi`, and developer-test paths are
advanced / in-development and inform the design but do not drive it. Phase 3
priority order is **3a, 3b, 3d** (the core dataset types); the rest are
deferred or optional.

__Outstanding__ (sequenced)

1. Phase 0 — research & design ([autoarray/jax_interface_audit_and_design.md](../autoarray/jax_interface_audit_and_design.md))
   ✅ **SHIPPED 2026-05-24.** Design doc at `admin_jammy/notes/jax_interface.md`
   (admin_jammy main commit `2f02bbf`). Issue PyAutoArray#331 closed. All later
   phases cite this doc as the source of truth.

2. Phase 1 — top-level __JAX__ sections (one prompt: `workspaces/jax_start_here_intros.md`, TBA — unblocked)
   - `autolens_workspace/start_here.py` — **add new __JAX__ block** post-__Tracer__ per Phase 0
     (no existing block at line 33; tracker note was about per-dataset start_here, not top-level)
   - `autogalaxy_workspace/start_here.py` — add new __JAX__ block post-__Galaxies__

3. Phase 2 — library: Simulator.use_jax (one prompt: `autoarray/simulator_use_jax.md`, TBA — unblocked)
   - PyAutoArray `SimulatorImaging` / `SimulatorInterferometer` signature gains `use_jax=True`
   - PyAutoLens / PyAutoGalaxy simulator subclasses thread it through; auto-register
     Tracer/Imaging/Interferometer as pytrees on first call
   - PyAutoLens `PointSolver` gains `use_jax=True`; defaults `remove_infinities=False` on the JAX path
   - Validation: new `autolens_workspace/scripts/imaging/simulator.py` `__JAX Variant__`
     (from Phase 3a) runs end-to-end. Cluster simulator migration is a secondary
     worked example (its 8-step ceremony collapsing to 1-2 user lines).

4. Phase 3 — per-dataset-type doc passes (autolens_workspace) — one prompt each
   **Priority — core API (do these first):**
   - 3a `autolens_workspace/jax_docs_imaging.md` (TBA)
   - 3b `autolens_workspace/jax_docs_interferometer.md` (TBA)
   - 3d `autolens_workspace/jax_docs_point_source.md` (TBA)

   **Deferred — advanced / in-dev (author only if 3a/3b/3d shipped clean):**
   - 3c `autolens_workspace/jax_docs_multi.md` (TBA — multi/start_here.py already has __JAX__)
   - 3e `autolens_workspace/jax_docs_group.md` (TBA — optional)
   - 3f `autolens_workspace/jax_docs_cluster.md` (TBA — defer until cluster is out of in-dev;
     when authored, mostly a migration showing cluster/simulator.py before/after Phase 2)

5. Phase 4 — mirror to autogalaxy_workspace — one prompt each
   - 4a `autogalaxy_workspace/jax_docs_imaging.md` (TBA — highest priority)
   - 4b `autogalaxy_workspace/jax_docs_interferometer.md` (TBA)
   - 4c `autogalaxy_workspace/jax_docs_multi.md` (TBA — defer behind 4a/4b)

6. Phase 5 — guides
   - 5a `autolens_workspace/jax_docs_guide_data_structures.md` (TBA)
   - 5b `autolens_workspace/jax_docs_guide_galaxies.md` (TBA)
   - 5c `autolens_workspace/jax_docs_guide_tracer.md` (TBA)
   - 5d `autolens_workspace/jax_docs_guide_lens_calc.md` (TBA — advanced xp story lives here)
   - 5e `autogalaxy_workspace/jax_docs_guides.md` (TBA — galaxies + data_structures mirror)

Per [[feedback_no_bulk_issue_queues]], sub-prompts past Phase 0 remain deliberately
unauthored. Each gets authored and issued only when its predecessor is close to
shipping, so the prompts don't age out against the design that emerged from Phase 0.

__Related__

- [autofit/on_the_fly_docs.md](../autofit/on_the_fly_docs.md) — workspace doc updates
  around background quick-update; cite from Phase 3+ where `fit.py` scripts mention
  visualization.
- `PyAutoPrompt/issued/visualizer_fit_for_visualization_dispatch.md` — sibling prompt
  on visualizer dispatch, already issued. (Earlier "link rot" note was incorrect —
  the file lives in `issued/`, not the autogalaxy/ source tree it was originally
  authored under.)

__Original prompt__

Verbatim from `workspaces/jax_user_intro.md` (2026-05-22):

> Currently, the only way a user knows about JAX is when they do modeling via objects like AnalysisImaging.
>
> There is an example of it being used in a simulator in autolens_workspace/scripts/cluster/simulator.py,
> but the truth is we havent really introduce JAX into the workspace yet.
>
> This also means the whole xp interfface, whereby a user has to both jax.jit a function and pass jnp to xp is
> not documented for most users cases.
>
> So frist, I want you to do deep research and assess our xp interface is suitable. My vision here is that,
> but default a user does workspace stuff without JAX being used without them manually setting it up, except
> for two use cases:
>
> 1) When they do lens modeling via Analysis objects, which already have the use_jax=True inputs.
> 2) When they do a simulation via a Simulator object, which should have the use_jax=True input.
>
> The main logic is that when JAX is used, the user often has to input JAX arrays and also they receive JAx arrays,
> these are specific and weird outputs a user kind of needs to be aware of for how to deal with.
>
> So, first give me your judgement on if the JAX interface fully makes sense.
>
> I am expecting the interface will alays have a user write the jax.jit of a function themselves, and then call
> the jitted function.
>
> We can omit gradients for now, whcih will come in a future pass of JAX functionality.
>
> I detail things for autolens_Workspace below but truth is all work will be mapped across autogalaxy_workspace.
>
> The work will then be the following:
>
> 1) in autolens_workspace/start_here.py and autogalaxy_workspace/start_here.py, early on (after __Tracer__ amd __Galaxies__)
> Add a section __JAX_ detailing this interface to the user.
> 2) In each fit.py, likelihood_function.py, simulator.py and start_here.py file put describes of the use of JAX (this is across cluster / group / imaging / interferometer / multi / point source)
> 3) In guides, include JAX interface describes in data_structures.py (e.g. explaiing how using JAX changes structure to JAX array which require specific manipulation) but also gaalxeis.py, lens_calc.py and tracer.py.
>
>
> Do deep research and dont be afraid to tell me how we could do it different or better.
>
> This ultimately will also make its way into the docs folder for each projct and its foundation to how users use JAX
> for speed up.
>
> PyAutoPrompt/autofit/on_the_fly_docs.md also seems very relevent to this issue.
> PyAutoPrompt/autogalaxy/visualizer_fit_for_visualization_dispatch.md also relevent.
