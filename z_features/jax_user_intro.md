JAX user introduction series — make JAX a first-class concept in autolens_workspace
and autogalaxy_workspace, so users discover it through start_here.py and per-dataset-type
scripts, understand the xp / jnp interface, the "use_jax=True in → JAX arrays out"
contract, and how to write their own `jax.jit` around library calls. Gradients are
explicitly out of scope for this series (future pass).

__Outstanding__ (sequenced)

1. Phase 0 — research & design ([autoarray/jax_interface_audit_and_design.md](../autoarray/jax_interface_audit_and_design.md))
   Deep audit of the xp interface and use_jax=True surface; judgement on whether it
   makes sense as a user-facing contract; design decision on Simulator.use_jax shape;
   output a reference design doc at `admin_jammy/notes/jax_interface.md` that all
   later phases cite.

2. Phase 1 — top-level __JAX__ sections (one prompt: `workspaces/jax_start_here_intros.md`, TBA)
   - `autolens_workspace/start_here.py` — expand existing __JAX__ block (line 33+) per Phase 0
   - `autogalaxy_workspace/start_here.py` — add new __JAX__ block post-__Galaxies__

3. Phase 2 — library: Simulator.use_jax (one prompt: `autoarray/simulator_use_jax.md`, TBA)
   - PyAutoArray `SimulatorImaging` / `SimulatorInterferometer` signature gains `use_jax=True`
   - PyAutoLens / PyAutoGalaxy simulator subclasses thread it through
   - PyAutoLens `point.SimulatorPoint` gains `use_jax=True` (replaces `cluster/simulator.py` manual jit)

4. Phase 3 — per-dataset-type doc passes (autolens_workspace) — one prompt each
   - 3a `autolens_workspace/jax_docs_imaging.md` (TBA)
   - 3b `autolens_workspace/jax_docs_interferometer.md` (TBA)
   - 3c `autolens_workspace/jax_docs_multi.md` (TBA)
   - 3d `autolens_workspace/jax_docs_point_source.md` (TBA)
   - 3e `autolens_workspace/jax_docs_group.md` (TBA)
   - 3f `autolens_workspace/jax_docs_cluster.md` (TBA — `cluster/simulator.py` already has detail)

5. Phase 4 — mirror to autogalaxy_workspace — one prompt each
   - 4a `autogalaxy_workspace/jax_docs_imaging.md` (TBA)
   - 4b `autogalaxy_workspace/jax_docs_interferometer.md` (TBA)
   - 4c `autogalaxy_workspace/jax_docs_multi.md` (TBA)

6. Phase 5 — guides
   - 5a `autolens_workspace/jax_docs_guide_data_structures.md` (TBA)
   - 5b `autolens_workspace/jax_docs_guide_galaxies.md` (TBA)
   - 5c `autolens_workspace/jax_docs_guide_tracer.md` (TBA)
   - 5d `autolens_workspace/jax_docs_guide_lens_calc.md` (TBA)
   - 5e `autogalaxy_workspace/jax_docs_guides.md` (TBA — galaxies + data_structures mirror)

Per [[feedback_no_bulk_issue_queues]], sub-prompts past Phase 0 are deliberately
unauthored. Each gets authored and issued only when its predecessor is close to
shipping, so the prompts don't age out against the design that emerges from Phase 0.

__Related__

- [autofit/on_the_fly_docs.md](../autofit/on_the_fly_docs.md) — workspace doc updates
  around background quick-update; cite from Phase 3+ where `fit.py` scripts mention
  visualization.
- (link rot) The original prompt referenced `PyAutoPrompt/autogalaxy/visualizer_fit_for_visualization_dispatch.md`
  which does not exist in this tree. Before authoring sub-prompts that touch the
  visualizer, grep `complete.md` for any visualization-dispatch prompt that may
  already have shipped.

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
