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

2. Phase 1 — top-level __JAX__ sections (one prompt: [`workspaces/jax_start_here_intros.md`](../workspaces/jax_start_here_intros.md), authored 2026-05-24 — ready to `/start_dev`)
   - `autolens_workspace/start_here.py` — **add new __JAX__ block** post-__Tracer__ per Phase 0
     (no existing block at line 33; tracker note was about per-dataset start_here, not top-level)
   - `autogalaxy_workspace/start_here.py` — add new __JAX__ block post-__Galaxies__

3. Phase 2 — library: Simulator.use_jax ([`autoarray/simulator_use_jax.md`](../autoarray/simulator_use_jax.md), authored 2026-05-24 — ready to `/start_dev`)
   - PyAutoArray `SimulatorImaging` / `SimulatorInterferometer` signature gains `use_jax=True`
   - PyAutoLens / PyAutoGalaxy simulator subclasses thread it through; auto-register
     Tracer/Imaging/Interferometer as pytrees on first call
   - PyAutoLens `PointSolver` gains `use_jax=True`; defaults `remove_infinities=False` on the JAX path
   - Validation: new `autolens_workspace/scripts/imaging/simulator.py` `__JAX Variant__`
     (from Phase 3a) runs end-to-end. Cluster simulator migration is a secondary
     worked example (its 8-step ceremony collapsing to 1-2 user lines).

4. Phase 3 — per-dataset-type doc passes (autolens_workspace) — one prompt each
   **Priority — core API (do these first):**
   - 3a [`autolens_workspace/jax_docs_imaging.md`](../autolens_workspace/jax_docs_imaging.md) (authored, ready)
   - 3b [`autolens_workspace/jax_docs_interferometer.md`](../autolens_workspace/jax_docs_interferometer.md) (authored, ready)
   - 3d [`autolens_workspace/jax_docs_point_source.md`](../autolens_workspace/jax_docs_point_source.md) (authored, ready)

   **Deferred — advanced / in-dev (author/issue only after 3a/3b/3d ship clean — see banner inside each prompt file):**
   - 3c [`autolens_workspace/jax_docs_multi.md`](../autolens_workspace/jax_docs_multi.md) (authored, deferred)
   - 3e [`autolens_workspace/jax_docs_group.md`](../autolens_workspace/jax_docs_group.md) (authored, deferred — optional)
   - 3f [`autolens_workspace/jax_docs_cluster.md`](../autolens_workspace/jax_docs_cluster.md) (authored, deferred — defer until cluster is out of in-dev)

5. Phase 4 — mirror to autogalaxy_workspace — one prompt each
   - 4a [`autogalaxy_workspace/jax_docs_imaging.md`](../autogalaxy_workspace/jax_docs_imaging.md) (authored, ready — highest priority)
   - 4b [`autogalaxy_workspace/jax_docs_interferometer.md`](../autogalaxy_workspace/jax_docs_interferometer.md) (authored, ready)
   - 4c [`autogalaxy_workspace/jax_docs_multi.md`](../autogalaxy_workspace/jax_docs_multi.md) (authored, deferred behind 4a/4b)

6. Phase 5 — guides
   - 5a [`autolens_workspace/jax_docs_guide_data_structures.md`](../autolens_workspace/jax_docs_guide_data_structures.md) (authored, ready)
   - 5b [`autolens_workspace/jax_docs_guide_galaxies.md`](../autolens_workspace/jax_docs_guide_galaxies.md) (authored, ready)
   - 5c [`autolens_workspace/jax_docs_guide_tracer.md`](../autolens_workspace/jax_docs_guide_tracer.md) (authored, ready)
   - 5d [`autolens_workspace/jax_docs_guide_lens_calc.md`](../autolens_workspace/jax_docs_guide_lens_calc.md) (authored, ready — advanced xp story lives here)
   - 5e [`autogalaxy_workspace/jax_docs_guides.md`](../autogalaxy_workspace/jax_docs_guides.md) (authored, ready — galaxies + data_structures mirror)

All 15 sub-prompts (Phase 1-5) were authored 2026-05-24 in a single pass by
explicit user request, then sequenced for execution. Per
[[feedback_no_bulk_issue_queues]], **issue them to GitHub one at a time
via `/start_dev`** when each is ready to ship — do NOT bulk-file the
GitHub issues. The "deferred" markers above + the in-file banners enforce
the sequencing: 3c/3e/3f and 4c wait until their core siblings ship clean.

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
