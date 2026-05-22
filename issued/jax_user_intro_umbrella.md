Currently, the only way a user knows about JAX is when they do modeling via objects like AnalysisImaging.

There is an example of it being used in a simulator in autolens_workspace/scripts/cluster/simulator.py,
but the truth is we havent really introduce JAX into the workspace yet.

This also means the whole xp interfface, whereby a user has to both jax.jit a function and pass jnp to xp is
not documented for most users cases.

So frist, I want you to do deep research and assess our xp interface is suitable. My vision here is that,
but default a user does workspace stuff without JAX being used without them manually setting it up, except
for two use cases:

1) When they do lens modeling via Analysis objects, which already have the use_jax=True inputs.
2) When they do a simulation via a Simulator object, which should have the use_jax=True input.

The main logic is that when JAX is used, the user often has to input JAX arrays and also they receive JAx arrays,
these are specific and weird outputs a user kind of needs to be aware of for how to deal with.

So, first give me your judgement on if the JAX interface fully makes sense.

I am expecting the interface will alays have a user write the jax.jit of a function themselves, and then call
the jitted function.

We can omit gradients for now, whcih will come in a future pass of JAX functionality.

I detail things for autolens_Workspace below but truth is all work will be mapped across autogalaxy_workspace.

The work will then be the following:

1) in autolens_workspace/start_here.py and autogalaxy_workspace/start_here.py, early on (after __Tracer__ amd __Galaxies__)
Add a section __JAX_ detailing this interface to the user.
2) In each fit.py, likelihood_function.py, simulator.py and start_here.py file put describes of the use of JAX (this is across cluster / group / imaging / interferometer / multi / point source)
3) In guides, include JAX interface describes in data_structures.py (e.g. explaiing how using JAX changes structure to JAX array which require specific manipulation) but also gaalxeis.py, lens_calc.py and tracer.py.


Do deep research and dont be afraid to tell me how we could do it different or better.

This ultimately will also make its way into the docs folder for each projct and its foundation to how users use JAX
for speed up.

PyAutoPrompt/autofit/on_the_fly_docs.md also seems very relevent to this issue.
PyAutoPrompt/autogalaxy/visualizer_fit_for_visualization_dispatch.md also relevent.