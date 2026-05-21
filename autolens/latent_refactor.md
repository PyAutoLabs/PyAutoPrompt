In euclid_lens_modeling_pipeline, but also autolens_workspace, we have started to build up quite a lot of calculations
in compute_latent_variables, and many now have different behaviour depending on if we are using JAX, numpy and
other variants.

Furthemore, these are hard coded into the code and users have change python code, commneting stuff out, in order
to disable or enable certain latent variables. The interface in general is not very user-facing and its not
a clear and concise API for users to interface with the latent variable API.

First, I think we should move all of the calculations to a dedicated source code module, which I think we can
call autolens/latent.py, albeit maybe this will become a package with subdivision. This will also ensure
we have documentation on each variable and some much needed unit tests can be added. We should avoid where possible 
the actual calculations being done here -- we dont want lensing to be stuck in a latent variable package.

- Config files customizing things.
- Workspace example explaining what latents are, what their errors and whatnot correspond to, explain posterior draws.
- Workspace example probably borrinwg from results showing how to load and use latent variables.
- Workspace example showing users how to extend Analysis object with their own latent variables.