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

These should all be paired with a config file, config/latent.yaml, which allows users to turn on and off via bools 
the output of every latent variable and means that when off compute_latent_variables does not waste compute on them. This may
required us to interface and update the autofit source code a bit, do an assessment of if thats worth it.
 
Create autolesn and autogalaxy Workspace example explaining what latent variables are (good descriptions already in autofit_workspace, 
what their errors and whatnot correspond to, explain posterior draws. Expand autofit workspace is key context is missing.

Workspace examples, probably borrinwg from results, showing how to load and use latent variables, I guess this could just
be a section at the end of the above workspace example rather than its own tutorials. Make it clear that adding
latents to the modeling enables this loading and inspection thereafter.

Workspace example also includes a section showing users how to extend Analysis object's with their own latent variables,
by niheriting the Analysis and over wrtiting the LATENT_KEYS And compute_latent_variables method. Encourage them
to submit source code extension for all users. 