The folloeing examples show simple searches using JAX gradients in autofit:

autofit_workspace_developer/searches_minimal

Wrtie a example nuts_jax.py, which runs blackjax's NUTS sampler on the examples in this folder and provide the
timing and whatnot information on how the run. goes.

This is BlackJAX:

https://github.com/blackjax-devs/blackjax

Once that is running and working, then go to the use case in @z_projects/concr/scripts/cancer_sim/graphical.py and 
write a file graphical_nuts.py which runs nuts on that.
