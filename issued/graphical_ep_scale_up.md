We are now going to begin scaling up the graphical model snad EP frameworks.

First, in autofit_workspace_developer, we need to make two packages with examples called `graphical` and `ep`.
Lets adapt the following examples to set these up:

z_projects/concr/scripts/toy

But lets do this in two separate packages in autofit_workspace_developer called `graphical` and `ep`..

These scripts are slow and do not scale to large samples for many reasons, for the graphical example it is because:

1) It uses the DynestySampler, which takes many samples as its nested sampling and does not scale well with dimensionality.
2) There are many overheads for example outputting lots of results and visuals on a per-factor basis.
3) Even if this output were fast, there is a point where it does not scale for a human because the information is spread out over
many folders in output and thus there are no single-point-of-references to inspect results or look at visualization.

There are likely other limitations of graphical, thus your first task is to write a prompt PyAutoPrompt/graphical_ep/graphical_scoping.md
which define a prompt which sets up autofit_workspace_developer/graphical, runs it for 3, 10, 30, datasets and does an assessment
of how we can improve it for scalability. The goal should be to target the bigger reasons for it being slow first (e.g. using a faster)
sampling, which likely exploits gradients and then goes on to follow up with other aspects like visualizaition. Its not just about
run-time and speed, its also about making sure a scientist can be confident that as they scale up results are accurate and
robust, so high level information on results is key. Hard-disk size, and time spent writing results to hard disk,
may also prove important both in keep size down and reducing time -- thus moving to a model where runs can be resumed 
but do not need to output so much stuff on the hard disk may be benefitial. anyway, have the prompt do the full scoping
with its end goal to break the work down into a series of sub tasks and prompts we can tackle one by one.

We will then do the same thing for EP. Note that I have already done an EP profiling using a IC50 cancer use case where I got
the information at the bottom of this prompt. However, rerun everything for the toy model, and confirm you agree with the assessment.
Again, remember for now we are just writing prompts in  PyAutoPrompt/graphical_ep/ep_scoping.md to break this down into a series of tasks.

For EP, one of the biggest overheads is probably just autofit internals running slowly and needing speeding up.

Another key point with this scaling up is I want us to always be sure that the changes we make do not change the results.
Therefore, can you have the `simulator.py` output a file `ground_truth.json` for every datasets AND ALWAYS have our graphical
and ep scripts do sanity checks, at the end after profiling, that they are recovering the ground truth values and that their
log likelihood values (whichc an be output by a simulator.py) by doing a fit are also being maximized or recovered correctly.

Do deep research and plan for a while on this one.

● Here's the bottleneck ranking with optimisation angles for each, ordered by importance. Each tier is a natural prompt scope.

  Tier 1 — Dynesty wrapper overhead (~86% of optimise time)

  Impact: ~5.5 s per Dynesty fit × (N+1)×M fits per run. This is ~5 × N seconds per EP iteration that has nothing to do with the math. At N=10000 this bucket alone is ~30 hours.

  What's actually in it (from the PyAutoFit log lines we saw on every fit):
  - corner_anesthetic plot attempt — even though it always emits "posterior estimate not yet sufficient" for EP factors and contributes no end-of-run artifact
  - Creating latent samples by drawing 100 from the PDF — latent-space resampling step
  - Removing search internal folder — per-fit cleanup
  - Generating initial samples of model using JAX LH Function cores — initial sample finding before sampling starts
  - Sampler construction, paths object setup, bound construction inside Dynesty itself

  Candidate prompts (rank order by likely payoff):

  1. Suppress per-fit plotting and visualisation during EP iterations. EP factors don't need per-iteration corner plots or latent-sample draws — only the final one matters. Add a paths.suppress_plots /
  paths.ep_mode = True flag (or whatever PyAutoFit calls it) that the EP loop sets on each factor's paths before each iteration's fit. Probably a single-digit-percent change in PyAutoFit but cuts the wrapper
  bucket noticeably.
  2. Reuse the Dynesty sampler / pool across fits. Right now each search.fit(...) re-instantiates the sampler from scratch. EP's structure is "fit the same factor with mildly different priors each iteration"
  — there's an obvious cache there. May need a small force_x1_cpu carve-out so the no-pool branch reuses state.
  3. Skip Removing search internal folder for in-memory EP runs. When output-to-disk is disabled (we already see Output to hard-disk disabled, input a search name to enable), the folder removal step is doing
  nothing useful but still costs wall time.
  4. cProfile/py-spy attribution pass on the same workload to confirm which of the above dominates inside the wrapper bucket. Easiest first prompt — non-invasive, just gives data. Optional but quick.

  Tier 2 — Local Hill LL evaluation (~6%)

  Impact: Linear in N. At N=5 it's 4.4 s; at N=10000 it'd be ~2.4 hours. Becomes dominant only past N≈1000.

  Lever:

  5. Parallelise the N independent local Dynesty fits across CPU cores per EP iteration. Each local fit is embarrassingly parallel — no message passing within an iteration. PyAutoFit's EP loop currently runs
  them serially. Either a process pool or shared-memory thread pool would help. Worth doing once Tier 1 is cut.

  (Could also vmap the Hill likelihood across datasets within a single fused Dynesty fit, but that's a bigger model-architecture change. The parallel-across-factors approach is more straightforward.)

  Tier 3 — Global LL evaluation (~5%)

  Impact: Constant in N — set_model_approx freezes hill_coef, so the global free-param count stays at 18 regardless of dataset count. ~1.7 s/iter forever.

  Lever:

  6. Replace Dynesty on the global factor with a Laplace approximation or a small-scale gradient-based optimiser. For 18 free params with a smooth Gaussian likelihood (_global_log_likelihood_jit), nested
  sampling is overkill. Laplace gives a Gaussian posterior approximation in a handful of Newton steps. PyAutoFit already has LaplaceOptimiser (we instantiate one in run_ep_fit) — investigate whether it can be
   wired up as the global factor's optimiser instead of search_global. If yes, this drops to ~0.

  Tier 4 — EP-loop orchestration (~4%)

  Impact: ~1.3 s/iter, currently roughly constant in N. Largest unknown — message-field updates iterate over all priors, which could be linear-in-N at large scale.

  Lever:

  7. Profile inside factor_graph.optimise(...) proper. This is cProfile/py-spy territory. A targeted prompt would say "drop a cProfile on ep_sim excluding search.fit(...) calls; report which
  autofit/graphical/ functions show up at >5% time." Cheap, informative, and may surface a quadratic loop we don't know about.

  Tier 5 — set_model_approx (~0%)

  Not worth optimising — 2 ms total at N=5. If you ever scale to N=10⁶ this might matter (the prior-walk is O(N)), but at that point everything else has long since blown up. Skip.

  Recommended prompt sequencing

  The fastest path to a usable 1000-dataset run:

  1. First write the cProfile attribution prompt (item 4) — it costs ~30 minutes and tells you which Tier 1 sub-optimisation to chase first. Don't guess; measure.
  2. Then suppress per-fit plotting + folder cleanup (items 1, 3) — likely the biggest single bang-for-buck once cProfile confirms.
  3. Then Laplace for the global factor (item 6) — small constant savings but easy.
  4. Then measure at N=100 to validate the projection model. If the wrapper bucket has shrunk as expected, the projection coefficients will be much friendlier.
  5. Then sampler reuse across fits (item 2) and parallelise local fits (item 5) — both are bigger lifts but address the still-dominant buckets once plotting is gone.

  Tier 4 (orchestration cProfile) is a "do once we're past Tier 1+2+3" item.