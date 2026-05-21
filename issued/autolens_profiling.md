The likelihood package in autolens_profiling has scripts which profile the likelihood functions of many
dataset types and models. However, I think each script is trying to do too many things:

1) It gives a step-by-step run trhough of the likelihood function.
2) It is used to compute run times on CPU, GPU and with different settings.

The truth is, I often want to do these takss separately, so I either want to know

"How fast is interferometer delaunay on an A100, and compare that to CPU, for ALMA and ALMA high res"

or

"Give me a step by step profiling of the imaging likelihood for JWST on GPU and suggest where we should optimize"

The combination of all this information in the python scripts and likelihood package also means that the results,
profiling information and interrpetation is confused because different ways of getting timings are being mixed.

Therefore, can we split this into two packages, `likelihood_break_down` (or suggest better term) and
`likelihood_total` (again feel free to suggest a better term).

Then, do some deep research about how the profiling results for these two different use cases should be collected
and presents. Like I said, one is about estimating how long the analysis of a certain type of science
data might take with given hardware, the other is about understanding source code bottlenecks.

This will also help a lot because the step-by-step profiling can take a long time, especially for things like
eager numpy. The overall run times and time to gather the information being requested will therefore also
reduce.