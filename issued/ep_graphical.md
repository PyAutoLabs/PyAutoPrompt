The project @z_projects/ic50_workspace is our IC50  use case which we are now aiming to scale up the EP framework
to the IC50 use case.

We have EP fits, which fit each Hill Curve one-by-one, and then do the AnalysisGlobal model set up, but we dont
have a graphical.py example which fits everything at once in one huge parameter space.

An example of the graphical modeling API, which should be adapted for this use, is given in
HowToFit/scritps/chapter_3_graphical_models, read all 5 tutorials to work out how individual models, graphical
models and EP fits are related. Then come up with a sensible way to make a graphicl variant of the existing EP
fit.