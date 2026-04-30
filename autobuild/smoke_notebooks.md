Smoke tests currently run on python scripts, but we want to know on the normal workspaces notebooks
are always running ok.

For each, can you add two smoke tests on notebooks:

autofit_workspace: overview/overivew_1 and searches/mcmc.ipynb

autogalaxy_workspace and autolens_workspace: imaging/modeling.ipynb (with test mode) and interferometer/simulator.ipynb