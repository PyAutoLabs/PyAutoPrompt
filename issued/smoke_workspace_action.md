I currently have source code repos (PyAutoConf, PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens) and
workspaces (autofit_workspace, autogalaxy_workspace, autolens_workspace).

The source code repos have github actions which build the projects based on their main, run the unit tests
and flag when something went wrong.

the workspaces do not have actions that do this, thus when I run ship_workspace and merge changes if I have broke 
anything I get no feedback.

Its not feasible to run all scripts in a workspace, there are many and theyd expire my github action usage. However,
we have smoke tests which run reduced and fast end-to-end variants. Currently, these runs when I choose during
a claude session, but its time these run via my github actions. Thus, can you set it up so that this is so,
making sure they spawn off the up to date and relevent builds from the soruce code repos. By inspection
autobuild you may find useful information on how to set this up.

Make sure, like the source rpeos, I get an email and slack notifications to channels when these fails.

Make sure these actions use the env vars tha smoke tests do, and double the number of things that run from
the smoke tests as we can be a bit more extensive as the run times are on a github server.

I think these have been added now so just check if thats the case or if any requested freaturs are missing.