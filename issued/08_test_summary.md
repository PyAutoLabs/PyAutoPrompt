currently, when smoke tests run they display only on that prompt but the ifnormation is not permenant and often lost.
Can you make it so smoke test runs store their summary to a file which then displays when pyauto-summary runs,
so I can see the state of all smoke tests when I activeate my venv? Can you make this text green or red depenending
on success or failure.

Can you also do a similar thing with the latest build and release, e.g. what version is the software,
whene did we last do a release and what is a summary (keep it quite concise, just say how many failures not
which ones) were there when we did the full autobuild release which runs all test workspaces? Again,
this probably means files need creating when they run/

I dont want loading the venv to be too slow so try and make sure this stays fast.