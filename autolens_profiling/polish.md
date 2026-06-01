autolens_profiling is now a mature project, with a good separation of different tasks into packages (instruments, latent, likelihood_breakdown, etc)

I have run it for imaging, interferometer and datacube all individually, and got good profiling results, but at times
I think context with Claudee gets switched and some results may not quite be right, or certain interpretatons
of results.

Can you therefore look at the project with a high level holistic overview and do a full run in order to get
profiling times for imaging, interferometer and datacube. Can you give me likelihood_runtime values for all
for all instrument types (e.g. for imaging do Ao, JWST, HST), whereas for likelihood_breakdown stick to HST for
imaging, Alma_high for interferometer and datacube. dont worry about point_source for now. For imaging
also do the whole sparse vs mapping and make sure this runs through packages like vram too. of course for
the _runtime we should also get info on using vmap primarily, I guess it shuld be vmap only using the batch
sizes that are computed via the vram package.

Can you first do a scan of the repo and make sure that when these results come in, they are saved as .md or
.json files and perhaps clearly displayed in eery GitHub .md file for browsing and viewing. This is really our
last profiling before we do some proper optimizations and speed ups, so this is a base set of run times to compare
to. Perhaps we can give them a name to mark it -- PreOptimizationTimes.

Before doing all this, do deep research on the project and have one last think about if ther are ways to improve it,
redesigns to make it omre concise and cleary. I'm really happy with how it all works now, but we will be extending upon it
with more datasets, packages and whatnot so a good opportunity to lock the core design in well now.

I am also going to want laptop GPU run times as part of this, but dont run those as I need to do it when my laptop 
isnt being used, so I will do that in afollow up prompt. Try to do run times for laptop CPU, HPC CPU and HPC A100.
Do the A100 mixed precision profiling too, where thre source code supports it.

Sometimes these profiling runs flag memory issues or VRAM issues and part of the issue with the claude context
has basically been how that often then side tracked us to do fixes to the soruce code before going back to
profiling. Perhaps we should run vram first to make sure we dont have memory issues or other bugs crop up.

Dont do any searches, we are profiling likelihood functions here.