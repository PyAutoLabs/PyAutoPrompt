Currently, the autogalaxy_workspace and autolens_workspace do not have a guide illustrating all available light
profiles with examples of how to use them either as instances to do things like make images / convergence maps
or how to compose them as a model.

Can you make a guide for both workspaces, in scripts/guides/profiles/light.py which gfives examples for all
support light profiles, where the first bit gives exampels for all light profiles in one section but does not
use them to make images (e.g. high level run through), it then shows the API for making an image from one
light profile (e.g. Sersic). 

It then explxains linear light profiles and operated light profiles, shown a one line API on how to use them
but then says they are documented in the scritps/*/features/linear light profile and operated_light_profile
packages.

it then shows the API for how to put this in a model (e.g. Sersic again),
it then shows the API for making an instance from this Sersic model, either from just the light profile or when its
in the galaxies model the modeling API shows, and then does a detailed run through of
all remainig light profiles but only showing their image_2d_from methods, emphaissing the API above is
translateable.

For the autolens_workspace this will use the Tracer object instead of Galaaxies object and put the model gernerated
instances in the Tracer where appropriate.

Do a comparison to the docs folder in PyAutoGalaxy and PyAutoLens to make sure all light profiles are in
sync and also point to the URL to the docs early on in this script.