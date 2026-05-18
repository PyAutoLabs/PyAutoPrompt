Currently, the autogalaxy_workspace and autolens_workspace do not have a guide illustrating all available mass
profiles with examples of how to use them either as instances to do things like make images / convergence maps
or how to compose them as a model.

Can you make a guide for both workspaces, in scripts/guides/profiles/mass.py which gfives examples for all
support mass profiles, where the first bit gives exampels for all mass profiles in one section but does not
use them to make images (e.g. high level run through), it then shows the API for making an image from one
mass profile (e.g. Sersic).

it then shows the API for how to put this in a model (e.g. Sersic again),
it then shows the API for making an instance from this Sersic model, either from just the mass profile or when its
in the galaxies model the modeling API shows, and then does a detailed run through of
all remainig mass profiles but only showing their image_2d_from methods, emphaissing the API above is
translateable.

For the autolens_workspace this will use the Tracer object instead of Galaaxies object and put the model gernerated
instances in the Tracer where appropriate.

Do a comparison to the docs folder in PyAutoGalaxy and PyAutoLens to make sure all mass profiles are in
sync and also point to the URL to the docs early on in this script.

This should be paried nicely against the light.py example in the same folder