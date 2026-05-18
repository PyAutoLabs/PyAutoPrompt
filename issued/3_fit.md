We are now going to add weak lensing Fit class

First, we need to create the fit.py module, so inspect @autolens_workspace/scripts/weak and
@PyAutoLens/autolens/imaging/fit modules . We are basically going to make everything weak does from here a "mirror" of
the imaging model API (and also interferoter.)

So, set up a FitWeak module which compute the same key quantities as other Fit objects, such as residuals, chi
squared and log likelihood. Put unit tests in following the imaging unit test stucture, baring in mind there should
be far fewer as there are no variants like linear light profiles of pixelizations. Have me eyeball a few unit tests
so I can see they make sense. 

In a second phase, inspect @PyAutoLens/autolens/plot and set up the FitWeak weak_plots.py, you may need to do
some research on how best to plot these quantities, it may be we want to plot the dataset values via quiver on top
of the model ones.