We are going to refactor the mass profile module to work with the following assumptions:

1) All mass profiles must have a deflections_yx_2d_from method defined, as they do now.
2) The best method they can use is an analytic function, however this could instead be an MGE decomposition
or CSE decomposition.
3) The potential_2d_from and convergence_2d_from methods can, optionally, be defined inside mass profiles. 
This is common if they have analytic solutions which are quick to comptue.

However, a potential or convergence does not need to be defined anlytically. In this case, we will "fall back to
an MGE or CSE method", which computes the potential via those funcgtions. These methods sitll need a dediciated
convergence function for the decomposition though, which not all functions have. Therefore,we will draw me up
a list of mass profiles, which currently need this MGE / CSE treatment applied to them. The red flag to catch this
is those where the potential_2d_from or convergence_2d_from method returns a numpy array of zeros. Once
we have this list, we'll address each profile one-by-one as best we can.

Do deep research on autoglaxy/profiles/mass to fully understand this design. Think hard about the transformer
and data structure decorators, which do things like allow a user to both inputs uniform grids which
map to and from a map and irregular grids for doing cauclalutions at specific points -- especially important
for point source modeling already used.

In autolens_workspace_test, I think you should esign a test set in scripts/mass, which runs on every mass profile 
individually, using a methods / algric equations indepedent of the source code implmenetation, verifies their 
convergence, potential, deflections, shear and any other key quantity are fully self consistent INTERNALLY WITH
THEMSELVES. This would ask as a key sanity check throughout development. This should be top of the development phases. This can run
slow as a first pass as numerical accuracy is key, but maybe we make environment variable PYAUTO_MASS_FAST
which ensures these run fast for testing in general.

The MGE mass profile module is mature and allows us to compute deflection angles using a multip Gaussian epxansion.
We will look att his module in more detail for any refactor / design choies, but most importantt then redesign
the CSE module to also be fully jax'd. For now, our goal will be to just have the existing implementations which
use CSE support JAX. This should come before the major redesign, as the "fall back to MGE / CSE method" requires
that both are implemented. So have a whole feature scoped out for CSE implmeentation.

Check that the CSE and MGE have fast and robust potential_2d_from methods.

The first milestone is to have the really extensive testin strcutre in autolens_workspace_test up and running,
and this should be extensive, it sould run for EVERY mass profile and not mush together ellitpcal and spherical.
I guess really, it should use one generic method for lensing which goes from one quantitiy (e.g. convegence) forward,
whichc an then compare to the deflections-first source code implementation.

Many methods in the source code are bypassed due laziness, often having an np.zeros for their convergence or potential
I guess the MGE / CSE trick will sort these but look out foir it.

A final task will be first class documentation on all functions, including latex readable code in docstrings.

Also, a feature should be o a "spring clean" of dodgy code and other improvemenets.

So yeah, in conclusion, break this down into phased tasks in z_features that begins to fully clean up and sort out
our mass profiles module. Do deep research and thorough planning before doing this, perhaps reading some scientific context
from @PyAutoPaper .
