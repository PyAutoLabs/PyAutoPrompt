Do deep research and think hard on planning this one.

I am going to add weak gravitational lens analysis to autolens, whcih is going to be a series of 5 or so 
claude prompts to get us there.

First, we need to be able to simulate weak lensing data. The good news is, the ability to generate a shear catalogue 
from the source code is already possible. All we need to do is use the shear_yx_2d_via_hessian_from in the
module @PyAutoGalaxy/autogalaxy/operate/lens_calc.py. This returns the shear field which can then be the data
in a simulator.

However, we still need to add noise and do other steps required for a simulator. Thus, can you inspect
@autolens_workspace/scripts/imaging/simulator.py and see how simulations are performed (in this case for CCD imaging
and adapt this to a simulation of a weak lensing shear catalogue?

This returns a shear field defined in @PyAutoGalaxy/autogalaxy/util/shear_field.py, which is how we store
shear fields of data in general.

Can you use an Isothermal profile, which means you can use the method shear_yx_2d_from in the profiles/mass/total/isothermal.py
module. This should have the same outptu as shear_yx_2d_via_hessian_from, but if there is no unit test afgainst that
please add one.

This request will likely require you to add noise sources to the simualted shear field. please look up how to do
this and give me suggestions, I suspect its just random noise. However, we all need a random distribution of background
galaxies whose weak lensing shears we are measuring. Have a look at how to do this, but just drawing a uniform
random is fine for now if needs be.  I guess we may end up adding a SimulatorShearYX to the source code to match
the API of other dataset types (e.g. imaging).

This therefore means a Dataset class for the ShearYX2D field will also probably need to be made, which will include
the noise_map of the dataset. I think for now you can look to @PyAutoLens/autolens/point/dataset.py for inspiriation
on the format, noting that we will likely extend this soon to include a .csv layer for large shear fields
(and maybe point dataset). SO, its a good API to base this on. We should call this WeakDataset, noting it contains
ShearYX things.

The simulator script wont be able to output visualization as those tools dont yet exists. Dont put them in this
chunk of work, but create a prompt 2_visualization.md which based on the simulator creates the visualization tools.
So put place holders ready to replace in the simulator. Put in the prompt that visualization will like user the
matplotlib quiver method to best represent weak lesning ellipticities and shears.

Can we also put some unit tests and integration tests on this?