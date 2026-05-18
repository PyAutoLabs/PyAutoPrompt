The API for the placement of galaxies on a mass light scaling relation is given autolens_workspace/scripts/group/scaling_relation.

For group scale lenses, this is an optional feature which group scale modeling odes not necessarily neeed.

For cluster lenses, I want it to be the default, with all default cluster scripts having 10 scaling galaxies 
whose masses are lower than the main lens and dark matter halo. I want them in that regime of have to be modeled collectively,
but dont contribute much individually.

Can you upddate the 3 cluster examples on the autolens workspace to this effect, always using the .csv
interface for scaling galaxies. Can you make sure the simulator still produces multiple images in the right
positions/