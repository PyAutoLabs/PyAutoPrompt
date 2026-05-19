For autolens_workspace/scripts/cluster/simulator.py, can we make it so that all parameters are in .csv form
and loaded from there, to establish that the base way to interact with the autolens API for clusters
is via csv.

I thin kthe way to make this work is to write a guide, autolens_workspace/scripts/cluster/csv_api.py,
which illustrates how to set up lens models using the normal autolens API and then output them to csv,
and showing how all those featres linked together.

This csv file can then act as an "Auto Simulate" type siutation for simulator.py, which loads the csv outputsof this file.
The simulator will put the csv files int he lens it simulates at the end, meaning other scripts only need the
auto simulate performed here.

Things I am still unclear on that this guide could help with are:

Shouild main galaxies, extra galaxies and scaling galaies use their own csv files or can they all be combined into one?
I think it would be good if they could all be combined into one, but hiswould mean the .csv needs to know a lot more
than just parameters, but feasible mass profile class, lens name (e.g. when its used to name light and mass profiles in the Galaxy), redshift and
others. I think I like the idea of a single .csv API being used for all cluster interfaces.

The flip side is this could get complicated because if the same galaxy has light and mass profiles then the notion of column
heads breaks down, so maybe the rule is "one csv file per light or mass profile", and the reuse of light profile names, mass profile
names and galaxy names is exploited when building the model? Most cluster models will apply the same thing over loads
of galaxies so I think that works, so we can just build it in an extensible way.

This would mean it also needs the galaxy names, even though in simulator.py galaxies are not named when used in a Tracer
these names would be used for performing model composition, again the csv_api.py script could explain and cover this.

I would then go so far as to make it so that this guide also explains point_datasets.csv, which functionally looks a lot
more complete to me and just needs an explanation. I would explain this before doing galaxy API, as convention is normally load
dataset before modeling. Note that point_datasets.csv itself is made by simulator.py, thus I think csv_api.py 
can just make an example one which is not paired to the model in the guide making it clear its for illustrative purposes
but that simultor.py makes the actual one.

There are also no .csv's used at all for defining the point source model, which obviously need to be paired
with the point dataset. 

This:

"""
__Source Galaxies__

The 2 background sources at *different* redshifts. Each carries a `SersicCore` light profile (used only
for visual confirmation of the lensed arcs — the cored profile changes gradually in the centre so explicit
source-plane over-sampling is unnecessary) and a `Point` model component whose multiple-image positions
we solve for and use as the modeling data.

Each source's redshift is taken from ``source_redshifts``, so source 0 sits at ``z = 1.0`` and source 1
at ``z = 2.0``. The `Tracer` ray-traces multi-plane through both planes automatically.
"""
source_galaxies = []
for i, (centre, src_z) in enumerate(zip(source_centres, source_redshifts)):
    bulge = al.lp.SersicCore(
        centre=centre,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
        intensity=2.0,
        effective_radius=0.3,
        sersic_index=1.0,
    )
    point = al.ps.Point(centre=centre)
    source_galaxies.append(
        al.Galaxy(redshift=src_z, bulge=bulge, **{f"point_{i}": point})
    )


And this:

_source_models = [
    af.Model(
        al.Galaxy,
        redshift=src_z,
        bulge=af.Model(
            al.lp.SersicCore,
            centre=src_centre,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
        **{
            f"point_{i}": af.Model(al.ps.Point, centre=src_centre),
        },
    )
    for i, (src_centre, src_z) in enumerate(zip(source_centres, source_redshifts))
]

Shoiuld all be put into two csv files (source_point_models.csv and source_light_models.csv) again making the whole
cluster experience a first-class csv experience. 

I guess at this point users should easily not just load .csv's into models but also be able to print the csv
contents in python / Notebook cells and have clear print statements of the loaded objects showing how their
csv load parameters link to autolens objects, I guess the csv_api does that.

Finally, do a quick scan through autolens_workspace of other csv uses but I think at the moment its just scaling galaxies
which are already implemented wellk..

Do deep research when coming to this csv API and once you're happy with it make csv interface the version used on all 3 cluster
scripts that exist. This is a huge issue -- the csv interface defines cluster modeling throughout, so dont be afraid
to ask hard questions about balancing the need for modeling large amoiunts of gaalxies to making a user friendly API.
Feel free to ask if some of this work would benefit going into the source code more so than it alredy has.