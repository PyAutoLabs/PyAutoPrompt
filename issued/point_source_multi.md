The PointDataset object in @PyAutoLens/autolens/point was designed with single lensed point sources in mind, which means
single inputs or a small  number of inputs are used to use it, see @autolens_workspace/scripts/point_source/simulator.py .

To model multiple point sources, we use a list of PointDatasets, which would be input into an AnalysisFactorGraph
which is illustrated in @autolens_workspace/scripts/multi/modeling.py. 

We have an example which simulates a lens with two point sourcs @autolens_workspace/scripts/point_source/features/double_einstein_cross
but it does not model it. Can you basically make it so the simulated data is modeled using the multi API? And, whilst we're 
here, can you rename it from double_einstein_cross to just multiple_sources, which is more generic.