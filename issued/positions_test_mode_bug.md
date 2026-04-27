The following error is caused when in PYAUTO_TEST_MODE=2 the mass model is some random awful garbage that does 
not creatr positions that are good or physical or reliable for the lens model:

((PyAuto) ) jammy@DESKTOP-H143S82:~/Code/PyAutoLabs/autolens_workspace$ PYAUTO_TEST_MODE=2 python scripts/imaging/features/pixelization/delaunay.py

Traceback (most recent call last):
  File "/home/jammy/Code/PyAutoLabs/autolens_workspace/scripts/imaging/features/pixelization/delaunay.py", line 983, in <module>
    source_pix_result_1 = source_pix_1(
                          ^^^^^^^^^^^^^
  File "/home/jammy/Code/PyAutoLabs/autolens_workspace/scripts/imaging/features/pixelization/delaunay.py", line 655, in source_pix_1
    source_lp_result.positions_likelihood_from(
  File "/home/jammy/Code/PyAutoLabs/PyAutoLens/autolens/analysis/result.py", line 333, in positions_likelihood_from
    threshold = self.positions_threshold_from(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jammy/Code/PyAutoLabs/PyAutoLens/autolens/analysis/result.py", line 254, in positions_threshold_from
    threshold = factor * np.nanmax(positions_fits.max_separation_of_plane_positions)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jammy/Code/PyAutoLabs/PyAutoLens/autolens/point/max_separation.py", line 87, in max_separation_of_plane_positions
    return max(self.furthest_separations_of_plane_positions)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jammy/Code/PyAutoLabs/PyAutoLens/autolens/point/max_separation.py", line 83, in furthest_separations_of_plane_positions
    return self.plane_positions.furthest_distances_to_other_coordinates
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jammy/Code/PyAutoLabs/PyAutoArray/autoarray/structures/grids/irregular_2d.py", line 250, in furthest_distances_to_other_coordinates
    furthest = self._xp.sqrt(self._xp.nanmax(sq_dists, axis=1))
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py", line 486, in nanmax
    res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation fmax which has no identity


Can you make it so that a check is performed to see if the positions computed in positions_likelihood_from are
sensible or exist or not nans (check what they are when this script runs) and put in a PYAUTO_TEST_MODE fix
which simply converts the positions to (1.0, 0.0) and (-1.0, 0.0), the rest of the code should then run functionally
as expected.




can you combine this solution with another relatrd issue I drafted:

This is in group/simulator.py and annoying for users:


small_datasets = os.environ.pop("PYAUTO_SMALL_DATASETS", None)

solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1),
    pixel_scale_precision=0.001,
    magnification_threshold=0.01,
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

if small_datasets is not None:
    os.environ["PYAUTO_SMALL_DATASETS"] = small_datasets

Can you make it so if PYAUTO_SMALL_DATASETS=1, the PointSOlver knows to just return two positions at some
small but convenient values (e.g. (1.0, 0.0), (0.0, 1.0)), so that testing doesnt break.