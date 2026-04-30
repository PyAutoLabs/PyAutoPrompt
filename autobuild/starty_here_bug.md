Run welcome.py and start_here.py in all workspsaces, look for the bug below and any others.

##############################
## LIGHT AND MASS PROFILES ###
##############################


    The image displayed on your screen shows a `LightProfile`, the object PyAutoLens uses to represent the
    luminous emission of galaxies. This emission is unlensed, which is why it looks like a fairly ordinary and
    boring galaxy.

    To perform ray-tracing, we need a `MassProfile`, which will be shown after you push [Enter]. The figures will
    show deflection angles of the `MassProfile`, vital quantities for performing lensing calculations.

    [Press Enter to continue]

Traceback (most recent call last):
  File "/tmp/autolens_workspace_verify_A_20260429_214047/welcome.py", line 109, in <module>
    deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
                    ^^
NameError: name 'aa' is not defined. Did you mean: 'al'?